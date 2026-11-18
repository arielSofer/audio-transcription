"""Progressive weights: a simulation of a nested bit-plane weight format.

Idea (Netflix-for-weights / virtual textures): every linear layer keeps a cheap
low-bit BASE in RAM. The remaining bit-planes ("refinement tiles") live on the
SSD. For every token we pick the tiles of input channels whose activations are
largest and add their refinement:

        y = W_base @ x  +  R[:, S_t] @ x[S_t]        (R = W_8bit - W_base)

This file only SIMULATES the numerics and the I/O accounting (which tiles would
have to be read, how many bytes, what a cache would absorb). It does not touch a
real disk. See README.md for what was measured and what was not.
"""
from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

GROUP = 128  # quantization group (one scale/min per output row per 128 input channels)
TILE = 32    # input channels per refinement tile. Kept small so a 0.5B model has about as many
#              tiles per matrix (28 for hidden=896) as a 70B model has with 256-channel tiles (32).


# --------------------------------------------------------------------------
# nested quantization: 8-bit grid, the top `bits` bits are the base
# --------------------------------------------------------------------------
def quant8(W: torch.Tensor, group: int = GROUP):
    out, inn = W.shape
    ng = inn // group
    Wg = W.reshape(out, ng, group)
    mn = Wg.amin(-1, keepdim=True)
    mx = Wg.amax(-1, keepdim=True)
    scale = (mx - mn).clamp_min(1e-8) / 255.0
    q = torch.round((Wg - mn) / scale).clamp_(0, 255)
    return q.to(torch.uint8), scale, mn


def deq(q, scale, mn, bits: int = 8):
    """Reconstruct from the top `bits` bits of the 8-bit code (mid-point)."""
    step = float(2 ** (8 - bits))
    qf = q.float()
    mid = torch.floor(qf / step) * step + (step - 1.0) / 2.0
    return mn + mid * scale


# --------------------------------------------------------------------------
# controller: global knobs + statistics shared by every module
# --------------------------------------------------------------------------
class Controller:
    def __init__(self):
        self.mode = "base"      # base | full | calib | dynamic | static | random | policy
        self.frac = 0.0         # fraction of tiles refined per token
        self.budget = None      # policy: max NEW tiles fetched per token per group
        self.cap_mult = None    # policy: cache capacity in units of k (tiles kept resident)
        self.bits = 4
        self.reset_stats()

    def reset_stats(self):
        self.tokens = 0
        self.fetch_bytes = 0.0
        self.refine_bytes = 0.0     # bytes of the tiles actually used (no cache)
        self.overlap = {}           # kind -> [sum, n]
        self.energy = {}            # kind -> [sum, n]  (fraction of score^2 covered)
        self.used_frac = {}         # kind -> [sum, n]  (policy: fraction of desired tiles served)


# --------------------------------------------------------------------------
# a group of linears that read the SAME input x (q/k/v, gate/up, o, down).
# They share one tile selection -> a tile on disk is [128 input ch x all outs].
# --------------------------------------------------------------------------
class TileGroup:
    def __init__(self, kind: str, ctl: Controller):
        self.kind = kind
        self.ctl = ctl
        self.members: list[ProgLinear] = []
        self.rn = None
        self.mean_xe = None
        self.calib_sum = None
        self.calib_n = 0
        self._x = None
        self._mask = None

    @property
    def nt(self):
        return self.members[0].nt

    @property
    def out_total(self):
        return sum(m.out_features for m in self.members)

    def refresh_norms(self):
        self.rn = torch.sqrt(sum(m.rn ** 2 for m in self.members))

    def tile_bytes(self, bits):
        return TILE * self.out_total * (8 - bits) / 8.0

    # ---- calibration (static baseline) ----
    def _accumulate(self, xe):
        s = xe.sum(0)
        self.calib_sum = s if self.calib_sum is None else self.calib_sum + s
        self.calib_n += xe.shape[0]
        self.mean_xe = self.calib_sum / self.calib_n

    def _stat(self, d, key, val, n=1):
        a = d.setdefault(key, [0.0, 0])
        a[0] += float(val)
        a[1] += n

    def mask_for(self, x: torch.Tensor):
        """Return a float mask [..., nt] for input x (cached per tensor object)."""
        if self._x is x:
            return self._mask
        ctl = self.ctl
        nt = self.nt
        lead = x.shape[:-1]
        xe = x.reshape(-1, nt, TILE).pow(2).sum(-1).sqrt()          # [N, nt]
        N = xe.shape[0]
        if ctl.mode == "calib":
            self._accumulate(xe)
            mask = torch.ones(N, nt)
        else:
            k = min(nt, max(1, int(round(ctl.frac * nt))))
            score = xe * self.rn
            if ctl.mode == "static":
                sel = torch.topk(self.mean_xe * self.rn, k).indices
                mask = torch.zeros(nt)
                mask[sel] = 1.0
                mask = mask.expand(N, nt).contiguous()
            elif ctl.mode == "random":
                idx = torch.rand(N, nt).topk(k, -1).indices
                mask = torch.zeros(N, nt).scatter_(-1, idx, 1.0)
            else:  # dynamic / policy: top-k by |x_tile| * ||R_tile||
                idx = score.topk(k, -1).indices
                D = torch.zeros(N, nt).scatter_(-1, idx, 1.0)
                s2 = score.pow(2)
                cov = ((s2 * D).sum(-1) / s2.sum(-1).clamp_min(1e-30)).mean()
                self._stat(ctl.energy, self.kind, cov)
                if lead[0] == 1 and N > 1:
                    ov = ((D[1:] * D[:-1]).sum(-1) / k).mean()
                    self._stat(ctl.overlap, self.kind, ov)
                if ctl.mode == "policy":
                    mask = self._policy(score, D, k, N)
                else:
                    mask = D
            ctl.refine_bytes += float(mask.sum()) * self.tile_bytes(ctl.bits)
        self._x, self._mask = x, mask.reshape(*lead, nt)
        return self._mask

    # ---- never-wait policy: bounded fetch budget + LRU cache of resident tiles ----
    def _policy(self, score, D, k, N):
        ctl = self.ctl
        sc = score.numpy()
        Dn = D.numpy().astype(bool)
        nt = self.nt
        cap = None if ctl.cap_mult is None else int(round(ctl.cap_mult * k))
        budget = ctl.budget
        used = np.zeros_like(Dn)
        in_cache = np.zeros(nt, bool)
        last = np.full(nt, -1, np.int64)
        fetched = 0
        wanted = 0
        served = 0
        for t in range(N):
            d = np.flatnonzero(Dn[t])
            hit = d[in_cache[d]]
            miss = d[~in_cache[d]]
            if budget is not None and miss.size > budget:
                miss = miss[np.argsort(-sc[t, miss], kind="stable")[:budget]]
            used[t, hit] = True
            used[t, miss] = True
            fetched += miss.size
            wanted += d.size
            served += hit.size + miss.size
            last[hit] = t
            last[miss] = t
            in_cache[miss] = True
            if cap is not None:
                cnt = int(in_cache.sum())
                if cnt > cap:
                    idx = np.flatnonzero(in_cache)
                    drop = idx[np.argsort(last[idx], kind="stable")[: cnt - cap]]
                    in_cache[drop] = False
        ctl.fetch_bytes += fetched * self.tile_bytes(ctl.bits)
        self._stat(ctl.used_frac, self.kind, served / max(wanted, 1))
        return torch.from_numpy(used.astype(np.float32))


class ProgLinear(nn.Module):
    def __init__(self, lin: nn.Linear, group: TileGroup, ctl: Controller):
        super().__init__()
        W = lin.weight.data.float()
        self.out_features, self.in_features = W.shape
        assert self.in_features % GROUP == 0 and GROUP % TILE == 0, self.in_features
        self.nt = self.in_features // TILE
        q, scale, mn = quant8(W)
        self.register_buffer("q", q)
        self.register_buffer("scale", scale)
        self.register_buffer("mn", mn)
        self.bias = None if lin.bias is None else nn.Parameter(lin.bias.data.float(), requires_grad=False)
        self.ctl = ctl
        self.group = group
        group.members.append(self)
        self.set_bits(ctl.bits)

    def q_tiles(self):
        """8-bit codes as [out, n_tiles, TILE] (a view; the on-disk tile unit)."""
        return self.q.reshape(self.out_features, self.nt, TILE)

    def set_bits(self, bits: int):
        shape = (self.out_features, self.in_features)
        w8 = deq(self.q, self.scale, self.mn, 8).reshape(shape)
        base = deq(self.q, self.scale, self.mn, bits).reshape(shape)
        self.register_buffer("base", base, persistent=False)
        self.register_buffer("R", w8 - base, persistent=False)
        self.rn = self.R.reshape(self.out_features, self.nt, TILE).pow(2).sum((0, 2)).sqrt()

    def forward(self, x):
        y = F.linear(x, self.base, self.bias)
        mode = self.ctl.mode
        if mode == "base":
            return y
        if mode == "full":
            return y + F.linear(x, self.R)
        mask = self.group.mask_for(x)                      # [..., nt]
        xm = (x.reshape(*x.shape[:-1], self.nt, TILE) * mask.unsqueeze(-1)).reshape(x.shape)
        return y + F.linear(xm, self.R)


# --------------------------------------------------------------------------
# model surgery + evaluation
# --------------------------------------------------------------------------
def convert_model(model, ctl: Controller):
    groups = []
    for layer in model.model.layers:
        a, m = layer.self_attn, layer.mlp
        spec = [("qkv", [(a, "q_proj"), (a, "k_proj"), (a, "v_proj")]),
                ("o", [(a, "o_proj")]),
                ("gu", [(m, "gate_proj"), (m, "up_proj")]),
                ("down", [(m, "down_proj")])]
        for kind, members in spec:
            g = TileGroup(kind, ctl)
            for parent, name in members:
                setattr(parent, name, ProgLinear(getattr(parent, name), g, ctl))
            groups.append(g)
    for g in groups:
        g.refresh_norms()
    return groups


def set_bits(groups, ctl: Controller, bits: int):
    ctl.bits = bits
    for g in groups:
        for m in g.members:
            m.set_bits(bits)
        g.refresh_norms()
        g._x = None


@torch.no_grad()
def nll_of(model, windows):
    tot, n = 0.0, 0
    for w in windows:
        logits = model(w).logits
        tot += F.cross_entropy(logits[0, :-1].float(), w[0, 1:], reduction="sum").item()
        n += w.shape[1] - 1
    return tot, n
