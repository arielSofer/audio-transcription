"""Progressive weights on MLX with REAL file I/O (a prototype runtime for a Mac).

What is real here: the 4-bit base lives in RAM in MLX's own quantized format and runs through
mx.quantized_matmul; the refinement bit-planes live in a file on disk (tiles of 32 input channels x all
output rows, 4096-byte aligned) and are read with pread (F_NOCACHE on macOS) by a thread pool; tiles are chosen
per token from the activation, kept in an LRU cache, with a fetch budget and a deadline (late tiles are skipped
this token and land in the cache for the next one).

What is NOT optimized: the tile selection needs the activation on the CPU, so every projection group costs one
GPU->CPU sync, and the small refinement matmul runs in numpy. It decodes ONE token at a time (prompt tokens are fed
one by one too). It measures the real end-to-end overhead of the idea; it is not a fast engine.

  python mlx_engine.py convert --model HF_DIR --out OUT_DIR           (offline, once)
  python mlx_engine.py bench   --model HF_DIR --out OUT_DIR --data wikitext2_test.parquet
"""
from __future__ import annotations

import argparse
import collections
import json
import math
import os
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, wait

import numpy as np
import mlx.core as mx
import mlx.nn as nn

import tilefile as TF

GROUP = 128   # quantization group = MLX group_size
TILE = 32     # input channels per refinement tile
ALIGN = 4096


# ----------------------------------------------------------------------------------------------
# offline conversion
# ----------------------------------------------------------------------------------------------
def quant8(W):
    out, inn = W.shape
    ng = inn // GROUP
    Wg = W.reshape(out, ng, GROUP)
    mn = Wg.min(-1, keepdims=True)
    mxv = Wg.max(-1, keepdims=True)
    scale = np.maximum(mxv - mn, 1e-8) / 255.0
    q = np.clip(np.round((Wg - mn) / scale), 0, 255).astype(np.uint8)
    return q.reshape(out, inn), scale[..., 0].astype(np.float32), mn[..., 0].astype(np.float32)


def pack4(qb):
    """[out, in] values 0..15 -> MLX uint32 packing (8 values per word, first value in the lowest nibble)."""
    out, inn = qb.shape
    v = qb.reshape(out, inn // 8, 8).astype(np.uint32)
    packed = np.zeros((out, inn // 8), np.uint32)
    for j in range(8):
        packed |= v[..., j] << (4 * j)
    return packed


def load_weights(model_dir):
    w = {}
    for f in sorted(os.listdir(model_dir)):
        if f.endswith(".safetensors"):
            w.update(mx.load(os.path.join(model_dir, f)))
    return w


def convert(model_dir, out_dir, n_layers=None):
    os.makedirs(out_dir, exist_ok=True)
    w = load_weights(model_dir)
    cfg = json.load(open(os.path.join(model_dir, "config.json")))
    n_layers = n_layers or cfg["num_hidden_layers"]
    arrays, groups = {}, []
    off = 0
    index = {"bits": 4, "tile": TILE, "align": ALIGN, "groups": {}}
    t0 = time.time()
    with open(os.path.join(out_dir, "tiles.bin"), "wb") as tf:
        for i in range(n_layers):
            p = f"model.layers.{i}."
            spec = [("qkv", [p + "self_attn.q_proj", p + "self_attn.k_proj", p + "self_attn.v_proj"]),
                    ("o", [p + "self_attn.o_proj"]),
                    ("gu", [p + "mlp.gate_proj", p + "mlp.up_proj"]),
                    ("down", [p + "mlp.down_proj"])]
            for kind, members in spec:
                name = f"L{i}.{kind}"
                rs, scale8s = [], []
                for m in members:
                    W = np.array(w[m + ".weight"].astype(mx.float32))
                    q, scale8, mn = quant8(W)
                    qb = q >> 4
                    scales4, biases4 = 16.0 * scale8, mn + 7.5 * scale8
                    arrays[m + ".wq"] = pack4(qb)
                    arrays[m + ".s4"] = scales4
                    arrays[m + ".b4"] = biases4
                    arrays[m + ".scale8"] = scale8
                    if m + ".bias" in w:
                        arrays[m + ".bias"] = np.array(w[m + ".bias"].astype(mx.float32))
                    rs.append(q & 15)
                    scale8s.append(scale8)
                codes = np.concatenate(rs, 0)                      # [out_total, in]
                sc = np.concatenate(scale8s, 0)                    # [out_total, ng]
                out_total, inn = codes.shape
                nt = inn // TILE
                raw = out_total * TILE // 2
                stride = (raw + ALIGN - 1) // ALIGN * ALIGN
                rn = np.zeros(nt, np.float32)
                for t in range(nt):
                    c = codes[:, t * TILE:(t + 1) * TILE]
                    r = (c.astype(np.float32) - 7.5) * sc[:, t * TILE // GROUP][:, None]
                    rn[t] = np.sqrt((r * r).sum())
                    blob = ((c[:, 0::2] << 4) | c[:, 1::2]).astype(np.uint8).tobytes()
                    tf.write(blob + b"\0" * (stride - raw))
                index["groups"][name] = dict(base=off, stride=stride, raw=raw, nt=nt, out_total=out_total)
                off += stride * nt
                arrays[name + ".rn"] = rn
            print(f"  layer {i} converted ({time.time() - t0:.0f}s)", flush=True)
    index["file_bytes"] = off
    json.dump(index, open(os.path.join(out_dir, "tiles.bin.json"), "w"))
    np.savez(os.path.join(out_dir, "members.npz"), **arrays)
    # check the MLX packing convention on one matrix: dequantize(packed) must equal the intended base
    m = "model.layers.0.self_attn.o_proj"
    W = np.array(w[m + ".weight"].astype(mx.float32))
    q, scale8, mn = quant8(W)
    base = (mn[..., None] + ((q.reshape(W.shape[0], -1, GROUP) >> 4) * 16 + 7.5) * scale8[..., None]).reshape(W.shape)
    deq = np.array(mx.dequantize(mx.array(arrays[m + ".wq"]), mx.array(arrays[m + ".s4"]), mx.array(arrays[m + ".b4"]),
                                 group_size=GROUP, bits=4))
    err = np.abs(deq - base).max() / np.abs(base).max()
    print(f"packing check: max relative error {err:.2e} (must be ~1e-6)")
    assert err < 1e-4, "MLX packing convention mismatch"
    print(f"wrote {index['file_bytes'] / 1e6:.0f} MB of tiles and {len(arrays)} arrays to {out_dir}")


# ----------------------------------------------------------------------------------------------
# runtime
# ----------------------------------------------------------------------------------------------
class Settings:
    mode = "dyn"          # base | dyn | full
    frac = 0.1
    budget = None         # max NEW tiles fetched per token per group, as a fraction of k (None = unlimited)
    cap_mult = 2.0        # cache capacity in multiples of k (None = unlimited)
    deadline = None       # seconds to wait for tiles per group (None = wait for all requested)
    plan_len = 64         # mode "plan": decide ONCE after this many prompt tokens, then no per-token decisions
    plan_refresh = 0      # mode "plan": rebuild the plan every N tokens from the last N tokens (0 = never)
    plan_quant = False    # mode "plan": keep the chosen residual as 4-bit (mx.quantized_matmul) instead of fp16
    dtype = None


class Stats:
    def __init__(self):
        self.reset()

    def snapshot(self):
        return dict(vars(self))

    def diff(self, snap):
        return {k: getattr(self, k) - v for k, v in snap.items()}

    def reset(self):
        self.tokens = 0
        self.wanted = 0
        self.hits = 0
        self.requested = 0
        self.served = 0
        self.bytes_read = 0
        self.t_select = 0.0
        self.t_io_wait = 0.0
        self.t_refine = 0.0
        self.t_rebuild = 0.0
        self.rebuilds = 0


class Store:
    """LRU tile cache over the tile file; misses are read by a thread pool."""

    def __init__(self, path, st: Settings, stats: Stats, workers=8):
        self.tf = TF.TileFile(path, direct=True)
        self.st, self.stats = st, stats
        self.pool = ThreadPoolExecutor(workers)
        self.cache: dict[str, collections.OrderedDict] = {}
        self.inflight: dict = {}
        self.lock = threading.Lock()

    def _read(self, name, t):
        blob = self.tf.read_tile(name, t)
        with self.lock:
            self.cache.setdefault(name, collections.OrderedDict())[t] = blob
            self.inflight.pop((name, t), None)
            self.stats.bytes_read += len(blob)

    def fetch(self, name, wanted, k):
        st, stats = self.st, self.stats
        cache = self.cache.setdefault(name, collections.OrderedDict())
        with self.lock:
            hits = [t for t in wanted if t in cache]
            miss = [t for t in wanted if t not in cache]
        if st.budget is not None:
            miss = miss[: int(round(st.budget * k))]
        futs = []
        for t in miss:
            with self.lock:
                f = self.inflight.get((name, t))
                if f is None:
                    f = self.pool.submit(self._read, name, t)
                    self.inflight[(name, t)] = f
            futs.append(f)
        t0 = time.perf_counter()
        if futs:
            wait(futs, timeout=st.deadline)
        stats.t_io_wait += time.perf_counter() - t0
        with self.lock:
            avail = [t for t in wanted if t in cache]
            for t in avail:
                cache.move_to_end(t)
            if st.cap_mult is not None:
                cap = int(round(st.cap_mult * k))
                while len(cache) > cap:
                    cache.popitem(last=False)
            blobs = [(t, cache[t]) for t in avail if t in cache]
        stats.wanted += len(wanted)
        stats.hits += len(hits)
        stats.requested += len(miss)
        stats.served += len(blobs)
        return blobs


class ProgGroup:
    """q/k/v (or gate/up, o, down): linears that read the same activation share one tile selection."""

    def __init__(self, name, rn, scale8, store: Store, st: Settings, stats: Stats):
        self.name, self.rn, self.scale8 = name, rn, scale8        # scale8 [out_total, ng] (numpy)
        self.nt = len(rn)
        self.store, self.st, self.stats = store, st, stats
        self.out_total = scale8.shape[0]
        self.reset_sequence()

    def reset_sequence(self):
        self.plan = None
        self.n_seen = 0
        self.score_sum = np.zeros(self.nt)
        self.acc = []
        self.since = 0
        self.plan_bytes = 0
        self._x = None
        self._y = None

    def _codes(self, blobs):
        m = len(blobs)
        codes = np.empty((self.out_total, TILE * m), np.uint8)
        for j, (t, blob) in enumerate(blobs):
            a = np.frombuffer(blob, np.uint8).reshape(self.out_total, TILE // 2)
            codes[:, j * TILE:(j + 1) * TILE:2] = a >> 4
            codes[:, j * TILE + 1:(j + 1) * TILE:2] = a & 15
        sc = self.scale8[:, [t * TILE // GROUP for t, _ in blobs]]          # one scale per (row, tile)
        return codes, sc

    def _decode(self, blobs):
        codes, sc = self._codes(blobs)
        return (codes.astype(np.float32) - 7.5) * np.repeat(sc, TILE, axis=1)   # residual R [out_total, 32m]

    def _xe(self, x):
        """Per-tile L2 norm of the activation, computed on the GPU (no sync)."""
        return mx.sqrt(mx.sum(mx.square(x.astype(mx.float32).reshape(self.nt, TILE)), axis=-1))

    def _build_plan(self, score):
        """Pick the k tiles with the highest score, read them (cache or SSD), keep the residual resident on the GPU.
        After this a token costs one gather + one small matmul: no sync, no I/O, no decisions."""
        st = self.st
        k = min(self.nt, max(1, int(round(st.frac * self.nt))))
        sel = np.argsort(-score, kind="stable")[:k].tolist()
        blobs = self.store.fetch(self.name, sel, k)
        if not blobs:
            self.plan, self.plan_bytes = ("empty",), 0
            return
        cols = mx.array(np.concatenate([np.arange(t * TILE, (t + 1) * TILE) for t, _ in blobs]).astype(np.int32))
        if st.plan_quant:      # 4-bit codes stay packed; MLX's fused dequantize-matmul reads them (group = one tile)
            codes, sc = self._codes(blobs)
            s = mx.array(sc).astype(st.dtype)
            wq, b = mx.array(pack4(codes)), -7.5 * s
            self.plan = ("q4", wq, s, b, cols)
            self.plan_bytes = wq.nbytes + s.nbytes + b.nbytes + cols.nbytes
        else:
            R_t = mx.array(np.ascontiguousarray(self._decode(blobs).T)).astype(st.dtype)
            self.plan = ("fp16", R_t, cols)
            self.plan_bytes = R_t.nbytes + cols.nbytes

    def _plan_apply(self, x):
        kind = self.plan[0]
        if kind == "empty":
            return None
        if kind == "fp16":
            _, R_t, cols = self.plan
            return mx.matmul(mx.take(x, cols, axis=-1), R_t)
        _, wq, s, b, cols = self.plan
        return mx.quantized_matmul(mx.take(x, cols, axis=-1), wq, s, b, transpose=True, group_size=TILE, bits=4)

    def refine(self, x):
        if self._x is x:
            return self._y
        st, stats = self.st, self.stats
        if st.mode == "plan" and self.plan is None and self.n_seen >= st.plan_len:
            self._build_plan(self.score_sum / max(self.n_seen, 1))
        if st.mode == "plan" and self.plan is not None:
            if st.plan_refresh:
                self.acc.append(self._xe(x))                 # stays on the GPU: no sync per token
                self.since += 1
                if self.since >= st.plan_refresh:             # one sync per refresh, then a new plan from the last N tokens
                    t0 = time.perf_counter()
                    xe = np.array(mx.mean(mx.stack(self.acc), axis=0))
                    self._build_plan(xe * self.rn)
                    stats.t_rebuild += time.perf_counter() - t0
                    stats.rebuilds += 1
                    self.acc, self.since = [], 0
            y = self._plan_apply(x)
            self._x, self._y = x, y
            return y
        t0 = time.perf_counter()
        xf = np.array(x.astype(mx.float32)).reshape(-1)             # GPU -> CPU sync
        xt = xf.reshape(self.nt, TILE)
        score = np.sqrt((xt * xt).sum(1)) * self.rn
        k = self.nt if st.mode == "full" else min(self.nt, max(1, int(round(st.frac * self.nt))))
        wanted = np.argsort(-score, kind="stable")[:k].tolist()
        stats.t_select += time.perf_counter() - t0
        if st.mode == "plan":                      # prompt phase: per-token selection, while collecting the mean score
            self.score_sum += score
            self.n_seen += 1
        blobs = self.store.fetch(self.name, wanted, k)
        t0 = time.perf_counter()
        y = None
        if blobs:
            R = self._decode(blobs)
            xs = np.concatenate([xt[t] for t, _ in blobs])
            y = R @ xs
        stats.t_refine += time.perf_counter() - t0
        self._x, self._y = x, y
        return y


class ProgLinear(nn.Module):
    def __init__(self, wq, s4, b4, bias, group: ProgGroup, lo, hi, st: Settings, dtype):
        super().__init__()
        self.wq = wq
        self.s4 = s4.astype(dtype)
        self.b4 = b4.astype(dtype)
        if bias is not None:
            self.bias = bias.astype(dtype)
        self._group, self._lo, self._hi, self._st, self._dtype = group, lo, hi, st, dtype

    def __call__(self, x):
        y = mx.quantized_matmul(x, self.wq, self.s4, self.b4, transpose=True, group_size=GROUP, bits=4)
        if "bias" in self:
            y = y + self.bias
        if self._st.mode != "base":
            yr = self._group.refine(x)
            if yr is not None:
                if isinstance(yr, np.ndarray):
                    y = y + mx.array(yr[self._lo:self._hi]).astype(self._dtype)
                else:
                    y = y + yr[..., self._lo:self._hi].astype(self._dtype)
        return y


def build_model(model_dir, out_dir, st: Settings, stats: Stats, dtype=mx.float32):
    from mlx_lm.utils import load_model
    model, _ = load_model(__import__("pathlib").Path(model_dir), lazy=False)
    model.set_dtype(dtype)
    arrs = np.load(os.path.join(out_dir, "members.npz"))
    store = Store(os.path.join(out_dir, "tiles.bin"), st, stats)
    st.dtype = dtype
    store.groups = []
    for i, layer in enumerate(model.model.layers):
        a, m = layer.self_attn, layer.mlp
        p = f"model.layers.{i}."
        spec = [("qkv", a, [("q_proj", p + "self_attn.q_proj"), ("k_proj", p + "self_attn.k_proj"), ("v_proj", p + "self_attn.v_proj")]),
                ("o", a, [("o_proj", p + "self_attn.o_proj")]),
                ("gu", m, [("gate_proj", p + "mlp.gate_proj"), ("up_proj", p + "mlp.up_proj")]),
                ("down", m, [("down_proj", p + "mlp.down_proj")])]
        for kind, parent, members in spec:
            name = f"L{i}.{kind}"
            scale8 = np.concatenate([arrs[k + ".scale8"] for _, k in members], 0)
            g = ProgGroup(name, arrs[name + ".rn"], scale8, store, st, stats)
            store.groups.append(g)
            lo = 0
            for attr, k in members:
                n = arrs[k + ".scale8"].shape[0]
                bias = mx.array(arrs[k + ".bias"]) if (k + ".bias") in arrs.files else None
                setattr(parent, attr, ProgLinear(mx.array(arrs[k + ".wq"]), mx.array(arrs[k + ".s4"]),
                                                 mx.array(arrs[k + ".b4"]), bias, g, lo, lo + n, st, dtype))
                lo += n
    mx.eval(model.parameters())
    return model, store


def teacher_forced(model, ids, stats: Stats, start=1):
    """Decode token by token with a KV cache. Only the tokens at positions >= start are scored and timed, and the
    returned stats delta covers only them (the first `start` tokens are the 'prompt')."""
    from mlx_lm.models.cache import make_prompt_cache
    cache = make_prompt_cache(model)
    nll, n, secs, snap = 0.0, 0, 0.0, None
    for t in range(len(ids) - 1):
        if t + 1 == start:
            snap = stats.snapshot()
        t0 = time.perf_counter()
        logits = model(mx.array([[ids[t]]]), cache=cache)[0, -1].astype(mx.float32)
        v = float((mx.logsumexp(logits) - logits[ids[t + 1]]).item())
        dt = time.perf_counter() - t0
        if t + 1 >= start:
            nll, n, secs = nll + v, n + 1, secs + dt
    return nll, n, secs, stats.diff(snap)


def bench(a):
    import pyarrow.parquet as pq
    from transformers import AutoTokenizer
    dtype = getattr(mx, a.dtype)
    st, stats = Settings(), Stats()
    model, store = build_model(a.model, a.out, st, stats, dtype)
    tok = AutoTokenizer.from_pretrained(a.model)
    text = "\n\n".join(pq.read_table(a.data).column("text").to_pylist())
    allids = tok(text).input_ids
    wins = [allids[i * 512:i * 512 + a.T] for i in range(a.windows)]     # same text positions as the simulation
    if a.suite == "drift":
        # the text changes topic at token a.drift_at: first part from the start of the test set, rest from far away
        OFF = 150000
        wins = [allids[i * 512:i * 512 + a.drift_at] + allids[OFF + i * 512:OFF + i * 512 + a.T - a.drift_at]
                for i in range(a.windows)]
        st.plan_len, start = a.plan_len, a.drift_at
        configs = [dict(mode="base"), dict(mode="dyn", frac=0.1, cap_mult=2.0), dict(mode="plan", frac=0.1),
                   dict(mode="plan", frac=0.1, plan_refresh=32), dict(mode="plan", frac=0.1, plan_refresh=64),
                   dict(mode="plan", frac=0.1, plan_refresh=64, plan_quant=True)]
    elif a.suite == "plan":
        st.plan_len, start = a.plan_len, a.plan_len
        configs = [dict(mode="base"), dict(mode="dyn", frac=0.1, cap_mult=2.0), dict(mode="plan", frac=0.1),
                   dict(mode="plan", frac=0.1, plan_quant=True),
                   dict(mode="dyn", frac=0.2, cap_mult=2.0), dict(mode="plan", frac=0.2),
                   dict(mode="plan", frac=0.2, plan_quant=True)]
    else:
        start = 1
        configs = [dict(mode="base"), dict(mode="full", cap_mult=None),
                   dict(mode="dyn", frac=0.1, cap_mult=None),
                   dict(mode="dyn", frac=0.1, cap_mult=2.0),
                   dict(mode="dyn", frac=0.1, cap_mult=2.0, budget=0.5),
                   dict(mode="dyn", frac=0.1, cap_mult=2.0, budget=0.25),
                   dict(mode="dyn", frac=0.1, cap_mult=2.0, deadline=0.0005)]
    if a.only:
        configs = [c for c in configs if c["mode"] in a.only.split(",")]
    out = []
    for c in configs:
        st.mode, st.frac, st.budget, st.cap_mult, st.deadline = c["mode"], c.get("frac", 0.1), c.get("budget"), c.get("cap_mult", 2.0), c.get("deadline")
        st.plan_refresh, st.plan_quant = c.get("plan_refresh", 0), c.get("plan_quant", False)
        tot, n, secs, agg = 0.0, 0, 0.0, collections.Counter()
        for w in wins:
            store.cache.clear()                    # every window starts with a cold cache and no plan
            for g in store.groups:
                g.reset_sequence()
            nll, k, s, d = teacher_forced(model, w, stats, start)
            tot, n, secs = tot + nll, n + k, secs + s
            agg.update(d)
        r = dict(config=c, scored_from_token=start, ppl=math.exp(tot / n), ms_per_token=1000 * secs / n,
                 tokens_per_s=n / secs,
                 read_mb_per_token=agg["bytes_read"] / n / 1e6,
                 io_wait_ms_per_token=1000 * agg["t_io_wait"] / n,
                 select_ms_per_token=1000 * agg["t_select"] / n,
                 refine_ms_per_token=1000 * agg["t_refine"] / n,
                 cache_hit_rate=(agg["hits"] / agg["wanted"]) if agg["wanted"] else None,
                 served_of_wanted=(agg["served"] / agg["wanted"]) if agg["wanted"] else None)
        r["plan_resident_mb"] = sum(g.plan_bytes for g in store.groups) / 1e6
        r["rebuild_ms_per_token"] = 1000 * agg["t_rebuild"] / n
        r["rebuilds_per_window"] = agg["rebuilds"] / len(wins)
        out.append(r)
        print(json.dumps({k: (round(v, 3) if isinstance(v, float) else v) for k, v in r.items()}), flush=True)
    json.dump(out, open(a.json, "w"), indent=2)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["convert", "bench"])
    ap.add_argument("--model", required=True, help="HF dir with config.json + *.safetensors (e.g. Qwen2.5-0.5B)")
    ap.add_argument("--out", required=True, help="where the converted base + tile file go")
    ap.add_argument("--data", default=None)
    ap.add_argument("--T", type=int, default=128)
    ap.add_argument("--windows", type=int, default=2)
    ap.add_argument("--dtype", default="float32")
    ap.add_argument("--json", default="mlx_bench.json")
    ap.add_argument("--suite", default="all", choices=["all", "plan", "drift"])
    ap.add_argument("--drift-at", type=int, default=128)
    ap.add_argument("--only", default="", help="comma-separated modes to run, e.g. base,plan")
    ap.add_argument("--plan-len", type=int, default=64)
    a = ap.parse_args()
    if a.cmd == "convert":
        convert(a.model, a.out)
    else:
        bench(a)


if __name__ == "__main__":
    main()
