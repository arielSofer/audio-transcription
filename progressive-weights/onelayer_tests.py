"""One-layer kill-tests for two "buy bandwidth with capacity" ideas from the agents, on real activations.

Both ideas try to answer y = W x while reading far fewer weight bytes per token than a dense int4 matrix,
by pre-computing things offline and storing MANY more bytes on the SSD:

  POD / reduced basis : cluster the inputs x into K neighbourhoods; per neighbourhood keep a mean mu_c and an
                        orthonormal basis U_c (d x r) and pre-store a_c = W mu_c, B_c = W U_c (n x r).
                        y_hat = a_c + B_c (U_c^T (x - mu_c)).   Bytes read per token ~ r (n + d) + n elements.
  PQ lookup tables    : split x into subvectors of `sub` dims; per subspace 256 centroids; pre-store the
                        product of W's matching columns with each centroid.  y_hat = sum_s T[s][code_s(x)].
                        Bytes read per token ~ (d / sub) * n int8 values.

Metric: relative output error ||W(x - x_hat)|| / ||W x||  on HELD-OUT tokens (fit on one set of windows,
tested on another).  Equal-bytes reference ("TEAL-style"): keep only the largest-|x| channels of a dense int4
matrix so that the same number of bytes is read, and zero the rest.

usage: python onelayer_tests.py --model DIR --train train.parquet --test test.parquet
"""
from __future__ import annotations

import argparse
import json
import time

import pyarrow.parquet as pq
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

LAYERS = (2, 12, 21)


def collect(model, windows):
    """Inputs of q/gate/o/down projections for LAYERS over all windows -> {(layer, name): [N, d]}."""
    store = {}
    handles = []
    for li in LAYERS:
        layer = model.model.layers[li]
        for name, mod in (("qkv", layer.self_attn.q_proj), ("o", layer.self_attn.o_proj),
                          ("gu", layer.mlp.gate_proj), ("down", layer.mlp.down_proj)):
            store[(li, name)] = []

            def hook(m, inp, out, key=(li, name)):
                store[key].append(inp[0].reshape(-1, inp[0].shape[-1]).float())
            handles.append(mod.register_forward_hook(hook))
    with torch.no_grad():
        for w in windows:
            model(w)
    for h in handles:
        h.remove()
    return {k: torch.cat(v) for k, v in store.items()}


def kmeans(x, k, iters=15, seed=0):
    g = torch.Generator().manual_seed(seed)
    c = x[torch.randperm(x.shape[0], generator=g)[:k]].clone()
    for _ in range(iters):
        a = torch.cdist(x, c).argmin(1)
        sums = torch.zeros_like(c).index_add_(0, a, x)
        cnt = torch.bincount(a, minlength=k).unsqueeze(1).float()
        c = torch.where(cnt > 0, sums / cnt.clamp_min(1), c)
    return c


def pod_error(W, Xtr, Xte, K, r):
    d = Xtr.shape[1]
    cents = kmeans(Xtr, K)
    atr = torch.cdist(Xtr, cents).argmin(1)
    bases = []
    for j in range(K):
        sel = Xtr[atr == j]
        if sel.shape[0] < 2:
            bases.append((cents[j], torch.zeros(d, 0)))
            continue
        mu = sel.mean(0)
        _, _, Vh = torch.linalg.svd(sel - mu, full_matrices=False)
        bases.append((mu, Vh[:r].T.contiguous()))
    ate = torch.cdist(Xte, cents).argmin(1)
    err2, ref2 = 0.0, 0.0
    for j in range(K):
        sel = Xte[ate == j]
        if sel.shape[0] == 0:
            continue
        mu, U = bases[j]
        xh = mu + (sel - mu) @ U @ U.T
        err2 += ((sel - xh) @ W.T).pow(2).sum().item()
        ref2 += (sel @ W.T).pow(2).sum().item()
    return (err2 / ref2) ** 0.5


def teal_error(W, Xte, frac):
    """Keep the `frac` largest-|x| channels per token, zero the rest."""
    d = Xte.shape[1]
    k = max(1, int(round(frac * d)))
    idx = Xte.abs().topk(k, -1).indices
    keep = torch.zeros_like(Xte).scatter_(-1, idx, 1.0)
    return (((Xte * (1 - keep)) @ W.T).pow(2).sum() / (Xte @ W.T).pow(2).sum()).sqrt().item()


def pq_error(W, Xtr, Xte, sub, ncent=256):
    d = Xtr.shape[1]
    S = d // sub
    err2, ref2 = 0.0, (Xte @ W.T).pow(2).sum().item()
    yh = torch.zeros(Xte.shape[0], W.shape[0])
    for s in range(S):
        sl = slice(s * sub, (s + 1) * sub)
        cents = kmeans(Xtr[:, sl], ncent, iters=10, seed=s)
        codes = torch.cdist(Xte[:, sl], cents).argmin(1)
        table = cents @ W[:, sl].T                      # [ncent, n] the pre-computed products
        yh += table[codes]
    return ((Xte @ W.T - yh).pow(2).sum().item() / ref2) ** 0.5


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--train", required=True)
    ap.add_argument("--test", required=True)
    ap.add_argument("--fit-windows", type=int, default=32)
    ap.add_argument("--test-windows", type=int, default=4)
    ap.add_argument("--out", default="results/onelayer_tests.json")
    ap.add_argument("--threads", type=int, default=4)
    a = ap.parse_args()
    torch.set_num_threads(a.threads)
    tok = AutoTokenizer.from_pretrained(a.model)
    model = AutoModelForCausalLM.from_pretrained(a.model, dtype=torch.float32).eval()
    T = 512
    ids = tok("\n\n".join(pq.read_table(a.train).column("text").to_pylist())[:1_500_000], return_tensors="pt").input_ids[0]
    fit = [ids[i * T:(i + 1) * T].unsqueeze(0) for i in range(a.fit_windows)]
    tids = tok("\n\n".join(pq.read_table(a.test).column("text").to_pylist())[:300_000], return_tensors="pt").input_ids[0]
    te = [tids[i * T:(i + 1) * T].unsqueeze(0) for i in range(a.test_windows)]
    t0 = time.time()
    Xfit, Xte = collect(model, fit), collect(model, te)
    print(f"collected activations in {time.time() - t0:.0f}s ({next(iter(Xfit.values())).shape[0]} fit tokens, "
          f"{next(iter(Xte.values())).shape[0]} test tokens)", flush=True)

    res = []
    for (li, name), Xtr in Xfit.items():
        layer = model.model.layers[li]
        W = {"qkv": layer.self_attn.q_proj, "o": layer.self_attn.o_proj,
             "gu": layer.mlp.gate_proj, "down": layer.mlp.down_proj}[name].weight.data.float()
        n, d = W.shape
        xt = Xte[(li, name)]
        # POD
        for K in (8, 32):
            for r in (32, 128):
                if r >= d:
                    continue
                e = pod_error(W, Xtr, xt, K, r)
                ratio = (r * (n + d) + n) / (n * d)                 # elements read vs the dense matrix
                eq = teal_error(W, xt, min(1.0, ratio))             # same element count read from a dense matrix
                rec = dict(layer=li, module=name, idea="pod", K=K, r=r, rel_error=e, bytes_vs_dense_fp16=ratio,
                           equal_bytes_teal_error=eq)
                res.append(rec)
                print(json.dumps(rec), flush=True)
        # PQ lookup tables
        for sub in (16, 8):
            if d % sub:
                continue
            e = pq_error(W, Xtr, xt, sub)
            ratio = (d / sub) * n / (n * d * 0.5)                   # int8 table rows vs dense int4
            eq = teal_error(W, xt, min(1.0, ratio))
            rec = dict(layer=li, module=name, idea="pq_lut", sub=sub, rel_error=e, bytes_vs_dense_int4=ratio,
                       equal_bytes_teal_error=eq)
            res.append(rec)
            print(json.dumps(rec), flush=True)
    json.dump(res, open(a.out, "w"), indent=1)


if __name__ == "__main__":
    main()
