"""Strong static-quantization baselines (GPTQ and plain RTN) for the SAME model and the SAME
eval windows as run_sweep.py, so they can be compared with the progressive-weights numbers.

GPTQ here = the standard algorithm (Frantar et al. 2022): per linear layer, accumulate the input
Hessian H = 2/n * X^T X from calibration text, then quantize the columns one by one and push each
column's rounding error onto the not-yet-quantized columns using the inverse Hessian.
Group size 128, asymmetric min/max per (row, group), no act-order, layers quantized sequentially
(the inputs of layer i come from the already-quantized layers < i).

usage: python gptq_baseline.py --model DIR --train train.parquet --test test.parquet --bits 4 --method gptq
"""
from __future__ import annotations

import argparse
import json
import math
import os
import time

import pyarrow.parquet as pq
import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM, AutoTokenizer

import progressive as P
from run_sweep import load_windows

GROUP = 128


def find_params(w, bits):
    maxq = 2 ** bits - 1
    mn = w.min(1, keepdim=True).values
    mx = w.max(1, keepdim=True).values
    scale = (mx - mn).clamp_min(1e-8) / maxq
    zero = torch.round(-mn / scale)
    return scale, zero, maxq


def quant(w, scale, zero, maxq):
    q = torch.clamp(torch.round(w / scale) + zero, 0, maxq)
    return scale * (q - zero)


def rtn(W, bits, group=GROUP):
    out = torch.empty_like(W)
    for i in range(0, W.shape[1], group):
        w = W[:, i:i + group]
        s, z, m = find_params(w, bits)
        out[:, i:i + group] = quant(w, s, z, m)
    return out


def gptq(W, H, bits, group=GROUP, blocksize=128, percdamp=0.01):
    W = W.clone().float()
    out_f, in_f = W.shape
    H = H.clone()
    dead = torch.diag(H) == 0
    H[dead, dead] = 1
    W[:, dead] = 0
    H += percdamp * torch.mean(torch.diag(H)) * torch.eye(in_f)
    L = torch.linalg.cholesky(H)
    Hinv = torch.linalg.cholesky(torch.cholesky_inverse(L), upper=True)
    Q = torch.zeros_like(W)
    scale = zero = None
    maxq = 2 ** bits - 1
    for i1 in range(0, in_f, blocksize):
        i2 = min(i1 + blocksize, in_f)
        W1 = W[:, i1:i2].clone()
        Q1 = torch.zeros_like(W1)
        E1 = torch.zeros_like(W1)
        Hinv1 = Hinv[i1:i2, i1:i2]
        for i in range(i2 - i1):
            if (i1 + i) % group == 0:
                scale, zero, maxq = find_params(W[:, i1 + i:i1 + i + group] if i == 0 else W1[:, i:i + group], bits)
            w = W1[:, i]
            d = Hinv1[i, i]
            q = quant(w.unsqueeze(1), scale, zero, maxq).flatten()
            Q1[:, i] = q
            err = (w - q) / d
            W1[:, i:] -= err.unsqueeze(1) @ Hinv1[i, i:].unsqueeze(0)
            E1[:, i] = err
        Q[:, i1:i2] = Q1
        W[:, i2:] -= E1 @ Hinv[i1:i2, i2:]
    return Q


@torch.no_grad()
def quantize_gptq(model, calib, bits):
    layers = model.model.layers
    inps, kw = [], {}

    class Catcher(nn.Module):
        def __init__(self, m):
            super().__init__()
            self.m = m

        def forward(self, x, **kwargs):
            inps.append(x)
            kw.update(kwargs)
            raise ValueError

    layers[0] = Catcher(layers[0])
    for w in calib:
        try:
            model(w, use_cache=False)
        except ValueError:
            pass
    layers[0] = layers[0].m
    for i, layer in enumerate(layers):
        t0 = time.time()
        a, m = layer.self_attn, layer.mlp
        plan = [("qkv", a.q_proj, [a.q_proj, a.k_proj, a.v_proj]), ("o", a.o_proj, [a.o_proj]),
                ("gu", m.gate_proj, [m.gate_proj, m.up_proj]), ("down", m.down_proj, [m.down_proj])]
        H = {k: torch.zeros(src.in_features, src.in_features) for k, src, _ in plan}
        n = {k: 0 for k, _, _ in plan}

        def mk(key):
            def hook(mod, inp, out):
                x = inp[0].reshape(-1, inp[0].shape[-1]).float()
                H[key].add_(x.T @ x)
                n[key] += x.shape[0]
            return hook

        handles = [src.register_forward_hook(mk(k)) for k, src, _ in plan]
        for x in inps:
            layer(x, **kw)
        for h in handles:
            h.remove()
        for k, _, mods in plan:
            Hk = H[k] * 2.0 / n[k]
            for mod in mods:
                mod.weight.data = gptq(mod.weight.data, Hk, bits)
        inps = [layer(x, **kw) for x in inps]
        print(f"  layer {i} done in {time.time() - t0:.1f}s", flush=True)


def quantize_rtn(model, bits):
    for layer in model.model.layers:
        a, m = layer.self_attn, layer.mlp
        for mod in (a.q_proj, a.k_proj, a.v_proj, a.o_proj, m.gate_proj, m.up_proj, m.down_proj):
            mod.weight.data = rtn(mod.weight.data.float(), bits)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--train", required=True)
    ap.add_argument("--test", required=True)
    ap.add_argument("--bits", type=int, default=4)
    ap.add_argument("--method", default="gptq", choices=["gptq", "rtn"])
    ap.add_argument("--nsamples", type=int, default=128)
    ap.add_argument("--T", type=int, default=512)
    ap.add_argument("--out", default="results/gptq_baseline.jsonl")
    ap.add_argument("--threads", type=int, default=4)
    a = ap.parse_args()
    torch.set_num_threads(a.threads)
    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    tok = AutoTokenizer.from_pretrained(a.model)
    model = AutoModelForCausalLM.from_pretrained(a.model, dtype=torch.float32).eval()
    ev, _ = load_windows(tok, a.test, a.T, 8, 4)           # identical eval windows to run_sweep.py
    t0 = time.time()
    if a.method == "gptq":
        text = "\n\n".join(pq.read_table(a.train).column("text").to_pylist())[:4_000_000]
        ids = tok(text, return_tensors="pt").input_ids[0]
        g = torch.Generator().manual_seed(0)
        starts = torch.randint(0, ids.numel() - a.T - 1, (a.nsamples,), generator=g)
        calib = [ids[s:s + a.T].unsqueeze(0) for s in starts.tolist()]
        quantize_gptq(model, calib, a.bits)
    else:
        quantize_rtn(model, a.bits)
    qtime = time.time() - t0
    nll, n = P.nll_of(model, ev)
    rec = dict(method=a.method, bits=a.bits, group=GROUP, nsamples=a.nsamples if a.method == "gptq" else 0,
               ppl=math.exp(nll / n), tokens=n, quant_seconds=round(qtime, 1))
    print(json.dumps(rec), flush=True)
    with open(a.out, "a") as f:
        f.write(json.dumps(rec) + "\n")


if __name__ == "__main__":
    main()
