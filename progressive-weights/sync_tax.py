"""What does one CPU<->GPU synchronisation cost per layer?   (decides whether the 'fewer syncs' idea has anything to win)

The streamed MoE engine stops the GPU after every layer's router. On gpt-oss-20b that made a token 2.4x slower than the
normal model even with every expert in RAM (31.0 vs 13.1 ms). This script runs the NORMAL resident model (no expert
streaming at all) and forces mx.eval() after every k-th decoder layer, for several k, and fits

        ms_per_token = base + cost_per_sync * syncs_per_token

* cost_per_sync large  -> the tax is the synchronisation itself; fewer syncs per token can win back most of it.
* cost_per_sync small  -> the tax is our own Python/bookkeeping in the streaming path, and the fix is boring engineering.

usage: python sync_tax.py --model <mlx model dir> --data wikitext2_test.parquet [--tokens 96] [--skip 16]
"""
import argparse
import json
import pathlib
import time

import mlx.core as mx
import numpy as np


class EvalAfter:
    """Wraps one decoder layer: evaluates its output right after it runs (one CPU<->GPU sync)."""

    def __init__(self, inner):
        self._inner = inner

    def __call__(self, *a, **k):
        out = self._inner(*a, **k)
        mx.eval(out)
        return out

    def __getattr__(self, name):
        return getattr(self._inner, name)


def find_layers(model):
    for holder in (getattr(model, "model", None), getattr(model, "language_model", None), model):
        if holder is not None and hasattr(holder, "layers"):
            return holder
    raise SystemExit("could not find .layers on the model")


def run(model, ids, skip):
    from mlx_lm.models.cache import make_prompt_cache
    cache = make_prompt_cache(model)
    times = []
    for t in range(len(ids) - 1):
        t0 = time.perf_counter()
        logits = model(mx.array([[ids[t]]]), cache=cache)[0, -1]
        mx.eval(logits)
        dt = time.perf_counter() - t0
        if t >= skip:
            times.append(dt)
    return 1000 * float(np.median(times)), 1000 * float(np.mean(times))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--data", required=True)
    ap.add_argument("--tokens", type=int, default=96)
    ap.add_argument("--skip", type=int, default=16, help="warm-up tokens not timed")
    ap.add_argument("--json")
    a = ap.parse_args()

    import pyarrow.parquet as pq
    from mlx_lm.utils import load_model
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(a.model)
    text = "\n\n".join(pq.read_table(a.data).column("text").to_pylist())
    ids = tok(text).input_ids[:a.tokens]
    model, _ = load_model(pathlib.Path(a.model), lazy=False)
    mx.eval(model.parameters())
    holder = find_layers(model)
    original = list(holder.layers)
    n = len(original)
    print(f"{n} layers, {len(ids) - 1 - a.skip} timed tokens per setting (median ms reported)", flush=True)

    rows = []
    for every in (0, 12, 6, 4, 3, 2, 1):                    # 0 = the normal model (one sync per token, at the end)
        every = min(every, n) if every else 0
        holder.layers = [EvalAfter(l) if every and (i + 1) % every == 0 else l for i, l in enumerate(original)]
        syncs = 1 + (sum(1 for i in range(n) if every and (i + 1) % every == 0) if every else 0)
        med, mean = run(model, ids, a.skip)
        rows.append(dict(eval_every=every, syncs_per_token=syncs, ms_median=round(med, 3), ms_mean=round(mean, 3)))
        print(json.dumps(rows[-1]), flush=True)
    holder.layers = original

    x = np.array([r["syncs_per_token"] for r in rows], dtype=float)
    y = np.array([r["ms_median"] for r in rows], dtype=float)
    slope, base = np.polyfit(x, y, 1)
    resid = y - (base + slope * x)
    print()
    print(f"fit: ms_per_token = {base:.2f} + {slope:.3f} * syncs_per_token   (max residual {np.abs(resid).max():.2f} ms)")
    full = y[-1]
    print(f"normal model {y[0]:.2f} ms/token; a sync after every layer {full:.2f} ms/token  ({full / y[0]:.2f}x)")
    print(f"cost of one sync ~ {slope:.3f} ms  ->  a model with {n} layers pays ~{slope * n:.1f} ms/token for per-layer syncs")
    print("compare with the measured streaming tax on this model: gpt-oss-20b 31.0 - 13.1 = 17.9 ms/token for 24 layers;"
          " Qwen3-30B 36.4 - 28.7 = 7.7 ms/token for 48 layers.")
    if a.json:
        json.dump(dict(rows=rows, base_ms=base, ms_per_sync=slope), open(a.json, "w"), indent=2)


if __name__ == "__main__":
    main()
