"""Expert streaming for MoE models on MLX: keep attention, router and embeddings in RAM, keep the experts on the SSD.

After the router of a layer has chosen its top-k experts for a token, only those k experts are brought in
(from an LRU cache in RAM, or read from the SSD), stacked, and run through the ORIGINAL SwitchGLU code. So the
output is identical to the normal model; only where the expert weights live changes.

  python moe_stream.py convert --model MODEL_DIR --out EXPERTS_FILE        (once; works shard by shard, any model size)
  python moe_stream.py bench   --model MODEL_DIR --experts EXPERTS_FILE --data wikitext2_test.parquet
                               [--cache-frac 0.25] [--verify] [--trace trace.json]

The model is loaded lazily, so the expert weights are never read into RAM and the model may be larger than memory.
"""
from __future__ import annotations

import argparse
import collections
import json
import math
import os
import re
import time
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import mlx.core as mx
import mlx.nn as nn

import tilefile as TF

ALIGN = 4096
PROJS = ("gate_proj", "up_proj", "down_proj")
ARRS = ("weight", "scales", "biases", "bias", "global_scale")
NPT = {"uint32": np.uint32, "float16": np.float16, "float32": np.float32, "bfloat16": np.uint16, "uint8": np.uint8}
# two checkpoint layouts: experts already stacked ("...mlp.switch_mlp.gate_proj.weight" with a leading expert axis)
# or one tensor set per expert ("...mlp.experts.17.gate_proj.weight"; mlx-lm stacks these when it loads the model)
KEY_STACKED = re.compile(r"^(?P<path>.*\.layers\.(?P<layer>\d+)\.(?:.*\.)?(?:switch_mlp|experts))"
                         r"\.(?P<proj>gate_proj|up_proj|down_proj)\.(?P<arr>weight|scales|biases|bias|global_scale)$")
KEY_PER_EXPERT = re.compile(r"^(?P<prefix>.*\.layers\.(?P<layer>\d+)\.(?:.*\.)?mlp)\.experts\.(?P<e>\d+)"
                            r"\.(?P<proj>gate_proj|up_proj|down_proj)\.(?P<arr>weight|scales|biases|bias|global_scale)$")


def dname(a):
    return str(a.dtype).split(".")[-1]


def to_np(a):
    return np.array(a.view(mx.uint16)) if dname(a) == "bfloat16" else np.array(a)


# ----------------------------------------------------------------------------------------------
# offline: write every expert of every layer as one contiguous, 4 KB-aligned blob
# ----------------------------------------------------------------------------------------------
def convert(model_dir, out_path):
    tensors = {}
    for f in sorted(os.listdir(model_dir)):
        if f.endswith(".safetensors"):
            tensors.update(mx.load(os.path.join(model_dir, f)))           # lazy: nothing is read yet
    stacked = collections.defaultdict(dict)          # path -> {"proj.arr": key}
    per_expert = collections.defaultdict(lambda: collections.defaultdict(dict))   # path -> e -> {"proj.arr": key}
    for k in tensors:
        m = KEY_STACKED.match(k)
        if m:
            stacked[m.group("path")][f"{m.group('proj')}.{m.group('arr')}"] = k
            continue
        m = KEY_PER_EXPERT.match(k)
        if m:                                         # after mlx-lm's sanitize the module is called switch_mlp
            per_expert[m.group("prefix") + ".switch_mlp"][int(m.group("e"))][f"{m.group('proj')}.{m.group('arr')}"] = k
    paths = sorted(set(stacked) | set(per_expert), key=lambda p: int(re.search(r"layers\.(\d+)\.", p).group(1)))
    if not paths:
        raise SystemExit("no expert tensors found in " + model_dir)
    index = {"bits": 4, "tile": 0, "align": ALIGN, "groups": {}, "arrays": {}}
    off, t0 = 0, time.time()
    with open(out_path, "wb") as f:
        for path in paths:
            if path in stacked:                       # one stacked tensor per array: slice it per expert
                names = [f"{p}.{a}" for p in PROJS for a in ARRS if f"{p}.{a}" in stacked[path]]
                arrs = {n: tensors[stacked[path][n]] for n in names}
                E = arrs[names[0]].shape[0]
                np_arrs = {n: to_np(arrs[n]) for n in names}
                shapes = {n: list(arrs[n].shape[1:]) for n in names}
                dts = {n: dname(arrs[n]) for n in names}
                get = lambda e: [np_arrs[n][e] for n in names]
            else:                                     # one tensor set per expert
                E = max(per_expert[path]) + 1
                assert sorted(per_expert[path]) == list(range(E)), (path, "missing experts")
                names = [f"{p}.{a}" for p in PROJS for a in ARRS if f"{p}.{a}" in per_expert[path][0]]
                shapes = {n: list(tensors[per_expert[path][0][n]].shape) for n in names}
                dts = {n: dname(tensors[per_expert[path][0][n]]) for n in names}
                get = lambda e: [to_np(tensors[per_expert[path][e][n]]) for n in names]
            spec, pos = [], 0
            for n in names:
                per = int(np.prod(shapes[n])) * np.dtype(NPT[dts[n]]).itemsize
                spec.append(dict(name=n, dtype=dts[n], shape=shapes[n], offset=pos, nbytes=per))
                pos += per
            raw = pos
            stride = (raw + ALIGN - 1) // ALIGN * ALIGN
            for e in range(E):
                blob = b"".join(a.tobytes() for a in get(e))
                assert len(blob) == raw, (path, e, len(blob), raw)
                f.write(blob + b"\0" * (stride - raw))
            index["groups"][path] = dict(base=off, stride=stride, raw=raw, nt=E, out_total=0)
            index["arrays"][path] = spec
            off += stride * E
            print(f"  {path}: {E} experts x {raw / 1e6:.2f} MB  ({time.time() - t0:.0f}s)", flush=True)
    index["file_bytes"] = off
    json.dump(index, open(out_path + ".json", "w"))
    print(f"wrote {off / 1e9:.2f} GB of experts for {len(paths)} layers to {out_path}")


# ----------------------------------------------------------------------------------------------
# runtime
# ----------------------------------------------------------------------------------------------
class Streamer:
    """Per-layer LRU cache of experts (as MLX arrays) over the expert file; misses are read in parallel."""

    def __init__(self, path, cache_frac=0.25, workers=8, trace=None):
        self.tf = TF.TileFile(path, direct=True)
        self.index = self.tf.index
        self.pool = ThreadPoolExecutor(workers)
        self.cache_frac = cache_frac
        self.cache = {p: collections.OrderedDict() for p in self.index["groups"]}
        self.trace = trace
        self.reset()

    def reset(self):
        self.calls = self.hits = self.misses = self.bytes_read = 0
        self.t_io = self.t_stack = self.t_sync = 0.0

    def snapshot(self):
        return dict(calls=self.calls, hits=self.hits, misses=self.misses, bytes_read=self.bytes_read,
                    t_io=self.t_io, t_stack=self.t_stack, t_sync=self.t_sync)

    def cap(self, path, k):
        E = self.index["groups"][path]["nt"]
        return E if self.cache_frac >= 1 else max(k, int(math.ceil(self.cache_frac * E)))

    def _arrays(self, path, blob):
        out = {}
        for s in self.index["arrays"][path]:
            npt = NPT[s["dtype"]]
            n = int(np.prod(s["shape"]))
            a = np.frombuffer(blob, dtype=npt, count=n, offset=s["offset"]).reshape(s["shape"])
            m = mx.array(a)
            out[s["name"]] = m.view(mx.bfloat16) if s["dtype"] == "bfloat16" else m
        return out

    def stacked(self, path, experts):
        """experts: list of distinct expert ids -> {array name: stacked mx array [len(experts), ...]}."""
        cache = self.cache[path]
        miss = [e for e in experts if e not in cache]
        self.calls += 1
        self.hits += len(experts) - len(miss)
        self.misses += len(miss)
        t0 = time.perf_counter()
        futs = {e: self.pool.submit(self.tf.read_tile, path, e) for e in miss}
        blobs = {e: f.result() for e, f in futs.items()}
        self.t_io += time.perf_counter() - t0
        for e, b in blobs.items():
            cache[e] = self._arrays(path, b)
            self.bytes_read += len(b)
        for e in experts:
            cache.move_to_end(e)
        cap = self.cap(path, len(experts))
        while len(cache) > cap:
            cache.popitem(last=False)
        t0 = time.perf_counter()
        names = list(cache[experts[0]].keys())
        out = {n: mx.stack([cache[e][n] for e in experts]) for n in names}
        self.t_stack += time.perf_counter() - t0
        return out


class _Box:
    pass


class StreamedSwitchGLU(nn.Module):
    """Drop-in for SwitchGLU: same call signature and same arithmetic, experts fetched on demand."""

    def __init__(self, inner, path, streamer: Streamer):
        super().__init__()
        self._box = _Box()
        self._box.inner, self._box.path, self._box.streamer = inner, path, streamer
        for p in PROJS:                                   # drop the stacked expert weights from the model
            mod = getattr(inner, p)
            for a in ARRS:
                if a in mod and mod[a] is not None:
                    mod[a] = mx.zeros((1,), dtype=mx.float32)

    def __call__(self, x, indices):
        b = self._box
        t0 = time.perf_counter()
        idx = np.array(indices)                           # GPU -> CPU: waits for attention + router of this layer
        b.streamer.t_sync += time.perf_counter() - t0
        uniq, inv = np.unique(idx, return_inverse=True)
        if b.streamer.trace is not None:
            b.streamer.trace.append((b.path, uniq.tolist()))
        sub = b.streamer.stacked(b.path, uniq.tolist())
        for name, arr in sub.items():
            proj, a = name.split(".", 1)
            getattr(b.inner, proj)[a] = arr
        local = mx.array(inv.reshape(idx.shape).astype(np.int32))
        return b.inner(x, local)


def wrap_model(model, streamer: Streamer):
    from mlx_lm.models.switch_layers import SwitchGLU
    found = [(n, m) for n, m in model.named_modules() if isinstance(m, SwitchGLU)]
    if not found:
        raise SystemExit("the model has no SwitchGLU (not a supported MoE)")
    for name, m in found:
        parts = name.split(".")
        parent = model
        for p in parts[:-1]:
            parent = parent[int(p)] if p.isdigit() else getattr(parent, p)
        wrapped = StreamedSwitchGLU(m, name, streamer)
        setattr(parent, parts[-1], wrapped)
    return len(found)


def decode_nll(model, ids, start, streamer: Streamer):
    """Token-by-token with a KV cache; returns NLL/count/seconds of the tokens at positions >= start + stats delta."""
    from mlx_lm.models.cache import make_prompt_cache
    cache = make_prompt_cache(model)
    nll, n, secs, snap = 0.0, 0, 0.0, None
    for t in range(len(ids) - 1):
        if t + 1 == start:
            snap = streamer.snapshot() if streamer else None
        t0 = time.perf_counter()
        logits = model(mx.array([[ids[t]]]), cache=cache)[0, -1].astype(mx.float32)
        v = float((mx.logsumexp(logits) - logits[ids[t + 1]]).item())
        dt = time.perf_counter() - t0
        if t + 1 >= start:
            nll, n, secs = nll + v, n + 1, secs + dt
    delta = None
    if streamer:
        cur = streamer.snapshot()
        delta = {k: cur[k] - snap[k] for k in cur}
    return nll, n, secs, delta


def bench(a):
    import pathlib
    import pyarrow.parquet as pq
    from mlx_lm.utils import load_model
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(a.model)
    text = "\n\n".join(pq.read_table(a.data).column("text").to_pylist())
    allids = tok(text).input_ids
    wins = [allids[i * 512:i * 512 + a.T] for i in range(a.windows)]
    trace = [] if a.trace else None
    out = []
    if a.verify:                                           # the resident model first, as the reference
        model, _ = load_model(pathlib.Path(a.model), lazy=False)
        tot, n, secs = 0.0, 0, 0.0
        for w in wins:
            nll, k, s, _ = decode_nll(model, w, a.start, None)
            tot, n, secs = tot + nll, n + k, secs + s
        r = dict(config="resident (all experts in RAM)", ppl=math.exp(tot / n), ms_per_token=1000 * secs / n,
                 tokens_per_s=n / secs)
        out.append(r)
        print(json.dumps({k: (round(v, 4) if isinstance(v, float) else v) for k, v in r.items()}), flush=True)
    else:
        model, _ = load_model(pathlib.Path(a.model), lazy=True)
    streamer = Streamer(a.experts, cache_frac=a.cache_frac, workers=a.workers, trace=trace)
    nl = wrap_model(model, streamer)
    mx.eval(model.parameters())                            # lazy weights: only the non-expert ones are read now
    tot, n, secs, agg = 0.0, 0, 0.0, collections.Counter()
    for w in wins:
        for c in streamer.cache.values():
            c.clear()
        nll, k, s, d = decode_nll(model, w, a.start, streamer)
        tot, n, secs = tot + nll, n + k, secs + s
        agg.update(d)
    r = dict(config=f"streamed, cache {a.cache_frac:.2f} of the experts per layer ({nl} MoE layers)", ppl=math.exp(tot / n),
             ms_per_token=1000 * secs / n, tokens_per_s=n / secs,
             hit_rate=agg["hits"] / max(agg["hits"] + agg["misses"], 1),
             experts_read_per_token=agg["misses"] / n, read_mb_per_token=agg["bytes_read"] / n / 1e6,
             io_ms_per_token=1000 * agg["t_io"] / n, router_sync_ms_per_token=1000 * agg["t_sync"] / n,
             stack_ms_per_token=1000 * agg["t_stack"] / n)
    out.append(r)
    print(json.dumps({k: (round(v, 4) if isinstance(v, float) else v) for k, v in r.items()}), flush=True)
    if a.json:
        json.dump(out, open(a.json, "w"), indent=2)
    if trace is not None:
        json.dump(trace, open(a.trace, "w"))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["convert", "bench"])
    ap.add_argument("--model", required=True)
    ap.add_argument("--out", help="convert: expert file to write")
    ap.add_argument("--experts", help="bench: expert file")
    ap.add_argument("--data")
    ap.add_argument("--T", type=int, default=256)
    ap.add_argument("--windows", type=int, default=2)
    ap.add_argument("--start", type=int, default=64)
    ap.add_argument("--cache-frac", type=float, default=0.25)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--verify", action="store_true", help="also run the resident model and compare")
    ap.add_argument("--json")
    ap.add_argument("--trace")
    a = ap.parse_args()
    if a.cmd == "convert":
        convert(a.model, a.out)
    else:
        bench(a)


if __name__ == "__main__":
    main()
