"""Check that expert streaming gives the SAME logits as the normal (all experts resident) model.

Runs a short token sequence both ways, token by token with a KV cache and as one batch, and prints the largest
difference between the two sets of logits. (The resident model is slow on a CPU, so keep --tokens small there.)
usage: python moe_verify.py --model DIR --experts FILE --data test.parquet [--tokens 8] [--cache-frac 0.25]
"""
import argparse, pathlib, time
import numpy as np
import mlx.core as mx
import pyarrow.parquet as pq
from mlx_lm.models.cache import make_prompt_cache
from mlx_lm.utils import load_model
from transformers import AutoTokenizer
import moe_stream as MS

ap = argparse.ArgumentParser()
ap.add_argument("--model", required=True)
ap.add_argument("--experts", required=True)
ap.add_argument("--data", required=True)
ap.add_argument("--tokens", type=int, default=8)
ap.add_argument("--cache-frac", type=float, default=0.25)
a = ap.parse_args()
tok = AutoTokenizer.from_pretrained(a.model)
ids = tok("\n\n".join(pq.read_table(a.data).column("text").to_pylist())[:20000]).input_ids[:a.tokens]
model, _ = load_model(pathlib.Path(a.model), lazy=False)

def run(model):
    cache = make_prompt_cache(model)
    step = [np.array(model(mx.array([[t]]), cache=cache)[0, -1].astype(mx.float32)) for t in ids]
    batch = np.array(model(mx.array([ids]))[0].astype(mx.float32))          # all tokens in one forward (prefill path)
    return np.stack(step), batch

t0 = time.time(); ref_step, ref_batch = run(model); print("resident model: %.0fs" % (time.time() - t0), flush=True)
streamer = MS.Streamer(a.experts, cache_frac=a.cache_frac)
n = MS.wrap_model(model, streamer)
mx.eval(model.parameters())
t0 = time.time(); new_step, new_batch = run(model); print("streamed model: %.0fs  (%d MoE layers)" % (time.time() - t0, n), flush=True)
print("max |logit difference|, token by token : %.3e   (logit scale %.1f)" % (np.abs(ref_step - new_step).max(), np.abs(ref_step).max()))
print("max |logit difference|, one batch      : %.3e" % np.abs(ref_batch - new_batch).max())
print("same argmax everywhere:", bool((ref_step.argmax(-1) == new_step.argmax(-1)).all() and (ref_batch.argmax(-1) == new_batch.argmax(-1)).all()))
print("cache stats: hits %d misses %d, read %.1f MB" % (streamer.hits, streamer.misses, streamer.bytes_read / 1e6))
