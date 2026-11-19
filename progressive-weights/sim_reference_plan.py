"""Torch simulation of the 'plan' (decide once from the prompt) vs per-token selection, on the tokens that
mlx_engine.py bench --suite plan uses.   usage: python sim_reference_plan.py --model DIR --data test.parquet [--T 256 --plan-len 64]"""
import argparse, json, math
import pyarrow.parquet as pq
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import progressive as P

ap = argparse.ArgumentParser()
ap.add_argument("--model", required=True)
ap.add_argument("--data", required=True)
ap.add_argument("--T", type=int, default=256)
ap.add_argument("--windows", type=int, default=2)
ap.add_argument("--plan-len", type=int, default=64)
a = ap.parse_args()
tok = AutoTokenizer.from_pretrained(a.model)
ids = tok("\n\n".join(pq.read_table(a.data).column("text").to_pylist()), return_tensors="pt").input_ids[0]
wins = [ids[i * 512:i * 512 + a.T].unsqueeze(0) for i in range(a.windows)]
model = AutoModelForCausalLM.from_pretrained(a.model, dtype=torch.float32).eval()
ctl = P.Controller()
ctl.prompt_len = a.plan_len
groups = P.convert_model(model, ctl)
for mode, f in (("base", 0.0), ("full", 0.0), ("dynamic", 0.1), ("prompt", 0.1), ("dynamic", 0.2), ("prompt", 0.2)):
    ctl.mode, ctl.frac = mode, f
    for g in groups:
        g._x = None
    nll, n = P.nll_of(model, wins, start=a.plan_len)
    print(json.dumps(dict(mode=mode, frac=f, ppl=math.exp(nll / n))), flush=True)
