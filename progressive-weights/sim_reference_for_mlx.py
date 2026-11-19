"""The torch SIMULATION evaluated on exactly the tokens mlx_engine.py bench uses (two 128-token windows),
so the real-I/O engine can be checked against it.   usage: python sim_reference_for_mlx.py --model DIR --data test.parquet"""
import argparse, json, math
import pyarrow.parquet as pq
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import progressive as P

ap = argparse.ArgumentParser()
ap.add_argument("--model", required=True)
ap.add_argument("--data", required=True)
ap.add_argument("--T", type=int, default=128)
ap.add_argument("--windows", type=int, default=2)
a = ap.parse_args()
tok = AutoTokenizer.from_pretrained(a.model)
ids = tok("\n\n".join(pq.read_table(a.data).column("text").to_pylist()), return_tensors="pt").input_ids[0]
wins = [ids[i * 512:i * 512 + a.T].unsqueeze(0) for i in range(a.windows)]
model = AutoModelForCausalLM.from_pretrained(a.model, dtype=torch.float32).eval()
ctl = P.Controller()
groups = P.convert_model(model, ctl)
cfgs = [dict(mode="base"), dict(mode="full"), dict(mode="dynamic", frac=0.1),
        dict(mode="policy", frac=0.1, cap_mult=2.0, budget=None), dict(mode="policy", frac=0.1, cap_mult=2.0, budget=0.5),
        dict(mode="policy", frac=0.1, cap_mult=2.0, budget=0.25)]
out = []
for c in cfgs:
    ctl.mode, ctl.frac, ctl.cap_mult, ctl.budget = c["mode"], c.get("frac", 0.0), c.get("cap_mult"), c.get("budget")
    ctl.reset_stats()
    for g in groups:
        g._x = None
    nll, n = P.nll_of(model, wins)
    r = dict(config=c, ppl=math.exp(nll / n), fetch_mb_per_token=ctl.fetch_bytes / n / 1e6)
    out.append(r)
    print(json.dumps(r), flush=True)
