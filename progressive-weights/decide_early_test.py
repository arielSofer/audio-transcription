"""Does it hurt to decide WHICH tiles to refine earlier than 'per token, per projection'?

Compared on the same tokens (only positions >= 64 of every 512-token window are scored; the first 64 act as the prompt):
  oracle   : per token, from that token's own activation at each projection   (what the engine does now)
  prevtok  : the set the PREVIOUS token selected (decided one token ahead, so it can be prefetched)
  prompt   : ONE set per projection chosen from the prompt's mean activation score, reused for all later tokens
  static   : one set per projection chosen on separate calibration text (no knowledge of this input at all)
  random   : control
usage: python decide_early_test.py --model DIR --data test.parquet
"""
import argparse, json, math
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import progressive as P
from run_sweep import load_windows

ap = argparse.ArgumentParser()
ap.add_argument("--model", required=True)
ap.add_argument("--data", required=True)
ap.add_argument("--bits", type=int, default=4)
ap.add_argument("--fracs", default="0.1,0.2")
ap.add_argument("--out", default="results/decide_early.json")
ap.add_argument("--threads", type=int, default=4)
a = ap.parse_args()
torch.set_num_threads(a.threads)
tok = AutoTokenizer.from_pretrained(a.model)
model = AutoModelForCausalLM.from_pretrained(a.model, dtype=torch.float32).eval()
ev, cal = load_windows(tok, a.data, 512, 8, 4)
ctl = P.Controller()
groups = P.convert_model(model, ctl)
ctl.mode = "calib"
P.nll_of(model, cal)                       # statistics for the 'static' baseline
for g in groups:
    g._x = None
P.set_bits(groups, ctl, a.bits)
START = ctl.prompt_len
res = []
def run(mode, frac):
    ctl.mode, ctl.frac = mode, frac
    ctl.reset_stats()
    for g in groups:
        g._x = None
    nll, n = P.nll_of(model, ev, start=START)
    r = dict(mode=mode, frac=frac, ppl=math.exp(nll / n), tokens=n)
    res.append(r)
    print(json.dumps(r), flush=True)
run("base", 0.0)
ctl.mode = "full"
nll, n = P.nll_of(model, ev, start=START)
print(json.dumps(dict(mode="full(8-bit)", ppl=math.exp(nll / n))), flush=True)
for f in [float(x) for x in a.fracs.split(",")]:
    for m in ("dynamic", "prevtok", "prompt", "static", "random"):
        run(m, f)
json.dump(res, open(a.out, "w"), indent=1)
