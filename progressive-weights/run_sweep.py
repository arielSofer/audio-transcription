"""Run the progressive-weights experiments on a real model (CPU).

usage:  python run_sweep.py --model DIR --data wikitext2_test.parquet \
            --suite sanity|main|policy|all --out results/results.jsonl
Every finished config is appended to the jsonl, so a rerun resumes.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import time

import pyarrow.parquet as pq
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

import progressive as P


def load_windows(tok, data, T, n_eval, n_calib):
    text = "\n\n".join(pq.read_table(data).column("text").to_pylist())
    ids = tok(text, return_tensors="pt").input_ids[0]
    need = (n_eval + n_calib) * T
    assert ids.numel() >= need, (ids.numel(), need)
    ws = [ids[i * T:(i + 1) * T].unsqueeze(0) for i in range(n_eval + n_calib)]
    return ws[:n_eval], ws[n_eval:]


def key_of(c):
    return json.dumps({k: c.get(k) for k in ("suite", "bits", "mode", "frac", "budget", "cap_mult")}, sort_keys=True)


def suites(bits_list):
    fr = [0.05, 0.10, 0.20, 0.40, 0.70]
    out = []
    if True:  # sanity first: floor/ceiling for every base width
        for b in bits_list:
            out.append(dict(suite="sanity", bits=b, mode="base", frac=0.0))
        out.append(dict(suite="sanity", bits=bits_list[0], mode="full", frac=1.0))
    for b in bits_list:
        for f in fr:
            for mode in ("dynamic", "static", "random"):
                out.append(dict(suite="main", bits=b, mode=mode, frac=f))
    return out


def policy_suite(bits, frac):
    out = []
    # unlimited cache and unlimited budget = the "ideal" numbers; then squeeze the budget
    for cap in (None, 3.0, 1.5, 1.0):
        for budget in (None, 4, 2, 1, 0):
            out.append(dict(suite="policy", bits=bits, mode="policy", frac=frac, budget=budget, cap_mult=cap))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--data", required=True)
    ap.add_argument("--suite", default="sanity")
    ap.add_argument("--out", default="results/results.jsonl")
    ap.add_argument("--bits", default="4,3,2")
    ap.add_argument("--T", type=int, default=512)
    ap.add_argument("--windows", type=int, default=8)
    ap.add_argument("--calib", type=int, default=4)
    ap.add_argument("--policy-bits", type=int, default=4)
    ap.add_argument("--policy-frac", type=float, default=0.2)
    ap.add_argument("--threads", type=int, default=4)
    a = ap.parse_args()
    torch.set_num_threads(a.threads)
    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    done = set()
    if os.path.exists(a.out):
        for line in open(a.out):
            done.add(key_of(json.loads(line)))

    tok = AutoTokenizer.from_pretrained(a.model)
    model = AutoModelForCausalLM.from_pretrained(a.model, dtype=torch.float32).eval()
    ev, cal = load_windows(tok, a.data, a.T, a.windows, a.calib)
    ntok = sum(w.shape[1] - 1 for w in ev)

    def log(rec):
        with open(a.out, "a") as f:
            f.write(json.dumps(rec) + "\n")
        print(json.dumps({k: rec[k] for k in rec if k not in ("overlap", "energy", "used_frac")}), flush=True)

    # reference: the untouched fp32 model (weights were bf16 originally)
    ref_cfg = dict(suite="sanity", bits=32, mode="fp32", frac=1.0)
    if key_of(ref_cfg) not in done:
        t0 = time.time()
        nll, n = P.nll_of(model, ev)
        log(dict(ref_cfg, nll=nll / n, ppl=math.exp(nll / n), tokens=n, seconds=round(time.time() - t0, 1)))

    ctl = P.Controller()
    groups = P.convert_model(model, ctl)
    print("converted", len(groups), "groups", flush=True)

    # calibration for the STATIC baseline: mean |x_tile| on windows disjoint from eval
    ctl.mode = "calib"
    P.nll_of(model, cal)
    for g in groups:
        g._x = None

    if a.suite == "sanity":
        todo = [c for c in suites([int(x) for x in a.bits.split(",")]) if c["suite"] == "sanity"]
    elif a.suite == "main":
        todo = [c for c in suites([int(x) for x in a.bits.split(",")]) if c["suite"] == "main"]
    elif a.suite == "policy":
        todo = policy_suite(a.policy_bits, a.policy_frac)
    else:
        todo = suites([int(x) for x in a.bits.split(",")]) + policy_suite(a.policy_bits, a.policy_frac)

    cur_bits = None
    for c in todo:
        if key_of(c) in done:
            continue
        if c["bits"] != cur_bits:
            P.set_bits(groups, ctl, c["bits"])
            cur_bits = c["bits"]
        ctl.mode, ctl.frac = c["mode"], c["frac"]
        ctl.budget, ctl.cap_mult = c.get("budget"), c.get("cap_mult")
        ctl.reset_stats()
        for g in groups:
            g._x = None
        t0 = time.time()
        nll, n = P.nll_of(model, ev)
        ctl.tokens = n
        rec = dict(c, nll=nll / n, ppl=math.exp(nll / n), tokens=n,
                   refine_bytes_per_token=ctl.refine_bytes / n,
                   fetch_bytes_per_token=ctl.fetch_bytes / n,
                   overlap={k: v[0] / v[1] for k, v in ctl.overlap.items()},
                   energy={k: v[0] / v[1] for k, v in ctl.energy.items()},
                   used_frac={k: v[0] / v[1] for k, v in ctl.used_frac.items()},
                   seconds=round(time.time() - t0, 1))
        log(rec)


if __name__ == "__main__":
    main()
