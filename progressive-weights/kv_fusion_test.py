"""Kill-test for "KV fusion" (an agent's idea): decode with a cheap low-bit copy B of the model, but let a
high-precision copy A write the KV cache of the already-known text (prompt / older tokens).

Variants measured on the SAME text windows (WikiText-2 test, window = 256 context + 256 continuation tokens),
NLL/perplexity of the continuation only:
    A alone      : high-precision model reads and continues the text (upper bound)
    B alone      : low-bit model does everything (lower bound)
    A->B         : A builds the KV cache of the context, B continues reading A's cache
Recovery = how much of the A-vs-B gap in NLL the fusion closes.  B = plain RTN b-bit (group 128).

usage: python kv_fusion_test.py --model DIR --data test.parquet
"""
from __future__ import annotations

import argparse
import copy
import json
import math

import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer

import gptq_baseline as G
from run_sweep import load_windows

CTX = 256


@torch.no_grad()
def cont_nll(logits, ids):
    """NLL of tokens CTX..end given logits aligned so that logits[:, j] predicts ids[:, CTX + j]."""
    tgt = ids[0, CTX:]
    return F.cross_entropy(logits[0, : tgt.shape[0]].float(), tgt, reduction="sum").item(), tgt.shape[0]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--data", required=True)
    ap.add_argument("--bits", default="4,3,2")
    ap.add_argument("--windows", type=int, default=8)
    ap.add_argument("--out", default="results/kv_fusion.json")
    ap.add_argument("--threads", type=int, default=4)
    a = ap.parse_args()
    torch.set_num_threads(a.threads)
    tok = AutoTokenizer.from_pretrained(a.model)
    A = AutoModelForCausalLM.from_pretrained(a.model, dtype=torch.float32).eval()
    ev, _ = load_windows(tok, a.data, 512, a.windows, 4)

    # A alone (same for every B)
    tot, n = 0.0, 0
    with torch.no_grad():
        for w in ev:
            lg = A(w).logits[:, CTX - 1:-1]          # predicts tokens CTX..511
            s, k = cont_nll(lg, w)
            tot += s
            n += k
    nll_A = tot / n
    print(f"A alone: continuation ppl {math.exp(nll_A):.3f}", flush=True)
    res = {"A_ppl": math.exp(nll_A), "runs": []}

    for bits in [int(b) for b in a.bits.split(",")]:
        B = copy.deepcopy(A)
        G.quantize_rtn(B, bits)
        tb, ta, n = 0.0, 0.0, 0
        with torch.no_grad():
            for w in ev:
                s, k = cont_nll(B(w).logits[:, CTX - 1:-1], w)
                tb += s
                n += k
                past = A(w[:, : CTX - 1], use_cache=True).past_key_values     # A writes the KV of ctx[:-1]
                lg = B(w[:, CTX - 1: -1], past_key_values=past, use_cache=True).logits   # B reads it and continues
                s, _ = cont_nll(lg, w)
                ta += s
        nll_B, nll_AB = tb / n, ta / n
        rec = (nll_B - nll_AB) / (nll_B - nll_A) if nll_B > nll_A else float("nan")
        r = dict(bits=bits, B_ppl=math.exp(nll_B), AtoB_ppl=math.exp(nll_AB), A_ppl=math.exp(nll_A),
                 recovery_of_nll_gap=rec)
        res["runs"].append(r)
        print(json.dumps(r), flush=True)
        del B
    json.dump(res, open(a.out, "w"), indent=1)


if __name__ == "__main__":
    main()
