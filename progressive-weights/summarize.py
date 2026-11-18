"""Turn results/results.jsonl into markdown tables (printed + results/SUMMARY.md)."""
from __future__ import annotations

import json
import sys
from collections import defaultdict

PARAMS_BLOCKS = 357_826_560  # linear weights inside the 24 blocks of Qwen2.5-0.5B (q,k,v,o,gate,up,down)


def load(path):
    return [json.loads(l) for l in open(path)]


def fmt(x, nd=2):
    return "-" if x is None else f"{x:.{nd}f}"


def main(path="results/results.jsonl", out="results/SUMMARY.md"):
    rs = load(path)
    L = []
    ref = next((r for r in rs if r["mode"] == "fp32"), None)
    full = next((r for r in rs if r["mode"] == "full"), None)
    L.append("## 0. Reference points (WikiText-2 test, 8 windows x 512 tokens, Qwen2.5-0.5B)\n")
    L.append("| model | perplexity |\n|---|---|")
    if ref:
        L.append(f"| original (bf16 weights, fp32 math) | {ref['ppl']:.3f} |")
    if full:
        L.append(f"| 8-bit RTN, all tiles refined | {full['ppl']:.3f} |")
    for r in sorted((r for r in rs if r["mode"] == "base"), key=lambda r: -r["bits"]):
        L.append(f"| base only, {r['bits']}-bit (no refinement) | {r['ppl']:.3f} |")

    L.append("\n## 1. Quality vs. fraction of tiles refined per token\n")
    L.append("`f` = fraction of the input tiles that get their refinement bits for each token (tiles are 32 input channels wide). "
             "dynamic = chosen per token by |x_tile|*||R_tile||, static = fixed set chosen on calibration text "
             "(needs that extra RAM permanently), random = control. "
             "`read %` = refinement bytes touched per token as % of ALL refinement bytes (before any cache).\n")
    by = defaultdict(dict)
    for r in rs:
        if r["suite"] == "main":
            by[r["bits"]][(r["mode"], r["frac"])] = r
    for bits in sorted(by, reverse=True):
        fracs = sorted({f for (_, f) in by[bits]})
        L.append(f"### base = {bits} bit (RAM), refinement = {8 - bits} bit planes on SSD\n")
        L.append("| f | dynamic | static | random | read % (dynamic) | read MB/token |\n|---|---|---|---|---|---|")
        total = PARAMS_BLOCKS * (8 - bits) / 8
        for f in fracs:
            d, s, ra = (by[bits].get((m, f)) for m in ("dynamic", "static", "random"))
            rb = d["refine_bytes_per_token"] if d else None
            L.append(f"| {f:.2f} | {fmt(d and d['ppl'])} | {fmt(s and s['ppl'])} | {fmt(ra and ra['ppl'])} | "
                     f"{fmt(rb and 100 * rb / total, 1)} | {fmt(rb and rb / 1e6, 1)} |")
        L.append("")

    L.append("## 2. Coherence between consecutive tokens and energy coverage (dynamic mode)\n")
    L.append("overlap = fraction of a token's selected tiles that were also selected for the previous token "
             "(random selection would give about f). energy = share of sum(score^2) captured by the selected tiles.\n")
    L.append("| base | f | kind | overlap | energy covered |\n|---|---|---|---|---|")
    for bits in sorted(by, reverse=True):
        for f in sorted({f for (_, f) in by[bits]}):
            d = by[bits].get(("dynamic", f))
            if not d:
                continue
            for kind in ("qkv", "o", "gu", "down"):
                if kind in d.get("overlap", {}):
                    L.append(f"| {bits} | {f:.2f} | {kind} | {d['overlap'][kind]:.2f} | {d['energy'][kind]:.2f} |")

    pol = [r for r in rs if r["suite"] == "policy"]
    if pol:
        L.append("\n## 3. Never-wait policy: bounded fetch budget + resident-tile cache\n")
        L.append("budget = max NEW tiles fetched per token per group, as a fraction of the k wanted tiles (0 = never fetch, only reuse what is cached); "
                 "cap = cache size in multiples of k (tiles kept resident). served = share of the wanted tiles that were usable.\n")
        bits = pol[0]["bits"]
        total = PARAMS_BLOCKS * (8 - bits) / 8
        L.append(f"base = {bits} bit, f = {pol[0]['frac']}\n")
        L.append("| cap | budget | perplexity | fetched MB/token | fetched % of refinement | served |\n|---|---|---|---|---|---|")
        for r in sorted(pol, key=lambda r: (r["cap_mult"] is None, r["cap_mult"] or 0, r["budget"] is None, r["budget"] or 0)):
            uf = r.get("used_frac", {})
            served = sum(uf.values()) / len(uf) if uf else None
            L.append(f"| {r['cap_mult'] if r['cap_mult'] is not None else 'inf'} | {r['budget'] if r['budget'] is not None else 'inf'} | "
                     f"{r['ppl']:.2f} | {r['fetch_bytes_per_token'] / 1e6:.2f} | {100 * r['fetch_bytes_per_token'] / total:.1f} | {fmt(served)} |")
    text = "\n".join(L) + "\n"
    open(out, "w").write(text)
    print(text)


if __name__ == "__main__":
    main(*sys.argv[1:])
