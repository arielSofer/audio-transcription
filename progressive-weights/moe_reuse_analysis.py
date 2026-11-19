"""How much do MoE experts repeat between tokens?  Reads the trace written by `moe_stream.py bench --trace`.

For every layer, simulates (a) an LRU cache of C experts and (b) a fixed set of the C experts the PROMPT used most
(the "decide once" idea), and reports the hit rate on the tokens after the prompt, for several C.
usage: python moe_reuse_analysis.py trace.json --experts-per-layer 128 --top-k 8 [--prompt 64]
"""
import argparse, collections, json

ap = argparse.ArgumentParser()
ap.add_argument("trace")
ap.add_argument("--experts-per-layer", type=int, required=True)
ap.add_argument("--top-k", type=int, required=True)
ap.add_argument("--prompt", type=int, default=64)
a = ap.parse_args()
E, K = a.experts_per_layer, a.top_k
trace = json.load(open(a.trace))
seqs, cur = [], collections.defaultdict(list)             # list of {layer: [expert sets per token]}
for path, ex in trace:
    if path == "__window__":
        if cur: seqs.append(cur)
        cur = collections.defaultdict(list)
    else:
        cur[path].append(ex)
if cur: seqs.append(cur)
print(f"{len(seqs)} sequences, {sum(len(v) for v in seqs[0].values()) // max(len(seqs[0]), 1)} tokens per layer in the first, "
      f"{len(seqs[0])} MoE layers, {E} experts per layer, top-{K}")
sizes = sorted({max(K, round(E * f)) for f in (0.0625, 0.125, 0.25, 0.5, 0.75, 1.0)})
print(f"{'cache C':>8} {'C/E':>6} | {'LRU hit rate':>12} {'(after prompt)':>15} | {'prompt-fixed set':>16}")
for C in sizes:
    lru_hit = lru_tot = lru_hit_p = lru_tot_p = fix_hit = fix_tot = 0
    for seq in seqs:
        for layer, toks in seq.items():
            cache = collections.OrderedDict()
            counts = collections.Counter(e for t in toks[:a.prompt] for e in t)
            fixed = {e for e, _ in counts.most_common(C)}
            for i, t in enumerate(toks):
                for e in t:
                    h = e in cache
                    lru_hit += h; lru_tot += 1
                    if i >= a.prompt: lru_hit_p += h; lru_tot_p += 1
                    if i >= a.prompt: fix_hit += (e in fixed); fix_tot += 1
                for e in t:
                    cache[e] = 1; cache.move_to_end(e)
                while len(cache) > C: cache.popitem(last=False)
    print(f"{C:>8} {C / E:>6.2f} | {lru_hit / max(lru_tot, 1):>12.3f} {lru_hit_p / max(lru_tot_p, 1):>15.3f} | {fix_hit / max(fix_tot, 1):>16.3f}")
print("\nhit rate = share of the (token, expert) activations whose expert was already in RAM; each miss costs one SSD read of one expert.")
