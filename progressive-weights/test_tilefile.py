"""Check that the on-disk tile format is bit-exact, then measure real reads.

usage: python test_tilefile.py --model DIR --bits 4 --workdir SCRATCH_DIR
(the tile file is written to --workdir, not into the repo)
"""
from __future__ import annotations

import argparse
import os
import random
import time
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import torch
from transformers import AutoModelForCausalLM

import progressive as P
import tilefile as TF


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--bits", type=int, default=4)
    ap.add_argument("--workdir", required=True)
    a = ap.parse_args()
    nb = 8 - a.bits

    model = AutoModelForCausalLM.from_pretrained(a.model, dtype=torch.float32).eval()
    ctl = P.Controller()
    ctl.bits = a.bits
    groups = P.convert_model(model, ctl)
    names = []
    for i in range(len(model.model.layers)):
        for kind in ("qkv", "o", "gu", "down"):
            names.append(f"L{i}.{kind}")
    path = os.path.join(a.workdir, f"tiles_b{a.bits}.bin")
    t0 = time.time()
    idx = TF.write_tilefile(path, [(n, [m.q for m in g.members]) for n, g in zip(names, groups)], a.bits)
    print(f"wrote {idx['file_bytes']/1e6:.1f} MB in {time.time()-t0:.1f}s -> {path}")

    # ---- 1. bit-exactness: base bits | tile bits == original 8-bit code ----
    tf = TF.TileFile(path, direct=True)
    print("O_DIRECT in use:", tf.direct)
    rng = random.Random(0)
    n_checked = 0
    for _ in range(300):
        gi = rng.randrange(len(groups))
        g, name = groups[gi], names[gi]
        t = rng.randrange(g.nt)
        low = tf.decode(name, tf.read_tile(name, t))
        q = np.concatenate([m.q[:, t, :].numpy() for m in g.members], axis=0)
        base_code = (q >> nb) << nb
        assert np.array_equal(low, q & ((1 << nb) - 1)), (name, t)
        assert np.array_equal(base_code | low, q), (name, t)
        n_checked += 1
    print(f"bit-exact: {n_checked} random tiles round-trip OK (base code | tile bits == 8-bit code)")

    # ---- 2. real read throughput (this sandbox disk, NOT a real NVMe) ----
    def bench(label, items, workers):
        t0 = time.perf_counter()
        nbytes = 0
        with ThreadPoolExecutor(workers) as ex:
            for blob in ex.map(lambda k: tf.read_tile(*k), items):
                nbytes += len(blob)
        dt = time.perf_counter() - t0
        print(f"{label:<34} workers={workers:<3} {nbytes/1e6:8.1f} MB in {dt:6.2f}s = {nbytes/1e6/dt:8.1f} MB/s")

    gu = [n for n in names if n.endswith(".gu")]
    seq = [(n, t) for n in gu for t in range(idx["groups"][n]["nt"])]
    rnd = seq[:]
    rng.shuffle(rnd)
    for w in (1, 8, 32):
        bench("sequential tiles (gate/up, ~600KB)", seq, w)
    for w in (1, 8, 32):
        bench("random tiles (gate/up, ~600KB)", rnd, w)
    small = [(n, t) for n in names if n.endswith(".o") for t in range(idx["groups"][n]["nt"])]
    rng.shuffle(small)
    for w in (1, 8, 32):
        bench("random tiles (o_proj, ~57KB)", small, w)

    # ---- 3. never-wait: deadline fetch against a throttled channel ----
    for bw_mb in (2000, 400, 100):
        df = TF.DeadlineFetcher(tf, workers=8, bandwidth=bw_mb * 1e6)
        items = [("L5.gu", t) for t in range(idx["groups"]["L5.gu"]["nt"])]   # 7 tiles ~4.2MB
        want_bytes = sum(idx["groups"][n]["raw"] for n, _ in items)
        deadline_ms = 10.0
        t0 = time.perf_counter()
        got = df.request(items, t0 + deadline_ms / 1e3)
        arrived_in_time = len(got)
        time.sleep(want_bytes / (bw_mb * 1e6) + 0.05)   # let late tiles land
        late = sum(1 for k in items if k in df.landed) - arrived_in_time
        print(f"channel {bw_mb:5d} MB/s, deadline {deadline_ms:.0f} ms, {len(items)} tiles ({want_bytes/1e6:.1f} MB): "
              f"{arrived_in_time} on time, {late} landed late (cached for next token)")
        df.close()
    tf.close()


if __name__ == "__main__":
    main()
