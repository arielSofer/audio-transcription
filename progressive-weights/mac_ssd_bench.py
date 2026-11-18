#!/usr/bin/env python3
"""Measure what YOUR disk can really deliver for progressive-weights tile reads.

Run it on the Mac (stdlib only, Python 3.8+):
    python3 mac_ssd_bench.py                 # 4 GB test file in a temp folder on the system disk
    python3 mac_ssd_bench.py --size-gb 8 --dir /path/on/the/ssd

What it does
  1. writes a random test file (page cache bypassed: F_NOCACHE on macOS)
  2. reads it back in random, tile-sized chunks with 1..32 threads (like the tile fetcher)
  3. prints MB/s per (tile size, threads) and a tokens/s estimate for several model sizes

If a number is above ~12000 MB/s the OS cache is probably serving the reads and the
result is NOT the disk -> rerun with a bigger --size-gb than your free RAM.
"""
from __future__ import annotations

import argparse
import os
import random
import sys
import tempfile
import time
from concurrent.futures import ThreadPoolExecutor

IS_MAC = sys.platform == "darwin"
F_NOCACHE = 48  # fcntl constant on macOS


def nocache(fd):
    if IS_MAC:
        import fcntl
        fcntl.fcntl(fd, F_NOCACHE, 1)


def make_file(path, size_bytes):
    chunk = 16 * 1024 * 1024
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
    nocache(fd)
    t0 = time.perf_counter()
    done = 0
    while done < size_bytes:
        n = min(chunk, size_bytes - done)
        os.write(fd, os.urandom(n))
        done += n
    os.fsync(fd)
    os.close(fd)
    dt = time.perf_counter() - t0
    print(f"wrote {size_bytes/1e9:.1f} GB in {dt:.1f}s ({size_bytes/1e6/dt:.0f} MB/s write, includes random generation)")


def read_random(path, tile, threads, budget_s, size):
    fd = os.open(path, os.O_RDONLY)
    nocache(fd)
    if not IS_MAC and hasattr(os, "posix_fadvise"):
        os.posix_fadvise(fd, 0, 0, os.POSIX_FADV_DONTNEED)
    slots = size // tile
    rng = random.Random(1)
    offsets = [rng.randrange(slots) * tile for _ in range(200000)]
    stop = time.perf_counter() + budget_s
    lock_i = [0]

    def worker(_):
        got = 0
        while time.perf_counter() < stop:
            i = lock_i[0]
            lock_i[0] = i + 1
            got += len(os.pread(fd, tile, offsets[i % len(offsets)]))
        return got

    t0 = time.perf_counter()
    with ThreadPoolExecutor(threads) as ex:
        total = sum(ex.map(worker, range(threads)))
    dt = time.perf_counter() - t0
    os.close(fd)
    return total / 1e6 / dt


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default=None, help="folder on the disk you want to measure")
    ap.add_argument("--size-gb", type=float, default=4.0)
    ap.add_argument("--seconds", type=float, default=3.0, help="per measurement")
    ap.add_argument("--tile-kb", default="32,128,512,2048")
    ap.add_argument("--threads", default="1,4,8,16,32")
    a = ap.parse_args()
    tiles = [int(x) * 1024 for x in a.tile_kb.split(",")]
    threads = [int(x) for x in a.threads.split(",")]
    size = int(a.size_gb * 1e9) // 4096 * 4096
    folder = a.dir or tempfile.gettempdir()
    path = os.path.join(folder, "pw_bench.bin")
    print(f"platform={sys.platform}  file={path}  size={size/1e9:.1f} GB  (page cache bypass: {'F_NOCACHE' if IS_MAC else 'posix_fadvise'})")
    make_file(path, size)
    try:
        best = {}
        print(f"\n{'tile':>8} " + " ".join(f"{t:>8}T" for t in threads) + "   (MB/s, random tile reads)")
        for tile in tiles:
            row = []
            for th in threads:
                mbs = read_random(path, tile, th, a.seconds, size)
                row.append(mbs)
            best[tile] = max(row)
            print(f"{tile//1024:>6}KB " + " ".join(f"{v:9.0f}" for v in row))
        peak = max(best.values())
        if peak > 12000:
            print("\nWARNING: >12000 MB/s -> probably served from the OS cache; use a larger --size-gb.")
        print("\nProjected tokens/s if every token has to fetch X MB of refinement tiles (using the best tile size above):")
        use = best[min(best, key=lambda t: abs(t - 512 * 1024))] if 512 * 1024 in best else peak
        print(f"(assuming {use:.0f} MB/s)\n")
        print(f"{'model (4-bit base)':<22}{'refinement total':>18}   tokens/s at f=5% / 10% / 20% of tiles fetched per token (no cache)")
        for name, params_b in (("8B", 8), ("32B", 32), ("70B", 70)):
            total_mb = params_b * 1e9 * 0.5 / 1e6          # 4 refinement bits per weight
            rates = [use / (total_mb * f) for f in (0.05, 0.10, 0.20)]
            print(f"{name:<22}{total_mb/1000:>15.1f} GB   " + " / ".join(f"{r:6.1f}" for r in rates))
        print("\nThe real number is higher when a cache keeps recently used tiles (results/SUMMARY.md shows how many tiles repeat between tokens).")
    finally:
        os.remove(path)


if __name__ == "__main__":
    main()
