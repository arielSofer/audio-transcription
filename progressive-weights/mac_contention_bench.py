#!/usr/bin/env python3
"""Do SSD reads slow the GPU on Apple Silicon?  (CPU, GPU and the SSD controller share one memory system)

The papers on this idea (DecDEC, StrataFlux) were measured on CUDA GPUs with separate VRAM. On a Mac the GPU
reads the same unified memory the SSD DMA writes into, so streaming refinement tiles might steal bandwidth from
the GPU's own weight reads. This script measures exactly that:

  GPU workload : repeated 4-bit quantized GEMV over a ~1.8 GB set of matrices (memory-bound, like decoding)
  SSD workload : N threads reading `--chunk-kb` chunks from a file larger than RAM (page cache bypassed)
  For each SSD intensity it runs SSD alone, GPU alone and both together, and prints each side's slowdown.

usage:  python3 mac_contention_bench.py --size-gb 56            (needs mlx; the file is created and removed)
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import signal
import sys
import tempfile
import threading
import time

import mac_ssd_bench as B

ROWS = COLS = 8192
GROUP = 64


def build_gpu_set(mx, n_mat):
    mats = []
    for i in range(n_mat):
        w = (mx.random.normal((ROWS, COLS)) * 0.02).astype(mx.float16)
        wq, s, b = mx.quantize(w, group_size=GROUP, bits=4)
        mx.eval(wq, s, b)
        mats.append((wq, s, b))
    x = mx.random.normal((1, COLS)).astype(mx.float16)
    mx.eval(x)
    bytes_per = sum(a.nbytes for a in mats[0])
    return mats, x, bytes_per


def gpu_loop(mx, mats, x, bytes_per, stop_at):
    """Run GEMVs until stop_at; return GB/s of weight bytes streamed through the GPU."""
    done = 0
    t0 = time.perf_counter()
    while time.perf_counter() < stop_at:
        ys = [mx.quantized_matmul(x, wq, s, b, transpose=True, group_size=GROUP, bits=4) for wq, s, b in mats]
        mx.eval(ys)
        done += len(mats)
    dt = time.perf_counter() - t0
    return done * bytes_per / 1e9 / dt


def ssd_loop(path, size, threads, chunk, stop_at):
    """Sequential chunk reads per thread over disjoint segments (wrapping); return MB/s."""
    fd = os.open(path, os.O_RDONLY)
    B.nocache(fd)
    seg = size // threads // chunk * chunk
    totals = [0] * threads

    def worker(i):
        off, base = 0, i * seg
        got = 0
        while time.perf_counter() < stop_at:
            got += len(os.pread(fd, chunk, base + off))
            off = (off + chunk) % seg
        totals[i] = got

    ts = [threading.Thread(target=worker, args=(i,)) for i in range(threads)]
    t0 = time.perf_counter()
    for t in ts:
        t.start()
    for t in ts:
        t.join()
    dt = time.perf_counter() - t0
    os.close(fd)
    return sum(totals) / 1e6 / dt


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default=None)
    ap.add_argument("--size-gb", type=float, default=56.0, help="test file; must be larger than your RAM")
    ap.add_argument("--seconds", type=float, default=6.0, help="per measurement")
    ap.add_argument("--chunk-kb", type=int, default=512)
    ap.add_argument("--threads", default="1,4,16")
    ap.add_argument("--n-mat", type=int, default=48, help="matrices in the GPU working set (~38 MB each)")
    ap.add_argument("--json", default="mac_contention.json")
    a = ap.parse_args()
    import mlx.core as mx
    print(f"MLX device: {mx.default_device()}   (this should say gpu on a Mac)")
    folder = a.dir or tempfile.gettempdir()
    path = os.path.join(folder, "pw_contention.bin")
    size = int(a.size_gb * 1e9) // 4096 * 4096
    free = shutil.disk_usage(folder).free
    if free < size + 15e9:
        sys.exit(f"Not enough free disk space: {free/1e9:.0f} GB free, need {(size + 15e9)/1e9:.0f} GB. Use a smaller --size-gb.")
    for sig in (signal.SIGTERM, signal.SIGHUP):
        signal.signal(sig, lambda *_: sys.exit(1))
    chunk = a.chunk_kb * 1024
    results = []
    try:
        B.make_file(path, size)
        region = min(2_000_000_000, size) // 4096 * 4096
        cold, again = B.seq_pass(path, region, 8), B.seq_pass(path, region, 8)
        print(f"cold sequential read: {cold:.0f} MB/s, again: {again:.0f} MB/s"
              + ("  -> WARNING: looks cached, SSD numbers are inflated" if (again > 1.5 * cold or cold > 12000) else ""))
        print("building the GPU working set ...", flush=True)
        mats, x, bytes_per = build_gpu_set(mx, a.n_mat)
        gpu_loop(mx, mats, x, bytes_per, time.perf_counter() + 1.5)          # warm-up
        gpu_alone = gpu_loop(mx, mats, x, bytes_per, time.perf_counter() + a.seconds)
        print(f"\nGPU alone: {gpu_alone:.1f} GB/s of weights streamed\n")
        print(f"{'SSD threads':>11} {'SSD alone':>12} {'SSD w/ GPU':>12} {'GPU w/ SSD':>12} {'GPU slowdown':>13} {'SSD slowdown':>13}")
        for th in [int(t) for t in a.threads.split(",")]:
            ssd_alone = ssd_loop(path, size, th, chunk, time.perf_counter() + a.seconds)
            box = {}
            stop_at = time.perf_counter() + a.seconds
            t = threading.Thread(target=lambda: box.__setitem__("ssd", ssd_loop(path, size, th, chunk, stop_at)))
            t.start()
            gpu_with = gpu_loop(mx, mats, x, bytes_per, stop_at)
            t.join()
            r = dict(ssd_threads=th, ssd_alone_mbs=ssd_alone, ssd_with_gpu_mbs=box["ssd"], gpu_alone_gbs=gpu_alone,
                     gpu_with_ssd_gbs=gpu_with, gpu_slowdown_pct=100 * (1 - gpu_with / gpu_alone),
                     ssd_slowdown_pct=100 * (1 - box["ssd"] / ssd_alone))
            results.append(r)
            print(f"{th:>11} {ssd_alone:>10.0f}MB {box['ssd']:>10.0f}MB {gpu_with:>10.1f}GB {r['gpu_slowdown_pct']:>12.1f}% {r['ssd_slowdown_pct']:>12.1f}%", flush=True)
        json.dump(dict(gpu_alone_gbs=gpu_alone, rows=results), open(a.json, "w"), indent=1)
        print("\nReading it: a GPU slowdown near 0% while the SSD runs at full speed means streaming is 'free' for the GPU;"
              "\n20-40% means the tiles you stream steal bandwidth from the weights the GPU reads.")
    finally:
        if os.path.exists(path):
            os.remove(path)
            print("test file removed")


if __name__ == "__main__":
    main()
