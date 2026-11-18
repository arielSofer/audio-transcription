"""On-disk tile format for progressive weights + a deadline-based reader.

File layout (data file = tiles only; a sidecar .json describes it):
  for every tile-group (e.g. "L3.gu" = gate+up of layer 3), tiles 0..nt-1 are
  stored back to back, every tile padded to a multiple of 4096 bytes:
      tile = bit-packed low (8-bits) bits of the 8-bit codes of
             [all output rows of the group] x [`tile` input channels]
  so one tile == one aligned, sequential read (a few KB .. hundreds of KB, depending on `tile`).

base (kept in RAM) = top `bits` bits;  tile on disk = the remaining 8-bits bits.
base + tile == the original 8-bit code, bit-exactly (see test_tilefile.py).
"""
from __future__ import annotations

import json
import mmap
import os
import threading
import time
from concurrent.futures import ThreadPoolExecutor

import numpy as np

ALIGN = 4096


def _round_up(n, a=ALIGN):
    return (n + a - 1) // a * a


def pack_codes(codes: np.ndarray, nbits: int) -> bytes:
    flat = codes.reshape(-1).astype(np.uint8)
    shifts = np.arange(nbits - 1, -1, -1, dtype=np.uint8)
    bits = ((flat[:, None] >> shifts) & 1).astype(np.uint8)
    return np.packbits(bits.reshape(-1)).tobytes()


def unpack_codes(buf, n: int, nbits: int) -> np.ndarray:
    bits = np.unpackbits(np.frombuffer(buf, dtype=np.uint8))[: n * nbits].reshape(n, nbits)
    weights = (1 << np.arange(nbits - 1, -1, -1)).astype(np.uint16)
    return (bits.astype(np.uint16) * weights).sum(1).astype(np.uint8)


def write_tilefile(path: str, groups, bits: int):
    """groups: list of (name, [uint8 code tensors shaped [out, n_tiles, tile]]).  Returns index dict."""
    nb = 8 - bits
    mask = (1 << nb) - 1
    tile = groups[0][1][0].shape[2]
    index = {"bits": bits, "tile": tile, "align": ALIGN, "groups": {}}
    off = 0
    with open(path, "wb") as f:
        for name, qs in groups:
            nt = qs[0].shape[1]
            out_total = sum(q.shape[0] for q in qs)
            raw = (out_total * tile * nb + 7) // 8
            stride = _round_up(raw)
            index["groups"][name] = dict(base=off, stride=stride, raw=raw, nt=nt, out_total=out_total)
            for t in range(nt):
                arr = np.concatenate([q[:, t, :].numpy() for q in qs], axis=0) & mask
                blob = pack_codes(arr, nb)
                assert len(blob) == raw
                f.write(blob + b"\0" * (stride - raw))
            off += stride * nt
    index["file_bytes"] = off
    with open(path + ".json", "w") as f:
        json.dump(index, f)
    return index


class TileFile:
    """Reads tiles with pread; tries O_DIRECT (bypasses page cache) and falls back."""

    def __init__(self, path: str, direct: bool = True):
        self.index = json.load(open(path + ".json"))
        self.nb = 8 - self.index["bits"]
        self.direct = False
        flags = os.O_RDONLY
        if direct and hasattr(os, "O_DIRECT"):
            try:
                self.fd = os.open(path, flags | os.O_DIRECT)
                self.direct = True
                # probe: O_DIRECT is refused (EINVAL) on some filesystems
                probe = mmap.mmap(-1, ALIGN)
                os.preadv(self.fd, [probe], 0)
            except OSError:
                os.close(self.fd)
                self.fd = os.open(path, flags)
                self.direct = False
        else:
            self.fd = os.open(path, flags)
        self._local = threading.local()

    def _buf(self, size):
        b = getattr(self._local, "buf", None)
        if b is None or len(b) < size:
            b = mmap.mmap(-1, size)           # page aligned -> valid for O_DIRECT
            self._local.buf = b
        return b

    def read_tile(self, name: str, t: int) -> bytes:
        g = self.index["groups"][name]
        buf = self._buf(g["stride"])
        n = os.preadv(self.fd, [memoryview(buf)[: g["stride"]]], g["base"] + t * g["stride"])
        assert n == g["stride"], (n, g["stride"])
        return bytes(buf[: g["raw"]])

    def decode(self, name: str, blob: bytes) -> np.ndarray:
        g = self.index["groups"][name]
        tile = self.index["tile"]
        return unpack_codes(blob, g["out_total"] * tile, self.nb).reshape(g["out_total"], tile)

    def close(self):
        os.close(self.fd)


class DeadlineFetcher:
    """Request tiles in priority order, wait until a deadline, return what arrived.

    Late tiles are NOT cancelled: they keep landing in `landed` (the resident
    cache), so a miss now becomes a hit for the next token. `bandwidth` (bytes/s)
    paces all reads through one shared channel to emulate a slower / throttled SSD.
    """

    def __init__(self, tf: TileFile, workers: int = 8, bandwidth: float | None = None):
        self.tf = tf
        self.pool = ThreadPoolExecutor(max_workers=workers)
        self.bandwidth = bandwidth
        self._lock = threading.Lock()
        self._next_free = time.perf_counter()
        self.landed: dict[tuple[str, int], bytes] = {}
        self.bytes_read = 0
        self.inflight: dict[tuple[str, int], object] = {}

    def _task(self, name, t):
        if self.bandwidth:
            need = self.tf.index["groups"][name]["raw"] / self.bandwidth
            with self._lock:
                start = max(time.perf_counter(), self._next_free)
                self._next_free = start + need
            delay = start + need - time.perf_counter()
            if delay > 0:
                time.sleep(delay)
        blob = self.tf.read_tile(name, t)
        with self._lock:
            self.landed[(name, t)] = blob
            self.bytes_read += len(blob)
            self.inflight.pop((name, t), None)
        return blob

    def request(self, items, deadline: float):
        """items: iterable of (name, t) already sorted by priority. Returns arrived keys."""
        for key in items:
            if key not in self.landed and key not in self.inflight:
                self.inflight[key] = self.pool.submit(self._task, *key)
        while time.perf_counter() < deadline:
            if all(k in self.landed for k in items):
                break
            time.sleep(0.0002)
        return [k for k in items if k in self.landed]

    def close(self):
        self.pool.shutdown(wait=True)
