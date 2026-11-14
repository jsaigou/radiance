#!/usr/bin/env python3
"""gguf_q6k_dump.py -- list a GGUF's Q6_K tensors, or dump the first rows of chosen ones as raw
block bytes (210 bytes/block, exactly as stored) plus the dequantised reference in f32 for
halo_q6k_bench's `real` mode. Read-only on the GGUF.

The reference is tools/ud_repack/ggml_blocks.py's split_q6_K / decode_q6_K (the Python reference
the task names), i.e. w = (d * sc) * (q - 32) with every product exact in f32.

  gguf_q6k_dump.py list  <gguf>
  gguf_q6k_dump.py dump  <gguf> <outdir> <max_rows> <tensor>...
      writes <outdir>/<stem>.q6k   the first max_rows row-major blocks, as stored
             <outdir>/<stem>.ref   max_rows*K f32 little-endian, decode_q6_K's values
             <outdir>/<stem>.shape "rows K Nfull\\n"

A 3-D tensor (an expert stack) is read as its first max_rows rows of the first expert: the blocks
are row-major over (expert, row, K), so that prefix is contiguous. K must be a multiple of 256.
"""
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
from ud_repack.ggufhdr import read_header                    # noqa: E402
from ud_repack.ggml_blocks import split_q6_K, decode_q6_K     # noqa: E402

BLOCK_BYTES = 210
BLOCK_VALUES = 256


def main():
    if len(sys.argv) < 3:
        sys.exit(__doc__)
    mode, path = sys.argv[1], sys.argv[2]
    h = read_header(path)
    if mode == "list":
        for t in h["tensors"]:
            if t["type"] == "Q6_K":
                print(t["name"], t["shape"], t["nbytes"])
        return
    if mode != "dump":
        sys.exit("unknown mode " + mode)
    outdir, max_rows, names = sys.argv[3], int(sys.argv[4]), sys.argv[5:]
    os.makedirs(outdir, exist_ok=True)
    found = set()
    with open(path, "rb") as f:
        for t in h["tensors"]:
            if t["name"] not in names:
                continue
            found.add(t["name"])
            if t["type"] != "Q6_K" or len(t["shape"]) < 2:
                sys.exit("%s is %s %s, not a 2-D Q6_K tensor" % (t["name"], t["type"], t["shape"]))
            K, N = int(t["shape"][0]), int(t["shape"][1])
            if K % BLOCK_VALUES:
                sys.exit("%s: K=%d is not a multiple of 256" % (t["name"], K))
            rows = min(N, max_rows)
            nb = rows * (K // BLOCK_VALUES)
            f.seek(h["data_start"] + t["offset"])
            blob = f.read(nb * BLOCK_BYTES)
            if len(blob) != nb * BLOCK_BYTES:
                sys.exit("%s: short read" % t["name"])
            blocks = np.frombuffer(blob, dtype=np.uint8).reshape(nb, BLOCK_BYTES)
            ref = decode_q6_K(split_q6_K(blocks)).astype("<f4").reshape(rows, K)
            if ref.shape != (rows, K):
                sys.exit("%s: reference shape %s" % (t["name"], ref.shape))
            stem = os.path.join(outdir, t["name"])
            open(stem + ".q6k", "wb").write(blob)
            ref.tofile(stem + ".ref")
            open(stem + ".shape", "w").write("%d %d %d\n" % (rows, K, N))
            print("dumped %s: rows %d of %d, K=%d, %d blocks, %d bytes"
                  % (t["name"], rows, N, K, nb, len(blob)))
    missing = [n for n in names if n not in found]
    if missing:
        sys.exit("not found: " + ", ".join(missing))


main()
