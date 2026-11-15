#!/usr/bin/env python3
"""extract.py -- the Unsloth UD release's embedding tables as the container's planes, and a small
sample with the reference's decode for rad-avx-iq4 and halo_selftest.

Two tensors are served by the same gather:

  per_layer_token_embd.weight  IQ4_NL  the 26.8 GiB n-gram table (the PLE lane)
  token_embd.weight            IQ4_XS  the 0.315 GiB main token embedding (this lane)

An IQ4_NL row is `n_embd / 32` blocks of 18 bytes: two f16 `d` bytes then sixteen `qs` bytes,
element j of the block the low nibble of qs[j] and j+16 the high one (tools/ud_repack/
ggml_blocks.py split_iq4_nl). The container stores the value-order nibbles packed two a byte
(`codes`, n_embd / 2 bytes a row) and the f16 `d` of each block (`scales`, one per 32 values),
which is what rad_encoding.h's IQ4_NL encoding names.

An IQ4_XS row is `n_embd / 256` blocks of 136 bytes: f16 `d` at 0, `scales_l[4]` at 4, a u16
`scales_h` at 2 and `qs[128]` at 8 (split_iq4_xs). The container stores `codes` the same way,
`scales` as the signed six-bit `ls - 32` (one per 32 values, packed along the row) and `scale1` as
the f16 `d` (one per 256) -- rad_encoding.h's IQ4_XS encoding.

  extract.py sample  --gguf SHARD --out DIR [--tensor NAME] [--rows 4096] [--seed 1]
      codes.bin, scales.bin, ids.bin, expected.bf16, plus scale1.bin for IQ4_XS.
      `expected.bf16` is u16 bf16 of the reference decode rounded to nearest even, the value the
      device kernel and libavx must both write.

  extract.py full --gguf SHARD --out DIR [--tensor NAME] [--rows N]
      the planes for the whole table, streamed.

The reference arithmetic is tools/ud_repack/ggml_blocks.py's decode_iq4_nl / decode_iq4_xs; this
file re-derives the planes from the block bytes and uses the same KVALUES table.
"""
import argparse
import os
import sys

import numpy as np

KVALUES_IQ4NL = np.array([-127, -104, -83, -65, -49, -35, -22, -10,
                          1, 13, 25, 38, 53, 69, 89, 113], np.int8)


def bf16_rne(f32):
    """rad_f32_to_bf16 (abi/rad_plugin.h): nearest even, NaN kept quiet, overflow to infinity."""
    u = np.ascontiguousarray(f32, dtype=np.float32).view(np.uint32)
    a = u & np.uint32(0x7fffffff)
    out = np.where((a > np.uint32(0x7f800000)),
                   (u >> np.uint32(16)) | np.uint32(0x0040),
                   (u + np.uint32(0x7fff) + ((u >> np.uint32(16)) & np.uint32(1))) >> np.uint32(16))
    return out.astype(np.uint16)


def split_row(b, n_embd):
    """A row's blocks (nb, 18) -> (codes (nb, 32) value order, d (nb,) f16 bits as uint16)."""
    nb = n_embd // 32
    blocks = np.ascontiguousarray(b[:nb * 18]).reshape(nb, 18)
    d = np.ascontiguousarray(blocks[:, 0:2]).view('<f2').reshape(nb)
    qs = blocks[:, 2:18]
    codes = np.concatenate([qs & 15, qs >> 4], axis=1).astype(np.uint8)   # value order
    return codes, d


def pack_codes(codes):
    """(rows, n_embd) value-order codes -> (rows, n_embd / 2) packed two a byte."""
    c = np.ascontiguousarray(codes, dtype=np.uint8)
    return (c[:, 0::2] & 15) | ((c[:, 1::2] & 15) << 4)


def decode(codes, d):
    """(rows, nb, 32) codes and (rows, nb) f16 bits -> (rows, n_embd) f32, through tools/ud_repack's
    own decode_iq4_nl (the Python reference), exact in f32."""
    try:
        import ggml_blocks
    except ImportError:
        sys.exit("ggml_blocks.py not found: keep tools/ud_repack beside tools/ple_iq4")
    rows, nb, _ = codes.shape
    p = {'codes': np.ascontiguousarray(codes).reshape(rows * nb, 32),
         'd': np.ascontiguousarray(d).reshape(rows * nb)}
    return ggml_blocks.decode_iq4_nl(p).reshape(rows, nb * 32)


def find_table(path, tensor='per_layer_token_embd.weight'):
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'ud_repack'))
    try:
        import ggufhdr
    except ImportError:
        sys.exit("ggufhdr.py not found: keep tools/ud_repack beside tools/ple_iq4, or pass the "
                 "GGUF header fields yourself")
    h = ggufhdr.read_header(path)
    for t in h['tensors']:
        if t['name'] == tensor:
            if t['type'] not in ('IQ4_NL', 'IQ4_XS'):
                sys.exit(f"{tensor} is {t['type']}, not IQ4_NL or IQ4_XS")
            n_embd = int(t['shape'][0])
            rows = int(t['shape'][1])
            return h['data_start'] + int(t['offset']), rows, n_embd, t['type']
    sys.exit(f"{path}: no {tensor}")


def pack6(v):
    """(rows, m) unsigned ints 0..63 -> (rows, ceil(m*6/8)) uint8, LSB first along the row."""
    rows, m = v.shape
    bits = ((v.astype(np.uint32)[:, :, None] >> np.arange(6, dtype=np.uint32)) & 1)
    return np.packbits(bits.reshape(rows, m * 6), axis=1, bitorder='little')


def sample(args):
    base, rows, n_embd, tname = find_table(args.gguf, args.tensor)
    if tname == 'IQ4_XS' and n_embd % 256:
        sys.exit(f"{args.tensor}: n_embd {n_embd} is not whole 256s")
    os.makedirs(args.out, exist_ok=True)
    rng = np.random.default_rng(args.seed)
    n = min(args.rows, rows)
    # The table rows to draw, then the sample is the small image: row k of the image is a random
    # real table row, and the ids the gather reads index the image (0..n-1).
    ids = rng.choice(rows, size=n, replace=False).astype(np.int64)
    if tname == 'IQ4_NL':
        nb = n_embd // 32
        row_bytes = nb * 18
        codes = np.empty((n, n_embd), np.uint8)
        d = np.empty((n, nb), '<f2')
        with open(args.gguf, 'rb') as f:
            for k, rid in enumerate(ids):
                f.seek(base + int(rid) * row_bytes)
                b = np.frombuffer(f.read(row_bytes), np.uint8)
                c, dd = split_row(b, n_embd)
                codes[k] = c.reshape(-1)
                d[k] = dd
        packed = pack_codes(codes)
        vals = decode(codes.reshape(n, nb, 32), d)
        expected = bf16_rne(vals)
        open(os.path.join(args.out, 'codes.bin'), 'wb').write(packed.tobytes())
        open(os.path.join(args.out, 'scales.bin'), 'wb').write(d.view(np.uint16).reshape(-1).tobytes())
    else:
        import ggml_blocks
        nb = n_embd // 256
        row_bytes = nb * 136
        blk = np.empty((n * nb, 136), np.uint8)
        with open(args.gguf, 'rb') as f:
            for k, rid in enumerate(ids):
                f.seek(base + int(rid) * row_bytes)
                blk[k * nb:(k + 1) * nb] = np.frombuffer(f.read(row_bytes), np.uint8).reshape(nb, 136)
        p = ggml_blocks.split_iq4_xs(blk)
        codes = np.concatenate([p['codes'][k * nb:(k + 1) * nb].reshape(-1)
                                for k in range(n)]).reshape(n, n_embd)
        # ls is stored in the four values' low nibbles and a u16's top two bits per sub-block; the
        # plane keeps it as a signed six-bit `ls - 32`, left to right along the row.
        ls = np.concatenate([p['ls'][k * nb:(k + 1) * nb].reshape(-1) for k in range(n)])
        d = p['d']
        packed = pack_codes(codes)
        vals = ggml_blocks.decode_iq4_xs({'codes': p['codes'], 'ls': p['ls'], 'd': p['d']})
        expected = bf16_rne(vals.reshape(n, n_embd))
        open(os.path.join(args.out, 'codes.bin'), 'wb').write(packed.tobytes())
        s6 = ls.reshape(n, n_embd // 32).astype(np.int64) - 32
        # The plane's elements are the signed six-bit values; pack their two's-complement codes.
        open(os.path.join(args.out, 'scales.bin'), 'wb').write(
            pack6((s6 & 63).astype(np.uint8)).tobytes())
        open(os.path.join(args.out, 'scale1.bin'), 'wb').write(d.astype('<f2').tobytes())
    open(os.path.join(args.out, 'ids.bin'), 'wb').write(np.arange(n, dtype=np.int32).tobytes())
    open(os.path.join(args.out, 'expected.bf16'), 'wb').write(expected.tobytes())
    print(f"sample: {n} rows of {n_embd} {tname} from {args.gguf} (table {rows} rows) -> {args.out}")
    print(f"  codes {packed.nbytes} B, ids {ids.nbytes} B, expected {expected.nbytes} B")


def full(args):
    base, rows, n_embd, tname = find_table(args.gguf, args.tensor)
    total = min(args.rows, rows) if args.rows else rows
    if tname == 'IQ4_XS':
        import ggml_blocks
        if n_embd % 256:
            sys.exit(f"{args.tensor}: n_embd {n_embd} is not whole 256s")
        nb = n_embd // 256
        row_bytes = nb * 136
        os.makedirs(args.out, exist_ok=True)
        cpath = os.path.join(args.out, 'codes.bin')
        spath = os.path.join(args.out, 'scales.bin')
        dpath = os.path.join(args.out, 'scale1.bin')
        chunk = 1 << 15
        done = 0
        with open(args.gguf, 'rb') as f, open(cpath, 'wb') as fc, open(spath, 'wb') as fs, \
                open(dpath, 'wb') as fd:
            f.seek(base)
            while done < total:
                n = min(chunk, total - done)
                b = np.frombuffer(f.read(n * row_bytes), np.uint8)
                if b.size != n * row_bytes:
                    sys.exit(f"short read at row {done}")
                p = ggml_blocks.split_iq4_xs(b.reshape(n * nb, 136))
                # value order, packed two a byte, per row
                fc.write(pack_codes(p['codes'].reshape(n, n_embd)).tobytes())
                s6 = (p['ls'].astype(np.int64).reshape(n, n_embd // 32) - 32) & 63
                fs.write(pack6(s6.astype(np.uint8)).tobytes())
                fd.write(p['d'].astype('<f2').tobytes())
                done += n
                if done % (chunk * 8) == 0 or done == total:
                    print(f"  {done}/{total} rows", flush=True)
        print(f"full: {total} rows {tname} -> {cpath} ({os.path.getsize(cpath)} B), {spath} "
              f"({os.path.getsize(spath)} B), {dpath} ({os.path.getsize(dpath)} B)")
        return
    if tname != 'IQ4_NL':
        sys.exit(f"full extraction is not implemented for {tname}")
    nb = n_embd // 32
    row_bytes = nb * 18
    os.makedirs(args.out, exist_ok=True)
    cpath = os.path.join(args.out, 'codes.bin')
    spath = os.path.join(args.out, 'scales.bin')
    chunk = 1 << 18
    done = 0
    with open(args.gguf, 'rb') as f, open(cpath, 'wb') as fc, open(spath, 'wb') as fs:
        f.seek(base)
        while done < total:
            n = min(chunk, total - done)
            b = np.frombuffer(f.read(n * row_bytes), np.uint8)
            if b.size != n * row_bytes:
                sys.exit(f"short read at row {done}")
            blk = b.reshape(n, nb, 18)
            qs = blk[:, :, 2:18]
            # value order: low nibbles of qs[j] are values 0..15, high nibbles 16..31
            lo = qs & 15
            hi = qs >> 4
            # pack value 2k low, 2k+1 high within each 32: pairs come from [lo, hi] interleaved
            both = np.concatenate([lo, hi], axis=2).reshape(n, nb, 32)
            packed = (both[:, :, 0::2] & 15) | ((both[:, :, 1::2] & 15) << 4)
            fc.write(packed.reshape(n, -1).tobytes())
            fs.write(np.ascontiguousarray(blk[:, :, 0:2]).view('<f2').reshape(n, nb).view(np.uint16).tobytes())
            done += n
            if done % (chunk * 16) == 0 or done == total:
                print(f"  {done}/{total} rows", flush=True)
    print(f"full: {total} rows -> {cpath} ({os.path.getsize(cpath)} B), {spath} "
          f"({os.path.getsize(spath)} B)")


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest='cmd', required=True)
    for name in ('sample', 'full'):
        p = sub.add_parser(name)
        p.add_argument('--gguf', required=True)
        p.add_argument('--out', required=True)
        p.add_argument('--tensor', default='per_layer_token_embd.weight')
        p.add_argument('--rows', type=int, default=4096 if name == 'sample' else 0)
        if name == 'sample':
            p.add_argument('--seed', type=int, default=1)
    a = ap.parse_args()
    (sample if a.cmd == 'sample' else full)(a)


if __name__ == '__main__':
    main()
