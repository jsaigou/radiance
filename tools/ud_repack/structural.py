#!/usr/bin/env python3
"""structural -- when could a GGUF block ALREADY lie on a grid an existing rad format can express?

For real tensors this counts, per block, the exact-representability conditions of the candidate
formats (not a fit: yes/no per block, from the integer fields of the block).
  Q8_0 -> i8*bf16[1x128] needs: the four d of a 128-group equal, and that d a bf16 (low 13 f16 bits ... see code)
  Q8_0 -> i8*f16[1x32]   is the canonical plane (always exact) but no libhalo kernel reads group 32
  Q4_K -> u4 f16 scale, integer zero, per 32: needs d*sc f16-representable and dmin*mm/(d*sc) an integer in 0..255
"""
import sys, os, json
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from ud_repack import load_all
from ggml_blocks import SPLIT
from ggufhdr import BLK


def f16_exact(x32):
    return x32.astype(np.float16).astype(np.float32) == x32


def bf16_exact(x32):
    return (np.ascontiguousarray(x32, np.float32).view(np.uint32) & 0xFFFF) == 0


def main():
    gguf, name, expert, rows = sys.argv[1], sys.argv[2], int(sys.argv[3]), int(sys.argv[4])
    kv, ts = load_all([gguf]); t = [x for x in ts if x['name'] == name][0]
    ty = t['type']; K = t['shape'][0]; N = t['shape'][1]; E = t['shape'][2] if len(t['shape']) > 2 else 1
    qk, bb = BLK[ty]
    raw = np.memmap(t['file'], np.uint8, 'r', t['data_start'] + t['offset'], (t['nbytes'],)).reshape(E, N, K // qk, bb)[expert if E > 1 else 0, :rows]
    pl = SPLIT[ty](np.ascontiguousarray(raw).reshape(-1, bb))
    out = dict(tensor=name, type=ty, rows=rows)
    if ty == 'Q8_0':
        d = pl['d'].astype(np.float32).reshape(rows, K // 32)
        g = d.reshape(rows, K // 128, 4)
        eq4 = (g == g[:, :, :1]).all(axis=2)
        out['frac_128groups_all_four_d_equal'] = float(eq4.mean())
        out['frac_d_f16_is_bf16_exact'] = float(bf16_exact(d).mean())
        out['frac_128groups_i8_bf16_exact'] = float((eq4 & bf16_exact(g[:, :, 0])).mean())
        out['frac_zero_blocks'] = float((d == 0).mean())
    elif ty in ('Q4_K', 'Q5_K'):
        d1 = pl['d'].astype(np.float32)[:, None] * pl['sc'].astype(np.float32)
        m1 = pl['dmin'].astype(np.float32)[:, None] * pl['mm'].astype(np.float32)
        with np.errstate(divide='ignore', invalid='ignore'):
            z = m1 / d1
        zint = (np.abs(z - np.round(z)) == 0) & (z >= -255) & (z <= 255)
        out['frac_subblocks_d1_f16_exact'] = float(f16_exact(d1).mean())
        out['frac_subblocks_zero_point_integer'] = float(zint.mean())
        out['frac_subblocks_both'] = float((f16_exact(d1) & zint).mean())
        out['frac_subblocks_min_is_zero(mm==0)'] = float((pl['mm'] == 0).mean())
        out['frac_32blocks_sc_power_of_two'] = float(np.isin(pl['sc'], [1, 2, 4, 8, 16, 32]).mean())
    print(json.dumps(out))


if __name__ == '__main__':
    main()
