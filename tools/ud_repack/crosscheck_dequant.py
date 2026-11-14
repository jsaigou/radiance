#!/usr/bin/env python3
"""crosscheck_dequant -- three independent dequantisers on real tensors of a GGUF, plus the .rad.

  ggml_blocks.dequant_ref   this directory's transcription of llama.cpp's dequantize_row_<type>
  gguf (python package)     gguf.quants.dequantize, the reference implementation of the gguf project
  kquant_ref (optional)     the DeepSeek-work numpy K-quant reference (~/radiance-test/ds-kquant/kquant_ref.py;
                            Q4_K there was fixed by the project lead); Q4_K, Q5_K, Q6_K only
  .rad                      radfile.decode_affine on the converted container (--rad), the same tensors

Compares f32 bit patterns on whole-tensor samples (a few thousand blocks of each type).
    python3 crosscheck_dequant.py MODEL.gguf [--rad MODEL.rad] [--kquant-ref PATH]
"""
import argparse, os, re, sys, importlib.util
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import ggufhdr
from ggml_blocks import BLOCK, dequant_ref


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('gguf'); ap.add_argument('--rad'); ap.add_argument('--kquant-ref')
    ap.add_argument('--blocks', type=int, default=4096)
    a = ap.parse_args()
    import gguf
    from gguf import quants
    kq = None
    if a.kquant_ref:
        spec = importlib.util.spec_from_file_location('kquant_ref', a.kquant_ref)
        kq = importlib.util.module_from_spec(spec); spec.loader.exec_module(kq)
    h = ggufhdr.read_header(a.gguf)
    mm = np.memmap(a.gguf, dtype=np.uint8, mode='r')
    rf = None
    if a.rad:
        from radfile import RadFile, decode_affine
        rf = RadFile(a.rad)
    seen = {}
    for t in h['tensors']:
        if t['type'] in ('Q4_K', 'Q5_K', 'Q6_K', 'Q8_0') and t['name'].endswith('weight'):
            seen.setdefault(t['type'], []).append(t)
    ok_all = True
    for ty, ts in sorted(seen.items()):
        qk, bb = BLOCK[ty]
        for t in (ts[0], ts[len(ts) // 2], ts[-1]):
            o = h['data_start'] + t['offset']
            nblk = t['nbytes'] // bb
            n = min(a.blocks, nblk)
            start = (nblk - n) // 2          # a window from the middle of the tensor
            expert = None
            if len(t['shape']) == 3:         # a stacked expert tensor: the window is the start of expert 3
                per = nblk // t['shape'][2]
                expert = 3
                n = min(a.blocks, per)
                start = expert * per
            blk = np.ascontiguousarray(mm[o + start * bb:o + (start + n) * bb]).reshape(n, bb)
            r0 = dequant_ref(ty, blk).astype(np.float32)
            r1 = quants.dequantize(blk.reshape(-1), getattr(gguf.GGMLQuantizationType, ty)).astype(np.float32).reshape(n, qk)
            res = {'ggml_blocks==gguf': np.array_equal(r0.view(np.uint32), r1.view(np.uint32))}
            if kq is not None and ty in ('Q4_K', 'Q5_K', 'Q6_K'):
                r2 = kq.dequant(blk, ty).astype(np.float32).reshape(n, qk)
                res['ggml_blocks==kquant_ref'] = np.array_equal(r0.view(np.uint32), r2.view(np.uint32))
                res['kquant_ref max|diff|'] = float(np.abs(r0 - r2).max())
            regrouped = re.search(r'attn_qkv|attn_gate|ssm_out', t['name'])    # rows / columns in another order in the .rad
            rname = t['name'] if expert is None else re.sub(r'(_exps)\.weight', r'\1.%d.weight' % expert, t['name'])
            if rf is not None and rname in rf.by_name and not regrouped and len(t['shape']) in (2, 3):
                e = rf.by_name[rname]
                cols = t['shape'][0]
                base = start - (expert * (nblk // t['shape'][2]) if expert is not None else 0)
                row0 = (base * qk) // cols
                row1 = ((base + n) * qk) // cols
                if row1 > row0 and (base * qk) % cols == 0 and ((base + n) * qk) % cols == 0:
                    d = decode_affine(rf, e, row0, row1 - row0).astype(np.float32)
                    res['rad==ggml_blocks'] = np.array_equal(d.reshape(n, qk).view(np.uint32), r0.view(np.uint32))
            ok_all &= all(v for k, v in res.items() if isinstance(v, (bool, np.bool_)))
            print(f'{ty:5s} {t["name"]:34s} blocks {start}..{start + n}: ' + ', '.join(f'{k}={v}' for k, v in res.items()))
    print('ALL EQUAL' if ok_all else 'MISMATCH')
    sys.exit(0 if ok_all else 1)


if __name__ == '__main__':
    main()
