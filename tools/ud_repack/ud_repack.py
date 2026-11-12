#!/usr/bin/env python3
"""ud_repack -- CPU-only prototype of the lossless GGUF -> kq-layout repack.

  ud_repack.py plan  MODEL.gguf [more shards...]   per-tensor plan: type, bytes, target layout, exactness,
                                                    bytes after repack, share of per-token reads
  ud_repack.py fingerprint MODEL.gguf --tensors REGEX   sha256 of the dequantised f32 words, from the GGUF and from the kq planes
  ud_repack.py check MODEL.gguf [--layers 0,1,40] [--experts 0,77] [--tensors REGEX]
                                                    repack real tensors and prove value equality
                                                    (bitwise f32 dequant, byte-exact inverse)

Nothing is written to disk and nothing is requantised. `check` is the proof obligation of the
lossless claim; `plan` is what a `rad-convert --gguf-plan` mode would print.
"""
import argparse, collections, hashlib, json, re, sys, os
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from ggufhdr import read_header, BLK
from ggml_blocks import BLOCK, SPLIT, DECODE, MERGE, dequant_ref
import kq_layout as kq

# type -> (target layout name, exactness class)
#   exact-planes : every ggml field is carried bit for bit; decode in f32 reproduces dequantize_row_<type>
#   exact-grid   : exact in principle, needs a codebook kernel (grid in LDS) that does not exist; not prototyped
#   copy         : floating-point tensor, kept as it is (or cast losslessly)
TARGET = {
    'Q8_0': ('kq8', 'exact-planes'), 'Q4_K': ('kq4k', 'exact-planes'), 'Q5_K': ('kq5k', 'exact-planes'),
    'Q6_K': ('kq6k', 'exact-planes'), 'Q5_0': ('kq5_0', 'exact-planes'), 'IQ4_NL': ('kiq4nl', 'exact-planes'),
    'IQ4_XS': ('kiq4xs', 'exact-planes'),
    'IQ3_XXS': ('kiq3xxs', 'exact-grid'), 'IQ2_S': ('kiq2s', 'exact-grid'), 'IQ2_XS': ('kiq2xs', 'exact-grid'),
    'IQ2_XXS': ('kiq2xxs', 'exact-grid'), 'IQ3_S': ('kiq3s', 'exact-grid'), 'IQ1_S': ('kiq1s', 'exact-grid'),
    'Q2_K': ('kq2k', 'exact-planes-unprototyped'), 'Q3_K': ('kq3k', 'exact-planes-unprototyped'),
    'Q4_0': ('kq4_0', 'exact-planes-unprototyped'), 'Q4_1': ('kq4_1', 'exact-planes-unprototyped'),
    'F32': ('f32', 'copy'), 'F16': ('f16', 'copy'), 'BF16': ('bf16', 'copy'),
}


def role_of(name):
    n = re.sub(r'^blk\.\d+\.', '', name)
    if re.match(r'ffn_(gate|up|down)_exps', n): return 'routed-expert ' + n.split('_')[1]
    if 'shexp' in n: return 'shared-expert'
    if n in ('attn_q.weight', 'attn_k.weight', 'attn_v.weight', 'attn_output.weight'): return 'attention'
    if n in ('attn_qkv.weight', 'attn_gate.weight', 'ssm_out.weight') or n.startswith('ssm_'): return 'delta-net'
    if n.startswith('nextn'): return 'mtp'
    if n == 'token_embd.weight' or n.startswith('per_layer_token_embd'): return 'embedding'
    if n == 'output.weight': return 'output'
    if 'norm' in n: return 'norm'
    if n.startswith('ffn_gate_inp'): return 'router'
    return 'other'


def layer_of(name):
    m = re.match(r'blk\.(\d+)\.', name)
    return int(m[1]) if m else -1


def load_all(paths):
    ts, kv = [], None
    for p in paths:
        h = read_header(p)
        if kv is None or 'general.architecture' in h['kv']: kv = kv or h['kv']
        # tensor sizes of unknown types come from the offset gap to the next tensor
        order = sorted(h['tensors'], key=lambda t: t['offset'])
        for i, t in enumerate(order):
            t['file'] = p; t['data_start'] = h['data_start']
            if t['nbytes'] == 0:
                nxt = order[i + 1]['offset'] if i + 1 < len(order) else os.path.getsize(p) - h['data_start']
                t['nbytes'] = nxt - t['offset']; t['nbytes_inferred'] = True
        ts += order
    return kv, ts


def tensor_bytes(t):
    return np.memmap(t['file'], dtype=np.uint8, mode='r', offset=t['data_start'] + t['offset'], shape=(t['nbytes'],))


def plan(args):
    kv, ts = load_all(args.gguf)
    arch = kv.get('general.architecture', '?')
    ne_all = kv.get(f'{arch}.expert_count', 0); ne_used = kv.get(f'{arch}.expert_used_count', 0)
    nl = kv.get(f'{arch}.block_count', 0); nextn = kv.get(f'{arch}.nextn_predict_layers', 0)
    rows = []
    for t in ts:
        ty = t['type']; tg, ex = TARGET.get(ty, ('?', 'unknown-type'))
        K = t['shape'][0]
        qk = BLOCK.get(ty, (1, 0))[0]
        if ex == 'exact-planes' and K % qk == 0:
            bpw_after = kq.bits_per_weight(ty, args.variant); after = int(t['nelem'] * bpw_after / 8)
        else:
            after = t['nbytes']
        exps = role_of(t['name']).startswith('routed-expert')
        main = layer_of(t['name']) < nl - nextn
        frac = (ne_used / ne_all if exps else 1.0) if main else 0.0
        if role_of(t['name']) == 'embedding': frac = 0.0   # a row lookup per token, not a streamed matrix
        rows.append(dict(name=t['name'], type=ty, shape=t['shape'], role=role_of(t['name']), bytes=t['nbytes'], target=tg,
                         exactness=ex, bytes_after=after, per_token_bytes=int(t['nbytes'] * frac), mtp_layer=not main and layer_of(t['name']) >= 0))
    if args.json:
        json.dump(rows, open(args.json, 'w'))
    tot = sum(r['bytes'] for r in rows); tok = sum(r['per_token_bytes'] for r in rows)
    print(f'{arch}: {len(rows)} tensors, {tot/2**30:.3f} GiB, ~{tok/2**30:.3f} GiB read per decoded token (main path, {ne_used}/{ne_all} experts)')
    by = collections.OrderedDict()
    for r in rows:
        k = (r['role'], r['type'], r['target'], r['exactness'])
        a = by.setdefault(k, [0, 0, 0, 0]); a[0] += 1; a[1] += r['bytes']; a[2] += r['bytes_after']; a[3] += r['per_token_bytes']
    print(f"{'role':22s}{'ggml':9s}{'target':9s}{'exactness':27s}{'n':>4s}{'MiB':>10s}{'%file':>7s}{'after MiB':>11s}{'read/tok MiB':>13s}{'%read':>7s}")
    for (role, ty, tg, ex), (n, b, a, p) in sorted(by.items(), key=lambda x: -x[1][1]):
        print(f'{role:22s}{ty:9s}{tg:9s}{ex:27s}{n:4d}{b/2**20:10.1f}{100*b/tot:7.2f}{a/2**20:11.1f}{p/2**20:13.1f}{100*p/max(tok,1):7.2f}')
    ex_tot = collections.Counter()
    for r in rows: ex_tot[r['exactness']] += r['bytes']
    print('bytes by exactness class:', {k: f'{v/2**30:.3f} GiB ({100*v/tot:.1f}%)' for k, v in ex_tot.items()})
    print(f"after repack ({args.variant} scales): {sum(r['bytes_after'] for r in rows)/2**30:.3f} GiB")


def _bits_equal(a, b):
    return bool(np.array_equal(a.view(np.uint32), b.view(np.uint32)))


def check(args):
    kv, ts = load_all(args.gguf)
    sel = re.compile(args.tensors) if args.tensors else None
    layers = set(int(x) for x in args.layers.split(',')) if args.layers else None
    experts = [int(x) for x in args.experts.split(',')]
    out = []
    for t in ts:
        ty = t['type']
        if ty not in SPLIT or TARGET[ty][1] != 'exact-planes': continue
        if sel and not sel.search(t['name']): continue
        if layers is not None and layer_of(t['name']) not in layers: continue
        K = t['shape'][0]; N = t['shape'][1] if len(t['shape']) > 1 else 1
        E = t['shape'][2] if len(t['shape']) > 2 else 1
        qk, bb = BLOCK[ty]
        if K % qk: continue
        raw = tensor_bytes(t).reshape(E, N, K // qk, bb)
        for e in (experts if E > 1 else [0]):
            if e >= E: continue
            blocks = np.ascontiguousarray(raw[e]).reshape(-1, bb)
            ref = dequant_ref(ty, blocks)
            pl = SPLIT[ty](blocks)
            mine = DECODE[ty](pl)
            # the information-preservation claim: blocks -> planes -> blocks is the identity
            inv_ok = bool(np.array_equal(MERGE[ty](pl), blocks))
            rec = dict(tensor=t['name'], type=ty, expert=e, rows=N, K=K, merge_inverse_exact=inv_ok,
                       decode_bit_equal_ref=_bits_equal(mine, ref))
            for var in ('P', 'A'):
                ser = kq.serialize(ty, pl, N, var)
                back = kq.deserialize(ty, ser, K, var)
                same = all(np.array_equal(back[k].view(np.uint8) if back[k].dtype != np.uint8 else back[k],
                                          pl[k].view(np.uint8) if pl[k].dtype != np.uint8 else pl[k]) for k in pl)
                bits_ok = _bits_equal(DECODE[ty](back), ref)
                rec[f'serialize_{var}_bytes_per_row'] = int(ser.shape[1])
                rec[f'serialize_{var}_roundtrip_planes'] = bool(same)
                rec[f'serialize_{var}_decode_bit_equal'] = bool(bits_ok)
            w = ref.reshape(-1)
            h = w.astype(np.float16).astype(np.float32); bf = (w.view(np.uint32) + 0x7FFF + ((w.view(np.uint32) >> 16) & 1)) & 0xFFFF0000
            bf = bf.astype(np.uint32).view(np.float32)
            nz = np.abs(w) > 0
            rec['frac_f16_exact'] = float((h == w).mean()); rec['frac_bf16_exact'] = float((bf == w).mean())
            rec['f16_stage_rel_rmse'] = float(np.sqrt(((h - w) ** 2).sum() / max((w ** 2).sum(), 1e-30)))
            rec['bf16_stage_rel_rmse'] = float(np.sqrt(((bf - w) ** 2).sum() / max((w ** 2).sum(), 1e-30)))
            rec['f16_max_rel_err_nonzero'] = float(np.max(np.abs(h[nz] - w[nz]) / np.abs(w[nz]))) if nz.any() else 0.0
            out.append(rec)
            print(json.dumps(rec), flush=True)
    bad = [r for r in out if not (r['merge_inverse_exact'] and r['decode_bit_equal_ref'] and r['serialize_P_roundtrip_planes']
                                   and r['serialize_A_roundtrip_planes'] and r['serialize_P_decode_bit_equal'] and r['serialize_A_decode_bit_equal'])]
    print(f'# checked {len(out)} (tensor, expert) slices; failures: {len(bad)}')
    return 1 if bad else 0


def fingerprint(args):
    """sha256 of a tensor's dequantised f32 words (little endian, expert-major row-major), computed twice:
    from the GGUF blocks with the reference dequantiser, and from the serialised kq planes with the
    kernel arithmetic. The two must match; the value is what a .rad would be verified against."""
    kv, ts = load_all(args.gguf)
    sel = re.compile(args.tensors)
    bad = 0
    for t in ts:
        ty = t['type']
        if ty not in SPLIT or not sel.search(t['name']): continue
        K = t['shape'][0]; N = t['shape'][1]; E = t['shape'][2] if len(t['shape']) > 2 else 1
        qk, bb = BLOCK[ty]
        raw = tensor_bytes(t).reshape(E, N, K // qk, bb)
        h1, h2 = hashlib.sha256(), hashlib.sha256()
        for e in range(E):
            blocks = np.ascontiguousarray(raw[e]).reshape(-1, bb)
            h1.update(np.ascontiguousarray(dequant_ref(ty, blocks), '<f4').tobytes())
            back = kq.deserialize(ty, kq.serialize(ty, SPLIT[ty](blocks), N, args.variant), K, args.variant)
            h2.update(np.ascontiguousarray(DECODE[ty](back), '<f4').tobytes())
        ok = h1.hexdigest() == h2.hexdigest(); bad += (not ok)
        print(json.dumps(dict(tensor=t['name'], type=ty, elements=t['nelem'], sha256_gguf_dequant=h1.hexdigest(),
                              sha256_kq_decode=h2.hexdigest(), equal=ok)), flush=True)
    return 1 if bad else 0


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sp = ap.add_subparsers(dest='cmd', required=True)
    p = sp.add_parser('plan'); p.add_argument('gguf', nargs='+'); p.add_argument('--variant', default='P', choices='PA'); p.add_argument('--json')
    c = sp.add_parser('check'); c.add_argument('gguf', nargs='+'); c.add_argument('--layers'); c.add_argument('--experts', default='0')
    c.add_argument('--tensors')
    f = sp.add_parser('fingerprint'); f.add_argument('gguf', nargs='+'); f.add_argument('--tensors', required=True); f.add_argument('--variant', default='P', choices='PA')
    a = ap.parse_args()
    if a.cmd == 'fingerprint': return fingerprint(a)
    if a.cmd == 'plan': plan(a); return 0
    return check(a)


if __name__ == '__main__':
    sys.exit(main())
