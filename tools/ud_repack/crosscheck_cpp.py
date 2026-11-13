#!/usr/bin/env python3
"""crosscheck_cpp -- compare the python 'P' planes with the fork's own C++ GGUF reader, byte for byte.
usage: crosscheck_cpp.py REPO_ROOT   (needs a C++17 compiler; synthetic blocks, no model file)"""
import os, subprocess, sys, tempfile
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from ggml_blocks import SPLIT, BLOCK
import kq_layout as kq
from test_ud_repack import random_blocks

# python plane name -> C++ plane role, per ggml type id
MAP = {'Q8_0': (8, {'codes': 'codes', 'd': 'scale'}),
       'Q4_K': (12, {'codes': 'codes', 'sc': 'scale', 'd': 'scale.1', 'mm': 'min', 'dmin': 'min.1'}),
       'Q5_K': (13, {'codes5': 'codes', 'sc': 'scale', 'd': 'scale.1', 'mm': 'min', 'dmin': 'min.1'}),
       'Q6_K': (14, {'lo4+hi2': 'codes', 'sc': 'scale', 'd': 'scale.1'})}


def main():
    root = sys.argv[1]
    exe = os.path.join(tempfile.mkdtemp(), 'ggml_planes_dump')
    subprocess.check_call(['c++', '-std=c++17', '-I.', '-Iabi', '-Icore', '-Icore/format', '-Icore/text',
                           'tools/ud_repack/ggml_planes_dump.cpp', '-o', exe], cwd=root)
    rng = np.random.default_rng(7); rows, K = 9, 1024; bad = 0
    for ty, (tid, m) in MAP.items():
        qk, bb = BLOCK[ty]
        b = random_blocks(ty, rows * (K // qk), rng)
        d = tempfile.mkdtemp(); fn = os.path.join(d, 'blocks.bin'); b.tofile(fn)
        subprocess.check_call([exe, str(tid), fn, str(rows), str(K), os.path.join(d, 'p')], stdout=subprocess.DEVNULL)
        pl = SPLIT[ty](b); mine = kq.plane_bytes(ty, pl, rows, 'P')
        if ty == 'Q5_K':   # C++ stores the whole 5-bit code in one plane; python's kq layout splits lo4 / hi1
            mine = {'codes5': kq.pack_bits(pl['codes'].reshape(rows, -1), 5), 'sc': mine['sc'], 'd': mine['d'],
                    'mm': mine['mm'], 'dmin': mine['dmin']}
        if ty == 'Q6_K':   # C++ stores q-32 as 6-bit two's complement; python stores q (0..63) split in lo4/hi2
            c6 = ((pl['codes'].astype(np.int32) - 32) & 63).reshape(rows, -1)
            mine = {'lo4+hi2': kq.pack_bits(c6, 6), 'sc': mine['sc'], 'd': mine['d']}
        for pname, role in m.items():
            theirs = np.fromfile(os.path.join(d, 'p.' + role), np.uint8).reshape(rows, -1)
            ok = theirs.shape == mine[pname].shape and np.array_equal(theirs, mine[pname])
            print(f'{ty:5s} plane {pname:8s} vs C++ {role:8s} {"IDENTICAL" if ok else "DIFFERENT"} ({theirs.shape[1]} B/row)')
            bad += (not ok)
    print('mismatched planes:', bad)
    return 1 if bad else 0


if __name__ == '__main__':
    sys.exit(main())
