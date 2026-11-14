"""Tests for the CPU-side UD repack: python3 -m unittest -v test_ud_repack   (from tools/ud_repack)
Synthetic blocks always run; set UD_GGUF=/path/model.gguf to also drive real tensors.
Every exactness claim is a bitwise comparison of f32 words, never a tolerance."""
import os, sys, unittest
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from ggml_blocks import BLOCK, SPLIT, DECODE, MERGE, dequant_ref
import kq_layout as kq

D_OFFS = {'Q8_0': [0], 'Q4_K': [0, 2], 'Q5_K': [0, 2], 'Q6_K': [208], 'Q5_0': [0], 'IQ4_NL': [0], 'IQ4_XS': [0],
          'Q2_0': [0]}


def random_blocks(ty, n, rng):
    qk, bb = BLOCK[ty]
    b = rng.integers(0, 256, size=(n, bb), dtype=np.uint8)
    for off in D_OFFS[ty]:   # finite f16 super-scales of a plausible size, both signs
        v = (rng.uniform(1e-4, 0.2, n) * rng.choice([-1, 1], n)).astype('<f2')
        b[:, off:off + 2] = v.view(np.uint8).reshape(n, 2)
    return b


def bits_equal(a, b):
    return np.array_equal(np.ascontiguousarray(a, np.float32).view(np.uint32), np.ascontiguousarray(b, np.float32).view(np.uint32))


class Synthetic(unittest.TestCase):
    def setUp(self):
        self.rng = np.random.default_rng(1234)

    def test_decode_from_planes_equals_reference_bitwise(self):
        for ty in SPLIT:
            b = random_blocks(ty, 2000, self.rng)
            self.assertTrue(bits_equal(DECODE[ty](SPLIT[ty](b)), dequant_ref(ty, b)), ty)

    def test_planes_merge_back_to_the_original_bytes(self):
        for ty in SPLIT:
            b = random_blocks(ty, 2000, self.rng)
            self.assertTrue(np.array_equal(MERGE[ty](SPLIT[ty](b)), b), ty)

    def test_serialised_rows_roundtrip_and_decode_equal(self):
        for ty in SPLIT:
            qk, bb = BLOCK[ty]; K = qk * (8 if qk == 256 else 64) if qk == 256 else 2048; rows = 7
            b = random_blocks(ty, rows * (K // qk), self.rng)
            pl = SPLIT[ty](b)
            for var in 'PA':
                ser = kq.serialize(ty, pl, rows, var)
                self.assertEqual(ser.shape[1], kq.row_bytes(ty, K, var), (ty, var))
                back = kq.deserialize(ty, ser, K, var)
                self.assertTrue(bits_equal(DECODE[ty](back), dequant_ref(ty, b)), (ty, var))
                self.assertTrue(np.array_equal(MERGE[ty](back), b), (ty, var))

    def test_bits_per_weight(self):
        want = {('Q4_K', 'P'): 4.5, ('Q4_K', 'A'): 4.625, ('Q5_K', 'P'): 5.5, ('Q5_K', 'A'): 5.625, ('Q6_K', 'P'): 6.5625,
                ('Q8_0', 'P'): 8.5, ('Q5_0', 'P'): 5.5, ('IQ4_NL', 'P'): 4.5, ('IQ4_XS', 'P'): 4.25, ('IQ4_XS', 'A'): 4.3125}
        for (ty, v), w in want.items():
            self.assertAlmostEqual(kq.bits_per_weight(ty, v), w, msg=(ty, v))
        # P matches the GGUF block size exactly for every type
        for ty, (qk, bb) in BLOCK.items():
            self.assertAlmostEqual(kq.bits_per_weight(ty, 'P'), 8 * bb / qk, msg=ty)

    def test_agrees_with_the_gguf_python_package_when_present(self):
        try:
            import gguf
            from gguf import quants
        except Exception:
            self.skipTest('gguf package not installed')
        T = gguf.GGMLQuantizationType
        for ty in SPLIT:
            if not hasattr(T, ty):
                continue    # the gguf package predates type 42 (Q2_0); the reference is llama.cpp's formula
            b = random_blocks(ty, 500, self.rng)
            ref = quants.dequantize(b, getattr(T, ty)).reshape(500, -1)
            self.assertTrue(bits_equal(DECODE[ty](SPLIT[ty](b)), ref), ty)


@unittest.skipUnless(os.environ.get('UD_GGUF'), 'set UD_GGUF to a model file')
class RealFile(unittest.TestCase):
    def test_sample_tensors(self):
        import ud_repack
        class A: gguf = [os.environ['UD_GGUF']]; layers = os.environ.get('UD_LAYERS', '0,1,40'); experts = '0,200'; tensors = None
        self.assertEqual(ud_repack.check(A), 0)


if __name__ == '__main__':
    unittest.main()
