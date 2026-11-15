"""Tests for the CPU-side UD repack: python3 -m unittest -v test_ud_repack   (from tools/ud_repack)
Synthetic blocks always run; set UD_GGUF=/path/model.gguf to also drive real tensors.
Every exactness claim is a bitwise comparison of f32 words, never a tolerance."""
import os, sys, unittest
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from ggml_blocks import BLOCK, SPLIT, DECODE, MERGE, dequant_ref
import kq_layout as kq
import radfile

D_OFFS = {'Q8_0': [0], 'Q4_K': [0, 2], 'Q5_K': [0, 2], 'Q6_K': [208], 'Q5_0': [0], 'IQ4_NL': [0], 'IQ4_XS': [0],
          'Q2_0': [0], 'IQ3_XXS': [0], 'IQ2_S': [0]}

# The kq row layouts cover the non-grid types; the grid types are decoded directly from their
# canonical planes (SPLIT/DECODE/MERGE) rather than serialised into a kq row.
KQ_TYPES = [t for t in SPLIT if t in kq.SPEC]


def random_blocks(ty, n, rng):
    qk, bb = BLOCK[ty]
    b = rng.integers(0, 256, size=(n, bb), dtype=np.uint8)
    for off in D_OFFS[ty]:   # finite f16 super-scales of a plausible size, both signs
        v = (rng.uniform(1e-4, 0.2, n) * rng.choice([-1, 1], n)).astype('<f2')
        b[:, off:off + 2] = v.view(np.uint8).reshape(n, 2)
    return b


def bits_equal(a, b):
    return np.array_equal(np.ascontiguousarray(a, np.float32).view(np.uint32), np.ascontiguousarray(b, np.float32).view(np.uint32))


class RadFilePlanes(unittest.TestCase):
    def test_64bit_planes_decode_without_overflow(self):
        vals = np.array([0, 1, 2**63, 2**64 - 1], dtype=np.uint64)
        rows = np.ascontiguousarray(vals.view(np.uint8).reshape(1, -1))
        got = radfile.unpack_stream(rows, 64, len(vals)).reshape(-1)
        self.assertTrue(np.array_equal(got, vals))
        signed = radfile.signed_from_bits(got, 64)
        self.assertEqual(signed.dtype, np.int64)
        self.assertTrue(np.array_equal(signed, np.array([0, 1, -(2**63), -1], np.int64)))

    def test_signed_extension_below_64_bits(self):
        v = np.array([0, 7, 8, 15], dtype=np.uint32)
        self.assertTrue(np.array_equal(radfile.signed_from_bits(v, 4),
                                       np.array([0, 7, -8, -1], np.int64)))

    def test_plugin_dtypes_are_named_and_refused(self):
        self.assertTrue(radfile.is_plugin_dtype(radfile.RAD_DT_PLUGIN_BASE + 7))
        self.assertEqual(radfile.dt_name(radfile.RAD_DT_PLUGIN_BASE + 7), 'halo_moew4')
        self.assertFalse(radfile.is_plugin_dtype(radfile.DT_BY_NAME['u4']))
        self.assertEqual(radfile.dt_bits(radfile.RAD_DT_PLUGIN_BASE + 7), 0)

    def test_encoding_with_a_plugin_plane_is_kernel_private(self):
        raw = bytearray(radfile.ENC_SZ)
        raw[0:6] = b'affine'
        import struct
        struct.pack_into('<ii', raw, 48, 1, 0)
        struct.pack_into('<24s II 2q 2q', raw, 56, b'codes', radfile.RAD_DT_PLUGIN_BASE + 7,
                         0, 1, 1, 0, 0)
        enc = radfile.Encoding(bytes(raw))
        self.assertEqual(len(enc.kernel_private_planes()), 1)
        self.assertIn('halo_moew4', enc.name())

    def test_decode_entry_refuses_a_plugin_plane(self):
        sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'ud_map'))
        import ud_map
        raw = bytearray(radfile.ENC_SZ)
        raw[0:6] = b'plain'
        import struct
        struct.pack_into('<ii', raw, 48, 1, 0)
        struct.pack_into('<24s II 2q 2q', raw, 56, b'codes', radfile.RAD_DT_PLUGIN_BASE + 9,
                         0, 1, 1, 0, 0)
        e = type('E', (), {})()
        e.enc = radfile.Encoding(bytes(raw))
        e.shape = (4, 4)
        with self.assertRaises(radfile.KernelPrivateDtype):
            ud_map.decode_entry(None, e)

    def test_policy_verdict_does_not_pass_kernel_private(self):
        sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'ud_map'))
        import ud_map
        self.assertEqual(ud_map._policy_verdict(True, 0), 'PASS')
        self.assertEqual(ud_map._policy_verdict(True, 3), 'PARTIAL')
        self.assertEqual(ud_map._policy_verdict(False, 3), 'FAIL')
        line = ud_map._policy_line('quantised', 'PARTIAL', 10, 3, 1 << 20, ['halo_moew4'])
        self.assertIn('kernel-private planes not Python-decoded', line)
        self.assertNotIn(': PASS', line)


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
        for ty in KQ_TYPES:
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
        for ty in KQ_TYPES:
            qk, bb = BLOCK[ty]
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
