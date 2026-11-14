"""Tests for ud_map's architecture map. The integration test runs only when the Flash-Next GGUF and
the declared plan are given: UD_MAP_GGUF (all shards, colon-separated) and UD_MAP_DECLARED.
"""
import json
import os
import unittest

import ud_map


class Qwen4Exp(unittest.TestCase):
    def test_delta_attention_fuses_qkv_and_gate(self):
        a = ud_map.qwen4exp("blk.0.attn_qkv.weight")
        b = ud_map.qwen4exp("blk.0.attn_gate.weight")
        self.assertEqual(a["target"], "blk.0.ssm_inz.weight")
        self.assertEqual(b["target"], a["target"])
        self.assertEqual(a["partner"], "blk.0.attn_gate.weight")

    def test_experts_fuse_gate_and_up(self):
        g = ud_map.qwen4exp("blk.7.ffn_gate_exps.weight")
        u = ud_map.qwen4exp("blk.7.ffn_up_exps.weight")
        d = ud_map.qwen4exp("blk.7.ffn_down_exps.weight")
        self.assertEqual(g["target"], "blk.7.ffn_gate_up_exps.*.weight")
        self.assertEqual(u["target"], g["target"])
        self.assertEqual(d["target"], "blk.7.ffn_down_exps.*.weight")

    def test_ple_table_is_one_declared_weight(self):
        m = ud_map.qwen4exp("per_layer_token_embd.weight")
        self.assertEqual(m["kind"], "special")
        self.assertEqual(m["target"], "blk.1.ple_ngram.weight")

    def test_hc_prefix_moves(self):
        self.assertEqual(ud_map.qwen4exp("blk.3.hc_attn_inject.weight")["target"],
                         "blk.3.attn_hc_inject.weight")
        self.assertEqual(ud_map.qwen4exp("blk.3.hc_ffn_up.weight")["target"],
                         "blk.3.ffn_hc_up.weight")

    def test_shared_expert_gate_renames(self):
        # llama.cpp calls the shared-expert gate ffn_gate_inp_shexp; the arch declares ffn_gate_shexp
        self.assertEqual(ud_map.qwen4exp("blk.4.ffn_gate_inp_shexp.weight")["target"],
                         "blk.4.ffn_gate_shexp.weight")

    def test_ssm_ab_fuses_alpha_and_beta(self):
        a = ud_map.qwen4exp("blk.2.ssm_alpha.weight")
        b = ud_map.qwen4exp("blk.2.ssm_beta.weight")
        self.assertEqual(a["target"], "blk.2.ssm_ab.weight")
        self.assertEqual(b["target"], a["target"])

    def test_unknown_names_are_reported_not_guessed(self):
        m = ud_map.qwen4exp("blk.9.mystery.weight")
        self.assertEqual(m["kind"], "unknown")
        self.assertIsNone(m["target"])


class Types(unittest.TestCase):
    def test_every_ud_v31_type_has_an_entry(self):
        for ty in ("Q8_0", "Q6_K", "Q5_0", "IQ4_NL", "IQ4_XS", "IQ3_XXS", "IQ2_S", "F32", "BF16"):
            self.assertIn(ty, ud_map.TYPES)

    def test_unknown_type_is_not_servable(self):
        self.assertNotIn("TYPE42", ud_map.TYPES)
        self.assertFalse(ud_map.UNKNOWN_TYPE[2])


@unittest.skipUnless(os.environ.get("UD_MAP_GGUF") and os.environ.get("UD_MAP_DECLARED"),
                     "set UD_MAP_GGUF and UD_MAP_DECLARED for the integration test")
class Integration(unittest.TestCase):
    def test_flash_next_map(self):
        import argparse
        shards = os.environ["UD_MAP_GGUF"].split(":")
        declared = os.environ["UD_MAP_DECLARED"]
        out = "/tmp/ud_map_test.json"
        a = argparse.Namespace(gguf=shards, declared=declared, source=None, recipe_out=None, json=out)
        self.assertEqual(ud_map.plan(a), 0)
        p = json.load(open(out))
        self.assertEqual(p["arch"], "qwen4exp")
        self.assertAlmostEqual(p["totals"]["map_bytes"] / 2**30, 83.31, places=2)
        self.assertEqual(p["fusion_problems"], [])
        # the PLE n-gram table is the single biggest mapped tensor and maps to one declared weight
        big = max(p["tensors"], key=lambda r: r["bytes"])
        self.assertEqual(big["target"], "blk.1.ple_ngram.weight")


if __name__ == "__main__":
    unittest.main()
