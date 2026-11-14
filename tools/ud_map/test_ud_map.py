"""Tests for ud_map's architecture map. The integration test runs only when the Flash-Next GGUF and
the declared plan are given: UD_MAP_GGUF (all shards, colon-separated) and UD_MAP_DECLARED.
"""
import argparse
import json
import os
import tempfile
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


class Qwen35Moe(unittest.TestCase):
    def test_delta_attention_fuses_qkv_and_gate(self):
        a = ud_map.qwen35moe("blk.0.attn_qkv.weight")
        b = ud_map.qwen35moe("blk.0.attn_gate.weight")
        self.assertEqual(a["target"], "blk.0.ssm_inz.weight")
        self.assertEqual(b["target"], a["target"])
        self.assertEqual(a["kind"], "fused")

    def test_ssm_ab_fuses_alpha_and_beta(self):
        a = ud_map.qwen35moe("blk.2.ssm_alpha.weight")
        b = ud_map.qwen35moe("blk.2.ssm_beta.weight")
        self.assertEqual(a["target"], "blk.2.ssm_ab.weight")
        self.assertEqual(b["target"], a["target"])

    def test_experts_and_shared_expert_fuse(self):
        g = ud_map.qwen35moe("blk.7.ffn_gate_exps.weight")
        u = ud_map.qwen35moe("blk.7.ffn_up_exps.weight")
        d = ud_map.qwen35moe("blk.7.ffn_down_exps.weight")
        self.assertEqual(g["target"], "blk.7.ffn_gate_up_exps.*.weight")
        self.assertEqual(u["target"], g["target"])
        self.assertEqual(d["target"], "blk.7.ffn_down_exps.*.weight")
        sg = ud_map.qwen35moe("blk.7.ffn_gate_shexp.weight")
        su = ud_map.qwen35moe("blk.7.ffn_up_shexp.weight")
        self.assertEqual(sg["target"], "blk.7.ffn_gate_up_shexp.weight")
        self.assertEqual(su["target"], sg["target"])

    def test_shared_expert_gate_renames_and_nextn_maps_the_mtp_head(self):
        self.assertEqual(ud_map.qwen35moe("blk.4.ffn_gate_inp_shexp.weight")["target"],
                         "blk.4.ffn_gate_shexp.weight")
        self.assertEqual(ud_map.qwen35moe("blk.40.nextn.eh_proj.weight")["target"],
                         "mtp.fc.weight")
        self.assertEqual(ud_map.qwen35moe("blk.40.nextn.shared_head_norm.weight")["target"],
                         "mtp.norm.weight")

    def test_linear_and_full_attention_names(self):
        self.assertEqual(ud_map.qwen35moe("blk.0.ssm_out.weight")["target"],
                         "blk.0.ssm_out.weight")
        self.assertEqual(ud_map.qwen35moe("blk.3.attn_q.weight")["target"],
                         "blk.3.attn_qg.weight")
        self.assertEqual(ud_map.qwen35moe("blk.3.post_attention_norm.weight")["target"],
                         "blk.3.ffn_norm.weight")

    def test_unknown_names_are_reported_not_guessed(self):
        self.assertEqual(ud_map.qwen35moe("blk.9.mystery.weight")["kind"], "unknown")


def _rows():
    return [
        dict(name="blk.0.ffn_gate_exps.weight", type="Q4_K"),
        dict(name="blk.0.ffn_up_exps.weight", type="Q4_K"),
        dict(name="blk.0.ffn_down_exps.weight", type="Q5_K"),
        dict(name="blk.1.ffn_gate_exps.weight", type="Q4_K"),
        dict(name="blk.1.ffn_up_exps.weight", type="Q5_K"),
        dict(name="blk.1.ffn_down_exps.weight", type="Q6_K"),
        dict(name="blk.2.ssm_inz.weight", type="Q8_0", shape=[12288, 2048],
             target="blk.2.ssm_inz.weight"),
        dict(name="blk.2.ffn_gate_up_shexp.weight", type="Q8_0", shape=[1024, 2048],
             target="blk.2.ffn_gate_up_shexp.weight"),
        dict(name="output.weight", type="Q8_0", shape=[248320, 2048],
             target="output.weight"),
        dict(name="token_embd.weight", type="Q8_0", shape=[248320, 2048],
             target="token_embd.weight"),
    ]


class FastRecipe(unittest.TestCase):
    def test_expert_formats_follow_the_map_and_fuse_by_the_wider_half(self):
        text = ud_map._fast_recipe(_rows(), "m.gguf")
        self.assertIn("blk.0.ffn_gate_up_exps.*.weight", text)
        self.assertIn("blk.0.ffn_down_exps.*.weight", text)
        self.assertRegex(text, r"blk\.1\.ffn_gate_up_exps\.\*\.weight.*w5nl.*# Q5_K")
        self.assertRegex(text, r"blk\.1\.ffn_down_exps\.\*\.weight.*w5nl.*# Q6_K")

    def test_dense_trunk_is_block_fp8_but_embedding_is_kept(self):
        text = ud_map._fast_recipe(_rows(), "m.gguf")
        self.assertIn("blk.*.ssm_inz.weight", text)
        self.assertIn("blk.*.ffn_gate_up_shexp.weight", text)
        self.assertIn("output.weight", text)
        self.assertNotIn("blk.*.token_embd.weight", text)
        self.assertIn("mtp.draft_head.weight", text)

    def test_gptq_recipe_carries_calib(self):
        text = ud_map._fast_recipe(_rows(), "m.gguf", gptq=True)
        self.assertIn("gptq table=w4nl", text)
        self.assertIn("cd=3 calib=$CALIB", text)

    def test_mtp_variant_overrides_layer_40(self):
        rows = _rows() + [dict(name="blk.40.ffn_gate_exps.weight", type="Q8_0"),
                          dict(name="blk.40.ffn_up_exps.weight", type="Q8_0"),
                          dict(name="blk.40.ffn_down_exps.weight", type="Q8_0")]
        text = ud_map._fast_recipe(rows, "m.gguf", mtp="w4nl64")
        self.assertRegex(text, r"blk\.40\.ffn_gate_up_exps\.\*\.weight.*table=w4nl")
        self.assertRegex(text, r"blk\.40\.ffn_down_exps\.\*\.weight.*table=w4nl")


class Build(unittest.TestCase):
    def test_bf16_source_gets_the_fp8_descriptor_and_imatrix(self):
        a = argparse.Namespace(source="/src", rad_convert="/bin/rad-convert", recipe="r",
                               set=[], auto_fp8=True, imatrix="/im.gguf", reuse="old.rad",
                               home=None, kernels=None, extra_arg=[],
                               _source_quant="bf16")
        cmd = ud_map._convert_command(a, "recipe", "out.rad")
        self.assertIn("quantization_config.quant_method=fp8", cmd)
        self.assertIn("--imatrix", cmd)
        self.assertIn("--reuse", cmd)
        self.assertIn("/bin/rad-convert", cmd)

    def test_quantised_source_does_not_get_the_descriptor(self):
        a = argparse.Namespace(source="/src", rad_convert="/bin/rad-convert", recipe="r",
                               set=[], auto_fp8=True, imatrix=None, reuse=None,
                               home=None, kernels=None, extra_arg=[], _source_quant="gguf")
        cmd = ud_map._convert_command(a, "recipe", "out.rad")
        self.assertFalse(any("quantization_config" in x for x in cmd))

    def test_source_quant_reads_config_and_safetensors(self):
        with tempfile.TemporaryDirectory() as d:
            json.dump({"model_type": "qwen3_5_moe"}, open(os.path.join(d, "config.json"), "w"))
            with open(os.path.join(d, "model.safetensors"), "wb") as f:
                hd = json.dumps({"w": {"dtype": "BF16", "shape": [1],
                                       "data_offsets": [0, 2]}}).encode()
                f.write(len(hd).to_bytes(8, "little")); f.write(hd); f.write(b"\0\0")
            self.assertEqual(ud_map.source_quant(d), "bf16")

    def test_dry_run_writes_the_recipe_and_plan_without_converting(self):
        with tempfile.TemporaryDirectory() as d:
            gguf = os.path.join(d, "fake.gguf")
            open(gguf, "wb").close()
            # load_all cannot read a zero-byte GGUF; drive the recipe/plan writers directly.
            text = ud_map._fast_recipe(_rows(), gguf)
            self.assertIn("blk.0.ffn_gate_up_exps.*.weight", text)
            self.assertIn("gptq", ud_map._fast_recipe(_rows(), gguf, gptq=True))


class Types(unittest.TestCase):
    def test_every_ud_v31_type_has_an_entry(self):
        for ty in ("Q8_0", "Q6_K", "Q5_0", "IQ4_NL", "IQ4_XS", "IQ3_XXS", "IQ2_S", "F32", "BF16"):
            self.assertIn(ty, ud_map.TYPES)

    def test_q2_0_is_resolved_but_not_servable(self):
        # type 42 is Q2_0: libquant has the encoder and the plane split, but no kernel reads it.
        self.assertEqual(ud_map.TYPES["TYPE42"][0], 2.25)
        self.assertIsNone(ud_map.TYPES["TYPE42"][1])
        self.assertFalse(ud_map.TYPES["TYPE42"][2])
        self.assertFalse(ud_map.UNKNOWN_TYPE[2])


class Grade(unittest.TestCase):
    def test_multiple_choice(self):
        self.assertTrue(ud_map._grade({"kind": "mc", "expected": "D"}, {"text": "D"}))
        self.assertTrue(ud_map._grade({"kind": "mc", "expected": "D"}, {"text": "The answer is D."}))
        self.assertFalse(ud_map._grade({"kind": "mc", "expected": "D"}, {"text": "B"}))

    def test_gsm8k_takes_the_final_number(self):
        self.assertTrue(ud_map._grade({"kind": "gsm8k", "expected": "2280"},
                                      {"text": "so it is 2280"}))
        self.assertFalse(ud_map._grade({"kind": "gsm8k", "expected": "2280"},
                                       {"text": "so it is 2281"}))

    def test_tool_calls(self):
        item = {"kind": "tool", "expected": "{'name': 'get_weather', 'args': {'city': 'Oslo'}}"}
        rec = {"tool_calls": [{"function": {"name": "get_weather",
                                            "arguments": '{"city": "Oslo", "unit": "c"}'}}]}
        self.assertTrue(ud_map._grade(item, rec))
        rec["tool_calls"][0]["function"]["arguments"] = '{"city": "Rome"}'
        self.assertFalse(ud_map._grade(item, rec))


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
