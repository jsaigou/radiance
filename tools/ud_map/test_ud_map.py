"""Tests for ud_map's architecture map. The integration test runs only when the Flash-Next GGUF and
the declared plan are given: UD_MAP_GGUF (all shards, colon-separated) and UD_MAP_DECLARED.
"""
import argparse
import json
import os
import re
import sys
import tempfile
import unittest

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

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


def _rows_4exp():
    """The Flash-Next map's shape: one short layer per UD type, at toy sizes."""
    sh = [256, 256, 2]
    return [
        dict(name="per_layer_token_embd.weight", type="IQ4_NL", shape=[1024, 160],
             role="ple-table", target="blk.1.ple_ngram.weight"),
        dict(name="blk.0.ffn_gate_exps.weight", type="IQ3_XXS", shape=sh,
             target="blk.0.ffn_gate_up_exps.*.weight"),
        dict(name="blk.0.ffn_up_exps.weight", type="IQ3_XXS", shape=sh,
             target="blk.0.ffn_gate_up_exps.*.weight"),
        dict(name="blk.0.ffn_down_exps.weight", type="IQ4_NL", shape=sh,
             target="blk.0.ffn_down_exps.*.weight"),
        dict(name="blk.1.ffn_gate_exps.weight", type="IQ4_XS", shape=sh,
             target="blk.1.ffn_gate_up_exps.*.weight"),
        dict(name="blk.1.ffn_up_exps.weight", type="IQ4_XS", shape=sh,
             target="blk.1.ffn_gate_up_exps.*.weight"),
        dict(name="blk.1.ffn_down_exps.weight", type="Q5_0", shape=sh,
             target="blk.1.ffn_down_exps.*.weight"),
        dict(name="blk.2.ffn_gate_exps.weight", type="IQ2_S", shape=sh,
             target="blk.2.ffn_gate_up_exps.*.weight"),
        dict(name="blk.2.ffn_up_exps.weight", type="IQ2_S", shape=sh,
             target="blk.2.ffn_gate_up_exps.*.weight"),
        dict(name="blk.2.ffn_down_exps.weight", type="TYPE42", shape=sh,
             target="blk.2.ffn_down_exps.*.weight"),
        dict(name="token_embd.weight", type="IQ4_XS", shape=[2048, 256],
             role="embedding", target="token_embd.weight"),
        dict(name="output.weight", type="Q6_K", shape=[2048, 256],
             role="output", target="output.weight"),
        dict(name="blk.0.ssm_out.weight", type="Q8_0", shape=[256, 256],
             role="ssm_out", target="blk.0.ssm_out.weight"),
        dict(name="blk.0.ffn_gate_up_shexp.weight", type="Q6_K", shape=[512, 256],
             role="ffn_gate_shexp", target="blk.0.ffn_gate_up_shexp.weight"),
        dict(name="blk.0.qsa_qk.weight", type="BF16", shape=[256, 256],
             role="indexer", target="blk.0.qsa_qk.weight"),
        dict(name="blk.0.ffn_gate_inp.weight", type="F32", shape=[16, 256],
             role="ffn_gate_inp", target="blk.0.ffn_gate_inp.weight"),
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


class MirrorRecipe(unittest.TestCase):
    def test_q6k_lm_head_and_q8_0_connections_take_the_served_encodings(self):
        rows = [
            dict(name="output.weight", type="Q6_K", shape=[2560, 248320],
                 target="output.weight"),
            dict(name="blk.0.hc_attn_down.weight", type="Q8_0", shape=[10240, 320],
                 target="blk.0.attn_hc_down.weight"),
            dict(name="blk.0.hc_attn_up.weight", type="Q8_0", shape=[320, 10240],
                 target="blk.0.attn_hc_up.weight"),
            dict(name="blk.0.ssm_out.weight", type="Q8_0", shape=[2560, 6144],
                 target="blk.0.ssm_out.weight"),
        ]
        text = ud_map._mirror_recipe(rows, "m.gguf")
        # The Q6_K head has no logits row; it is mapped to the int8 head, not mirrored as q6_K.
        self.assertRegex(text, r"output\.weight\s+rtn\s+codes=i8")
        self.assertNotRegex(text, r"output\.weight\s+ggml type=q6_K")
        # hc_read serves bf16/E4M3 rows, so the Q8_0 connection planes are not mirrored.
        self.assertIn("*_hc_down.weight", text)
        self.assertIn("*_hc_up.weight", text)
        self.assertNotIn("blk.0.attn_hc_down.weight", text)
        self.assertNotIn("blk.0.attn_hc_up.weight", text)
        # The dense Q8_0 trunk is served natively, so its plane rule stays exact.
        self.assertRegex(text, r"blk\.0\.ssm_out\.weight\s+ggml type=q8_0")

class FastFlashNext(unittest.TestCase):
    def test_recipe_maps_every_ud_format_to_a_served_form(self):
        text = ud_map._fast_recipe(_rows_4exp(), "m.gguf", arch="qwen4exp")
        self.assertRegex(text, r"blk\.0\.ffn_gate_up_exps\.\*\.weight.*table=w3nl.*"
                               r"# IQ3_XXS -> w3nl64")
        self.assertRegex(text, r"blk\.0\.ffn_down_exps\.\*\.weight.*table=w4nl.*"
                               r"# IQ4_NL -> w4nl64")
        self.assertRegex(text, r"blk\.1\.ffn_gate_up_exps\.\*\.weight.*table=w4nl.*"
                               r"# IQ4_XS -> w4nl64")
        self.assertRegex(text, r"blk\.1\.ffn_down_exps\.\*\.weight.*table=w5nl.*"
                               r"# Q5_0 -> w5nl64")
        self.assertRegex(text, r"blk\.2\.ffn_gate_up_exps\.\*\.weight.*table=w3nl.*"
                               r"# IQ2_S -> w3nl64")
        # Q2_0's native q20a16 is smaller but unrotated; its layers' gate/up are IQ2_S, and the
        # arch keeps one encoding family per layer, so it takes the rotated w3nl form.
        self.assertRegex(text, r"blk\.2\.ffn_down_exps\.\*\.weight.*table=w3nl.*"
                               r"# TYPE42 -> w3nl64")
        self.assertNotIn("ggml type=q2_0", text)
        self.assertIn("blk.1.ple_ngram.weight", text)
        self.assertIn("ggml type=iq4_nl", text)
        self.assertRegex(text, r"output\.weight.*codes=i8")
        self.assertRegex(text, r"blk\.0\.ssm_out\.weight.*ggml type=q8_0")
        self.assertNotIn("has no fast format", text)

    def test_embedding_is_kept_and_no_draft_head_is_invented(self):
        text = ud_map._fast_recipe(_rows_4exp(), "m.gguf", arch="qwen4exp")
        rules = [line for line in text.splitlines()
                 if line.startswith("token_embd.weight") and "ggml" in line]
        self.assertEqual(rules, [])
        self.assertNotIn("mtp.draft_head.weight", text)
        self.assertIn("token_embd.weight (IQ4_XS) is kept bf16", text)

    def test_fast_dtypes_and_dependencies(self):
        by = {r["name"]: ud_map._fast_dtype("qwen4exp", r) for r in _rows_4exp()}
        self.assertEqual(by["per_layer_token_embd.weight"][0], "iq4_nl")
        self.assertTrue(by["per_layer_token_embd.weight"][2])
        self.assertEqual(by["blk.0.ffn_gate_exps.weight"][0], "w3nl64")
        self.assertFalse(by["blk.0.ffn_gate_exps.weight"][2])
        self.assertIn("w3nl64a8h", by["blk.0.ffn_gate_exps.weight"][3])
        self.assertEqual(by["blk.0.ffn_down_exps.weight"][0], "w4nl64")
        self.assertTrue(by["blk.0.ffn_down_exps.weight"][2])
        self.assertEqual(by["blk.1.ffn_down_exps.weight"][0], "w5nl64")
        self.assertEqual(by["blk.2.ffn_down_exps.weight"][0], "w3nl64")
        self.assertFalse(by["blk.2.ffn_down_exps.weight"][2])
        self.assertEqual(by["token_embd.weight"][0], "bf16")
        self.assertEqual(by["output.weight"][0], "q6_k")
        self.assertFalse(by["output.weight"][2])
        self.assertIn("L3", by["output.weight"][3])
        self.assertEqual(by["blk.0.ssm_out.weight"][0], "q8_0")
        self.assertEqual(by["blk.0.qsa_qk.weight"][0], "bf16")
        self.assertEqual(by["blk.0.ffn_gate_inp.weight"][0], "f32")

    def test_fast_bytes_scale_with_the_codebook(self):
        row = dict(type="Q5_0", shape=[256, 256, 2], bytes=256 * 256 * 2 * 2)
        self.assertEqual(ud_map._fast_bytes(row, "w5nl64", ud_map.FAST_BPW["w5nl64"]),
                         int(256 * 256 * 2 * 5.125 / 8))
        row = dict(type="Q8_0", shape=[256, 256], bytes=256 * 256)
        self.assertEqual(ud_map._fast_bytes(row, "q8_0", ud_map.FAST_BPW["q8_0"]),
                         256 * 256)

    def test_mtp_variant_targets_the_kept_layer(self):
        text = ud_map._fast_recipe(_rows_4exp(), "m.gguf", arch="qwen4exp",
                                   mtp="w4nl64", mtp_layer=48)
        self.assertRegex(text, r"blk\.48\.ffn_gate_up_exps\.\*\.weight.*table=w4nl")
        self.assertRegex(text, r"blk\.48\.ffn_down_exps\.\*\.weight.*table=w4nl")

    def test_kept_mtp_layer_defaults_to_w4(self):
        # A whole bf16 expert layer is refused by the arch, so the fast recipe writes blk.48.
        text = ud_map._fast_recipe(_rows_4exp(), "m.gguf", arch="qwen4exp", mtp_layer=48)
        self.assertRegex(text, r"blk\.48\.ffn_gate_up_exps\.\*\.weight.*table=w4nl")
        self.assertRegex(text, r"blk\.48\.ffn_down_exps\.\*\.weight.*table=w4nl")


class MirrorFlashNext(unittest.TestCase):
    """The mirror policy is the UD-faithful path: each tensor keeps the GGUF's own format."""

    def test_every_ud_type_gets_its_own_ggml_spelling(self):
        text = ud_map._mirror_recipe(_rows_4exp(), "m.gguf")
        expected = {
            "blk.0.ffn_gate_up_exps.*.weight": "iq3_xxs",   # IQ3_XXS gate/up
            "blk.0.ffn_down_exps.*.weight": "iq4_nl",       # IQ4_NL down
            "blk.1.ffn_gate_up_exps.*.weight": "iq4_xs",    # IQ4_XS gate/up
            "blk.1.ffn_down_exps.*.weight": "q5_0",         # Q5_0 down
            "blk.2.ffn_gate_up_exps.*.weight": "iq2_s",     # IQ2_S gate/up
            "blk.2.ffn_down_exps.*.weight": "q2_0",         # TYPE42 down
            "blk.1.ple_ngram.weight": "iq4_nl",             # PLE n-gram table
            "token_embd.weight": "iq4_xs",                  # token embedding
            "blk.0.ssm_out.weight": "q8_0",
            "blk.0.ffn_gate_up_shexp.weight": "q6_k",
        }
        for target, spell in expected.items():
            self.assertRegex(text, re.escape(target) + r"\s+ggml type=" + re.escape(spell),
                             f"{target} should keep {spell}")
        # The Q6_K head has no logits row; it is the served int8 head, not a q6_K mirror.
        self.assertRegex(text, r"output\.weight\s+rtn\s+codes=i8")
        self.assertNotRegex(text, r"output\.weight\s+ggml type=q6_k")

    def test_mixed_projection_formats_are_preserved(self):
        # UD mixes gate/up and down within a layer; the mirror recipe must not flatten it.
        text = ud_map._mirror_recipe(_rows_4exp(), "m.gguf")
        self.assertRegex(text, r"blk\.0\.ffn_gate_up_exps\.\*\.weight\s+ggml type=iq3_xxs")
        self.assertRegex(text, r"blk\.0\.ffn_down_exps\.\*\.weight\s+ggml type=iq4_nl")
        self.assertRegex(text, r"blk\.2\.ffn_gate_up_exps\.\*\.weight\s+ggml type=iq2_s")
        self.assertRegex(text, r"blk\.2\.ffn_down_exps\.\*\.weight\s+ggml type=q2_0")

    def test_fused_halves_share_one_rule_and_kept_types_get_none(self):
        text = ud_map._mirror_recipe(_rows_4exp(), "m.gguf")
        # blk.0 gate and up are both IQ3_XXS and fuse into one declared weight: one rule.
        self.assertEqual(sum(1 for line in text.splitlines()
                             if line.startswith("blk.0.ffn_gate_up_exps.*.weight")), 1)
        # F32/BF16 have no libquant type: they keep the checkpoint.
        self.assertNotIn("qsa_qk", text)
        self.assertNotIn("ffn_gate_inp", text)
        # The mirror recipe is not the fast path.
        self.assertNotIn("table=", text)
        self.assertNotIn("gptq ", text)
        # output.weight is the one served int8 exception, not a fast expert table.
        self.assertNotIn("ffn", "".join(line for line in text.splitlines()
                                        if line.startswith("output.weight")))

    def test_one_rule_per_target_and_type(self):
        text = ud_map._mirror_recipe(_rows_4exp(), "m.gguf")
        valid = {(r["target"], r["type"]) for r in _rows_4exp()
                 if r.get("target") and ud_map.TYPES.get(r["type"], ud_map.UNKNOWN_TYPE)[1]}
        rules = [line for line in text.splitlines() if line and not line.startswith("#")]
        self.assertEqual(len(rules), len(valid))

    def test_qwen4exp_defaults_to_mirror_not_fast(self):
        a = argparse.Namespace(policy="auto")
        self.assertEqual(ud_map._choose_policy(a, "qwen4exp"), "mirror")
        self.assertEqual(ud_map._choose_policy(a, "qwen35moe"), "fast")


class SizePolicy(unittest.TestCase):
    def _rows(self):
        rows = _rows_4exp()
        for r in rows:
            elems = 1
            for x in r["shape"]:
                elems *= x
            r["bytes"] = int(elems * ud_map.TYPES[r["type"]][0] / 8)
        return rows

    def test_per_layer_choice_keeps_the_small_native_family(self):
        choices = ud_map._expert_choices(self._rows())
        # blk.0 (IQ3_XXS gate/up + IQ4_NL down) takes the smaller rotated family.
        self.assertTrue(choices[0]["use_fast"])
        # blk.1 (IQ4_XS gate/up + Q5_0 down) also takes it.
        self.assertTrue(choices[1]["use_fast"])
        # blk.2 (IQ2_S gate/up + Q2_0 down) is much smaller native.
        self.assertFalse(choices[2]["use_fast"])

    def test_size_map_is_below_both_families(self):
        rows = self._rows()
        native = sum(r["bytes"] for r in rows)
        size_map = ud_map._size_map_bytes(rows)
        self.assertLess(size_map, native)
        self.assertLess(size_map, sum(
            ud_map._fast_bytes(r, ud_map.FAST_EXPERTS[r["type"]][0],
                               ud_map.FAST_BPW[ud_map.FAST_EXPERTS[r["type"]][0]])
            if r["name"].endswith("_exps.weight") else r["bytes"] for r in rows))

    def test_recipe_overrides_only_the_chosen_layers(self):
        text = ud_map._size_recipe_4exp(self._rows(), "m.gguf")
        self.assertRegex(text, r"blk\.0\.ffn_gate_up_exps\.\*\.weight.*table=w3nl")
        self.assertRegex(text, r"blk\.0\.ffn_down_exps\.\*\.weight.*table=w4nl")
        self.assertRegex(text, r"blk\.1\.ffn_gate_up_exps\.\*\.weight.*table=w4nl")
        self.assertRegex(text, r"blk\.2\.ffn_gate_up_exps\.\*\.weight\s+ggml type=iq2_s")
        self.assertRegex(text, r"blk\.2\.ffn_down_exps\.\*\.weight\s+ggml type=q2_0")
        self.assertRegex(text, r"token_embd\.weight\s+ggml type=iq4_xs")
        self.assertNotIn("blk.2.ffn_gate_up_exps.*.weight          rtn", text)

    def test_choose_policy_accepts_size(self):
        a = argparse.Namespace(policy="size")
        self.assertEqual(ud_map._choose_policy(a, "qwen4exp"), "size")


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

    def test_q2_0_is_resolved_and_served(self):
        # type 42 is Q2_0: libquant's q2_0 kind and the q20a16 expert GEMM now serve it.
        self.assertEqual(ud_map.TYPES["TYPE42"][0], 2.25)
        self.assertEqual(ud_map.TYPES["TYPE42"][1], "q2_0")
        self.assertTrue(ud_map.TYPES["TYPE42"][2])
        self.assertFalse(ud_map.UNKNOWN_TYPE[2])

    def test_served_flags_match_the_moe_merge(self):
        # IQ4_NL (iq4nl32a16 + PLE gather), Q5_0 (q50a16), Q2_0 (q20a16) are served by base
        # 2a3200f; Q6_K/Q8_0 wait on the dense wiring (L3) and IQ4_XS/IQ3_XXS/IQ2_S on kernels.
        for ty in ("IQ4_NL", "Q5_0", "Q2_0"):
            self.assertTrue(ud_map.TYPES[ty][2], ty)
        for ty in ("Q6_K", "Q8_0", "IQ4_XS", "IQ3_XXS", "IQ2_S"):
            self.assertFalse(ud_map.TYPES[ty][2], ty)

    def test_fast_experts_cover_the_ud_types(self):
        for ty in ("Q4_K", "Q5_K", "Q6_K", "Q8_0", "IQ4_NL", "IQ4_XS", "Q5_0",
                   "IQ3_XXS", "IQ2_S", "Q2_0", "TYPE42"):
            self.assertIn(ty, ud_map.FAST_EXPERTS, ty)


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
        a = argparse.Namespace(gguf=shards, declared=declared, source=None, recipe_out=None,
                               json=out, policy="mirror")
        self.assertEqual(ud_map.plan(a), 0)
        p = json.load(open(out))
        self.assertEqual(p["arch"], "qwen4exp")
        self.assertEqual(p["policy"], "mirror")
        self.assertAlmostEqual(p["totals"]["map_bytes"] / 2**30, 83.31, places=2)
        self.assertAlmostEqual(p["totals"]["kept_bytes"] / 2**30, 5.692, places=2)
        self.assertAlmostEqual(p["totals"]["total_bytes"] / 2**30, 89.003, places=2)
        self.assertEqual(p["totals"]["derived_bytes"], 0)
        self.assertEqual(p["coverage"]["declared"], 1044)
        self.assertEqual(p["coverage"]["missing"], {})
        self.assertEqual(p["fusion_problems"], [])
        # the PLE n-gram table is the single biggest mapped tensor and maps to one declared weight
        big = max(p["tensors"], key=lambda r: r["bytes"])
        self.assertEqual(big["target"], "blk.1.ple_ngram.weight")
        # The mirror recipe keeps Unsloth's per-tensor formats, not the fast codebooks.
        for spell in ("iq4_nl", "iq4_xs", "iq3_xxs", "iq2_s", "q5_0", "q2_0", "q6_k", "q8_0"):
            self.assertIn(f"ggml type={spell}", p["recipe"], spell)
        self.assertNotIn("table=", p["recipe"])
        # blk.48 (the MTP layer) is not in the UD text GGUF: it is kept from bf16, not mapped.
        self.assertNotIn("blk.48", p["recipe"])
        kept = {k["name"].split(".")[0] for k in p["kept"]}
        self.assertEqual(kept, {"blk", "mtp", "v"})

    def test_flash_next_fast_plan(self):
        import argparse
        shards = os.environ["UD_MAP_GGUF"].split(":")
        declared = os.environ["UD_MAP_DECLARED"]
        out = "/tmp/ud_map_test_fast.json"
        a = argparse.Namespace(gguf=shards, declared=declared, source=None, recipe_out=None,
                               json=out, policy="fast")
        self.assertEqual(ud_map.plan(a), 0)
        p = json.load(open(out))
        self.assertEqual(p["policy"], "fast")
        fast = p["fast"]
        self.assertEqual(fast["coverage"]["covered"], 1044)
        self.assertEqual(fast["coverage"]["targets"], 1044)
        self.assertEqual(fast["coverage"]["missing"], [])
        # The fast remap is close to the UD size: the codebook swaps roughly cancel, and the
        # kept-bf16 token embedding (IQ4_XS has no gather yet) is the one net addition.
        self.assertLess(abs(fast["total_bytes"] - p["totals"]["total_bytes"]), 1 * 2**30)
        self.assertTrue(all(r["fast_dtype"] for r in p["tensors"]))
        self.assertIn("table=w3nl", p["recipe"])
        self.assertIn("table=w4nl", p["recipe"])
        self.assertIn("table=w5nl", p["recipe"])
        self.assertNotIn("ggml type=q2_0", p["recipe"])
        self.assertIn("ggml type=iq4_nl", p["recipe"])
        self.assertIn("blk.48.ffn_gate_up_exps.*.weight", p["recipe"])
        self.assertLess(fast["build_bytes"], fast["total_bytes"])
        ple = [r for r in p["tensors"] if r["target"] == "blk.1.ple_ngram.weight"][0]
        self.assertEqual(ple["fast_dtype"], "iq4_nl")
        self.assertTrue(any("L3" in (r["fast_dep"] or "") for r in p["tensors"]))


if __name__ == "__main__":
    unittest.main()
