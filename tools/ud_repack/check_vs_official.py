#!/usr/bin/env python3
"""check_vs_official -- the small tensors of a converted .rad against the official bf16 safetensors,
bit for bit: proves the conventions the plugin undoes (value-head order, A_log) and keeps (norm w + 1).

    python3 check_vs_official.py MODEL.rad BF16_DIR
"""
import os, sys
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from radfile import RadFile, plane_values
from route_b import Safetensors


def main():
    rf = RadFile(sys.argv[1]); st = Safetensors(sys.argv[2])
    L = 40
    pre = 'model.language_model.layers.%d.'
    res = {}

    def chk(key, ok, note=''):
        r = res.setdefault(key, [0, 0, note])
        r[0] += 1; r[1] += bool(ok)

    def val(name):
        return plane_values(rf, rf.by_name[name], 0).astype(np.float32)

    for l in range(L):
        p = pre % l
        full = (l + 1) % 4 == 0
        chk('attn_norm == bf16 + 1 (f32 add)', np.array_equal(val(f'blk.{l}.attn_norm.weight').reshape(-1), st.get(p + 'input_layernorm.weight') + np.float32(1)))
        chk('ffn_norm == bf16 + 1', np.array_equal(val(f'blk.{l}.ffn_norm.weight').reshape(-1), st.get(p + 'post_attention_layernorm.weight') + np.float32(1)))
        chk('router == bf16', np.array_equal(val(f'blk.{l}.ffn_gate_inp.weight').reshape(256, 2048), st.get(p + 'mlp.gate.weight')))
        chk('shared-expert gate == bf16', np.array_equal(val(f'blk.{l}.ffn_gate_inp_shexp.weight').reshape(-1), st.get(p + 'mlp.shared_expert_gate.weight').reshape(-1)))
        if full:
            chk('q_norm == bf16 + 1', np.array_equal(val(f'blk.{l}.attn_q_norm.weight').reshape(-1), st.get(p + 'self_attn.q_norm.weight') + np.float32(1)))
            chk('k_norm == bf16 + 1', np.array_equal(val(f'blk.{l}.attn_k_norm.weight').reshape(-1), st.get(p + 'self_attn.k_norm.weight') + np.float32(1)))
        else:
            q = p + 'linear_attn.'
            chk('A_log == bf16 (recovered, bit-equal)', np.array_equal(val(f'blk.{l}.ssm_a_log').reshape(-1), st.get(q + 'A_log')))
            chk('dt_bias == bf16 (regrouped)', np.array_equal(val(f'blk.{l}.ssm_dt.bias').reshape(-1), st.get(q + 'dt_bias')))
            chk('in_proj_a == ssm_alpha (regrouped rows)', np.array_equal(val(f'blk.{l}.ssm_alpha.weight').reshape(32, 2048), st.get(q + 'in_proj_a.weight')))
            chk('in_proj_b == ssm_beta (regrouped rows)', np.array_equal(val(f'blk.{l}.ssm_beta.weight').reshape(32, 2048), st.get(q + 'in_proj_b.weight')))
            chk('conv1d == bf16 (regrouped V channels)', np.array_equal(val(f'blk.{l}.ssm_conv1d.weight').reshape(8192, 4), st.get(q + 'conv1d.weight').reshape(8192, 4)))
            chk('ssm_norm == bf16 (no +1)', np.array_equal(val(f'blk.{l}.ssm_norm.weight').reshape(-1), st.get(q + 'norm.weight')))
    chk('output_norm == bf16 + 1', np.array_equal(val('output_norm.weight').reshape(-1), st.get('model.language_model.norm.weight') + np.float32(1)))
    for n, h in [('mtp.norm.weight', 'mtp.norm.weight'), ('mtp.pre_fc_norm_hidden.weight', 'mtp.pre_fc_norm_hidden.weight'),
                 ('mtp.pre_fc_norm_embedding.weight', 'mtp.pre_fc_norm_embedding.weight')]:
        chk('mtp norms == bf16 + 1', np.array_equal(val(n).reshape(-1), st.get(h) + np.float32(1)))
    chk('blk.40 attn_norm == bf16 + 1', np.array_equal(val('blk.40.attn_norm.weight').reshape(-1), st.get('mtp.layers.0.input_layernorm.weight') + np.float32(1)))
    bad = 0
    for k, (n, ok, _) in res.items():
        print(f'{ok:3d}/{n:3d}  {k}')
        bad += n - ok
    sys.exit(1 if bad else 0)


if __name__ == '__main__':
    main()
