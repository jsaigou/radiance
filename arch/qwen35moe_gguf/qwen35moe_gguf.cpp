/* qwen35moe_gguf -- Qwen3.5 / 3.6 MoE (35B-A3B) from a llama.cpp GGUF, as the GGUF's own numbers.
 *
 * THIS PLUGIN IS A CONVERTER, NOT YET A SERVER. It declares every weight of the model, the name
 * map from llama.cpp's tensor names to them, and nothing else: no buffers, no ops, no logits. A
 * container written through it holds the GGUF's tensors value for value (docs/
 * radiance-ud-lossless-repack.md, docs/radiance-gguf-plugin.md), and `radiance --model` refuses
 * it at startup ("a Program with no logits buffer cannot serve") until a graph and kernels for
 * the planes exist. Everything the graph will need to read these weights correctly is decided
 * here, at convert, and recorded below.
 *
 * ============================== WHAT llama.cpp CHANGED, AND WHAT THIS UNDOES ==============================
 *
 * Measured against the official FP8 container (Qwen3.6-35B-A3B-FP8-RAD), tensor by tensor:
 *
 *  1. Norms. llama.cpp stores every RMSNorm gain of this family as w + 1 (the model's norms are
 *     x_hat * (1 + w)), in f32. That f32 is exactly bf16(w) + 1 for every norm tested, so the file
 *     already holds what the graph multiplies by. NOT undone: the values are copied verbatim and a
 *     graph over them must run its rmsnorm with wadd = 0 (the fp8 plugin's g.wadd = 1 is the other
 *     half of the same arithmetic). The one gain that is not +1 is ssm_norm (RMSNormGated), copied
 *     verbatim like the rest and read with wadd = 0 there too. Recorded as the container metadata
 *     key `qwen35moe_gguf.norm_wadd = 0`.
 *
 *  2. Value-head order. llama.cpp reorders the delta net's value heads from the checkpoint's
 *     grouped order (head k * a + r of the model, k a key head, r in [0, a)) to a tiled one (head
 *     r * b + k in the file; a = 2 value heads per key head, b = 16 key heads here), so that
 *     ggml_repeat broadcasts the key heads. Affected: the V rows of attn_qkv, all of attn_gate
 *     (the z gate), ssm_alpha, ssm_beta, ssm_a, ssm_dt.bias, the V channels of ssm_conv1d, and the
 *     COLUMNS of ssm_out. UNDONE here, losslessly, by RAD_MAP_REGROUP: a row permutation (or a
 *     permutation of whole 128-column runs of each row, which for Q8_0 is whole blocks), so every
 *     weight is in the order the fp8 plugin's graph and libr4d's delta-net kernels expect.
 *
 *  3. ssm_a. llama.cpp stores -exp(A_log). The kernels take A_log and exponentiate themselves
 *     (rad_block_gdn.h). RAD_MAP_NEGLOG recovers A_log = log(-ssm_a), regrouped, rounded to the
 *     bf16 whose exponential is ssm_a to two f32 ulps (the checkpoint's A_log is bf16) and kept as
 *     f32 otherwise. Not bit-lossless to the f32 ssm_a (exp(A_log) differs from it by <= ~1.6e-7
 *     relative, the rounding of the converter's own exp); it is the checkpoint's own value.
 *
 *  Not changed by llama.cpp (Measured): the Q and K channels of the conv1d, attn_q/k/v/o, the
 *  router, the shared-expert gate, the experts (gate, up, down are separate stacked tensors).
 *
 * ============================== NAMES ==============================
 *
 * Declared names follow the fp8 container's (blk.N.attn_k.weight, blk.N.ffn_norm.weight,
 * mtp.fc.weight), except where the fp8 container fuses what the GGUF keeps apart or the regroup
 * makes a fusion impossible to express:
 *   attn_qkv + attn_gate  -> blk.N.ssm_qkv.weight and blk.N.ssm_z.weight (fp8: ssm_inz)
 *   ssm_alpha, ssm_beta   -> blk.N.ssm_alpha.weight, blk.N.ssm_beta.weight (fp8: ssm_ab)
 *   gate, up of a shared expert and of each routed expert stay apart (fp8: gate_up), because a
 *   fusion needs the two to share a ggml type and the planes need no fusion to be read.
 *   attn_q                -> blk.N.attn_qg.weight (it holds q and its output gate, as fp8's)
 * The MTP head (M1): blk.40.* keeps its GGUF name and is declared like any layer; nextn.eh_proj,
 * hnorm, enorm and shared_head_norm are mtp.fc, mtp.pre_fc_norm_hidden, mtp.pre_fc_norm_embedding
 * and mtp.norm, as in the fp8 container.
 *
 * Quantisation descriptor: "gguf_ftype_*", any GGUF file type (core/build/startup.cpp matches the
 * prefix). The container's descriptor is the file's own (gguf_ftype_15 for Unsloth's UD-Q4_K_XL).
 */
#include <arch/rad_arch.h>

#include <cstring>
#include <deque>
#include <string>
#include <vector>

namespace qwen35moe_gguf {

using namespace rad::arch;

namespace {

struct Cfg {
    int64_t L = 0, n_embd = 0, n_head = 0, n_head_kv = 0, head_dim = 0, n_expert = 0;
    int64_t ff_exp = 0, ff_shexp = 0, interval = 4, nextn = 0;
    int64_t nk = 0, nv = 0, hk = 0, hv = 0, conv = 0;
    int64_t qkv_dim() const { return 2 * nk * hk + nv * hv; }
};

int geti(const RadModelMeta* m, const char* k, int64_t* out) {
    std::string key = std::string("qwen35moe.") + k;
    const long long v = rad_meta_geti(m, key.c_str(), -1);
    if (v < 0) {
        fprintf(stderr, "radiance: qwen35moe_gguf: the GGUF carries no %s\n", key.c_str());
        return RAD_E_UNSUPPORTED;
    }
    *out = v;
    return RAD_OK;
}

/* Declares weight `name` over the map just made for it, at the extents the checkpoint gives it
 * (rad_weight_encoding) after checking them against `expect` (a convention that did not hold
 * would otherwise show as a silent shape change). `dim1` of 0 means rank 1. */
int weight(RadBuilder* b, const char* name, std::initializer_list<int64_t> expect, int access,
           RadWeightGroup grp, uint32_t dflt_dtype) {
    RadEncoding enc{};
    int64_t shape[RAD_MAX_RANK] = { 0 };
    uint32_t rank = 0;
    const int rc = rad_weight_encoding(b, name, &enc, shape, &rank);
    if (rc != RAD_OK) {
        fprintf(stderr, "radiance: qwen35moe_gguf: '%s': no encoding (%d); the GGUF lacks a tensor "
                        "its name map reads\n", name, rc);
        return rc;
    }
    /* A rank-1 gain and a [1, n] one are the same n values. */
    int64_t want = 1, have = 1;
    for (int64_t s : expect) want *= s;
    for (uint32_t i = 0; i < rank; ++i) have *= shape[i];
    bool same = want == 0 || (want == have && (int)rank >= 1);   /* 0: any extent (the vocabulary) */
    if (want && same && expect.size() == rank) {
        size_t i = 0;
        for (int64_t s : expect) same = same && shape[i++] == s;
    }
    if (!same) {
        fprintf(stderr, "radiance: qwen35moe_gguf: '%s' has %u dim(s), %lld element(s); the model "
                        "needs %lld\n", name, rank, (long long)have, (long long)want);
        return RAD_E_SHAPE;
    }
    RadWeightDecl d{};
    d.dtype = enc.n_planes > 0 ? enc.plane[0].dtype : dflt_dtype;
    d.rank = rank;
    for (uint32_t i = 0; i < rank; ++i) d.shape[i] = shape[i];
    d.access = access;
    d.shard = RAD_SHARD_NONE;
    d.group = grp;
    if (enc.n_planes <= 1) return rad_decl_weight(b, name, &d) ? RAD_OK : RAD_E_INVAL;
    /* A weight of several planes is declared as one view per plane. Declared as one operand it
     * would need a kernel's layout to say how the planes join (rad_builder.cpp: "no op reads it,
     * so nothing says how they are stored"); there is no kernel yet, and a container stores the
     * planes separately anyway. Each view is that plane's own dtype and extents. */
    int64_t rows = 1, cols = shape[rank - 1];
    for (uint32_t i = 0; i + 1 < rank; ++i) rows *= shape[i];
    for (int k = 0; k < enc.n_planes; ++k) {
        int64_t pr = 0, pc = 0;
        rad_enc_plane_dims(&enc.plane[k], rows, cols, &pr, &pc);
        RadWeightDecl v{};
        v.dtype = enc.plane[k].dtype;
        v.rank = 2;
        v.shape[0] = pr;
        v.shape[1] = pc;
        v.access = access;
        v.shard = RAD_SHARD_NONE;
        v.group = grp;
        v.source = name;
        v.planes = enc.plane[k].role;
        static thread_local std::deque<std::string> names;
        names.push_back(std::string(name) + "." + enc.plane[k].role);
        if (!rad_decl_weight(b, names.back().c_str(), &v)) return RAD_E_INVAL;
    }
    return RAD_OK;
}

}  /* namespace */

static int declare(RadBuilder* b, const RadModelMeta* meta, const RadBuildCtx* ctx) {
    if (!b || !meta || !ctx) return RAD_E_INVAL;
    if (!meta->quant || std::strncmp(meta->quant, "gguf_ftype_", 11) != 0) {
        fprintf(stderr, "radiance: qwen35moe_gguf reads a GGUF (quantisation gguf_ftype_*); this "
                        "is '%s'\n", meta->quant ? meta->quant : "");
        return RAD_E_UNSUPPORTED;
    }
    if (ctx->world_size != 1) {
        fprintf(stderr, "radiance: qwen35moe_gguf declares weights for one rank only\n");
        return RAD_E_UNSUPPORTED;
    }
    Cfg c;
    int64_t blocks = 0, group_count = 0, rank_ssm = 0, inner = 0, nextn = 0;
    RAD_ARCH_TRY(geti(meta, "block_count", &blocks));
    RAD_ARCH_TRY(geti(meta, "embedding_length", &c.n_embd));
    RAD_ARCH_TRY(geti(meta, "attention.head_count", &c.n_head));
    RAD_ARCH_TRY(geti(meta, "attention.head_count_kv", &c.n_head_kv));
    RAD_ARCH_TRY(geti(meta, "attention.key_length", &c.head_dim));
    RAD_ARCH_TRY(geti(meta, "expert_count", &c.n_expert));
    RAD_ARCH_TRY(geti(meta, "expert_feed_forward_length", &c.ff_exp));
    RAD_ARCH_TRY(geti(meta, "expert_shared_feed_forward_length", &c.ff_shexp));
    RAD_ARCH_TRY(geti(meta, "full_attention_interval", &c.interval));
    RAD_ARCH_TRY(geti(meta, "ssm.conv_kernel", &c.conv));
    RAD_ARCH_TRY(geti(meta, "ssm.state_size", &c.hk));
    RAD_ARCH_TRY(geti(meta, "ssm.group_count", &group_count));
    RAD_ARCH_TRY(geti(meta, "ssm.time_step_rank", &rank_ssm));
    RAD_ARCH_TRY(geti(meta, "ssm.inner_size", &inner));
    nextn = rad_meta_geti(meta, "qwen35moe.nextn_predict_layers", 0);
    c.nk = group_count;
    c.nv = rank_ssm;
    c.hv = c.nv ? inner / c.nv : 0;
    c.nextn = nextn;
    c.L = blocks - nextn;
    if (c.L <= 0 || c.interval <= 0 || c.nk <= 0 || c.nv % c.nk || c.hv <= 0 || nextn > 1) {
        fprintf(stderr, "radiance: qwen35moe_gguf: block_count %lld, nextn %lld, interval %lld, "
                        "ssm heads %lld/%lld: not the Qwen3.5/3.6 layout\n", (long long)blocks,
                (long long)nextn, (long long)c.interval, (long long)c.nk, (long long)c.nv);
        return RAD_E_UNSUPPORTED;
    }
    const int64_t a = c.nv / c.nk;                  /* value heads per key head */
    const int64_t q_out = 2 * c.n_head * c.head_dim; /* q and its output gate */
    const int64_t kv_out = c.n_head_kv * c.head_dim;
    const int64_t o_in = c.n_head * c.head_dim;
    const int64_t v_dim = c.nv * c.hv;

    Names nm("");

    const uint32_t F32 = RAD_F32;
    /* ---- the model-level weights */
    RAD_ARCH_TRY(map_copy(b, nm.f("token_embd.weight"), nm.ckpt("token_embd.weight")));
    RAD_ARCH_TRY(map_copy(b, nm.f("output_norm.weight"), nm.ckpt("output_norm.weight")));
    RAD_ARCH_TRY(map_copy_alt(b, nm.f("output.weight"), nm.ckpt("output.weight"),
                              nm.ckpt("token_embd.weight")));
    RAD_ARCH_TRY(weight(b, "token_embd.weight", { 0, 0 }, RAD_ACCESS_VOCAB, grp_model(), F32));
    RAD_ARCH_TRY(weight(b, "output_norm.weight", { c.n_embd }, RAD_ACCESS_PER_TOKEN, grp_model(), F32));
    RAD_ARCH_TRY(weight(b, "output.weight", { 0, 0 }, RAD_ACCESS_VOCAB, grp_model(), F32));

    int n_decl = 3;
    const int64_t n_layers_all = c.L + c.nextn;
    for (int64_t l = 0; l < n_layers_all; ++l) {
        const bool mtp = l >= c.L;
        const bool full = mtp || (l + 1) % c.interval == 0;
        const RadWeightGroup g = grp_layer((int)l);
        auto D = [&](const char* suffix) { return nm.f("blk.%lld.%s", (long long)l, suffix); };
        auto G = [&](const char* suffix) { return nm.ckpt("blk.%lld.%s", (long long)l, suffix); };
        auto P = [&](std::initializer_list<int64_t> ex, const char* suffix, int access = RAD_ACCESS_PER_TOKEN) {
            ++n_decl;
            return weight(b, D(suffix), ex, access, g, F32);
        };
        /* norms: verbatim, GGUF's w + 1 */
        RAD_ARCH_TRY(map_copy(b, D("attn_norm.weight"), G("attn_norm.weight")));
        RAD_ARCH_TRY(P({ c.n_embd }, "attn_norm.weight"));
        RAD_ARCH_TRY(map_copy(b, D("ffn_norm.weight"), G("post_attention_norm.weight")));
        RAD_ARCH_TRY(P({ c.n_embd }, "ffn_norm.weight"));
        if (full) {
            RAD_ARCH_TRY(map_copy(b, D("attn_qg.weight"), G("attn_q.weight")));
            RAD_ARCH_TRY(map_copy(b, D("attn_k.weight"), G("attn_k.weight")));
            RAD_ARCH_TRY(map_copy(b, D("attn_v.weight"), G("attn_v.weight")));
            RAD_ARCH_TRY(map_copy(b, D("attn_output.weight"), G("attn_output.weight")));
            RAD_ARCH_TRY(map_copy(b, D("attn_q_norm.weight"), G("attn_q_norm.weight")));
            RAD_ARCH_TRY(map_copy(b, D("attn_k_norm.weight"), G("attn_k_norm.weight")));
            RAD_ARCH_TRY(P({ q_out, c.n_embd }, "attn_qg.weight"));
            RAD_ARCH_TRY(P({ kv_out, c.n_embd }, "attn_k.weight"));
            RAD_ARCH_TRY(P({ kv_out, c.n_embd }, "attn_v.weight"));
            RAD_ARCH_TRY(P({ c.n_embd, o_in }, "attn_output.weight"));
            RAD_ARCH_TRY(P({ c.head_dim }, "attn_q_norm.weight"));
            RAD_ARCH_TRY(P({ c.head_dim }, "attn_k_norm.weight"));
        } else {
            /* the delta net, value heads back in the model's grouped order */
            RAD_ARCH_TRY(map_regroup(b, D("ssm_qkv.weight"), G("attn_qkv.weight"), 0,
                                     (int)(2 * c.nk * c.hk), v_dim, (int)a, c.hv));
            RAD_ARCH_TRY(map_regroup(b, D("ssm_z.weight"), G("attn_gate.weight"), 0, 0, v_dim,
                                     (int)a, c.hv));
            RAD_ARCH_TRY(map_regroup(b, D("ssm_alpha.weight"), G("ssm_alpha.weight"), 0, 0, c.nv,
                                     (int)a, 1));
            RAD_ARCH_TRY(map_regroup(b, D("ssm_beta.weight"), G("ssm_beta.weight"), 0, 0, c.nv,
                                     (int)a, 1));
            RAD_ARCH_TRY(map_regroup(b, D("ssm_conv1d.weight"), G("ssm_conv1d.weight"), 0,
                                     (int)(2 * c.nk * c.hk), v_dim, (int)a, c.hv));
            RAD_ARCH_TRY(map_neglog(b, D("ssm_a_log"), G("ssm_a"), 0, 0, c.nv, (int)a, 1));
            RAD_ARCH_TRY(map_regroup(b, D("ssm_dt.bias"), G("ssm_dt.bias"), 0, 0, c.nv, (int)a, 1));
            RAD_ARCH_TRY(map_copy(b, D("ssm_norm.weight"), G("ssm_norm.weight")));
            RAD_ARCH_TRY(map_regroup(b, D("ssm_out.weight"), G("ssm_out.weight"), 1, 0, v_dim,
                                     (int)a, c.hv));
            RAD_ARCH_TRY(P({ c.qkv_dim(), c.n_embd }, "ssm_qkv.weight"));
            RAD_ARCH_TRY(P({ v_dim, c.n_embd }, "ssm_z.weight"));
            RAD_ARCH_TRY(P({ c.nv, c.n_embd }, "ssm_alpha.weight"));
            RAD_ARCH_TRY(P({ c.nv, c.n_embd }, "ssm_beta.weight"));
            RAD_ARCH_TRY(P({ c.qkv_dim(), c.conv }, "ssm_conv1d.weight"));
            RAD_ARCH_TRY(P({ c.nv }, "ssm_a_log"));
            RAD_ARCH_TRY(P({ c.nv }, "ssm_dt.bias"));
            RAD_ARCH_TRY(P({ c.hv }, "ssm_norm.weight"));
            RAD_ARCH_TRY(P({ c.n_embd, v_dim }, "ssm_out.weight"));
        }
        /* the routed feed-forward */
        RAD_ARCH_TRY(map_copy(b, D("ffn_gate_inp.weight"), G("ffn_gate_inp.weight")));
        RAD_ARCH_TRY(map_copy(b, D("ffn_gate_inp_shexp.weight"), G("ffn_gate_inp_shexp.weight")));
        RAD_ARCH_TRY(map_copy(b, D("ffn_gate_shexp.weight"), G("ffn_gate_shexp.weight")));
        RAD_ARCH_TRY(map_copy(b, D("ffn_up_shexp.weight"), G("ffn_up_shexp.weight")));
        RAD_ARCH_TRY(map_copy(b, D("ffn_down_shexp.weight"), G("ffn_down_shexp.weight")));
        RAD_ARCH_TRY(P({ c.n_expert, c.n_embd }, "ffn_gate_inp.weight"));
        RAD_ARCH_TRY(P({ c.n_embd }, "ffn_gate_inp_shexp.weight"));
        RAD_ARCH_TRY(P({ c.ff_shexp, c.n_embd }, "ffn_gate_shexp.weight"));
        RAD_ARCH_TRY(P({ c.ff_shexp, c.n_embd }, "ffn_up_shexp.weight"));
        RAD_ARCH_TRY(P({ c.n_embd, c.ff_shexp }, "ffn_down_shexp.weight"));
        const char* stems[3][2] = { { "ffn_gate_exps", "ffn_gate_exps.weight" },
                                    { "ffn_up_exps", "ffn_up_exps.weight" },
                                    { "ffn_down_exps", "ffn_down_exps.weight" } };
        for (int64_t e = 0; e < c.n_expert; ++e)
            for (int s = 0; s < 3; ++s) {
                const char* dname = nm.f("blk.%lld.%s.%lld.weight", (long long)l, stems[s][0],
                                         (long long)e);
                RAD_ARCH_TRY(map_slice(b, dname, G(stems[s][1]), (int)e));
                RAD_ARCH_TRY(weight(b, dname, s < 2 ? std::initializer_list<int64_t>{ c.ff_exp, c.n_embd }
                                                    : std::initializer_list<int64_t>{ c.n_embd, c.ff_exp },
                                    RAD_ACCESS_CONDITIONAL, grp_expert_slot((int)l, (int)e, s), F32));
                ++n_decl;
            }
    }
    if (c.nextn > 0) {
        const RadWeightGroup g = grp_layer((int)c.L);
        const char* pairs[4][2] = { { "mtp.fc.weight", "blk.%lld.nextn.eh_proj.weight" },
                                    { "mtp.pre_fc_norm_hidden.weight", "blk.%lld.nextn.hnorm.weight" },
                                    { "mtp.pre_fc_norm_embedding.weight", "blk.%lld.nextn.enorm.weight" },
                                    { "mtp.norm.weight", "blk.%lld.nextn.shared_head_norm.weight" } };
        for (int i = 0; i < 4; ++i) {
            RAD_ARCH_TRY(map_copy(b, nm.f("%s", pairs[i][0]), nm.ckpt(pairs[i][1], (long long)c.L)));
            RAD_ARCH_TRY(weight(b, pairs[i][0], i == 0 ? std::initializer_list<int64_t>{ c.n_embd, 2 * c.n_embd }
                                                       : std::initializer_list<int64_t>{ c.n_embd },
                                RAD_ACCESS_PER_TOKEN, g, F32));
            ++n_decl;
        }
    }
    rad_note(b, "qwen35moe_gguf: %d weights from a GGUF, value for value. CONVERT ONLY: no graph, "
                "no buffers, no logits -- nothing serves this container yet. Norm gains are the "
                "GGUF's w + 1 (run rmsnorm with wadd 0); value heads are back in grouped order; "
                "ssm_a_log is log(-ssm_a).", n_decl);
    return RAD_OK;
}

static void step(RadCtx*, const RadBatch*) {}

}  /* namespace qwen35moe_gguf */

RAD_ARCH_PLUGIN(qwen35moe_gguf, "qwen35moe", "gguf_ftype_*", "0.1.0",
                "Qwen3.5 / 3.6 MoE (35B-A3B) from a llama.cpp GGUF as its own planes: the "
                "lossless import of an Unsloth Dynamic file. Convert-only until kernels read the "
                "planes")
RAD_ARCH_SERVES_IMPORTS(0)
