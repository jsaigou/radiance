/* haru_lm.cpp -- the implementation of arch/haru_common/haru_lm.h.
 *
 * Compiled into the plugin that uses it (as hyv3_common's implementation is): the file has no
 * exports and no global state.  The declaration and issue order mirror, with the linear weights at
 * the checkpoint's dtype instead of RAD_BF16,
 *
 *   arch/common/rad_block_attn_gated.h:78-308   AttnGatedBF16
 *   arch/common/rad_block_gdn.h:114-354        GdnBF16
 *   arch/common/rad_block_mlp.h:47-116         MlpBF16
 *
 * at the qwen35_bf16 revision of the H4 note (docs/HARU-H4-DESIGN.md section 1.3).  A change to a
 * block's operand order or arithmetic should be diffed against those ranges.
 */
#include "haru_lm.h"

#include <cmath>
#include <cstdio>
#include <cstring>

namespace haru_lm {

int HaruLm::declare(RadBuilder* b, const RadModelMeta* meta, const RadBuildCtx* ctx,
                    const Options& opts) {
    if (!b || !meta || !ctx) return RAD_E_INVAL;
    if (ctx->rank < 0 || ctx->rank >= MAX_RANKS) return RAD_E_INVAL;
    if (ctx->is_draft) {
        fprintf(stderr, "radiance: haru_lm has no draft scope\n");
        return RAD_E_UNSUPPORTED;
    }

    nm = Names(ctx->scope);
    want_logits = opts.logits;
    want_taps   = opts.taps;

    /* The checkpoint's own dtype decides the linear weights, the activations and the KV cache.
     * The name map must exist before weight_codes_dtype can resolve it (rad_arch.h:424-427), and
     * the map must precede the weight (rad_block_gdn.h:145-147). */
    RAD_ARCH_TRY(map_copy(b, nm.f("token_embd.weight"), nm.ckpt("lm.embed_tokens.weight")));
    wd = weight_codes_dtype(b, "token_embd.weight", RAD_BF16);
    if (wd != RAD_F32 && wd != RAD_BF16) {
        fprintf(stderr, "radiance: haru_lm: the embedding table is %s; this block serves f32 or "
                        "bf16 weights\n", rad_dtype_name(wd));
        return RAD_E_UNSUPPORTED;
    }
    dt = rad_dtype_name(wd);

    if (meta->n_expert > 0) {
        fprintf(stderr, "radiance: haru_lm refuses a checkpoint with %lld experts; Haru's LM is "
                        "dense\n", (long long)meta->n_expert);
        return RAD_E_UNSUPPORTED;
    }

    Geom& g = this->g;
    RAD_ARCH_TRY(geom_from(g, meta, ctx, dt, wd));
    /* Qwen3.5's norms are zero-centred, `x_hat * (1 + w)`: every norm except the delta net's
     * RMSNormGated, whose gain is a plain weight (qwen35_bf16.cpp:149-157). */
    g.wadd    = 1.0f;
    g.wadd_qk = 1.0f;

    bool have_interval = false;
    const int64_t interval = meta_int2(meta, "qwen35.full_attention_interval",
                                       "full_attention_interval", 4, &have_interval);
    if (interval <= 0) {
        fprintf(stderr, "radiance: haru_lm: full_attention_interval %lld for %lld layers\n",
                (long long)interval, (long long)g.n_layer);
        return RAD_E_INVAL;
    }
    RAD_ARCH_TRY(check_layer_schedule("haru", meta_text_gets(meta, "layer_types", nullptr),
                                      g.n_layer, interval, {"full_attention"}));

    /* Partial rotary: 0.25 of head_dim (16 of 64 tiny, 64 of 256 real). */
    n_rot = meta_int2(meta, "qwen35.rope.dimension_count", "rope_dimension_count", 0);
    if (n_rot <= 0) {
        double prf = rad_meta_getf(meta, "partial_rotary_factor", 0.0);
        if (!(prf > 0.0)) prf = rad_meta_getf(meta, "rope_parameters.partial_rotary_factor", 0.0);
        if (prf > 0.0) n_rot = (int64_t)((double)g.head_dim * prf);
    }
    if (n_rot <= 0 || n_rot > g.head_dim || (n_rot & 1)) {
        fprintf(stderr, "radiance: haru_lm: no usable rotary width (head_dim %lld); give the "
                        "container qwen35.rope.dimension_count or partial_rotary_factor\n",
                (long long)g.head_dim);
        return RAD_E_INVAL;
    }
    RAD_ARCH_TRY(yarn_from(g, meta, "", n_rot));

    /* The GDN geometry, from the HuggingFace keys the lm.* alias makes bare (or their llama.cpp
     * spellings). */
    const int64_t n_head_k_all = meta_int2(meta, "qwen35.ssm.group_count",
                                           "linear_num_key_heads", 0);
    const int64_t n_head_v_all = meta_int2(meta, "qwen35.ssm.time_step_rank",
                                           "linear_num_value_heads", 0);
    gc.head_k     = meta_int2(meta, "qwen35.ssm.state_size", "linear_key_head_dim", 0);
    gc.head_v     = meta_int2(meta, "qwen35.ssm.value_head_dim", "linear_value_head_dim", 0);
    gc.conv_width = meta_int2(meta, "qwen35.ssm.conv_kernel", "linear_conv_kernel_dim", 0);
    if (gc.head_v <= 0) {
        const int64_t inner = meta_int2(meta, "qwen35.ssm.inner_size", "linear_inner_size", 0);
        if (inner > 0 && n_head_v_all > 0) gc.head_v = inner / n_head_v_all;
    }
    if (n_head_k_all <= 0 || n_head_v_all <= 0 || gc.head_k <= 0 || gc.head_v <= 0 ||
        gc.conv_width <= 0) {
        fprintf(stderr, "radiance: haru_lm: the container does not describe the linear layers "
                        "(key heads %lld, value heads %lld, key width %lld, value width %lld, "
                        "conv width %lld)\n", (long long)n_head_k_all, (long long)n_head_v_all,
                (long long)gc.head_k, (long long)gc.head_v, (long long)gc.conv_width);
        return RAD_E_INVAL;
    }
    gc.chunk = 64;   /* the chunked scan's tile, as qwen35_bf16.cpp:204-208 */
    int s;
    if ((s = divide_or_fail(n_head_k_all, g.world, "linear_num_key_heads", &gc.n_head_k)) < 0)
        return s;
    if ((s = divide_or_fail(n_head_v_all, g.world, "linear_num_value_heads", &gc.n_head_v)) < 0)
        return s;
    if (gc.n_head_v % gc.n_head_k != 0) {
        fprintf(stderr, "radiance: haru_lm: %lld value heads is not a multiple of %lld key "
                        "heads\n", (long long)gc.n_head_v, (long long)gc.n_head_k);
        return RAD_E_INVAL;
    }

    const int64_t q_dim    = g.q_dim();
    const int64_t kv_dim   = g.kv_dim();
    const int64_t v_dim    = gc.v_dim();

    /* The KV cache's dtype is the checkpoint's for the f32 test path (the reference's cache is
     * the model dtype, modeling_qwen3_5.py:785-793) and the deployment's setting otherwise.
     * rad_kv_cache_dtype() cannot express the first: the engine always fills ctx->kv_dtype
     * (bf16 by default), so its fallback never applies and a bf16 cache cost the f32 goldens
     * about 2e-4 at the full-attention layer. */
    const uint32_t kvd = wd == RAD_F32 ? (uint32_t)RAD_F32 : rad_kv_cache_dtype(ctx, RAD_BF16);
    g.kv_dtype = rad_dtype_name(kvd);

    /* ---- the three groups, as qwen35_bf16.cpp:227-254. */
    RadKVGroupDecl fd{};
    fd.kind      = RAD_KV_FULL;
    fd.dtype     = kvd;
    fd.n_head_kv = g.n_head_kv;
    fd.head_dim  = g.head_dim;
    kv_full = rad_decl_kv_group(b, nm.f("kv_attn"), &fd);
    if (!kv_full) return RAD_E_INVAL;

    RadKVGroupDecl sd{};
    sd.kind         = RAD_KV_LINEAR;
    sd.dtype        = RAD_F32;   /* mamba_ssm_dtype: float32 */
    sd.n_head_kv    = gc.n_head_v;
    sd.state_dim[0] = gc.head_v;
    sd.state_dim[1] = gc.head_k;
    kv_state = rad_decl_kv_group(b, nm.f("kv_gdn_state"), &sd);
    if (!kv_state) return RAD_E_INVAL;

    RadKVGroupDecl cd{};
    cd.kind       = RAD_KV_CONV;
    cd.dtype      = wd;          /* the reference's conv_state is the model dtype */
    cd.n_head_kv  = 2 * gc.n_head_k + gc.n_head_v;
    cd.head_dim   = gc.head_k;
    cd.conv_width = gc.conv_width;
    kv_conv = rad_decl_kv_group(b, nm.f("kv_gdn_conv"), &cd);
    if (!kv_conv) return RAD_E_INVAL;

    /* ---- buffers, shared by every layer of a kind (qwen35_bf16.cpp:259-294). */
    b_x  = decl_b(b, nm.f("x"), g.act_dtype, {g.max_tok, g.n_embd});
    b_h  = decl_b(b, nm.f("h"), g.act_dtype, {g.max_tok, g.n_embd});
    b_qg = decl_b(b, nm.f("attn_qg"),  g.act_dtype, {g.max_tok, g.n_head, 2 * g.head_dim});
    b_k  = decl_b(b, nm.f("attn_k"),   g.act_dtype, {g.max_tok, kv_dim});
    b_v  = decl_b(b, nm.f("attn_v"),   g.act_dtype, {g.max_tok, kv_dim});
    b_q  = decl_b(b, nm.f("attn_q"),   g.act_dtype, {g.max_tok, q_dim});
    b_gate = decl_b(b, nm.f("attn_gate"), g.act_dtype, {g.max_tok, q_dim});
    b_attn = decl_b(b, nm.f("attn_out"), g.act_dtype, {g.max_tok, q_dim});
    b_in = decl_b(b, nm.f("gdn_in"), g.act_dtype, {g.max_tok, gc.in_dim()});
    b_z  = decl_b(b, nm.f("gdn_z"),  g.act_dtype, {g.max_tok, v_dim});
    b_gq = decl_b(b, nm.f("gdn_q"),  g.act_dtype, {g.max_tok, gc.n_head_k, gc.head_k});
    b_gk = decl_b(b, nm.f("gdn_k"),  g.act_dtype, {g.max_tok, gc.n_head_k, gc.head_k});
    b_gv = decl_b(b, nm.f("gdn_v"),  g.act_dtype, {g.max_tok, gc.n_head_v, gc.head_v});
    b_gdec = decl_b(b, nm.f("gdn_g"),    RAD_F32, {g.max_tok, gc.n_head_v});
    b_beta = decl_b(b, nm.f("gdn_beta"), RAD_F32, {g.max_tok, gc.n_head_v});
    b_kkt  = decl_b(b, nm.f("gdn_kkt"),  wd,      {g.max_tok, gc.n_head_v, gc.chunk});
    b_go   = decl_b(b, nm.f("gdn_o"),    g.act_dtype, {g.max_tok, gc.n_head_v, gc.head_v});
    b_gate_up = decl_b(b, nm.f("gate_up"), g.act_dtype, {g.max_tok, 2 * g.n_ff});
    b_ffn     = decl_b(b, nm.f("ffn"),     g.act_dtype, {g.max_tok, g.n_ff});
    if (!b_x || !b_h || !b_qg || !b_k || !b_v || !b_q || !b_gate || !b_attn || !b_in ||
        !b_z || !b_gq || !b_gk || !b_gv || !b_gdec || !b_beta || !b_kkt || !b_go ||
        !b_gate_up || !b_ffn)
        return RAD_E_INVAL;

    if (want_logits) {
        b_hf32    = decl_b(b, nm.f("h_f32"),    RAD_F32, {g.max_seqs, g.n_embd});
        b_logits  = decl_b(b, nm.f("logits"),   RAD_F32, {g.max_seqs, g.n_vocab});
        b_pad_eop = decl_b(b, nm.f("pad_eop"),  RAD_F32, {g.max_seqs, 2});
        if (!b_hf32 || !b_logits || !b_pad_eop) return RAD_E_INVAL;
    }
    if (want_taps) {
        tap_cols = g.n_layer * g.n_embd;
        b_tap = decl_b(b, nm.f("taps"), RAD_F32, {1, tap_cols});
        if (!b_tap) return RAD_E_INVAL;
        /* Declared here, before the layers, although the plugin issues it inside the layer loop:
         * the buffer planner orders ops by declaration, and a tap declared after the layers could
         * be laid over a layer buffer that is still live when the tap is written. */
        op_tap = rw(b, RAD_OP(b, "cast",
                         RAD_PARAMS(RAD_RANGE("M", 1, g.max_tok), RAD_INT("n", g.n_embd),
                                    RAD_STR("from", g.dtype), RAD_STR("to", "f32")),
                         RAD_NOWEIGHTS),
                   {b_x}, {b_tap});
        if (!op_tap) return RAD_E_NOKERNEL;
    }

    /* ---- the vocabulary edges.  The token table is the tied head as well: one declared weight
     * feeds embed_lookup and logits_gemm.  The 2-row PAD/EOP view is a separate tiny weight at
     * f32, so the reference's float32 recomputation (haru.py:117-123) is exact on a bf16 model. */
    w_tok = decl_w(b, nm.f("token_embd.weight"), wd, {g.n_vocab, g.n_embd},
                   RAD_ACCESS_VOCAB, RAD_SHARD_ROW, grp_model());
    RAD_ARCH_TRY(map_copy(b, nm.f("output_norm.weight"),
                          nm.ckpt("lm.norm.weight")));
    w_out_norm = decl_w(b, nm.f("output_norm.weight"), RAD_F32, {g.n_embd},
                        RAD_ACCESS_PER_TOKEN, RAD_SHARD_NONE, grp_model());
    if (!w_tok || !w_out_norm) return RAD_E_INVAL;

    if (want_logits) {
        pad_id = rad_meta_geti(meta, "pad_token_id", -1);
        eop_id = rad_meta_geti(meta, "eop_token_id", -1);
        if (pad_id < 0 || eop_id < 0 || pad_id >= g.n_vocab_all || eop_id >= g.n_vocab_all) {
            fprintf(stderr, "radiance: haru_lm: pad_token_id %lld / eop_token_id %lld are not "
                            "two ids under the %lld-token vocabulary\n", (long long)pad_id,
                    (long long)eop_id, (long long)g.n_vocab_all);
            return RAD_E_INVAL;
        }
        const char* emb = nm.ckpt("lm.embed_tokens.weight");
        RAD_ARCH_TRY(map_rows(b, nm.f("pad_eop.weight"),
                              { { emb, (int)pad_id, 1 }, { emb, (int)eop_id, 1 } }));
        w_pad_eop = decl_w(b, nm.f("pad_eop.weight"), RAD_F32, {2, g.n_embd},
                           RAD_ACCESS_PER_TOKEN, RAD_SHARD_NONE, grp_model());
        if (!w_pad_eop) return RAD_E_INVAL;
    }

    /* ---- the token lookup, for the caller's prompt row (the assembled plugin adds the projector
     * outputs to b_x after it). */
    op_embed = rw(b, RAD_OP(b, "embed_lookup",
                       RAD_PARAMS(RAD_RANGE("M", 1, g.max_tok), RAD_INT("n_embd", g.n_embd),
                                  RAD_INT("n_vocab", g.n_vocab), RAD_STR("dtype", g.dtype),
                                  RAD_INT("vocab_offset", g.vocab_off)),
                       RAD_WEIGHTS(w_tok)),
                 {}, {b_x});
    if (!op_embed) return RAD_E_NOKERNEL;
    if (g.world > 1)
        op_embed_ar = rw(b, RAD_OP(b, "all_reduce",
                              RAD_PARAMS(RAD_INT("world_size", g.world),
                                         RAD_RANGE("numel", g.n_embd, g.max_tok * g.n_embd),
                                         RAD_STR("dtype", g.dtype), RAD_INT("exact", g.wire_exact),
                                         RAD_INT("min_bytes", g.wire_min_bytes)),
                              RAD_NOWEIGHTS),
                        {b_x}, {b_x});

    /* ---- the layers. */
    layers.resize((size_t)g.n_layer);
    for (int64_t l = 0; l < g.n_layer; ++l) {
        Layer& lay = layers[(size_t)l];
        lay.layer = (int)l;
        lay.full = is_full_attention(l, interval) ? 1 : 0;
        if (lay.full) {
            RAD_ARCH_TRY(rad_bind_layer_kv(b, (int)l, kv_full));
            RAD_ARCH_TRY(declare_attn(b, (int)l, lay));
        } else {
            RAD_ARCH_TRY(rad_bind_layer_kv(b, (int)l, kv_state));
            RAD_ARCH_TRY(rad_bind_layer_kv(b, (int)l, kv_conv));
            RAD_ARCH_TRY(declare_gdn(b, (int)l, lay));
        }
        RAD_ARCH_TRY(declare_mlp(b, (int)l, lay));
    }

    op_final_norm = rw(b, RAD_OP(b, "rmsnorm",
                            RAD_PARAMS(RAD_RANGE("M", 1, g.max_tok), RAD_INT("n", g.n_embd),
                                       RAD_F64("eps", g.eps), RAD_F64("wadd", 1.0f),
                                       RAD_STR("dtype", g.dtype)),
                            RAD_WEIGHTS(w_out_norm)),
                       {b_x}, {b_h});
    if (!op_final_norm) return RAD_E_NOKERNEL;

    if (want_logits) {
        op_hf32 = rw(b, RAD_OP(b, "cast",
                         RAD_PARAMS(RAD_RANGE("M", 1, g.max_seqs), RAD_INT("n", g.n_embd),
                                    RAD_STR("from", g.dtype), RAD_STR("to", "f32")),
                         RAD_NOWEIGHTS),
                   {b_h}, {b_hf32});
        op_logits = rw(b, RAD_OP(b, "logits_gemm",
                            RAD_PARAMS(RAD_RANGE("M", 1, g.max_seqs),
                                       RAD_INT("n_vocab", g.n_vocab),
                                       RAD_INT("n_embd", g.n_embd), RAD_STR("dtype", g.dtype)),
                            RAD_WEIGHTS(w_tok)),
                       {b_h}, {b_logits});
        op_pad_eop = rw(b, RAD_OP(b, "gemm_nt",
                             RAD_PARAMS(RAD_RANGE("M", 1, g.max_seqs), RAD_INT("N", 2),
                                        RAD_INT("K", g.n_embd), RAD_STR("dtype", "f32")),
                             RAD_WEIGHTS(w_pad_eop)),
                        {b_hf32}, {b_pad_eop});
        if (!op_hf32 || !op_logits || !op_pad_eop) return RAD_E_NOKERNEL;
    }
    rad_note(b, "haru_lm: %lld layers (%lld full attention + %lld gated delta net), n_head=%lld "
                "n_head_kv=%lld head_dim=%lld n_rot=%lld, ffn %lld, vocab=%lld, %s weights, %s "
                "cache; GDN %lld key heads x %lld, %lld value heads x %lld, conv %lld",
             (long long)g.n_layer, (long long)(g.n_layer / interval),
             (long long)(g.n_layer - g.n_layer / interval), (long long)g.n_head,
             (long long)g.n_head_kv, (long long)g.head_dim, (long long)n_rot,
             (long long)g.n_ff, (long long)g.n_vocab, dt, g.kv_dtype,
             (long long)gc.n_head_k, (long long)gc.head_k, (long long)gc.n_head_v,
             (long long)gc.head_v, (long long)gc.conv_width);
    if (!have_interval)
        rad_note(b, "haru_lm: the container stated no full_attention_interval; assuming %lld",
                 (long long)interval);
    return RAD_OK;
}

/* ================================================================== full attention */
int HaruLm::declare_attn(RadBuilder* b, int l, Layer& lay) {
    Attn& a = lay.a;
    const char* pre = "lm.layers";
    const int64_t q_dim  = g.q_dim();
    const int64_t kv_dim = g.kv_dim();
    const int64_t qg_dim = 2 * q_dim;

    /* Maps first, then the weights they name (rad_block_attn_gated.h:111-133). */
    RAD_ARCH_TRY(map_copy(b, nm.f("blk.%d.attn_norm.weight", l),
                          nm.ckpt("%s.%d.input_layernorm.weight", pre, l)));
    RAD_ARCH_TRY(map_copy(b, nm.f("blk.%d.attn_qg.weight", l),
                          nm.ckpt("%s.%d.self_attn.q_proj.weight", pre, l)));
    RAD_ARCH_TRY(map_copy(b, nm.f("blk.%d.attn_k.weight", l),
                          nm.ckpt("%s.%d.self_attn.k_proj.weight", pre, l)));
    RAD_ARCH_TRY(map_copy(b, nm.f("blk.%d.attn_v.weight", l),
                          nm.ckpt("%s.%d.self_attn.v_proj.weight", pre, l)));
    RAD_ARCH_TRY(map_copy(b, nm.f("blk.%d.attn_q_norm.weight", l),
                          nm.ckpt("%s.%d.self_attn.q_norm.weight", pre, l)));
    RAD_ARCH_TRY(map_copy(b, nm.f("blk.%d.attn_k_norm.weight", l),
                          nm.ckpt("%s.%d.self_attn.k_norm.weight", pre, l)));
    RAD_ARCH_TRY(map_copy(b, nm.f("blk.%d.attn_output.weight", l),
                          nm.ckpt("%s.%d.self_attn.o_proj.weight", pre, l)));

    a.w_norm = decl_w(b, nm.f("blk.%d.attn_norm.weight", l), RAD_F32, {g.n_embd},
                      RAD_ACCESS_PER_TOKEN, RAD_SHARD_NONE, grp_layer(l));
    a.w_qg = decl_w(b, nm.f("blk.%d.attn_qg.weight", l), wd, {qg_dim, g.n_embd},
                    RAD_ACCESS_PER_TOKEN, RAD_SHARD_ROW, grp_layer(l));
    a.w_k = decl_w(b, nm.f("blk.%d.attn_k.weight", l), wd, {kv_dim, g.n_embd},
                   RAD_ACCESS_PER_TOKEN, RAD_SHARD_ROW, grp_layer(l));
    a.w_v = decl_w(b, nm.f("blk.%d.attn_v.weight", l), wd, {kv_dim, g.n_embd},
                   RAD_ACCESS_PER_TOKEN, RAD_SHARD_ROW, grp_layer(l));
    a.w_q_norm = decl_w(b, nm.f("blk.%d.attn_q_norm.weight", l), RAD_F32, {g.head_dim},
                        RAD_ACCESS_PER_TOKEN, RAD_SHARD_NONE, grp_layer(l));
    a.w_k_norm = decl_w(b, nm.f("blk.%d.attn_k_norm.weight", l), RAD_F32, {g.head_dim},
                        RAD_ACCESS_PER_TOKEN, RAD_SHARD_NONE, grp_layer(l));
    a.w_o = decl_w(b, nm.f("blk.%d.attn_output.weight", l), wd, {g.n_embd, q_dim},
                   RAD_ACCESS_PER_TOKEN, RAD_SHARD_COL, grp_layer(l));
    if (!a.w_norm || !a.w_qg || !a.w_k || !a.w_v || !a.w_q_norm || !a.w_k_norm || !a.w_o)
        return RAD_E_INVAL;

    a.op_norm = rw(b, RAD_OP(b, "rmsnorm",
                        RAD_PARAMS(RAD_RANGE("M", 1, g.max_tok), RAD_INT("n", g.n_embd),
                                   RAD_F64("eps", g.eps), RAD_F64("wadd", g.wadd),
                                   RAD_STR("dtype", g.dtype)),
                        RAD_WEIGHTS(a.w_norm)),
                  {b_x}, {b_h});
    a.op_qg = rw(b, RAD_OP(b, "gemm_nt",
                      RAD_PARAMS(RAD_RANGE("M", 1, g.max_tok), RAD_INT("N", qg_dim),
                                 RAD_INT("K", g.n_embd), RAD_STR("dtype", g.dtype)),
                      RAD_WEIGHTS(a.w_qg)),
                {b_h}, {b_qg});
    a.op_kp = rw(b, RAD_OP(b, "gemm_nt",
                      RAD_PARAMS(RAD_RANGE("M", 1, g.max_tok), RAD_INT("N", kv_dim),
                                 RAD_INT("K", g.n_embd), RAD_STR("dtype", g.dtype)),
                      RAD_WEIGHTS(a.w_k)),
                {b_h}, {b_k});
    a.op_vp = rw(b, RAD_OP(b, "gemm_nt",
                      RAD_PARAMS(RAD_RANGE("M", 1, g.max_tok), RAD_INT("N", kv_dim),
                                 RAD_INT("K", g.n_embd), RAD_STR("dtype", g.dtype)),
                      RAD_WEIGHTS(a.w_v)),
                {b_h}, {b_v});

    /* The q norm de-interleaves the query out of q|gate; the k norm is in place. */
    a.op_q_norm = rw(b, RAD_OP(b, "rmsnorm",
                          RAD_PARAMS(RAD_RANGE("M", 1, g.max_tok * g.n_head),
                                     RAD_INT("n", g.head_dim), RAD_F64("eps", g.eps),
                                     RAD_F64("wadd", g.wadd_qk), RAD_STR("dtype", g.dtype)),
                          RAD_WEIGHTS(a.w_q_norm)),
                    {b_qg}, {b_q});
    a.op_k_norm = rw(b, RAD_OP(b, "rmsnorm",
                          RAD_PARAMS(RAD_RANGE("M", 1, g.max_tok * g.n_head_kv),
                                     RAD_INT("n", g.head_dim), RAD_F64("eps", g.eps),
                                     RAD_F64("wadd", g.wadd_qk), RAD_STR("dtype", g.dtype)),
                          RAD_WEIGHTS(a.w_k_norm)),
                    {b_k}, {b_k});
    a.op_rope_q = rw(b, decl_rope(b, g, g.max_tok, g.head_dim, g.n_head, 0, n_rot, "mrope"),
                     {b_q}, {b_q});
    a.op_rope_k = rw(b, decl_rope(b, g, g.max_tok, g.head_dim, g.n_head_kv, 0, n_rot, "mrope"),
                     {b_k}, {b_k});

    a.op_attn = rw(b, decl_attn(b, g, "attn_paged",
                          RAD_PARAMS(RAD_RANGE("q_len", 1, g.max_tok),
                                     RAD_INT("head_dim", g.head_dim), RAD_INT("gqa", g.gqa()),
                                     RAD_INT("causal", 1), RAD_INT("window", 0),
                                     RAD_INT("n_head", g.n_head),
                                     RAD_INT("block_size", RAD_KV_BLOCK),
                                     RAD_INT("max_seqs", g.max_seqs),
                                     RAD_INT("max_ctx", g.max_ctx),
                                     RAD_STR("q_dtype", g.dtype),
                                     RAD_STR("kv_dtype", g.kv_dtype)),
                          RAD_NOWEIGHTS),
                    {b_q}, {b_attn});
    /* Declared after attn_paged, which fixes the group's block size. */
    a.op_kv_store = rw(b, RAD_OP(b, "kv_store",
                           RAD_PARAMS(RAD_RANGE("M", 1, g.max_tok),
                                      RAD_INT("head_dim", g.head_dim),
                                      RAD_INT("n_head_kv", g.n_head_kv),
                                      RAD_INT("block_size", rad_kv_block_size(b, kv_full)),
                                      RAD_STR("kv_dtype", g.kv_dtype)),
                           RAD_NOWEIGHTS),
                     {b_k, b_v}, {});
    a.op_sigmoid = rw(b, RAD_OP(b, "sigmoid",
                          RAD_PARAMS(RAD_RANGE("M", 1, g.max_tok * g.n_head),
                                     RAD_INT("n", g.head_dim), RAD_STR("dtype", g.dtype)),
                          RAD_NOWEIGHTS),
                    {b_qg}, {b_gate});
    a.op_gate = rw(b, RAD_OP(b, "mul",
                       RAD_PARAMS(RAD_RANGE("M", 1, g.max_tok), RAD_INT("n", q_dim),
                                  RAD_STR("dtype", g.dtype)),
                       RAD_NOWEIGHTS),
                 {b_attn, b_gate}, {b_attn});
    a.op_o = rw(b, RAD_OP(b, "gemm_nt",
                    RAD_PARAMS(RAD_RANGE("M", 1, g.max_tok), RAD_INT("N", g.n_embd),
                               RAD_INT("K", q_dim), RAD_STR("dtype", g.dtype)),
                    RAD_WEIGHTS(a.w_o)),
              {b_attn}, {b_h});
    if (g.world > 1)
        a.op_ar = rw(b, RAD_OP(b, "all_reduce",
                         RAD_PARAMS(RAD_INT("world_size", g.world),
                                    RAD_RANGE("numel", g.n_embd, g.max_tok * g.n_embd),
                                    RAD_STR("dtype", g.dtype), RAD_INT("exact", g.wire_exact),
                                    RAD_INT("min_bytes", g.wire_min_bytes)),
                         RAD_NOWEIGHTS),
                   {b_h}, {b_h});
    a.op_add = rw(b, RAD_OP(b, "add",
                      RAD_PARAMS(RAD_RANGE("M", 1, g.max_tok), RAD_INT("n", g.n_embd),
                                 RAD_STR("dtype", g.dtype)),
                      RAD_NOWEIGHTS),
                {b_x, b_h}, {b_x});
    if (!a.op_norm || !a.op_qg || !a.op_kp || !a.op_vp || !a.op_q_norm || !a.op_k_norm ||
        !a.op_rope_q || !a.op_rope_k || !a.op_attn || !a.op_kv_store || !a.op_sigmoid ||
        !a.op_gate || !a.op_o || !a.op_add)
        return RAD_E_NOKERNEL;
    return RAD_OK;
}

/* ================================================================== gated delta net */
int HaruLm::declare_gdn(RadBuilder* b, int l, Layer& lay) {
    Gdn& d = lay.d;
    const char* pre = "lm.layers";
    const int64_t v_dim    = gc.v_dim();
    const int64_t conv_dim = gc.conv_dim();
    const int64_t in_dim   = gc.in_dim();

    RAD_ARCH_TRY(map_copy(b, nm.f("blk.%d.attn_norm.weight", l),
                          nm.ckpt("%s.%d.input_layernorm.weight", pre, l)));
    /* q, k, v, a and b in one row, as gdn_conv_prep reads it. */
    RAD_ARCH_TRY(map_concat(b, nm.f("blk.%d.ssm_in.weight", l),
                            { nm.ckpt("%s.%d.linear_attn.in_proj_qkv.weight", pre, l),
                              nm.ckpt("%s.%d.linear_attn.in_proj_a.weight", pre, l),
                              nm.ckpt("%s.%d.linear_attn.in_proj_b.weight", pre, l) }, 0));
    RAD_ARCH_TRY(map_copy(b, nm.f("blk.%d.ssm_z.weight", l),
                          nm.ckpt("%s.%d.linear_attn.in_proj_z.weight", pre, l)));
    RAD_ARCH_TRY(map_copy(b, nm.f("blk.%d.ssm_conv1d.weight", l),
                          nm.ckpt("%s.%d.linear_attn.conv1d.weight", pre, l)));
    RAD_ARCH_TRY(map_copy(b, nm.f("blk.%d.ssm_a_log", l),
                          nm.ckpt("%s.%d.linear_attn.A_log", pre, l)));
    RAD_ARCH_TRY(map_copy(b, nm.f("blk.%d.ssm_dt.bias", l),
                          nm.ckpt("%s.%d.linear_attn.dt_bias", pre, l)));
    RAD_ARCH_TRY(map_copy(b, nm.f("blk.%d.ssm_norm.weight", l),
                          nm.ckpt("%s.%d.linear_attn.norm.weight", pre, l)));
    RAD_ARCH_TRY(map_copy(b, nm.f("blk.%d.ssm_out.weight", l),
                          nm.ckpt("%s.%d.linear_attn.out_proj.weight", pre, l)));

    d.w_norm = decl_w(b, nm.f("blk.%d.attn_norm.weight", l), RAD_F32, {g.n_embd},
                      RAD_ACCESS_PER_TOKEN, RAD_SHARD_NONE, grp_layer(l));
    d.w_in = decl_w(b, nm.f("blk.%d.ssm_in.weight", l), wd, {in_dim, g.n_embd},
                    RAD_ACCESS_PER_TOKEN, RAD_SHARD_ROW, grp_layer(l), 0,
                    {gc.k_dim(), gc.k_dim(), v_dim, gc.n_head_v, gc.n_head_v});
    d.w_z = decl_w(b, nm.f("blk.%d.ssm_z.weight", l), wd, {v_dim, g.n_embd},
                   RAD_ACCESS_PER_TOKEN, RAD_SHARD_ROW, grp_layer(l));
    d.w_conv = decl_w(b, nm.f("blk.%d.ssm_conv1d.weight", l), wd, {conv_dim, gc.conv_width},
                      RAD_ACCESS_PER_TOKEN, RAD_SHARD_ROW, grp_layer(l), 0,
                      {gc.k_dim(), gc.k_dim(), v_dim});
    d.w_a_log = decl_w(b, nm.f("blk.%d.ssm_a_log", l), RAD_F32, {gc.n_head_v},
                       RAD_ACCESS_PER_TOKEN, RAD_SHARD_ROW, grp_layer(l));
    d.w_dt_bias = decl_w(b, nm.f("blk.%d.ssm_dt.bias", l), RAD_F32, {gc.n_head_v},
                         RAD_ACCESS_PER_TOKEN, RAD_SHARD_ROW, grp_layer(l));
    d.w_out_norm = decl_w(b, nm.f("blk.%d.ssm_norm.weight", l), RAD_F32, {gc.head_v},
                          RAD_ACCESS_PER_TOKEN, RAD_SHARD_NONE, grp_layer(l));
    d.w_out = decl_w(b, nm.f("blk.%d.ssm_out.weight", l), wd, {g.n_embd, v_dim},
                     RAD_ACCESS_PER_TOKEN, RAD_SHARD_COL, grp_layer(l));
    if (!d.w_norm || !d.w_in || !d.w_z || !d.w_conv || !d.w_a_log || !d.w_dt_bias ||
        !d.w_out_norm || !d.w_out)
        return RAD_E_INVAL;

    const double q_scale = 1.0 / std::sqrt((double)gc.head_k);

    d.op_norm = rw(b, RAD_OP(b, "rmsnorm",
                        RAD_PARAMS(RAD_RANGE("M", 1, g.max_tok), RAD_INT("n", g.n_embd),
                                   RAD_F64("eps", g.eps), RAD_F64("wadd", g.wadd),
                                   RAD_STR("dtype", g.dtype)),
                        RAD_WEIGHTS(d.w_norm)),
                  {b_x}, {b_h});
    d.op_in = rw(b, RAD_OP(b, "gemm_nt",
                      RAD_PARAMS(RAD_RANGE("M", 1, g.max_tok), RAD_INT("N", in_dim),
                                 RAD_INT("K", g.n_embd), RAD_STR("dtype", g.dtype)),
                      RAD_WEIGHTS(d.w_in)),
                {b_h}, {b_in});
    d.op_z = rw(b, RAD_OP(b, "gemm_nt",
                     RAD_PARAMS(RAD_RANGE("M", 1, g.max_tok), RAD_INT("N", v_dim),
                                RAD_INT("K", g.n_embd), RAD_STR("dtype", g.dtype)),
                     RAD_WEIGHTS(d.w_z)),
               {b_h}, {b_z});

    d.op_conv_prep = rw(b, RAD_OP(b, "gdn_conv_prep",
                            RAD_PARAMS(RAD_RANGE("M", 1, g.max_tok),
                                       RAD_INT("head_k", gc.head_k),
                                       RAD_INT("head_v", gc.head_v),
                                       RAD_INT("chunk", gc.chunk),
                                       RAD_INT("conv_width", gc.conv_width)),
                            RAD_WEIGHTS(d.w_conv, d.w_a_log, d.w_dt_bias)),
                      {b_in}, {b_gq, b_gk, b_gv, b_gdec, b_beta});
    d.op_kkt = rw(b, RAD_OP(b, "gdn_kkt_solve",
                       RAD_PARAMS(RAD_RANGE("M", 1, g.max_tok), RAD_INT("head_k", gc.head_k)),
                       RAD_NOWEIGHTS),
                 {b_gk, b_beta, b_gdec}, {b_kkt});
    d.op_scan = rw(b, RAD_OP(b, "gdn_chunk_scan",
                        RAD_PARAMS(RAD_RANGE("M", 1, g.max_tok), RAD_INT("head_k", gc.head_k),
                                   RAD_INT("head_v", gc.head_v), RAD_F64("scale", q_scale)),
                        RAD_NOWEIGHTS),
                  {b_gq, b_gk, b_gv, b_kkt, b_gdec, b_beta}, {b_go});
    d.op_gnorm = rw(b, RAD_OP(b, "gdn_gated_rmsnorm",
                         RAD_PARAMS(RAD_RANGE("M", 1, g.max_tok),
                                    RAD_INT("channels", gc.head_v),
                                    RAD_F64("eps", g.eps), RAD_STR("act", "silu")),
                         RAD_WEIGHTS(d.w_out_norm)),
                   {b_go, b_z}, {b_go});
    d.op_conv_update = rw(b, RAD_OP(b, "gdn_conv_update",
                               RAD_PARAMS(RAD_RANGE("q_len", 1, g.max_tok),
                                          RAD_INT("head_k", gc.head_k),
                                          RAD_INT("head_v", gc.head_v),
                                          RAD_INT("conv_width", gc.conv_width)),
                               RAD_WEIGHTS(d.w_conv)),
                         {b_in}, {b_gq, b_gk, b_gv});
    d.op_recur = rw(b, RAD_OP(b, "gdn_recurrent_update",
                         RAD_PARAMS(RAD_RANGE("q_len", 1, g.max_tok),
                                    RAD_INT("head_k", gc.head_k),
                                    RAD_INT("head_v", gc.head_v),
                                    RAD_INT("n_head_v", gc.n_head_v),
                                    RAD_F64("scale", q_scale), RAD_F64("eps", g.eps),
                                    RAD_STR("act", "silu"),
                                    RAD_STR("state_form",
                                            g.max_spec > 0 ? "anchor" : "per_candidate")),
                         RAD_WEIGHTS(d.w_a_log, d.w_dt_bias, d.w_out_norm)),
                   {b_gq, b_gk, b_gv, b_in, b_z}, {b_go});
    d.op_out = rw(b, RAD_OP(b, "gemm_nt",
                       RAD_PARAMS(RAD_RANGE("M", 1, g.max_tok), RAD_INT("N", g.n_embd),
                                  RAD_INT("K", v_dim), RAD_STR("dtype", g.dtype)),
                       RAD_WEIGHTS(d.w_out)),
                 {b_go}, {b_h});
    if (g.world > 1)
        d.op_ar = rw(b, RAD_OP(b, "all_reduce",
                          RAD_PARAMS(RAD_INT("world_size", g.world),
                                     RAD_RANGE("numel", g.n_embd, g.max_tok * g.n_embd),
                                     RAD_STR("dtype", g.dtype), RAD_INT("exact", g.wire_exact),
                                     RAD_INT("min_bytes", g.wire_min_bytes)),
                          RAD_NOWEIGHTS),
                    {b_h}, {b_h});
    d.op_add = rw(b, RAD_OP(b, "add",
                       RAD_PARAMS(RAD_RANGE("M", 1, g.max_tok), RAD_INT("n", g.n_embd),
                                  RAD_STR("dtype", g.dtype)),
                       RAD_NOWEIGHTS),
                 {b_x, b_h}, {b_x});
    if (!d.op_norm || !d.op_in || !d.op_z || !d.op_conv_prep || !d.op_kkt || !d.op_scan ||
        !d.op_gnorm || !d.op_conv_update || !d.op_recur || !d.op_out || !d.op_add)
        return RAD_E_NOKERNEL;
    return RAD_OK;
}

/* ================================================================== feed-forward */
int HaruLm::declare_mlp(RadBuilder* b, int l, Layer& lay) {
    Mlp& m = lay.m;
    const char* pre = "lm.layers";
    RAD_ARCH_TRY(map_copy(b, nm.f("blk.%d.ffn_norm.weight", l),
                          nm.ckpt("%s.%d.post_attention_layernorm.weight", pre, l)));
    RAD_ARCH_TRY(map_concat(b, nm.f("blk.%d.ffn_gate_up.weight", l),
                            { nm.ckpt("%s.%d.mlp.gate_proj.weight", pre, l),
                              nm.ckpt("%s.%d.mlp.up_proj.weight", pre, l) }, 0));
    RAD_ARCH_TRY(map_copy(b, nm.f("blk.%d.ffn_down.weight", l),
                          nm.ckpt("%s.%d.mlp.down_proj.weight", pre, l)));

    m.w_norm = decl_w(b, nm.f("blk.%d.ffn_norm.weight", l), RAD_F32, {g.n_embd},
                      RAD_ACCESS_PER_TOKEN, RAD_SHARD_NONE, grp_layer(l));
    m.w_gate_up = decl_w(b, nm.f("blk.%d.ffn_gate_up.weight", l), wd, {2 * g.n_ff, g.n_embd},
                         RAD_ACCESS_PER_TOKEN, RAD_SHARD_ROW, grp_layer(l), 0,
                         {g.n_ff, g.n_ff});
    m.w_down = decl_w(b, nm.f("blk.%d.ffn_down.weight", l), wd, {g.n_embd, g.n_ff},
                      RAD_ACCESS_PER_TOKEN, RAD_SHARD_COL, grp_layer(l));
    if (!m.w_norm || !m.w_gate_up || !m.w_down) return RAD_E_INVAL;

    m.op_norm = rw(b, RAD_OP(b, "rmsnorm",
                        RAD_PARAMS(RAD_RANGE("M", 1, g.max_tok), RAD_INT("n", g.n_embd),
                                   RAD_F64("eps", g.eps), RAD_F64("wadd", g.wadd),
                                   RAD_STR("dtype", g.dtype)),
                        RAD_WEIGHTS(m.w_norm)),
                  {b_x}, {b_h});
    m.op_gate_up = rw(b, RAD_OP(b, "gemm_nt",
                           RAD_PARAMS(RAD_RANGE("M", 1, g.max_tok), RAD_INT("N", 2 * g.n_ff),
                                      RAD_INT("K", g.n_embd), RAD_STR("dtype", g.dtype)),
                           RAD_WEIGHTS(m.w_gate_up)),
                     {b_h}, {b_gate_up});
    m.op_act = rw(b, RAD_OP(b, "silu_mul",
                       RAD_PARAMS(RAD_RANGE("M", 1, g.max_tok), RAD_INT("n", g.n_ff),
                                  RAD_STR("dtype", g.dtype)),
                       RAD_NOWEIGHTS),
                 {b_gate_up}, {b_ffn});
    m.op_down = rw(b, RAD_OP(b, "gemm_nt",
                        RAD_PARAMS(RAD_RANGE("M", 1, g.max_tok), RAD_INT("N", g.n_embd),
                                   RAD_INT("K", g.n_ff), RAD_STR("dtype", g.dtype)),
                        RAD_WEIGHTS(m.w_down)),
                  {b_ffn}, {b_h});
    if (g.world > 1)
        m.op_ar = rw(b, RAD_OP(b, "all_reduce",
                          RAD_PARAMS(RAD_INT("world_size", g.world),
                                     RAD_RANGE("numel", g.n_embd, g.max_tok * g.n_embd),
                                     RAD_STR("dtype", g.dtype), RAD_INT("exact", g.wire_exact),
                                     RAD_INT("min_bytes", g.wire_min_bytes)),
                          RAD_NOWEIGHTS),
                    {b_h}, {b_h});
    m.op_add = rw(b, RAD_OP(b, "add",
                       RAD_PARAMS(RAD_RANGE("M", 1, g.max_tok), RAD_INT("n", g.n_embd),
                                  RAD_STR("dtype", g.dtype)),
                       RAD_NOWEIGHTS),
                 {b_x, b_h}, {b_x});
    if (!m.op_norm || !m.op_gate_up || !m.op_act || !m.op_down || !m.op_add)
        return RAD_E_NOKERNEL;
    return RAD_OK;
}

/* ================================================================== the run */

void HaruLm::step_attn(RadCtx* c, const RadBatch* batch, const Layer& lay) const {
    const Attn& a = lay.a;
    const RadKVGroupBatch* kvb = kv_batch(batch, kv_full);
    const int32_t* slot  = kvb ? kvb->slot_mapping : nullptr;
    const int32_t* table = kvb ? kvb->block_table  : nullptr;
    const int32_t* used  = kvb ? kvb->seqused      : nullptr;
    const int64_t T = batch->n_tok;

    RAD_ISSUE(c, a.op_norm, brows(b_x, T), RAD_W(a.w_norm), brows(b_h, T));
    RAD_ISSUE(c, a.op_qg, brows(b_h, T), RAD_W(a.w_qg), brows(b_qg, T));
    RAD_ISSUE(c, a.op_kp, brows(b_h, T), RAD_W(a.w_k), brows(b_k, T));
    RAD_ISSUE(c, a.op_vp, brows(b_h, T), RAD_W(a.w_v), brows(b_v, T));

    RAD_ISSUE_N(c, a.op_q_norm, T * g.n_head,
                bcol(b_qg, 0, g.head_dim, T), RAD_W(a.w_q_norm), brows(b_q, T));
    RAD_ISSUE_N(c, a.op_k_norm, T * g.n_head_kv,
                brows(b_k, T), RAD_W(a.w_k_norm), brows(b_k, T));
    RAD_ISSUE(c, a.op_rope_q, brows(b_q, T), rope_posmc(g, batch, T));
    RAD_ISSUE(c, a.op_rope_k, brows(b_k, T), rope_posmc(g, batch, T));

    RAD_ISSUE(c, a.op_kv_store,
              brows(b_k, T), brows(b_v, T), praw(slot, RAD_I32, T), kv_cache(kv_full, lay.layer));
    RAD_ISSUE_N(c, a.op_attn, batch->max_q_len,
                brows(b_q, T), kv_cache(kv_full, lay.layer),
                praw2(table, RAD_I32, batch->n_seq, kvb ? kvb->block_table_pitch : 0),
                praw(used, RAD_I32, batch->n_seq),
                RAD_NONE, RAD_NONE,
                praw(batch->cu_seqlens, RAD_I32, batch->n_seq + 1),
                brows(b_attn, T));

    RAD_ISSUE_N(c, a.op_sigmoid, T * g.n_head,
                bcol(b_qg, g.head_dim, g.head_dim, T), brows(b_gate, T));
    RAD_ISSUE(c, a.op_gate, brows(b_attn, T), brows(b_gate, T), brows(b_attn, T));

    RAD_ISSUE(c, a.op_o, brows(b_attn, T), RAD_W(a.w_o), brows(b_h, T));
    if (a.op_ar) RAD_ISSUE_N(c, a.op_ar, T * g.n_embd, brows(b_h, T), RAD_NONE);
    RAD_ISSUE(c, a.op_add, brows(b_x, T), brows(b_h, T), brows(b_x, T));
}

void HaruLm::step_gdn(RadCtx* c, const RadBatch* batch, const Layer& lay) const {
    const Gdn& d = lay.d;
    const RadKVGroupBatch* st = kv_batch(batch, kv_state);
    const RadKVGroupBatch* cv = kv_batch(batch, kv_conv);
    const int32_t* st_idx = st ? st->state_index : nullptr;
    const int64_t st_w = st && st->state_index_pitch > 0 ? st->state_index_pitch : 1;
    const int32_t* cv_idx = cv ? cv->state_index : nullptr;

    const int64_t T = batch->n_tok;
    const int64_t S = batch->n_seq;
    const int64_t conv_dim = gc.conv_dim();
    const int64_t v_dim = gc.v_dim();
    const int64_t H = gc.n_head_v;

    RAD_ISSUE(c, d.op_norm, brows(b_x, T), RAD_W(d.w_norm), brows(b_h, T));
    RAD_ISSUE(c, d.op_in,   brows(b_h, T), RAD_W(d.w_in),   brows(b_in, T));
    RAD_ISSUE(c, d.op_z,    brows(b_h, T), RAD_W(d.w_z),    brows(b_z, T));

    int64_t D = 0, DT = 0;
    batch_split(batch, &D, &DT);
    const int32_t qd = batch->phase == RAD_PHASE_MIXED ? batch->max_q_len_decode
                                                       : batch->max_q_len;
    const int64_t P  = S - D;
    const int64_t PT = T - DT;

    if (D > 0) {
        RAD_ISSUE_N(c, d.op_conv_update, qd,
                    brows(b_in, DT), RAD_W(d.w_conv), RAD_NONE,
                    kv_cache(kv_conv, lay.layer),
                    praw(cv_idx, RAD_I32, D), praw(batch->num_accepted, RAD_I32, D),
                    praw(batch->cu_seqlens, RAD_I32, D + 1),
                    brows(b_gq, DT), brows(b_gk, DT), brows(b_gv, DT));
        RAD_ISSUE_N(c, d.op_recur, qd,
                    brows(b_gq, DT), brows(b_gk, DT), brows(b_gv, DT),
                    bcol(b_in, conv_dim, H, DT), bcol(b_in, conv_dim + H, H, DT),
                    RAD_W(d.w_a_log), RAD_W(d.w_dt_bias), kv_cache(kv_state, lay.layer),
                    praw2(st_idx, RAD_I32, D, st_w), praw(batch->num_accepted, RAD_I32, D),
                    praw(batch->cu_seqlens, RAD_I32, D + 1),
                    brows(b_z, DT), RAD_W(d.w_out_norm), brows(b_go, DT),
                    RAD_NONE, RAD_NONE);
    }

    if (P > 0) {
        RAD_ISSUE(c, d.op_conv_prep,
                  brows(b_in, T), RAD_W(d.w_conv), RAD_NONE, RAD_W(d.w_a_log), RAD_W(d.w_dt_bias),
                  kv_cache(kv_conv, lay.layer), praw(batch->cu_seqlens + D, RAD_I32, P + 1),
                  bcol(b_in, conv_dim, H, T), bcol(b_in, conv_dim + H, H, T),
                  praw(cv_idx ? cv_idx + D : nullptr, RAD_I32, P),
                  praw(batch->ctx_lens + D, RAD_I32, P),
                  brows(b_gq, T), brows(b_gk, T), brows(b_gv, T),
                  brows(b_gdec, T), brows(b_beta, T));
        RAD_ISSUE(c, d.op_kkt,
                  brows(b_gk, T), brows(b_beta, T), brows(b_gdec, T),
                  praw(batch->cu_seqlens + D, RAD_I32, P + 1), brows(b_kkt, T));
        RAD_ISSUE(c, d.op_scan,
                  brows(b_gq, T), brows(b_gk, T), brows(b_gv, T), brows(b_kkt, T),
                  brows(b_gdec, T), brows(b_beta, T),
                  kv_cache(kv_state, lay.layer), praw(batch->cu_seqlens + D, RAD_I32, P + 1),
                  brows(b_go, T), kv_cache(kv_state, lay.layer),
                  praw2(st_idx ? st_idx + D * st_w : nullptr, RAD_I32, P, st_w));
        RAD_ISSUE_N(c, d.op_gnorm, PT,
                    brow_slice(b_go, DT, PT, v_dim), brow_slice(b_z, DT, PT, v_dim),
                    RAD_W(d.w_out_norm), brow_slice(b_go, DT, PT, v_dim));
    }
    RAD_ISSUE(c, d.op_out, brows(b_go, T), RAD_W(d.w_out), brows(b_h, T));
    if (d.op_ar) RAD_ISSUE_N(c, d.op_ar, T * g.n_embd, brows(b_h, T), RAD_NONE);
    RAD_ISSUE(c, d.op_add, brows(b_x, T), brows(b_h, T), brows(b_x, T));
}

void HaruLm::step_mlp(RadCtx* c, const RadBatch* batch, const Layer& lay) const {
    const Mlp& m = lay.m;
    const int64_t T = batch->n_tok;
    RAD_ISSUE(c, m.op_norm,    brows(b_x, T),       RAD_W(m.w_norm),    brows(b_h, T));
    RAD_ISSUE(c, m.op_gate_up, brows(b_h, T),       RAD_W(m.w_gate_up), brows(b_gate_up, T));
    RAD_ISSUE(c, m.op_act,     brows(b_gate_up, T), brows(b_ffn, T));
    RAD_ISSUE(c, m.op_down,    brows(b_ffn, T),     RAD_W(m.w_down),    brows(b_h, T));
    if (m.op_ar) RAD_ISSUE_N(c, m.op_ar, T * g.n_embd, brows(b_h, T), RAD_NONE);
    RAD_ISSUE(c, m.op_add,     brows(b_x, T),       brows(b_h, T),      brows(b_x, T));
}

void HaruLm::step_layers(RadCtx* c, const RadBatch* batch) const {
    const int64_t T = batch->n_tok;
    if (T <= 0 || T > g.max_tok) return;

    for (int64_t l = 0; l < (int64_t)layers.size(); ++l) {
        const Layer& lay = layers[(size_t)l];
        if (lay.full) step_attn(c, batch, lay);
        else          step_gdn(c, batch, lay);
        step_mlp(c, batch, lay);
        /* The reference's per-layer product is the layer's whole output, after the MLP residual
         * (gen_haru_tiny.py:459-463 hooks model.lm.layers[i]), and only a one-row pass has a
         * single row to keep. */
        if (op_tap && T == 1)
            RAD_ISSUE_N(c, op_tap, 1, brow_slice(b_x, 0, 1, g.n_embd),
                        bcol_at(b_tap, 0, tap_cols, l * g.n_embd, g.n_embd, 1));
    }

    RAD_ISSUE(c, op_final_norm, brows(b_x, T), RAD_W(w_out_norm), brows(b_h, T));

    if (op_logits) {
        /* One row a lane of a STEP; the last row of a prompt pass (the duplex driver runs one
         * lane's prompt at a time, and the last row is the one that samples). */
        int64_t n = (batch->gen == RAD_GEN_PASS_DUPLEX_STEP && batch->dup_b)
                        ? batch->dup_b->n_lane
                        : 1;
        if (n > T) n = T;
        if (n <= 0) return;
        RAD_ISSUE_N(c, op_hf32, n, brow_slice(b_h, T - n, n, g.n_embd),
                    brow_slice(b_hf32, 0, n, g.n_embd));
        RAD_ISSUE_N(c, op_logits, n, brow_slice(b_h, T - n, n, g.n_embd), RAD_W(w_tok),
                    brow_slice(b_logits, 0, n, g.n_vocab));
        RAD_ISSUE_N(c, op_pad_eop, n, brow_slice(b_hf32, 0, n, g.n_embd), RAD_W(w_pad_eop),
                    brow_slice(b_pad_eop, 0, n, 2));
    }
}

}  /* namespace haru_lm */
