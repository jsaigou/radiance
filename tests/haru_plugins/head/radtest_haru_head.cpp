/* radtest_haru_head.cpp -- the duplex tenant that wraps Haru's MeanFlow head and its two
 * audio-input projectors for tests/haru_head_test.cpp.
 *
 * The engine's duplex ABI has one tenant a program (builder: "declare_duplex_tenant: ... was
 * already declared"), while this block needs two tunnel shapes (a head row and an embedding row).
 * The test therefore starts one engine per shape, each on a temporary copy of the tiny checkpoint
 * whose config.json names the shape in the free-form key "haru_test_tenant"; the plugin declares
 * the tenant under that name. This is test-only glue, recorded in the H2 report; the architecture
 * id is still the checkpoint's "haru".
 *
 * Tensor tunnelling (tests/HARU_HARNESS.md): a STEP's audio_in carries the block's inputs in its
 * first samples and audio_out returns them at the same lane offset. Nothing else is read.
 */
#include <arch/rad_arch.h>

#include "../../../arch/haru_common/haru_head.h"
#include "../../../arch/haru_common/haru_proj.h"

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>

namespace radtestharuhead {

using namespace rad::arch;

constexpr int64_t kMaxLanes = 4;
constexpr int64_t kSamples = 3200;
constexpr int64_t kMaxPrompt = 64;
constexpr int64_t kMaxRefFrames = 16;
constexpr int64_t kMaxCtx = 512;

enum Mode { MD_HEAD = 0, MD_EMBED = 1 };

static Mode tenant_mode(const RadModelMeta* meta) {
    const char* t = rad_meta_gets(meta, "haru_test_tenant", "head");
    return (t && std::strcmp(t, "embed") == 0) ? MD_EMBED : MD_HEAD;
}

struct Model {
    Mode mode = MD_HEAD;
    rad_buf audio_in = 0, audio_out = 0, status = 0;
    rad_buf mark = 0, token_i32 = 0;
    rad_kvgroup kv = 0;
    rad_op op_attn = 0, op_mark = 0, op_status = 0, op_zero = 0, op_cast = 0;
    rad_weight embed_tokens = 0;
    HaruHead head;
    HaruProj proj;
};

static Model g_model[MAX_RANKS];

static int probe(const RadModelMeta*, RadArchProbe* out) {
    if (!out) return RAD_E_INVAL;
    out->gen_duplex = 1;
    out->gen_rows = kMaxPrompt;
    out->gen_ctx = kMaxCtx;
    return RAD_OK;
}

/* The tiny checkpoint keeps its LM dimensions under "lm", which the engine flattens to free-form
 * keys; the head and the projectors read the keys they need here. */
static int64_t meta_i(const RadModelMeta* m, const char* key, int64_t d) {
    return (int64_t)rad_meta_geti(m, key, (long long)d);
}

static int declare(RadBuilder* b, const RadModelMeta* meta, const RadBuildCtx* ctx) {
    if (!b || !meta || !ctx || ctx->rank < 0 || ctx->rank >= MAX_RANKS) return RAD_E_INVAL;
    Model& m = g_model[ctx->rank];
    m = Model{};
    m.mode = tenant_mode(meta);

    m.audio_in  = decl_b(b, "haru.audio_in",  RAD_F32, { kMaxLanes * kSamples }, RAD_BUF_PERSIST);
    m.audio_out = decl_b(b, "haru.audio_out", RAD_F32, { kMaxLanes * kSamples }, RAD_BUF_PERSIST);
    m.status    = decl_b(b, "haru.status",    RAD_I32, { kMaxLanes }, RAD_BUF_PERSIST);
    m.mark      = decl_b(b, "haru.mark",      RAD_F32, { 1 }, RAD_BUF_PERSIST);
    m.token_i32 = decl_b(b, "haru.token",     RAD_I32, { 1 }, RAD_BUF_PERSIST);
    if (!m.audio_in || !m.audio_out || !m.status || !m.mark || !m.token_i32) return RAD_E_INVAL;

    /* The duplex driver stages exactly one paged full-attention group. This tenant does not read
     * it, but the driver's lane bookkeeping does; the attention op also pins the group's block
     * size, which the builder reads from the first op after the group that carries one. */
    RadKVGroupDecl kd{};
    kd.kind = RAD_KV_FULL;
    kd.dtype = RAD_F32;
    kd.n_head_kv = 1;
    kd.head_dim = 1;
    m.kv = rad_decl_kv_group(b, "haru.kv", &kd);
    if (!m.kv) return RAD_E_INVAL;

    const RadParam attn_p[] = {
        RAD_RANGE("q_len", 1, kMaxPrompt), RAD_INT("head_dim", 1), RAD_INT("gqa", 1),
        RAD_INT("causal", 1), RAD_INT("window", 0), RAD_INT("n_head", 1),
        RAD_INT("block_size", 16), RAD_INT("max_seqs", ctx->max_seqs),
        RAD_INT("max_ctx", ctx->max_ctx), RAD_STR("q_dtype", "f32"),
        RAD_STR("kv_dtype", "f32"), RAD_F64("scale", 1.0),
    };
    m.op_attn = rad_decl_op(b, "attn_paged", attn_p, (int)(sizeof(attn_p) / sizeof(attn_p[0])),
                            nullptr, 0);
    if (!m.op_attn) return RAD_E_NOKERNEL;
    RAD_ARCH_TRY(rad_bind_layer_kv(b, 0, m.kv));

    const RadParam mark_p[] = {
        RAD_RANGE("M", 1, 1), RAD_INT("n", 1), RAD_STR("dtype", "f32"),
    };
    m.op_mark = rad_decl_op(b, "fill", mark_p, (int)(sizeof(mark_p) / sizeof(mark_p[0])),
                            nullptr, 0);
    const RadParam status_p[] = {
        RAD_RANGE("M", 1, 1), RAD_INT("n", 1), RAD_STR("dtype", "i32"),
    };
    m.op_status = rad_decl_op(b, "fill", status_p, (int)(sizeof(status_p) / sizeof(status_p[0])),
                              nullptr, 0);
    const RadParam zero_p[] = {
        RAD_RANGE("M", 1, 1), RAD_INT("n", (int)kSamples), RAD_STR("dtype", "f32"),
    };
    m.op_zero = rad_decl_op(b, "fill", zero_p, (int)(sizeof(zero_p) / sizeof(zero_p[0])),
                            nullptr, 0);
    const RadParam cast_p[] = {
        RAD_RANGE("M", 1, 1), RAD_INT("n", 1), RAD_STR("from", "f32"), RAD_STR("to", "i32"),
    };
    m.op_cast = rad_decl_op(b, "cast", cast_p, (int)(sizeof(cast_p) / sizeof(cast_p[0])),
                            nullptr, 0);
    if (!m.op_mark || !m.op_status || !m.op_zero || !m.op_cast) return RAD_E_NOKERNEL;

    /* Geometry from the checkpoint's free-form keys. The nested `lm` and `head` objects arrive as
     * "lm.<key>" and "head.<key>". */
    const int64_t lm_hidden = meta_i(meta, "lm.hidden_size", -1);
    const int64_t head_hidden = meta_i(meta, "head.hidden_size", lm_hidden);
    const int64_t latent = meta_i(meta, "head.latent_dim", meta_i(meta, "latent_dim", -1));
    const int64_t n_vocab = meta_i(meta, "lm.vocab_size", meta->n_vocab);
    if (lm_hidden <= 0 || head_hidden <= 0 || latent <= 0 || n_vocab <= 0) return RAD_E_INVAL;

    m.head.g.hidden = head_hidden;
    m.head.g.latent = latent;
    m.head.g.layers = meta_i(meta, "head.num_layers", 4);
    m.head.g.norm_eps = rad_meta_getf(meta, "head.norm_eps", 1e-5);
    m.head.g.interval_embed = rad_meta_getf(meta, "head.interval_embed", 7.0);
    const double ffn_ratio = rad_meta_getf(meta, "head.ffn_ratio", 3.0);
    m.head.g.ffn = (int64_t)((double)head_hidden * ffn_ratio);
    if (m.head.g.ffn <= 0) return RAD_E_INVAL;

    m.proj.g.latent = latent;
    m.proj.g.hidden = lm_hidden;
    m.proj.g.n_vocab = n_vocab;

    /* The embedding table is the LM's, so the assembling plugin declares it once; this block
     * only reads it. The name map entry precedes the weight. */
    RAD_ARCH_TRY(map_copy(b, "lm.embed_tokens.weight", "lm.embed_tokens.weight"));
    m.embed_tokens = decl_w(b, "lm.embed_tokens.weight", RAD_F32, { n_vocab, lm_hidden },
                            RAD_ACCESS_VOCAB, RAD_SHARD_NONE, grp_model());
    if (!m.embed_tokens) return RAD_E_INVAL;

    RAD_ARCH_TRY(m.head.declare(b));
    RAD_ARCH_TRY(m.proj.declare(b, m.embed_tokens));

    RadDuplexDecl d{};
    d.name = m.mode == MD_EMBED ? "embed" : "head";
    d.tmpl = "<c>{card}</c>{references}";
    d.max_prompt = kMaxPrompt;
    d.max_ref_frames = kMaxRefFrames;
    d.ref_rate = 24000;
    d.ref_hop = kSamples;
    d.max_lanes = kMaxLanes;
    d.sample_rate = 24000;
    d.channels = 1;
    d.samples_per_frame = kSamples;
    d.latent_dim = latent;
    d.frame_rate_num = 15;
    d.frame_rate_den = 2;
    d.n_audio_in = 0;
    d.text = 0;
    d.audio_in = m.audio_in;
    d.audio_out = m.audio_out;
    d.status = m.status;
    RAD_ARCH_TRY(rad_declare_duplex_tenant(b, &d));

    rad_note(b, "radtest_haru_head: tenant '%s', head %lld/%lld/%lld x %lld, proj %lld <- %lld",
             d.name, (long long)m.head.g.hidden, (long long)m.head.g.latent,
             (long long)m.head.g.ffn, (long long)m.head.g.layers,
             (long long)m.proj.g.hidden, (long long)m.proj.g.latent);
    return RAD_OK;
}

static void step(RadCtx* c, const RadBatch* batch) {
    if (!c || !batch) return;
    const Model& m = g_model[rad_rank(c)];

    if (batch->gen == RAD_GEN_PASS_DUPLEX_PROMPT) {
        /* The prompt rows only open the lane for this block; one op keeps the pass non-empty. */
        RAD_ISSUE_N(c, m.op_mark, 1, RAD_B(m.mark));
        return;
    }
    if (batch->gen != RAD_GEN_PASS_DUPLEX_STEP || !batch->dup_b) return;
    const RadDuplexBatch* d = batch->dup_b;

    for (int64_t i = 0; i < d->n_lane; ++i) {
        const int64_t lane = d->lane_h[i];
        const int64_t base = lane * kSamples;
        RAD_ISSUE_N(c, m.op_status, 1, RAD_B_N(m.status, lane, 1));
        /* The tunnel's unused tail is zero, so a reply carries only the tensor it computed. */
        RAD_ISSUE_N(c, m.op_zero, 1, RAD_B_N(m.audio_out, base, kSamples));

        if (m.mode == MD_EMBED) {
            const int64_t D = m.proj.g.latent;
            /* [token | model_latent | user_latent | model_mask | user_mask] */
            RAD_ISSUE_N(c, m.op_cast, 1, RAD_B_N(m.audio_in, base, 1), RAD_B(m.token_i32));
            m.proj.issue(c, RAD_B(m.token_i32),
                         RAD_B_N(m.audio_in, base + 1, D),
                         RAD_B_N(m.audio_in, base + 1 + D, D),
                         RAD_B_N(m.audio_in, base + 1 + 2 * D, 1),
                         RAD_B_N(m.audio_in, base + 2 + 2 * D, 1),
                         RAD_B_N(m.audio_out, base, m.proj.g.hidden));
        } else {
            const int64_t H = m.head.g.hidden;
            const int64_t D = m.head.g.latent;
            /* [hidden | noise | guidance] */
            m.head.issue(c, RAD_B_N(m.audio_in, base, H),
                         RAD_B_N(m.audio_in, base + H, D),
                         RAD_B_N(m.audio_in, base + H + D, 1),
                         RAD_B_N(m.audio_out, base, D));
        }
    }
}

}  /* namespace radtestharuhead */

#ifndef RAD_ARCH_NO_EXPORTS
RAD_ARCH_PLUGIN(radtestharuhead, "haru", "", "0.0.1",
                "test-only duplex tenant: Haru's MeanFlow head and audio-input projectors")
RAD_ARCH_PROBE(radtestharuhead)
#endif
