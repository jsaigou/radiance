/* radtest_duplex.cpp -- the smallest real architecture plugin that declares a duplex tenant, for
 * tests/duplex_plugin_test.cpp. It exists to exercise the engine's duplex branch, its KV planning
 * and the pass loop with a built .so, not to model anything: PROMPT runs one causal attention op
 * over the prompt rows (enough to give the one KV group a block size and a consumer), and STEP
 * copies audio_in to audio_out through the real graph, writes status 0 and a text token
 * lane * 1000 + frame.
 *
 * The text tokens come from two i32 weight tables in the tiny checkpoint, one a lane. A gather
 * reads the frame value from the table, so the value is produced by a kernel (gather_rows2), not by
 * the driver or the engine. The tables hold 1024 frames, enough for this test; a real plugin would
 * compute the token from the bound or read a longer table.
 */
#include <arch/rad_arch.h>

#include <cstdint>
#include <cstring>

namespace radtestduplex {

using namespace rad::arch;

constexpr int64_t kMaxLanes = 2;
constexpr int64_t kSamples = 3200;
constexpr int64_t kMaxPrompt = 64;
constexpr int64_t kTextRows = 1024;

struct Model {
    rad_buf audio_in = 0, audio_out = 0, text_out = 0, status = 0;
    rad_buf q = 0, attn = 0, text_tmp = 0;
    rad_weight text0 = 0, text1 = 0, one = 0;
    rad_kvgroup kv = 0;
    rad_op op_fill_q = 0, op_attn = 0, op_echo = 0, op_status = 0, op_text0 = 0, op_text1 = 0,
           op_cast = 0;
};

static Model g_model[MAX_RANKS];

static int probe(const RadModelMeta*, RadArchProbe* out) {
    if (!out) return RAD_E_INVAL;
    out->gen_duplex = 1;
    out->gen_rows = kMaxPrompt;   /* the largest prompt pass; a STEP carries lanes, fewer rows */
    out->gen_ctx = 13500;         /* 30 min at 7.5 frames a second */
    return RAD_OK;
}

static int declare(RadBuilder* b, const RadModelMeta*, const RadBuildCtx* ctx) {
    if (!b || !ctx || ctx->rank < 0 || ctx->rank >= MAX_RANKS) return RAD_E_INVAL;
    Model& m = g_model[ctx->rank];
    m = Model{};

    m.audio_in  = decl_b(b, "duplex.audio_in",  RAD_F32, { kMaxLanes, kSamples },
                         RAD_BUF_PERSIST);
    m.audio_out = decl_b(b, "duplex.audio_out", RAD_F32, { kMaxLanes, kSamples },
                         RAD_BUF_PERSIST);
    m.text_out  = decl_b(b, "duplex.text_out",  RAD_I32, { kMaxLanes }, RAD_BUF_PERSIST);
    m.status    = decl_b(b, "duplex.status",    RAD_I32, { kMaxLanes }, RAD_BUF_PERSIST);
    m.q         = decl_b(b, "duplex.q",         RAD_F32, { kMaxPrompt, 1 }, RAD_BUF_PERSIST);
    m.attn      = decl_b(b, "duplex.attn",      RAD_F32, { kMaxPrompt, 1 }, RAD_BUF_PERSIST);
    m.text_tmp  = decl_b(b, "duplex.text_tmp",  RAD_F32, { 1, 1 }, RAD_BUF_PERSIST);
    if (!m.audio_in || !m.audio_out || !m.text_out || !m.status || !m.q || !m.attn ||
        !m.text_tmp)
        return RAD_E_INVAL;

    /* The one full-attention group the driver requires, and the user's prompt pass's consumer. */
    RadKVGroupDecl kd{};
    kd.kind = RAD_KV_FULL;
    kd.dtype = RAD_F32;
    kd.n_head_kv = 1;
    kd.head_dim = 1;
    m.kv = rad_decl_kv_group(b, "duplex.kv", &kd);
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

    const RadParam fill_q_p[] = {
        RAD_RANGE("M", 1, kMaxPrompt), RAD_INT("n", 1), RAD_STR("dtype", "f32"),
    };
    m.op_fill_q = rad_decl_op(b, "fill", fill_q_p, (int)(sizeof(fill_q_p) / sizeof(fill_q_p[0])),
                              nullptr, 0);
    const RadParam status_p[] = {
        RAD_RANGE("M", 1, 1), RAD_INT("n", (int)kMaxLanes), RAD_STR("dtype", "i32"),
    };
    m.op_status = rad_decl_op(b, "fill", status_p, (int)(sizeof(status_p) / sizeof(status_p[0])),
                              nullptr, 0);

    /* The twitch: the per-frame tables are in the checkpoint; `one` makes mul_weight an exact
     * copy (x * 1, including -0). */
    /* The loader maps checkpoint tensors to declared weights by the plugin's own name map: with no
     * entry a declared weight is 'absent from the container' and the model cannot run. The tiny
     * checkpoint's tensors are named as the weights are. The entries come BEFORE the weights they
     * name: a weight declared first is read as plain f32 and the engine refuses the program. */
    RAD_ARCH_TRY(map_copy(b, "duplex.text0", "duplex.text0"));
    RAD_ARCH_TRY(map_copy(b, "duplex.text1", "duplex.text1"));
    RAD_ARCH_TRY(map_copy(b, "duplex.one", "duplex.one"));
    m.text0 = decl_w(b, "duplex.text0", RAD_F32, { kTextRows, 1 }, RAD_ACCESS_PER_TOKEN,
                     RAD_SHARD_NONE, grp_model());
    m.text1 = decl_w(b, "duplex.text1", RAD_F32, { kTextRows, 1 }, RAD_ACCESS_PER_TOKEN,
                     RAD_SHARD_NONE, grp_model());
    m.one   = decl_w(b, "duplex.one",   RAD_F32, { 1 }, RAD_ACCESS_PER_TOKEN,
                     RAD_SHARD_NONE, grp_model());
    if (!m.text0 || !m.text1 || !m.one) return RAD_E_INVAL;

    const RadParam echo_p[] = {
        RAD_RANGE("M", 1, kMaxLanes), RAD_INT("n", (int)kSamples), RAD_STR("dtype", "f32"),
        RAD_INT("scalar", 0),
    };
    m.op_echo = rad_decl_op(b, "mul_weight", echo_p, (int)(sizeof(echo_p) / sizeof(echo_p[0])),
                            &m.one, 1);
    const RadParam text_p[] = {
        RAD_RANGE("M", 1, 1), RAD_INT("n_embd", 1), RAD_INT("n_vocab", (int)kTextRows),
        RAD_STR("dtype", "f32"), RAD_INT("vocab_offset", 0), RAD_F64("mult", 1.0),
    };
    m.op_text0 = rad_decl_op(b, "embed_lookup", text_p,
                             (int)(sizeof(text_p) / sizeof(text_p[0])), &m.text0, 1);
    m.op_text1 = rad_decl_op(b, "embed_lookup", text_p,
                             (int)(sizeof(text_p) / sizeof(text_p[0])), &m.text1, 1);
    const RadParam cast_p[] = {
        RAD_RANGE("M", 1, 1), RAD_INT("n", 1), RAD_STR("from", "f32"), RAD_STR("to", "i32"),
    };
    m.op_cast = rad_decl_op(b, "cast", cast_p, (int)(sizeof(cast_p) / sizeof(cast_p[0])),
                            nullptr, 0);
    if (!m.op_fill_q || !m.op_status || !m.op_echo || !m.op_text0 || !m.op_text1 || !m.op_cast)
        return RAD_E_NOKERNEL;

    RadDuplexDecl d{};
    d.name = "radtest-duplex";
    d.tmpl = "<c>{card}</c>{references}";
    d.max_prompt = kMaxPrompt;
    d.max_ref_frames = 2;
    d.ref_rate = 24000;
    d.ref_hop = kSamples;
    d.max_lanes = kMaxLanes;
    d.sample_rate = 24000;
    d.channels = 1;
    d.samples_per_frame = kSamples;
    d.latent_dim = 64;
    d.frame_rate_num = 15;
    d.frame_rate_den = 2;
    d.n_audio_in = 0;
    d.text = 1;
    d.audio_in = m.audio_in;
    d.audio_out = m.audio_out;
    d.text_out = m.text_out;
    d.status = m.status;
    RAD_ARCH_TRY(rad_declare_duplex_tenant(b, &d));
    rad_note(b, "radtest_duplex: one KV group, %lld lanes, %lld-sample frames, echo graph",
             (long long)kMaxLanes, (long long)kSamples);
    return RAD_OK;
}

static void step(RadCtx* c, const RadBatch* batch) {
    if (!c || !batch) return;
    const Model& m = g_model[rad_rank(c)];
    if (batch->gen == RAD_GEN_PASS_DUPLEX_PROMPT) {
        const int64_t T = batch->n_tok;
        if (T <= 0 || T > kMaxPrompt) return;
        RAD_ISSUE(c, m.op_fill_q, RAD_B_N(m.q, 0, T));
        const RadKVGroupBatch* kb = batch->n_kv_groups > 0 ? &batch->kv[0] : nullptr;
        RAD_ISSUE_N(c, m.op_attn, batch->max_q_len,
                    RAD_B_N(m.q, 0, T), RAD_KV(m.kv, 0),
                    praw2(kb ? kb->block_table : nullptr, RAD_I32, batch->n_seq,
                          kb ? kb->block_table_pitch : 0),
                    praw(kb ? kb->seqused : nullptr, RAD_I32, batch->n_seq),
                    RAD_NONE, RAD_NONE,
                    praw(batch->cu_seqlens, RAD_I32, batch->n_seq + 1),
                    RAD_B_N(m.attn, 0, T));
        return;
    }
    if (batch->gen != RAD_GEN_PASS_DUPLEX_STEP || !batch->dup_b) return;
    const RadDuplexBatch* d = batch->dup_b;

    /* Every lane's frame, bit for bit: audio_out = audio_in * 1. */
    RAD_ISSUE_N(c, m.op_echo, kMaxLanes, RAD_B(m.audio_in), RAD_W(m.one), RAD_B(m.audio_out));
    RAD_ISSUE_N(c, m.op_status, 1, RAD_B(m.status));

    for (int64_t i = 0; i < d->n_lane; ++i) {
        const int32_t lane = d->lane_h[i];
        const rad_op op = lane == 1 ? m.op_text1 : m.op_text0;
        const rad_weight w = lane == 1 ? m.text1 : m.text0;
        RAD_ISSUE_N(c, op, 1, praw(d->frame + i, RAD_I32, 1), RAD_W(w),
                    RAD_B(m.text_tmp));
        RAD_ISSUE_N(c, m.op_cast, 1, RAD_B(m.text_tmp), RAD_B_N(m.text_out, lane, 1));
    }
}

}  /* namespace radtestduplex */

#ifndef RAD_ARCH_NO_EXPORTS
RAD_ARCH_PLUGIN(radtestduplex, "radtestduplex", "", "0.0.1",
                "test-only duplex tenant: a one-group echo graph for the engine's pass loop")
RAD_ARCH_PROBE(radtestduplex)
#endif
