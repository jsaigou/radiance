/* radtest_haru_lm.cpp -- the H4 test plugin: Haru's LM trunk as a duplex tenant, fed one
 * embedding row a step through the tensor tunnel.
 *
 * The plugin is the block's only caller.  It declares the trunk (arch/haru_common/haru_lm.cpp,
 * compiled in by path below) with its logits and layer taps, then one duplex tenant:
 *
 *   PROMPT  one row, the card the engine renders from "<|im_start|>{card}" (an empty card makes it
 *           the single special token 482, which is `prompt.embed[0]`'s token).  The row's
 *           embedding is the token lookup, so the pass checks the trunk's chunked prefill form.
 *   STEP    one row a lane, the golden `prompt.embed[i]` / `frame{t}.embed` in audio_in[0..63].
 *           The row is cast into the trunk's `b_x`; the trunk runs; the reply packs the final
 *           normed hidden, then the four layer taps, then the 512 logits, then the two f32
 *           PAD/EOP logits, in that order:
 *
 *             [0, 64)          hidden
 *             [64, 320)        layer 0..3 outputs (only on a one-row pass)
 *             [320, 832)       512 logits
 *             [832, 834)       PAD, EOP logits
 *
 * The remaining samples are the driver's zeroed audio_out.
 *
 * What the row-by-row STEP form does and does not prove: it checks the GDN's *recurrent* form
 * (gdn_conv_update + gdn_recurrent_update) and a one-row paged attention against the reference
 * goldens, whose prompt was produced by one 37-row chunked prefill.  The two GDN forms are the
 * same recurrence algebraically, so the comparison is meaningful, but it is not a bit-level check
 * of gdn_conv_prep/kkt_solve/chunk_scan; a true 37-row PROMPT would need a per-row embedding
 * channel the duplex ABI does not have (docs/HARU-H4-DESIGN.md section 8.5).
 */
#include <arch/rad_arch.h>

#include "../../../arch/haru_common/haru_lm.cpp"

#include <cstdint>

namespace radtest_harulm {

using namespace rad::arch;

constexpr int64_t kMaxLanes = 2;
constexpr int64_t kSamples   = 3200;
constexpr int64_t kMaxRows   = 64;
constexpr int64_t kMaxCtx    = 128;

struct Model {
    haru_lm::HaruLm lm;
    rad_buf audio_in = 0, audio_out = 0, status = 0;
    rad_op op_in = 0, op_hidden = 0, op_taps = 0, op_logits = 0, op_pad = 0, op_status = 0;
};

static Model g_model[MAX_RANKS];

static int probe(const RadModelMeta*, RadArchProbe* out) {
    if (!out) return RAD_E_INVAL;
    out->gen_duplex = 1;
    out->gen_rows = kMaxRows;   /* the largest pass: a prompt row or one row a lane */
    out->gen_ctx = kMaxCtx;     /* a lane's prompt and live frames */
    return RAD_OK;
}

/* A cast op over 1..kMaxLanes rows of `n` elements. */
static rad_op row_copy(RadBuilder* b, int64_t n, const char* from, const char* to,
                       rad_buf src, rad_buf dst) {
    const RadParam p[] = {
        RAD_RANGE("M", 1, kMaxLanes), RAD_INT("n", n), RAD_STR("from", from), RAD_STR("to", to),
    };
    return rw(b, RAD_OP(b, "cast", p, (int)(sizeof(p) / sizeof(p[0])), nullptr, 0), { src }, { dst });
}

static int declare(RadBuilder* b, const RadModelMeta* meta, const RadBuildCtx* ctx) {
    if (!b || !meta || !ctx) return RAD_E_INVAL;
    if (ctx->rank < 0 || ctx->rank >= MAX_RANKS) return RAD_E_INVAL;
    Model& m = g_model[ctx->rank];
    m = Model{};

    haru_lm::Options o;
    o.logits = true;
    o.taps   = true;
    RAD_ARCH_TRY(m.lm.declare(b, meta, ctx, o));
    if (m.lm.tap_cols <= 0) return RAD_E_INVAL;

    Names nm(ctx->scope);
    m.audio_in  = decl_b(b, nm.f("audio_in"),  RAD_F32, { kMaxLanes, kSamples }, RAD_BUF_PERSIST);
    m.audio_out = decl_b(b, nm.f("audio_out"), RAD_F32, { kMaxLanes, kSamples }, RAD_BUF_PERSIST);
    m.status    = decl_b(b, nm.f("status"),    RAD_I32, { kMaxLanes }, RAD_BUF_PERSIST);
    if (!m.audio_in || !m.audio_out || !m.status) return RAD_E_INVAL;

    const int64_t ne = m.lm.g.n_embd;
    m.op_in = row_copy(b, ne, "f32", m.lm.g.dtype, m.audio_in, m.lm.b_x);
    m.op_hidden = row_copy(b, ne, m.lm.g.dtype, "f32", m.lm.b_h, m.audio_out);
    m.op_taps = row_copy(b, m.lm.tap_cols, "f32", "f32", m.lm.b_tap, m.audio_out);
    m.op_logits = row_copy(b, m.lm.g.n_vocab, "f32", "f32", m.lm.b_logits, m.audio_out);
    m.op_pad = row_copy(b, 2, "f32", "f32", m.lm.b_pad_eop, m.audio_out);
    const RadParam status_p[] = {
        RAD_RANGE("M", 1, 1), RAD_INT("n", (int)kMaxLanes), RAD_STR("dtype", "i32"),
    };
    m.op_status = rw(b, RAD_OP(b, "fill", status_p, (int)(sizeof(status_p) / sizeof(status_p[0])),
                               nullptr, 0),
                     {}, { m.status });
    if (!m.op_in || !m.op_hidden || !m.op_taps || !m.op_logits || !m.op_pad || !m.op_status)
        return RAD_E_NOKERNEL;

    RadDuplexDecl d{};
    d.name = "haru-lm";
    /* One row for an empty card: "<|im_start|>" is the tiny tokenizer's single special token 482,
     * the first golden prompt token.  max_prompt 1 makes a tokenizer surprise fail the open rather
     * than silently shift every position. */
    d.tmpl = "<|im_start|>{card}";
    d.max_prompt = 1;
    d.max_ref_frames = 0;
    d.ref_rate = 0;
    d.ref_hop = 0;
    d.max_lanes = kMaxLanes;
    d.sample_rate = 24000;
    d.channels = 1;
    d.samples_per_frame = kSamples;
    d.latent_dim = 64;
    d.frame_rate_num = 15;
    d.frame_rate_den = 2;
    d.n_audio_in = 0;
    d.text = 0;
    d.audio_in = m.audio_in;
    d.audio_out = m.audio_out;
    d.text_out = 0;
    d.status = m.status;
    RAD_ARCH_TRY(rad_declare_duplex_tenant(b, &d));

    rad_note(b, "radtest_haru_lm: one tenant, %lld lanes, one row a step, reply = %lld hidden + "
                "%lld taps + %lld logits + 2 PAD/EOP",
             (long long)kMaxLanes, (long long)ne, (long long)m.lm.tap_cols,
             (long long)m.lm.g.n_vocab);
    return RAD_OK;
}

static void step(RadCtx* c, const RadBatch* batch) {
    if (!c || !batch) return;
    Model& m = g_model[rad_rank(c)];
    const int64_t ne = m.lm.g.n_embd;

    if (batch->gen == RAD_GEN_PASS_DUPLEX_PROMPT) {
        const int64_t T = batch->n_tok;
        if (T <= 0 || T > m.lm.g.max_tok) return;
        RAD_ISSUE_N(c, m.lm.op_embed, T, praw(batch->token_ids, RAD_I32, T),
                    RAD_W(m.lm.w_tok), brows(m.lm.b_x, T));
        if (m.lm.op_embed_ar)
            RAD_ISSUE_N(c, m.lm.op_embed_ar, T * ne, brows(m.lm.b_x, T), RAD_NONE);
        m.lm.step_layers(c, batch);
        return;
    }
    if (batch->gen != RAD_GEN_PASS_DUPLEX_STEP || !batch->dup_b) return;

    const RadDuplexBatch* d = batch->dup_b;
    const int64_t L = d->n_lane;
    if (L <= 0 || L > kMaxLanes || batch->n_tok != L) return;

    for (int64_t i = 0; i < L; ++i) {
        const int32_t lane = d->lane_h[i];
        if (lane < 0 || lane >= kMaxLanes) return;
        RAD_ISSUE_N(c, m.op_in, 1, bcol_at(m.audio_in, lane, kSamples, 0, ne, 1),
                    brow_slice(m.lm.b_x, i, 1, ne));
    }

    m.lm.step_layers(c, batch);

    /* Every lane of the pass replied with audio. */
    RAD_ISSUE_N(c, m.op_status, 1, RAD_B(m.status));

    const int64_t tap_cols = m.lm.tap_cols;
    const int64_t logits_off = ne + tap_cols;
    const int64_t pad_off = logits_off + m.lm.g.n_vocab;
    for (int64_t i = 0; i < L; ++i) {
        const int32_t lane = d->lane_h[i];
        RAD_ISSUE_N(c, m.op_hidden, 1, brow_slice(m.lm.b_h, i, 1, ne),
                    bcol_at(m.audio_out, lane, kSamples, 0, ne, 1));
        /* The block keeps the taps only for a one-row pass. */
        if (batch->n_tok == 1)
            RAD_ISSUE_N(c, m.op_taps, 1, brow_slice(m.lm.b_tap, 0, 1, tap_cols),
                        bcol_at(m.audio_out, lane, kSamples, ne, tap_cols, 1));
        RAD_ISSUE_N(c, m.op_logits, 1, brow_slice(m.lm.b_logits, i, 1, m.lm.g.n_vocab),
                    bcol_at(m.audio_out, lane, kSamples, logits_off, m.lm.g.n_vocab, 1));
        RAD_ISSUE_N(c, m.op_pad, 1, brow_slice(m.lm.b_pad_eop, i, 1, 2),
                    bcol_at(m.audio_out, lane, kSamples, pad_off, 2, 1));
    }
}

}  /* namespace radtest_harulm */

#ifndef RAD_ARCH_NO_EXPORTS
RAD_ARCH_PLUGIN(radtest_harulm, "haru", "", "0.0.1",
                "test-only Haru LM trunk: one embedding row a step, hidden + layer taps + logits")
RAD_ARCH_PROBE(radtest_harulm)
#endif
