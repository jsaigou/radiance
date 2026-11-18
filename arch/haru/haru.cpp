/* haru.cpp -- the assembled Haru duplex architecture plugin.
 *
 * One duplex tenant: the Qwen3.5 hybrid LM (haru_lm), the two audio-input projectors (haru_proj),
 * the MeanFlow head (haru_head) and the VibeVoice causal codec (haru_codec). PROMPT runs the card
 * and reference rows, samples the host-side text row and produces model frame 0 (decoded only to
 * advance the decoder state; frame 0 is not sent). STEP runs one user frame and writes the next
 * model frame, its logits row and its PAD/EOP pair. The host owns text sampling (ABI 39); the
 * plugin never draws a token or a noise vector.
 *
 * The state/lifetime rules are docs/HARU-H5-DESIGN.md:
 *   - codec reset in PROMPT (before references and before frame 0's decode), never on FIRST;
 *   - the one-ahead pairing: reply to wire step s is model frame s+1;
 *   - frame 0 is decoded but not written to audio_out;
 *   - reference rows use PAD for all but the last reference frame and EOP for the last.
 */
#include <arch/rad_arch.h>

#include "../haru_common/haru_codec.h"
#include "../haru_common/haru_head.h"
#include "../haru_common/haru_lm.h"
#include "../haru_common/haru_proj.h"

#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <initializer_list>
#include <string>

namespace haru_asm {

using namespace rad::arch;

struct Model {
    haru_lm::HaruLm lm;
    HaruProj proj;
    HaruHead head;
    haru::Codec codec;

    rad_buf audio_in = 0, audio_out = 0, text_out = 0, status = 0;
    rad_buf logits_out = 0, pad_eop_out = 0, text_in = 0;
    rad_buf prev_latent = 0, head_norm = 0;
    rad_buf mask_one = 0, mask_zero = 0, zero_latent = 0, guidance = 0;
    rad_buf pad_tok = 0, eop_tok = 0;

    rad_op op_copy_ref = 0, op_copy_mean = 0, op_copy_noise = 0, op_copy_prev = 0;
    rad_op op_copy_logits = 0, op_copy_pad = 0;
    rad_op op_fill_mask_one = 0, op_fill_mask_zero = 0, op_fill_zero_latent = 0;
    rad_op op_fill_guidance = 0, op_fill_pad = 0, op_fill_eop = 0;
    rad_op op_fill_status = 0, op_fill_text = 0;

    int64_t max_lanes = 0, frame_samples = 0, latent_dim = 0, n_vocab = 0;
    int32_t pad_id = -1, eop_id = -1, eos_id = -1;
};

static Model g_model[MAX_RANKS];

static int64_t meta_i(const RadModelMeta* m, const char* key, int64_t d) {
    return (int64_t)rad_meta_geti(m, key, (long long)d);
}

static int probe(const RadModelMeta*, RadArchProbe* out) {
    if (!out) return RAD_E_INVAL;
    out->gen_duplex = 1;
    out->gen_rows = 4096;   /* the largest prompt pass; a STEP carries lanes, fewer rows */
    out->gen_ctx = 13500;   /* 30 minutes at 7.5 frames a second */
    return RAD_OK;
}

/* One cast op of `n` f32 elements. `src` may be a declaration buffer; an external batch field
 * has no declared handle, so it is an operand at issue time. */
static rad_op copy_f32(RadBuilder* b, int64_t max_rows, int64_t n, rad_buf src, rad_buf dst) {
    const RadParam p[] = {
        RAD_RANGE("M", 1, max_rows), RAD_INT("n", n), RAD_STR("from", "f32"), RAD_STR("to", "f32"),
    };
    rad_op op = RAD_OP(b, "cast", p, (int)(sizeof(p) / sizeof(p[0])), nullptr, 0);
    if (src) rw(b, op, { src }, { dst });
    else     rw(b, op, {}, { dst });
    return op;
}

static rad_op fill_op(RadBuilder* b, int64_t n, const char* dtype, double value, rad_buf dst) {
    const RadParam p[] = {
        RAD_RANGE("M", 1, 1), RAD_INT("n", n), RAD_STR("dtype", dtype), RAD_F64("value", value),
    };
    rad_op op = RAD_OP(b, "fill", p, (int)(sizeof(p) / sizeof(p[0])), nullptr, 0);
    return rw(b, op, {}, { dst });
}

static void fill_constants(RadCtx* c, const Model& m) {
    RAD_ISSUE_N(c, m.op_fill_mask_one, 1, RAD_B(m.mask_one));
    RAD_ISSUE_N(c, m.op_fill_mask_zero, 1, RAD_B(m.mask_zero));
    RAD_ISSUE_N(c, m.op_fill_zero_latent, 1, RAD_B(m.zero_latent));
    RAD_ISSUE_N(c, m.op_fill_guidance, 1, RAD_B(m.guidance));
    RAD_ISSUE_N(c, m.op_fill_pad, 1, RAD_B(m.pad_tok));
    RAD_ISSUE_N(c, m.op_fill_eop, 1, RAD_B(m.eop_tok));
}

static int declare(RadBuilder* b, const RadModelMeta* meta, const RadBuildCtx* ctx) {
    if (!b || !meta || !ctx || ctx->rank < 0 || ctx->rank >= MAX_RANKS) return RAD_E_INVAL;
    if (ctx->shape_probe) return RAD_OK;
    Model& m = g_model[ctx->rank];
    m = Model{};

    const int64_t max_lanes = std::max<int64_t>(1, ctx->max_seqs);
    const int64_t max_ctx = ctx->max_ctx > 0 ? ctx->max_ctx : 4096;
    const int64_t max_tok = ctx->max_tok > 0 ? ctx->max_tok : 4096;
    const int64_t max_prompt = std::max<int64_t>(1, std::min<int64_t>(std::min(max_ctx, max_tok),
                                                                     4096));
    const int64_t max_ref_frames = std::max<int64_t>(1, std::min<int64_t>(max_prompt, 1400));

    haru_lm::Options lo;
    lo.logits = true;
    lo.taps = false;
    RAD_ARCH_TRY(m.lm.declare(b, meta, ctx, lo));

    haru::CodecGeometry cg{};
    std::string why;
    if (haru::codec_geometry_from_meta(meta, &cg, &why) < 0) {
        rad_note(b, "haru: codec geometry: %s", why.c_str());
        return RAD_E_INVAL;
    }
    if (cg.frame_samples <= 0 || cg.latent_dim <= 0) return RAD_E_INVAL;
    cg.max_lanes = max_lanes;

    const int64_t H = m.lm.g.n_embd;
    const int64_t D = cg.latent_dim;
    const int64_t S = cg.frame_samples;
    const int64_t V = m.lm.g.n_vocab;
    m.max_lanes = max_lanes;
    m.frame_samples = S;
    m.latent_dim = D;
    m.n_vocab = V;

    m.pad_id = (int32_t)meta_i(meta, "pad_token_id", -1);
    m.eop_id = (int32_t)meta_i(meta, "eop_token_id", -1);
    m.eos_id = (int32_t)meta_i(meta, "eos_token_id", -1);
    if (m.pad_id < 0 || m.eop_id < 0 || m.eop_id >= V) {
        rad_note(b, "haru: pad/eop ids %d/%d are not usable with vocab %lld", m.pad_id, m.eop_id,
                 (long long)V);
        return RAD_E_INVAL;
    }

    m.proj.g.latent = D;
    m.proj.g.hidden = H;
    m.proj.g.n_vocab = V;
    RAD_ARCH_TRY(m.proj.declare(b, m.lm.w_tok));

    m.head.g.hidden = meta_i(meta, "head.hidden_size", H);
    m.head.g.latent = meta_i(meta, "head.latent_dim", D);
    m.head.g.layers = meta_i(meta, "head.num_layers", 4);
    m.head.g.norm_eps = rad_meta_getf(meta, "head.norm_eps", 1e-5);
    m.head.g.interval_embed = rad_meta_getf(meta, "head.interval_embed", 7.0);
    const double ffn_ratio = rad_meta_getf(meta, "head.ffn_ratio", 3.0);
    m.head.g.ffn = (int64_t)((double)m.head.g.hidden * ffn_ratio);
    if (m.head.g.hidden <= 0 || m.head.g.latent <= 0 || m.head.g.layers <= 0 ||
        m.head.g.ffn <= 0)
        return RAD_E_INVAL;
    RAD_ARCH_TRY(m.head.declare(b));

    Names nm(ctx->scope);
    m.audio_in = decl_b(b, nm.f("duplex.audio_in"), RAD_F32, { max_lanes, S }, RAD_BUF_PERSIST);
    m.audio_out = decl_b(b, nm.f("duplex.audio_out"), RAD_F32, { max_lanes, S },
                         RAD_BUF_PERSIST);
    m.text_out = decl_b(b, nm.f("duplex.text_out"), RAD_I32, { max_lanes }, RAD_BUF_PERSIST);
    m.status = decl_b(b, nm.f("duplex.status"), RAD_I32, { max_lanes }, RAD_BUF_PERSIST);
    m.logits_out = decl_b(b, nm.f("duplex.logits_out"), RAD_F32, { max_lanes, V },
                          RAD_BUF_PERSIST);
    m.pad_eop_out = decl_b(b, nm.f("duplex.pad_eop_out"), RAD_F32, { max_lanes, 2 },
                           RAD_BUF_PERSIST);
    m.text_in = decl_b(b, nm.f("duplex.text_in"), RAD_I32, { max_lanes }, RAD_BUF_PERSIST);
    m.prev_latent = decl_b(b, nm.f("duplex.prev_latent"), RAD_F32, { max_lanes, D },
                           RAD_BUF_PERSIST);
    m.head_norm = decl_b(b, nm.f("duplex.head_norm"), RAD_F32, { max_lanes, D },
                         RAD_BUF_PERSIST);
    m.mask_one = decl_b(b, nm.f("duplex.mask_one"), RAD_F32, { max_lanes }, RAD_BUF_PERSIST);
    m.mask_zero = decl_b(b, nm.f("duplex.mask_zero"), RAD_F32, { max_lanes }, RAD_BUF_PERSIST);
    m.zero_latent = decl_b(b, nm.f("duplex.zero_latent"), RAD_F32, { 1, D }, RAD_BUF_PERSIST);
    m.guidance = decl_b(b, nm.f("duplex.guidance"), RAD_F32, { 1 }, RAD_BUF_PERSIST);
    m.pad_tok = decl_b(b, nm.f("duplex.pad_tok"), RAD_I32, { 1 }, RAD_BUF_PERSIST);
    m.eop_tok = decl_b(b, nm.f("duplex.eop_tok"), RAD_I32, { 1 }, RAD_BUF_PERSIST);
    if (!m.audio_in || !m.audio_out || !m.text_out || !m.status || !m.logits_out ||
        !m.pad_eop_out || !m.text_in || !m.prev_latent || !m.head_norm || !m.mask_one ||
        !m.mask_zero || !m.zero_latent || !m.guidance || !m.pad_tok || !m.eop_tok)
        return RAD_E_INVAL;

    RAD_ARCH_TRY(haru::codec_declare(b, cg, m.audio_in, m.audio_out, &m.codec));

    m.op_copy_ref = copy_f32(b, S, 1, 0, m.audio_in);
    m.op_copy_mean = copy_f32(b, 1, D, m.audio_out, m.audio_in);
    m.op_copy_noise = copy_f32(b, 1, D, 0, m.audio_in);
    m.op_copy_prev = copy_f32(b, 1, D, m.prev_latent, m.audio_in);
    m.op_copy_logits = copy_f32(b, 1, V, m.lm.b_logits, m.logits_out);
    m.op_copy_pad = copy_f32(b, 1, 2, m.lm.b_pad_eop, m.pad_eop_out);
    m.op_fill_mask_one = fill_op(b, max_lanes, "f32", 1.0, m.mask_one);
    m.op_fill_mask_zero = fill_op(b, max_lanes, "f32", 0.0, m.mask_zero);
    m.op_fill_zero_latent = fill_op(b, D, "f32", 0.0, m.zero_latent);
    m.op_fill_guidance =
        fill_op(b, 1, "f32", rad_meta_getf(meta, "head.guidance_scale", 1.5), m.guidance);
    m.op_fill_pad = fill_op(b, 1, "i32", (double)m.pad_id, m.pad_tok);
    m.op_fill_eop = fill_op(b, 1, "i32", (double)m.eop_id, m.eop_tok);
    m.op_fill_status = fill_op(b, max_lanes, "i32", 0.0, m.status);
    m.op_fill_text = fill_op(b, max_lanes, "i32", -1.0, m.text_out);
    if (!m.op_copy_ref || !m.op_copy_mean || !m.op_copy_noise || !m.op_copy_prev ||
        !m.op_copy_logits || !m.op_copy_pad || !m.op_fill_mask_one || !m.op_fill_mask_zero ||
        !m.op_fill_zero_latent || !m.op_fill_guidance || !m.op_fill_pad || !m.op_fill_eop ||
        !m.op_fill_status || !m.op_fill_text)
        return RAD_E_NOKERNEL;

    RadDuplexDecl d{};
    d.name = "haru";
    d.tmpl = "<|im_start|>system\n{card}<|im_end|>\n{references}";
    d.max_prompt = max_prompt;
    d.max_ref_frames = max_ref_frames;
    d.ref_rate = 24000;
    d.ref_hop = S;
    d.max_lanes = max_lanes;
    d.sample_rate = 24000;
    d.channels = 1;
    d.samples_per_frame = S;
    d.latent_dim = D;
    d.frame_rate_num = 15;
    d.frame_rate_den = 2;
    d.n_audio_in = 2;
    d.audio_ins[0] = RadDuplexAudioIn{ RAD_DUPLEX_AUDIO_USER, D, H, H, 1 };
    d.audio_ins[1] = RadDuplexAudioIn{ RAD_DUPLEX_AUDIO_MODEL, D, H, H, 1 };
    d.text = 1;
    d.audio_in = m.audio_in;
    d.audio_out = m.audio_out;
    d.text_out = m.text_out;
    d.status = m.status;
    d.n_vocab = V;
    d.host_sampling = 1;
    d.pad_id = m.pad_id;
    d.eop_id = m.eop_id;
    d.eos_id = m.eos_id;
    d.logits_out = m.logits_out;
    d.pad_eop_out = m.pad_eop_out;
    d.text_in = m.text_in;
    RAD_ARCH_TRY(rad_declare_duplex_tenant(b, &d));

    rad_note(b, "haru: %lld lanes, %lld-sample frames, latent %lld, vocab %lld; one-ahead "
                "duplex tenant", (long long)max_lanes, (long long)S, (long long)D,
             (long long)V);
    return RAD_OK;
}

/* Copy the lane's raw latent from prev_latent into audio_in[lane][0..D), the decoder's input. */
static void copy_prev_to_decoder(RadCtx* c, const Model& m, int32_t lane) {
    RAD_ISSUE_N(c, m.op_copy_prev, 1,
                bcol_at(m.prev_latent, lane, m.latent_dim, 0, m.latent_dim, 1),
                bcol_at(m.audio_in, lane, m.frame_samples, 0, m.latent_dim, 1));
}

static void prompt_pass(RadCtx* c, const RadBatch* batch) {
    Model& m = g_model[rad_rank(c)];
    const RadDuplexBatch* d = batch->dup_b;
    if (!d || d->n_lane < 1) return;
    const int64_t T = batch->n_tok;
    const int32_t lane = d->lane_h[0];
    if (T <= 0 || lane < 0 || lane >= m.max_lanes) return;

    fill_constants(c, m);
    haru::codec_prepare(c, m.codec);
    haru::codec_reset_lane(c, m.codec, lane);

    const int64_t n_frames =
        d->ref_samples > 0 ? (d->ref_samples + m.frame_samples - 1) / m.frame_samples : 0;
    for (int64_t t = 0; t < T; ++t) {
        const int32_t rf = d->ref_frame_h ? d->ref_frame_h[t] : -1;
        RadOperand model_lat = RAD_B(m.zero_latent);
        RadOperand user_lat = RAD_B(m.zero_latent);
        RadOperand model_mask = RAD_B_N(m.mask_zero, lane, 1);
        RadOperand user_mask = RAD_B_N(m.mask_zero, lane, 1);
        RadOperand token = praw(batch->token_ids + t, RAD_I32, 1);
        if (rf >= 0) {
            const int64_t frame = rf;
            const int64_t off = frame * m.frame_samples;
            int64_t n = m.frame_samples;
            if (off >= d->ref_samples) return;
            if (off + n > d->ref_samples) n = d->ref_samples - off;
            RAD_ISSUE_N(c, m.op_copy_ref, n, praw(d->ref_audio + off, RAD_F32, n),
                        bcol_at(m.audio_in, lane, m.frame_samples, 0, n, 1));
            haru::codec_encode(c, m.codec, lane, false, m.audio_in, m.audio_out, n, -1);

            /* mean + the host's combined codec noise -> raw reference latent in audio_out. */
            RAD_ISSUE_N(c, m.op_copy_mean, 1,
                        bcol_at(m.audio_out, lane, m.frame_samples, 0, m.latent_dim, 1),
                        bcol_at(m.audio_in, lane, m.frame_samples, 0, m.latent_dim, 1));
            RAD_ISSUE_N(c, m.op_copy_noise, 1,
                        praw(d->codec_noise + t * m.latent_dim, RAD_F32, m.latent_dim),
                        bcol_at(m.audio_in, lane, m.frame_samples, m.latent_dim, m.latent_dim, 1));
            haru::codec_sample(c, m.codec, lane, m.audio_in, m.audio_out);

            model_lat = bcol_at(m.audio_out, lane, m.frame_samples, 0, m.latent_dim, 1);
            model_mask = RAD_B_N(m.mask_one, lane, 1);
            token = (frame == n_frames - 1) ? RAD_B(m.eop_tok) : RAD_B(m.pad_tok);
        }
        m.proj.issue(c, token, model_lat, user_lat, model_mask, user_mask,
                     brow_slice(m.lm.b_x, t, 1, m.lm.g.n_embd));
    }

    m.lm.step_layers(c, batch);

    /* The last prompt row is frame 0's row. Its text row goes to the host; frame 0's audio is
     * decoded only to advance the decoder state and is not written to audio_out. */
    RAD_ISSUE_N(c, m.op_copy_logits, 1,
                brow_slice(m.lm.b_logits, 0, 1, m.n_vocab),
                bcol_at(m.logits_out, lane, m.n_vocab, 0, m.n_vocab, 1));
    RAD_ISSUE_N(c, m.op_copy_pad, 1,
                brow_slice(m.lm.b_pad_eop, 0, 1, 2),
                bcol_at(m.pad_eop_out, lane, 2, 0, 2, 1));
    m.head.issue(c, brow_slice(m.lm.b_h, T - 1, 1, m.lm.g.n_embd),
                 praw(d->head_noise, RAD_F32, m.latent_dim), RAD_B(m.guidance),
                 bcol_at(m.head_norm, lane, m.latent_dim, 0, m.latent_dim, 1));
    m.proj.unscale(c, bcol_at(m.head_norm, lane, m.latent_dim, 0, m.latent_dim, 1),
                   bcol_at(m.prev_latent, lane, m.latent_dim, 0, m.latent_dim, 1));

    /* The reference/user streams are separate in the reference; reset once more, then let frame
     * 0's decode set the streaming decoder's state for frame 1. */
    haru::codec_reset_lane(c, m.codec, lane);
    copy_prev_to_decoder(c, m, lane);
    haru::codec_decode(c, m.codec, lane, false, m.audio_in, m.audio_out, -1);
}

static void step_pass(RadCtx* c, const RadBatch* batch) {
    Model& m = g_model[rad_rank(c)];
    const RadDuplexBatch* d = batch->dup_b;
    if (!d) return;
    const int64_t L = d->n_lane;
    if (L <= 0 || batch->n_tok != L) return;

    fill_constants(c, m);
    RAD_ISSUE_N(c, m.op_fill_status, 1, RAD_B(m.status));
    RAD_ISSUE_N(c, m.op_fill_text, 1, RAD_B(m.text_out));

    for (int64_t i = 0; i < L; ++i) {
        const int32_t lane = d->lane_h[i];
        if (lane < 0 || lane >= m.max_lanes) continue;
        haru::codec_encode(c, m.codec, lane, false, m.audio_in, m.audio_out, m.frame_samples, -1);

        RAD_ISSUE_N(c, m.op_copy_mean, 1,
                    bcol_at(m.audio_out, lane, m.frame_samples, 0, m.latent_dim, 1),
                    bcol_at(m.audio_in, lane, m.frame_samples, 0, m.latent_dim, 1));
        RAD_ISSUE_N(c, m.op_copy_noise, 1,
                    praw(d->codec_noise + i * m.latent_dim, RAD_F32, m.latent_dim),
                    bcol_at(m.audio_in, lane, m.frame_samples, m.latent_dim, m.latent_dim, 1));
        haru::codec_sample(c, m.codec, lane, m.audio_in, m.audio_out);

        m.proj.issue(c, RAD_B_N(m.text_in, lane, 1),
                     bcol_at(m.prev_latent, lane, m.latent_dim, 0, m.latent_dim, 1),
                     bcol_at(m.audio_out, lane, m.frame_samples, 0, m.latent_dim, 1),
                     RAD_B_N(m.mask_one, lane, 1), RAD_B_N(m.mask_one, lane, 1),
                     brow_slice(m.lm.b_x, i, 1, m.lm.g.n_embd));
    }

    m.lm.step_layers(c, batch);

    for (int64_t i = 0; i < L; ++i) {
        const int32_t lane = d->lane_h[i];
        if (lane < 0 || lane >= m.max_lanes) continue;
        RAD_ISSUE_N(c, m.op_copy_logits, 1,
                    brow_slice(m.lm.b_logits, i, 1, m.n_vocab),
                    bcol_at(m.logits_out, lane, m.n_vocab, 0, m.n_vocab, 1));
        RAD_ISSUE_N(c, m.op_copy_pad, 1,
                    brow_slice(m.lm.b_pad_eop, i, 1, 2),
                    bcol_at(m.pad_eop_out, lane, 2, 0, 2, 1));
        m.head.issue(c, brow_slice(m.lm.b_h, i, 1, m.lm.g.n_embd),
                     praw(d->head_noise + i * m.latent_dim, RAD_F32, m.latent_dim),
                     RAD_B(m.guidance),
                     bcol_at(m.head_norm, lane, m.latent_dim, 0, m.latent_dim, 1));
        m.proj.unscale(c, bcol_at(m.head_norm, lane, m.latent_dim, 0, m.latent_dim, 1),
                       bcol_at(m.prev_latent, lane, m.latent_dim, 0, m.latent_dim, 1));
        copy_prev_to_decoder(c, m, lane);
        haru::codec_decode(c, m.codec, lane, false, m.audio_in, m.audio_out, -1);
    }
}

static void step(RadCtx* c, const RadBatch* batch) {
    if (!c || !batch) return;
    if (batch->gen == RAD_GEN_PASS_DUPLEX_PROMPT) {
        prompt_pass(c, batch);
    } else if (batch->gen == RAD_GEN_PASS_DUPLEX_STEP) {
        step_pass(c, batch);
    }
}

}  /* namespace haru_asm */

#ifndef RAD_ARCH_NO_EXPORTS
RAD_ARCH_PLUGIN(haru_asm, "haru", "", "0.0.1",
                "Haru full-duplex speech-to-speech: Qwen3.5 LM, projectors, MeanFlow head, codec")
RAD_ARCH_PROBE(haru_asm)
#endif
