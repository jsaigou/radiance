/* radtest_haru_codec.cpp -- the Haru codec block's test tenant.
 *
 * One duplex tenant (the engine keeps one declaration per program), so the operation is selected
 * through the mode channel tests/HARU_HARNESS.md describes: the plugin's template is exactly
 * "{card}", the PROMPT pass sees the prompt's row count in the host-readable RadDuplexBatch::n_tok,
 * and records it as mode[lane]. A test opens a lane with a card of m distinct ASCII letters for
 * mode m: 1 encode, 2 decode, 3 debug stages, 4 whole reference clip, 6 sampling, 8 mode probe.
 * Any other mode also reports its observed n_tok through the probe, so a tokenizer that does not
 * map m letters to m tokens fails the probe case with the observed number rather than a codec
 * mismatch.
 *
 * STEP runs the block on that one lane's frame. The codec's state is plugin-private and reset on
 * RAD_DUPLEX_F_FIRST (the lane index is reused after close, so that flag is the reset). The
 * minimal KV group and PROMPT attention are copied from tests/duplex_plugin/radtest_duplex.cpp:
 * the engine's duplex driver requires one full-attention group and a nonzero block size, which
 * comes from that group's first consumer even though this tenant does no language modelling.
 */
#include <arch/rad_arch.h>

#include <array>
#include <cstdio>
#include <cstring>
#include <string>

#include <haru_common/haru_codec.h>

namespace radtest_haru_codec {

using namespace rad::arch;

constexpr int64_t kMaxLanes = 2;
constexpr int64_t kSamples = 3200;
constexpr int64_t kMaxPrompt = 32;
constexpr int64_t kProbeMax = 31;
constexpr int64_t kClipFrames = 14;
constexpr int64_t kClipSamples = 44000;
constexpr int64_t kClipLastSamples = kClipSamples - (kClipFrames - 1) * kSamples;

enum {
    kModeEncode = 1,
    kModeDecode = 2,
    kModeDebug = 3,
    kModeClip = 4,
    kModeSample = 6,
    kModeProbe = 8
};

struct Model {
    haru::Codec codec;
    rad_buf audio_in = 0, audio_out = 0, status = 0;
    rad_buf q = 0, attn = 0;
    rad_kvgroup kv = 0;
    rad_op op_fill_q = 0, op_attn = 0, op_status_err = 0;
    std::array<rad_op, kProbeMax + 1> probe_fill{};
    int32_t mode[kMaxLanes] = {};
};

static Model g_model[MAX_RANKS];

static int probe(const RadModelMeta*, RadArchProbe* out) {
    if (!out) return RAD_E_INVAL;
    out->gen_duplex = 1;
    out->gen_rows = kMaxPrompt;
    out->gen_ctx = 13500;
    return RAD_OK;
}

static int declare(RadBuilder* b, const RadModelMeta* meta, const RadBuildCtx* ctx) {
    if (!b || !meta || !ctx || ctx->rank < 0 || ctx->rank >= MAX_RANKS) return RAD_E_INVAL;
    Model& m = g_model[ctx->rank];
    m = Model{};

    haru::CodecGeometry g{};
    std::string why;
    if (haru::codec_geometry_from_meta(meta, &g, &why) < 0) {
        rad_note(b, "haru codec: %s", why.c_str());
        return RAD_E_INVAL;
    }
    if (g.frame_samples != kSamples) {
        rad_note(b, "haru codec: frame_samples is %lld, this tenant takes %lld",
                 (long long)g.frame_samples, (long long)kSamples);
        return RAD_E_INVAL;
    }
    g.max_lanes = kMaxLanes;

    m.audio_in = decl_b(b, "haru.codec.audio_in", RAD_F32, { kMaxLanes, kSamples },
                        RAD_BUF_PERSIST);
    m.audio_out = decl_b(b, "haru.codec.audio_out", RAD_F32, { kMaxLanes, kSamples },
                         RAD_BUF_PERSIST);
    m.status = decl_b(b, "haru.codec.status", RAD_I32, { kMaxLanes }, RAD_BUF_PERSIST);
    if (!m.audio_in || !m.audio_out || !m.status) return RAD_E_INVAL;

    RAD_ARCH_TRY(haru::codec_declare(b, g, m.audio_in, m.audio_out, &m.codec));

    m.op_status_err = rw(b, RAD_OP(b, "fill",
                                   RAD_PARAMS(RAD_INT("M", 1), RAD_INT("n", 1),
                                              RAD_STR("dtype", "i32"), RAD_F64("value", -1.0)),
                                   RAD_NOWEIGHTS),
                         {}, { m.status });

    for (int v = 0; v <= (int)kProbeMax; ++v) {
        char name[64];
        std::snprintf(name, sizeof name, "haru.codec.probe.%d", v);
        m.probe_fill[(size_t)v] =
            rw(b, RAD_OP(b, "fill",
                         RAD_PARAMS(RAD_INT("M", 1), RAD_INT("n", 1), RAD_STR("dtype", "f32"),
                                    RAD_F64("value", (double)v)),
                         RAD_NOWEIGHTS),
               {}, { m.audio_out });
    }

    /* The one full-attention group the driver requires, and the PROMPT pass's consumer. */
    RadKVGroupDecl kd{};
    kd.kind = RAD_KV_FULL;
    kd.dtype = RAD_F32;
    kd.n_head_kv = 1;
    kd.head_dim = 1;
    m.kv = rad_decl_kv_group(b, "haru.codec.kv", &kd);
    if (!m.kv) return RAD_E_INVAL;
    m.q = decl_b(b, "haru.codec.q", RAD_F32, { kMaxPrompt, 1 }, RAD_BUF_PERSIST);
    m.attn = decl_b(b, "haru.codec.attn", RAD_F32, { kMaxPrompt, 1 }, RAD_BUF_PERSIST);
    if (!m.q || !m.attn) return RAD_E_INVAL;

    const RadParam attn_p[] = {
        RAD_RANGE("q_len", 1, kMaxPrompt), RAD_INT("head_dim", 1), RAD_INT("gqa", 1),
        RAD_INT("causal", 1), RAD_INT("window", 0), RAD_INT("n_head", 1),
        RAD_INT("block_size", 16), RAD_INT("max_seqs", ctx->max_seqs),
        RAD_INT("max_ctx", ctx->max_ctx), RAD_STR("q_dtype", "f32"),
        RAD_STR("kv_dtype", "f32"), RAD_F64("scale", 1.0),
    };
    m.op_attn = rad_decl_op(b, "attn_paged", attn_p,
                            (int)(sizeof(attn_p) / sizeof(attn_p[0])), nullptr, 0);
    if (!m.op_attn) return RAD_E_NOKERNEL;
    RAD_ARCH_TRY(rad_bind_layer_kv(b, 0, m.kv));

    const RadParam fill_q_p[] = {
        RAD_RANGE("M", 1, kMaxPrompt), RAD_INT("n", 1), RAD_STR("dtype", "f32"),
    };
    m.op_fill_q = rad_decl_op(b, "fill", fill_q_p,
                              (int)(sizeof(fill_q_p) / sizeof(fill_q_p[0])), nullptr, 0);
    if (!m.op_fill_q || !m.op_status_err) return RAD_E_NOKERNEL;
    for (rad_op op : m.probe_fill)
        if (!op) return RAD_E_NOKERNEL;

    RadDuplexDecl d{};
    d.name = "haru-codec";
    d.tmpl = "{card}";
    d.max_prompt = kMaxPrompt;
    d.max_ref_frames = 0;
    d.ref_rate = 24000;
    d.ref_hop = kSamples;
    d.max_lanes = kMaxLanes;
    d.sample_rate = 24000;
    d.channels = 1;
    d.samples_per_frame = kSamples;
    d.latent_dim = g.latent_dim;
    d.frame_rate_num = 15;
    d.frame_rate_den = 2;
    d.n_audio_in = 0;
    d.text = 0;
    d.audio_in = m.audio_in;
    d.audio_out = m.audio_out;
    d.text_out = 0;
    d.status = m.status;
    RAD_ARCH_TRY(rad_declare_duplex_tenant(b, &d));

    rad_note(b, "haru codec test tenant: %lld stages, base %lld, latent %lld, %lld lanes",
             (long long)haru::kCodecStages, (long long)g.base_channels, (long long)g.latent_dim,
             (long long)kMaxLanes);
    return RAD_OK;
}

static void step(RadCtx* c, const RadBatch* batch) {
    if (!c || !batch) return;
    Model& m = g_model[rad_rank(c)];

    if (batch->gen == RAD_GEN_PASS_DUPLEX_PROMPT) {
        const RadDuplexBatch* d = batch->dup_b;
        if (!d) return;
        const int64_t T = batch->n_tok;
        if (T <= 0 || T > kMaxPrompt) return;
        for (int64_t i = 0; i < d->n_lane; ++i) {
            const int32_t lane = d->lane_h[i];
            if (lane >= 0 && lane < kMaxLanes) m.mode[lane] = (int32_t)d->n_tok;
        }
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
        haru::codec_prepare(c, m.codec);
        return;
    }

    if (batch->gen != RAD_GEN_PASS_DUPLEX_STEP || !batch->dup_b) return;
    const RadDuplexBatch* d = batch->dup_b;
    for (int64_t i = 0; i < d->n_lane; ++i) {
        const int32_t lane = d->lane_h[i];
        if (lane < 0 || lane >= kMaxLanes) continue;
        const bool first = (d->flags_h[i] & RAD_DUPLEX_F_FIRST) != 0;
        const int32_t mode = m.mode[lane];
        const int64_t frame = d->frame_h[i];

        switch (mode) {
        case kModeEncode:
            haru::codec_encode(c, m.codec, lane, first, m.audio_in, m.audio_out, kSamples, -1);
            break;
        case kModeDecode:
            haru::codec_decode(c, m.codec, lane, first, m.audio_in, m.audio_out, -1);
            break;
        case kModeDebug:
            if (frame >= 0 && frame < haru::kCodecStages)
                haru::codec_encode(c, m.codec, lane, true, m.audio_in, m.audio_out, kSamples,
                                   frame);
            else if (frame >= haru::kCodecStages && frame < 2 * haru::kCodecStages)
                haru::codec_decode(c, m.codec, lane, true, m.audio_in, m.audio_out,
                                   frame - haru::kCodecStages);
            else
                RAD_ISSUE(c, m.op_status_err, bcol_at(m.status, lane, 1, 0, 1, 1));
            break;
        case kModeClip:
            /* The golden clip is 44,000 samples: 13 whole frames and a 2,400-sample last one.
             * The harness pads the short step to 3200, so the true length is the test's
             * convention, not inferred from a zero tail. */
            haru::codec_encode(c, m.codec, lane, first, m.audio_in, m.audio_out,
                               frame == kClipFrames - 1 ? kClipLastSamples : kSamples, -1);
            break;
        case kModeSample:
            haru::codec_sample(c, m.codec, lane, m.audio_in, m.audio_out);
            break;
        default: {
            /* Includes kModeProbe, and any token count the tokenizer produced that is not one of
             * the modes above: echo the observed n_tok so the test can name the mismatch. */
            int v = (int)mode;
            if (v < 0) v = 0;
            if (v > (int)kProbeMax) v = (int)kProbeMax;
            RAD_ISSUE(c, m.probe_fill[(size_t)v], bcol_at(m.audio_out, lane, kSamples, 0, 1, 1));
            RAD_ISSUE(c, m.probe_fill[(size_t)v], bcol_at(m.audio_out, lane, kSamples, 1, 1, 1));
            break;
        }
        }
    }
}

}  /* namespace radtest_haru_codec */

#ifndef RAD_ARCH_NO_EXPORTS
RAD_ARCH_PLUGIN(radtest_haru_codec, "haru", "", "0.0.1",
                "test-only duplex tenant for the Haru codec block")
RAD_ARCH_PROBE(radtest_haru_codec)
#endif
