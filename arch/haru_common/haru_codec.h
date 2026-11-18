/* haru_codec.h -- the VibeVoice causal codec as one streaming Radiance block.
 *
 * The reference is haru/models/vibevoice_codec.py (pinned 7fe6b76): seven resolution stages of
 * causal convolution, a depthwise causal mixer with a scaled residual and a two-linear GELU FFN
 * per Block, and a transposed convolution back to 3200 samples. The block keeps every layer's left
 * context in a plugin-private per-lane arena and runs one frame at a time; docs/HARU-H1-DESIGN.md
 * records the exact layer/state/op mapping and the two compositions this file implements.
 *
 * No convolution op takes persistent history, so each conv copies its `context` input rows in
 * front of the new frame and reads the valid tail of the result. For a strided conv the buffer is
 * padded to the stage's full-frame row count with zeros and only the first ceil(row/stride)
 * valid outputs are used; the reference's last partial hop is exactly those valid outputs (an
 * appendix check against the reference is in the design note). A transposed conv stores one
 * previous input row and runs the two-frame window the Triton backend uses.
 *
 * Launch count, for the device budget: a full 3200-sample frame is 8 non-block convs and 26
 * Blocks in each direction. Each conv is 4 launches (state in, input, conv, state out); each Block
 * is 13 (two norms, the depthwise conv, two scaled-residual adds, two linears, the exact GELU and
 * four copies), so an encode plus a decode is ~740 launches, plus the once-a-lane 68 state fills
 * and the once-a-program 68 `iota` initialisations of the fixed cu_seqlens. That is the cost of
 * the stateless-conv composition and must be measured on the target.
 */
#pragma once
#include <arch/rad_arch.h>

#include <array>
#include <cstdint>
#include <string>

namespace haru {

enum {
    kCodecStages = 7,   /* one more than the ratio count */
    kCodecRatios = 6,
    kCodecMaxDepth = 8, /* the real encoder's deepest stage */
    kCodecGroup = 100   /* weight group, beside fishomni's codec group */
};

/* The config's codec geometry. The tiny test is base 1 / depth 1; the real model is base 32 with
 * depths (3,3,3,3,3,3,8) and (8,3,3,3,3,3,3). */
struct CodecGeometry {
    int64_t latent_dim = 64;
    int64_t base_channels = 1;
    std::array<int64_t, kCodecRatios> ratios{ 8, 5, 5, 4, 2, 2 };
    std::array<int64_t, kCodecStages> encoder_depths{ 1, 1, 1, 1, 1, 1, 1 };
    std::array<int64_t, kCodecStages> decoder_depths{ 1, 1, 1, 1, 1, 1, 1 };
    double norm_eps = 1e-5;
    double latent_std = 0.5;
    int64_t frame_samples = 3200;
    int64_t max_lanes = 2;
    uint32_t dtype = RAD_F32;
};

/* Reads codec.*, frame_samples and dtype from the container header. `why` is set on refusal. */
int codec_geometry_from_meta(const RadModelMeta* m, CodecGeometry* g, std::string* why);

/* One CausalConv1d (regular) or CausalConvTranspose1d. `context` is the number of input rows kept
 * between calls (kernel - stride for a conv, 1 for a transposed conv). `cu_rows` is the fixed
 * number of rows the op is issued over: context + max_in for a conv, 1 + max_in for a transposed
 * one. `drop` is the first valid output row and max_out the most valid outputs a full frame makes. */
struct ConvLayer {
    rad_weight w = 0, b = 0;
    rad_op op = 0;
    rad_buf combined = 0, out = 0, state = 0, cu = 0;
    rad_op fill_state = 0, copy_state_in = 0, copy_input = 0, fill_pad = 0,
           copy_state_out = 0, iota_cu = 0;
    int64_t cin = 0, cout = 0, k = 0, stride = 1, context = 0;
    int64_t cu_rows = 0, drop = 0, max_in = 0, max_out = 0;
    bool transposed = false;
};

/* One Block: RMSNorm -> depthwise causal conv -> scaled residual -> RMSNorm -> 4x GELU FFN ->
 * scaled residual. `state` holds the last six rows of the conv's input (the normed x). */
struct BlockLayer {
    rad_weight norm = 0, conv_w = 0, conv_b = 0, gamma = 0;
    rad_weight ffn_norm = 0, fc1_w = 0, fc1_b = 0, fc2_w = 0, fc2_b = 0, ffn_gamma = 0;
    rad_op rms1 = 0, dw = 0, scale1 = 0, add1 = 0, rms2 = 0, fc1 = 0, gelu = 0, fc2 = 0,
           scale2 = 0, add2 = 0;
    rad_buf combined = 0, dw_out = 0, state = 0, cu = 0;
    rad_buf n1 = 0, scaled1 = 0, x1 = 0, n2 = 0, h1 = 0, h2 = 0, h3 = 0, scaled2 = 0, x2 = 0;
    rad_op fill_state = 0, copy_state_in = 0, copy_input = 0, fill_pad = 0,
           copy_state_out = 0, iota_cu = 0;
    int64_t dim = 0, max_rows = 0, context = 6, cu_rows = 0;
};

/* One direction. `trans[i]` is down[i] for the encoder and up[i] for the decoder; `blocks[i][j]`
 * are stage i's Blocks. `debug_copy[i]` is the flat copy of stage i's last-block output into the
 * driver's audio_out (the codec_dbg test; the tiny geometry fits 3200, wider geometries must not
 * call the capture path). */
struct Stack {
    ConvLayer stem;
    std::array<ConvLayer, kCodecRatios> trans;
    std::array<std::array<BlockLayer, kCodecMaxDepth>, kCodecStages> blocks;
    std::array<int64_t, kCodecStages> depths{};
    ConvLayer head;
    std::array<rad_op, kCodecStages> debug_copy{};
    rad_op head_out = 0;
    bool encoder = true;
};

/* Everything the block declares. The plugin owns one Codec per rank. */
struct Codec {
    CodecGeometry g;
    Stack enc, dec;
    rad_buf in_audio = 0, in_latent = 0, out_latent = 0, out_audio = 0;
    rad_buf sample_mean = 0, sample_noise = 0;
    rad_op copy_frame_in = 0, copy_frame_out = 0, copy_latent_in = 0, copy_latent_out = 0;
    rad_op copy_sample_mean = 0, copy_sample_noise = 0, sample_add = 0;
};

/* Declares the weights (name maps before the weights), buffers and ops. `audio_in`/`audio_out`
 * are the duplex declaration's f32 buffers; the block's internal audio/latent buffers use the
 * codec dtype and the frame/latent copies cast between them. */
int codec_declare(RadBuilder* b, const CodecGeometry& g, rad_buf audio_in, rad_buf audio_out,
                  Codec* out);

/* Writes the fixed cu_seqlens [0, rows] into every layer's cu buffer. Call once per PROMPT pass
 * (the values are constant; repeating it is harmless). */
int codec_prepare(RadCtx* c, const Codec& k);

/* Zeros one lane's whole state arena (both stacks). The assembled duplex tenant calls this in its
 * PROMPT pass: once before encoding the reference frames and once before decoding model frame 0,
 * so the live encoder/decoder starts from zero exactly as the reference's streaming codec does.
 * Do NOT call it on RAD_DUPLEX_F_FIRST: that would erase the decoder state frame 0 produced before
 * the first live STEP. A lane index is reused after close, so the PROMPT reset is the reset. */
int codec_reset_lane(RadCtx* c, const Codec& k, int64_t lane);

/* Streaming encode: copies the lane's first `n_samples` (<= frame_samples) from audio_in and runs
 * the encoder with that lane's state. The 64 latent means go to the first latent_dim floats of
 * audio_out[lane]. `capture_stage` >= 0 runs a fresh-state encode and writes that stage's last-block
 * output (flattened [rows, channels]) to audio_out instead; the caller must have checked it fits. */
int codec_encode(RadCtx* c, const Codec& k, int64_t lane, bool first, rad_buf audio_in,
                 rad_buf audio_out, int64_t n_samples, int64_t capture_stage);

/* Streaming decode: reads the first latent_dim floats of audio_in[lane] as one latent row and
 * writes 3200 raw samples to audio_out[lane]. `capture_stage` as in codec_encode. */
int codec_decode(RadCtx* c, const Codec& k, int64_t lane, bool first, rad_buf audio_in,
                 rad_buf audio_out, int64_t capture_stage);

/* mean + noise, both [1, latent_dim] taken from audio_in[lane] (mean at 0, noise at latent_dim),
 * into the first latent_dim floats of audio_out[lane]. The caller forms the reference's noise:
 * scale * (latent_std / 0.8) * raw. No RNG lives here. */
int codec_sample(RadCtx* c, const Codec& k, int64_t lane, rad_buf audio_in, rad_buf audio_out);

/* The reference's combined noise factor (vibevoice_codec.py:196-200), for the caller that draws
 * the raw noise. */
inline float codec_noise_factor(float scale, double latent_std) {
    return scale * (float)(latent_std / 0.8);
}

}  /* namespace haru */
