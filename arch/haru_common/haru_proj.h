/* haru_proj.h -- Haru's audio-input projectors, the latent scale/bias and the embedding sum.
 *
 * The reference (haru/models/haru.py, pinned commit 7fe6b76) sums three streams at every LM
 * position:
 *
 *   embed = embed_tokens[token]
 *         + model_audio_proj(scale(model_latent)) * model_mask
 *         + user_audio_proj (scale(user_latent))  * user_mask
 *   scale(raw) = (raw + latent_bias) * latent_scale
 *
 * An AudioProjector is fc1 (bias) -> RMSNorm (learnable gain, eps 1e-6) -> fc2 (bias). The token
 * embedding table is the LM's (`lm.embed_tokens.weight`), so it is not declared here: the plugin
 * that assembles the LM declares it once and passes the handle in, and this block only reads it.
 *
 * The projectors serve the checkpoint's own dtype (read from the container at declare): bf16 in
 * the real model, where the reference runs them in bf16, and f32 in the tiny test model. The
 * embedding sum and the two projector GEMMs are that dtype, and the result lands directly in the
 * LM's input row in the same dtype. `latent_scale`/`latent_bias` and the unscale path stay f32
 * (the Newton reciprocal needs f32 exactness), so the scaled latent is cast once before fc1.
 *
 * One call is one position: one token id, one row of each latent, one row of each mask.
 */
#ifndef HARU_PROJ_H
#define HARU_PROJ_H

#include <arch/rad_arch.h>

#include <cstdint>

namespace rad {
namespace arch {

struct HaruProjGeom {
    int64_t latent = 0;      /* the codec latent (64) */
    int64_t hidden = 0;      /* the LM embedding width */
    int64_t n_vocab = 0;     /* the token embedding's rows */
    double  norm_eps = 1e-6; /* AudioProjector's RMSNorm eps, hard-coded in the reference */
};

struct HaruAudioProjW {
    rad_weight fc1 = 0, fc1_b = 0;
    rad_weight norm = 0;
    rad_weight fc2 = 0, fc2_b = 0;
};

/* Everything the two projectors, the latent scalars and the embedding sum own. */
struct HaruProj {
    HaruProjGeom g;

    uint32_t    wd = RAD_F32;   /* the checkpoint's own dtype: the projector weights and rows */
    const char* dt = "f32";     /* the operand-combination string: "bf16" or "f32" */

    rad_weight latent_scale = 0, latent_bias = 0;
    HaruAudioProjW model{}, user{};
    /* The LM's table: passed in at declare, not owned by this block. */
    rad_weight embed_tokens = 0;

    rad_op op_embed = 0;
    rad_op op_fill_d = 0, op_mulw_bias = 0, op_mulw_scale = 0, op_add_d = 0;
    rad_op op_gemm1_m = 0, op_gemm2_m = 0, op_norm_m = 0;
    rad_op op_gemm1_u = 0, op_gemm2_u = 0, op_norm_u = 0;
    rad_op op_mask = 0, op_add_h = 0;
    rad_op op_cast_lat = 0;

    /* unscale: raw = z / latent_scale - latent_bias. The reciprocal is Newton's iteration on
     * the device (no host read of a weight). From r0 = 0.75 and kUnscaleSteps = 12 steps the
     * relative error is below f32 rounding for latent_scale in [0.05, 2.6]; it diverges at
     * 2.67 and above. The real checkpoint's value is 0.1963. A checkpoint outside the range
     * needs another step count; the value is not host-readable at declare, so this is
     * documented and covered by a test, not checked. */
    static constexpr int kUnscaleSteps = 12;
    rad_op op_mul_1 = 0, op_add_1 = 0, op_mulw_scale1 = 0;
    rad_op op_scale_row = 0, op_mul_d = 0;
    rad_op op_fill_r0 = 0, op_fill_two = 0, op_fill_minus1 = 0;
    rad_buf b_r0 = 0, b_r1 = 0, b_u = 0, b_v = 0, b_s = 0, b_two = 0, b_minus1 = 0;
    rad_buf b_bias_d = 0, b_neg_bias_d = 0, b_recip_d = 0, b_unscaled = 0;

    rad_buf b_x = 0, b_x2 = 0, b_y = 0, b_mask = 0;
    rad_buf b_fc1 = 0, b_norm = 0;
    rad_buf b_bias = 0, b_scaled = 0, b_scaled2 = 0, b_ones = 0;
    rad_buf b_scaled_bf16 = 0;

    /* Declares the projectors' weights and name maps, the latent scale/bias, the workspace and
     * the ops. `tokens` is the LM's `lm.embed_tokens.weight`, which must already be declared. */
    int declare(RadBuilder* b, rad_weight tokens);

    /* Issues one position's embedding in the serving dtype. `token` is an i32 view of [1];
     * `model_latent` and `user_latent` are [1, latent] f32; `model_mask` and `user_mask` are [1]
     * float views holding 0 or 1; `out` is [1, hidden] in the serving dtype. */
    void issue(RadCtx* c, RadOperand token, RadOperand model_latent, RadOperand user_latent,
               RadOperand model_mask, RadOperand user_mask, RadOperand out) const;

    /* normalized [1, latent] -> raw codec space [1, latent]: z / latent_scale - latent_bias. */
    void unscale(RadCtx* c, RadOperand normalized, RadOperand out) const;
};

}  /* namespace arch */
}  /* namespace rad */

#endif /* HARU_PROJ_H */
