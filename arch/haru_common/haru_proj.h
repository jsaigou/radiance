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

    rad_weight latent_scale = 0, latent_bias = 0;
    HaruAudioProjW model{}, user{};
    /* The LM's table: passed in at declare, not owned by this block. */
    rad_weight embed_tokens = 0;

    rad_op op_embed = 0;
    rad_op op_fill_d = 0, op_mulw_bias = 0, op_mulw_scale = 0, op_add_d = 0;
    rad_op op_gemm1_m = 0, op_gemm2_m = 0, op_norm_m = 0;
    rad_op op_gemm1_u = 0, op_gemm2_u = 0, op_norm_u = 0;
    rad_op op_mask = 0, op_add_h = 0;

    rad_buf b_x = 0, b_x2 = 0, b_y = 0, b_mask = 0;
    rad_buf b_fc1 = 0, b_norm = 0;
    rad_buf b_bias = 0, b_scaled = 0, b_scaled2 = 0, b_ones = 0;

    /* Declares the projectors' weights and name maps, the latent scale/bias, the workspace and
     * the ops. `tokens` is the LM's `lm.embed_tokens.weight`, which must already be declared. */
    int declare(RadBuilder* b, rad_weight tokens);

    /* Issues one position's embedding. `token` is an i32 view of [1]; `model_latent` and
     * `user_latent` are [1, latent]; `model_mask` and `user_mask` are [1] float views holding
     * 0 or 1; `out` is [1, hidden]. */
    void issue(RadCtx* c, RadOperand token, RadOperand model_latent, RadOperand user_latent,
               RadOperand model_mask, RadOperand user_mask, RadOperand out) const;
};

}  /* namespace arch */
}  /* namespace rad */

#endif /* HARU_PROJ_H */
