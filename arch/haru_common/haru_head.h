/* haru_head.h -- Haru's MeanFlow head, declared and issued as a block another plugin can use.
 *
 * The reference (haru/models/meanflow_head.py, pinned commit 7fe6b76) turns one LM hidden state
 * and one noise vector into the next normalized audio latent in a single MeanFlow step:
 *
 *   cond = cond_proj(hidden)
 *   c    = silu(cond + t_embed(0) + dt_embed(interval_embed) + h_embed(1) + omega_embed(1 - 1/guidance))
 *   x    = latent_proj(noise)
 *   for each layer: x += gate * ffn(rmsnorm(x) * norm.weight * (1 + scale) + shift), (shift,scale,gate) = ada(c)
 *   velocity = final_proj(rmsnorm(x) * (1 + scale) + shift), (shift,scale) = final.ada(c)
 *   z    = noise + velocity
 *
 * The four scalar embeddings are ScalarEmbed: the [cos | sin] sinusoidal features of the scalar
 * through a two-layer MLP. The first three scalars are constants of the checkpoint; omega depends
 * on the runtime guidance, so it is computed on the device. We build 1 - 1/guidance with a
 * Newton iteration (five steps from 0.75, which converges to f32 for guidance in [1, 2]) instead
 * of baking a constant; guidance is a [1, 2] runtime input (docs/DUPLEX-PLUGIN-ABI.md).
 *
 * One call is one lane-step: one row of `hidden`, `noise` and `guidance`, one row of output. The
 * caller owns the buffers and passes buffer views, so this header holds no global state.
 */
#ifndef HARU_HEAD_H
#define HARU_HEAD_H

#include <arch/rad_arch.h>

#include <cstdint>
#include <vector>

namespace rad {
namespace arch {

/* The sinusoidal feature width is the reference's ScalarEmbed._FREQUENCY_DIM, not a config key. */
constexpr int64_t kHaruScalarFreqs = 256;

struct HaruHeadGeom {
    int64_t hidden = 0;          /* LM hidden and head hidden */
    int64_t latent = 0;          /* audio latent */
    int64_t ffn = 0;             /* int(hidden * ffn_ratio) */
    int64_t layers = 0;          /* num_layers */
    double  norm_eps = 1e-5;     /* head.norm_eps */
    double  interval_embed = 7.0;/* head.interval_embed */
};

struct HaruHeadLayerW {
    rad_weight norm = 0;   /* head.layers.i.norm.weight */
    rad_weight ada = 0;    /* head.layers.i.ada.weight */
    rad_weight gate = 0;   /* head.layers.i.ffn.gate_proj.weight */
    rad_weight up = 0;     /* head.layers.i.ffn.up_proj.weight */
    rad_weight down = 0;   /* head.layers.i.ffn.down_proj.weight */
};

struct HaruHeadLayerOp {
    rad_op norm = 0;
    rad_op ada = 0;
    rad_op gate = 0;
    rad_op up = 0;
    rad_op down = 0;
};

/* Everything one head owns: the geometry it was declared at, its weights, the ops over them and
 * the workspace they write. A plugin keeps one of these per rank (declare runs once a rank). */
struct HaruHead {
    HaruHeadGeom g;

    rad_weight cond_proj = 0, latent_proj = 0;
    rad_weight t_fc1 = 0, t_fc2 = 0;
    rad_weight dt_fc1 = 0, dt_fc2 = 0;
    rad_weight h_fc1 = 0, h_fc2 = 0;
    rad_weight omega_fc1 = 0, omega_fc2 = 0;
    std::vector<HaruHeadLayerW>  lw;
    rad_weight final_ada = 0, final_proj = 0;

    rad_op op_cond = 0, op_latent = 0;
    rad_op op_t_fc1 = 0, op_t_fc2 = 0;
    rad_op op_dt_fc1 = 0, op_dt_fc2 = 0;
    rad_op op_h_fc1 = 0, op_h_fc2 = 0;
    rad_op op_omega_fc1 = 0, op_omega_fc2 = 0;
    std::vector<HaruHeadLayerOp> lop;
    rad_op op_final_ada = 0, op_final_proj = 0;

    rad_op op_timestep = 0;
    rad_op op_silu_h = 0, op_silu_f = 0;
    rad_op op_add_h = 0, op_mul_h = 0, op_mul_f = 0, op_add_lat = 0;
    rad_op op_add_1 = 0, op_mul_1 = 0;
    rad_op op_rmsnorm_p = 0;
    rad_op op_fill_zero = 0, op_fill_interval = 0, op_fill_one = 0;
    rad_op op_fill_two = 0, op_fill_minus1 = 0, op_fill_r0 = 0;
    rad_op op_fill_one_h = 0;

    rad_buf b_cond = 0, b_c = 0, b_emb = 0, b_emb2 = 0;
    rad_buf b_x0 = 0, b_x1 = 0;
    rad_buf b_hn = 0, b_t1 = 0, b_t2 = 0;
    rad_buf b_ada = 0, b_fada = 0;
    rad_buf b_g = 0, b_up = 0, b_sg = 0, b_gu = 0, b_fn = 0, b_gated = 0;
    rad_buf b_vel = 0, b_feat = 0;
    rad_buf b_scalar = 0, b_one = 0, b_two = 0, b_minus1 = 0;
    rad_buf b_r0 = 0, b_r1 = 0, b_u = 0, b_v = 0, b_s = 0;
    rad_buf b_oneh = 0;

    /* Declares the head's weights (and their checkpoint name maps, each entry before its weight),
     * its ops and its workspace. Returns a negative RAD_E_* on the first refusal. */
    int declare(RadBuilder* b);

    /* Issues one head step. `hidden` is [1, hidden], `noise` [1, latent], `guidance` [1] and
     * `out` [1, latent]; all are buffer views the caller built with RAD_B_N. */
    void issue(RadCtx* c, RadOperand hidden, RadOperand noise, RadOperand guidance,
               RadOperand out) const;
};

}  /* namespace arch */
}  /* namespace rad */

#endif /* HARU_HEAD_H */
