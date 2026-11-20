/* haru_head.cpp -- declare and issue the MeanFlow head (haru_head.h has the arithmetic).
 *
 * The reference stores every head linear without a bias (nn.Linear(..., bias=False)) and applies
 * two weight-free norms: HeadLayer takes the learned `norm.weight` gain, FinalLayer takes none.
 * The head serves the checkpoint's own dtype, read from the container: bf16 in the real model
 * (whose reference runs the head in bf16) and f32 in the tiny model. The norm gains stay f32
 * either way; the linears and the activation workspace are the serving dtype.
 */
#include "haru_head.h"

#include <cstdio>
#include <utility>

namespace rad {
namespace arch {

namespace {

/* The identity name map that finds a weight. The map entry must precede the weight: a weight
 * declared before its map is read as "absent from the container". */
int map_weight(RadBuilder* b, const char* name) { return map_copy(b, name, name); }

/* A weight over its already-mapped name, at the given dtype. */
int plain_weight(RadBuilder* b, const char* name, std::initializer_list<int64_t> shape,
                 uint32_t dtype, rad_weight* out) {
    *out = decl_w(b, name, dtype, shape, RAD_ACCESS_PER_TOKEN, RAD_SHARD_NONE, grp_model());
    return *out ? RAD_OK : RAD_E_INVAL;
}

#define HEAD_OP(field, expr)      \
    do {                          \
        (field) = (expr);         \
        if (!(field)) return RAD_E_NOKERNEL; \
    } while (0)

/* The activation workspace is the serving dtype; the scalars stay f32 (timestep_embed reads its
 * input t as f32 and the Newton scalars are one element). */
#define HEAD_BUF(field, name, n)                                                    \
    do {                                                                            \
        (field) = decl_b(b, name, wd, { (n) }, RAD_BUF_PERSIST);                    \
        if (!(field)) return RAD_E_INVAL;                                           \
    } while (0)

#define HEAD_BUF_F32(field, name, n)                                                \
    do {                                                                            \
        (field) = decl_b(b, name, RAD_F32, { (n) }, RAD_BUF_PERSIST);               \
        if (!(field)) return RAD_E_INVAL;                                           \
    } while (0)

}  /* namespace */

int HaruHead::declare(RadBuilder* b) {
    if (!b) return RAD_E_INVAL;
    const int64_t H = g.hidden;
    const int64_t D = g.latent;
    const int64_t F = g.ffn;
    if (H <= 0 || D <= 0 || F <= 0 || g.layers < 0) return RAD_E_INVAL;

    /* The serving dtype, from the checkpoint's own encoding (bf16 real / f32 tiny). The map must
     * precede the weight, so cond_proj's map comes first and every later weight is mapped right
     * before its declaration. */
    RAD_ARCH_TRY(map_weight(b, "head.cond_proj.weight"));
    wd = weight_codes_dtype(b, "head.cond_proj.weight", RAD_F32);
    if (wd != RAD_F32 && wd != RAD_BF16) {
        fprintf(stderr, "radiance: haru_head: cond_proj is %s; this block serves f32 or bf16 "
                        "weights\n", rad_dtype_name(wd));
        return RAD_E_UNSUPPORTED;
    }
    dt = rad_dtype_name(wd);

    /* Workspace. All persistent: the head runs once a lane-step and a transient plan would not
     * gain anything a few tens of KiB could not. */
    HEAD_BUF(b_cond, "head.cond", H);
    HEAD_BUF(b_c, "head.c", H);
    HEAD_BUF(b_emb, "head.emb", H);
    HEAD_BUF(b_emb2, "head.emb2", H);
    HEAD_BUF(b_x0, "head.x0", H);
    HEAD_BUF(b_x1, "head.x1", H);
    HEAD_BUF(b_hn, "head.hn", H);
    HEAD_BUF(b_t1, "head.t1", H);
    HEAD_BUF(b_t2, "head.t2", H);
    HEAD_BUF(b_ada, "head.ada", 3 * H);
    HEAD_BUF(b_fada, "head.fada", 2 * H);
    HEAD_BUF(b_g, "head.gate", F);
    HEAD_BUF(b_up, "head.up", F);
    HEAD_BUF(b_sg, "head.sg", F);
    HEAD_BUF(b_gu, "head.gu", F);
    HEAD_BUF(b_fn, "head.fn", H);
    HEAD_BUF(b_gated, "head.gated", H);
    HEAD_BUF(b_vel, "head.vel", D);
    HEAD_BUF(b_feat, "head.feat", kHaruScalarFreqs);
    HEAD_BUF_F32(b_scalar, "head.scalar", 1);
    HEAD_BUF_F32(b_one, "head.one", 1);
    HEAD_BUF_F32(b_two, "head.two", 1);
    HEAD_BUF_F32(b_minus1, "head.minus1", 1);
    HEAD_BUF_F32(b_r0, "head.r0", 1);
    HEAD_BUF_F32(b_r1, "head.r1", 1);
    HEAD_BUF_F32(b_u, "head.u", 1);
    HEAD_BUF_F32(b_v, "head.v", 1);
    HEAD_BUF_F32(b_s, "head.s", 1);
    HEAD_BUF(b_oneh, "head.oneh", H);
    HEAD_BUF(b_noise, "head.noise", D);
    HEAD_BUF(b_z, "head.z", D);

    /* Weights. Every weight -- linears and the rmsnorm gains -- is the checkpoint's own dtype, as
     * the reference runs the head. cond_proj's map already exists; each other weight is mapped
     * right before its declaration. */
    RAD_ARCH_TRY(plain_weight(b, "head.cond_proj.weight", { H, H }, wd, &cond_proj));
    RAD_ARCH_TRY(map_weight(b, "head.latent_proj.weight"));
    RAD_ARCH_TRY(plain_weight(b, "head.latent_proj.weight", { H, D }, wd, &latent_proj));
    RAD_ARCH_TRY(map_weight(b, "head.t_embed.fc1.weight"));
    RAD_ARCH_TRY(plain_weight(b, "head.t_embed.fc1.weight", { H, kHaruScalarFreqs }, wd, &t_fc1));
    RAD_ARCH_TRY(map_weight(b, "head.t_embed.fc2.weight"));
    RAD_ARCH_TRY(plain_weight(b, "head.t_embed.fc2.weight", { H, H }, wd, &t_fc2));
    RAD_ARCH_TRY(map_weight(b, "head.dt_embed.fc1.weight"));
    RAD_ARCH_TRY(plain_weight(b, "head.dt_embed.fc1.weight", { H, kHaruScalarFreqs }, wd, &dt_fc1));
    RAD_ARCH_TRY(map_weight(b, "head.dt_embed.fc2.weight"));
    RAD_ARCH_TRY(plain_weight(b, "head.dt_embed.fc2.weight", { H, H }, wd, &dt_fc2));
    RAD_ARCH_TRY(map_weight(b, "head.h_embed.fc1.weight"));
    RAD_ARCH_TRY(plain_weight(b, "head.h_embed.fc1.weight", { H, kHaruScalarFreqs }, wd, &h_fc1));
    RAD_ARCH_TRY(map_weight(b, "head.h_embed.fc2.weight"));
    RAD_ARCH_TRY(plain_weight(b, "head.h_embed.fc2.weight", { H, H }, wd, &h_fc2));
    RAD_ARCH_TRY(map_weight(b, "head.omega_embed.fc1.weight"));
    RAD_ARCH_TRY(plain_weight(b, "head.omega_embed.fc1.weight", { H, kHaruScalarFreqs }, wd, &omega_fc1));
    RAD_ARCH_TRY(map_weight(b, "head.omega_embed.fc2.weight"));
    RAD_ARCH_TRY(plain_weight(b, "head.omega_embed.fc2.weight", { H, H }, wd, &omega_fc2));

    lw.assign((size_t)g.layers, HaruHeadLayerW{});
    lop.assign((size_t)g.layers, HaruHeadLayerOp{});
    for (int64_t l = 0; l < g.layers; ++l) {
        HaruHeadLayerW& w = lw[(size_t)l];
        char n[128];
        std::snprintf(n, sizeof n, "head.layers.%lld.norm.weight", (long long)l);
        RAD_ARCH_TRY(map_weight(b, n));
        RAD_ARCH_TRY(plain_weight(b, n, { H }, wd, &w.norm));
        std::snprintf(n, sizeof n, "head.layers.%lld.ada.weight", (long long)l);
        RAD_ARCH_TRY(map_weight(b, n));
        RAD_ARCH_TRY(plain_weight(b, n, { 3 * H, H }, wd, &w.ada));
        std::snprintf(n, sizeof n, "head.layers.%lld.ffn.gate_proj.weight", (long long)l);
        RAD_ARCH_TRY(map_weight(b, n));
        RAD_ARCH_TRY(plain_weight(b, n, { F, H }, wd, &w.gate));
        std::snprintf(n, sizeof n, "head.layers.%lld.ffn.up_proj.weight", (long long)l);
        RAD_ARCH_TRY(map_weight(b, n));
        RAD_ARCH_TRY(plain_weight(b, n, { F, H }, wd, &w.up));
        std::snprintf(n, sizeof n, "head.layers.%lld.ffn.down_proj.weight", (long long)l);
        RAD_ARCH_TRY(map_weight(b, n));
        RAD_ARCH_TRY(plain_weight(b, n, { H, F }, wd, &w.down));
    }
    RAD_ARCH_TRY(map_weight(b, "head.final.ada.weight"));
    RAD_ARCH_TRY(plain_weight(b, "head.final.ada.weight", { 2 * H, H }, wd, &final_ada));
    RAD_ARCH_TRY(map_weight(b, "head.final.proj.weight"));
    RAD_ARCH_TRY(plain_weight(b, "head.final.proj.weight", { D, H }, wd, &final_proj));

    /* Linear layers. Every GEMM is one row: a lane-step. */
    auto gemm = [&](int64_t N, int64_t K, rad_weight w) {
        return RAD_OP(b, "gemm_nt",
                      RAD_PARAMS(RAD_RANGE("M", 1, 1), RAD_INT("N", N), RAD_INT("K", K),
                                 RAD_STR("dtype", dt)),
                      RAD_WEIGHTS(w));
    };
    auto unary = [&](const char* name, int64_t n) {
        return RAD_OP(b, name,
                      RAD_PARAMS(RAD_RANGE("M", 1, 1), RAD_INT("n", n), RAD_STR("dtype", dt)),
                      RAD_NOWEIGHTS);
    };
    auto binary = [&](const char* name, int64_t n) {
        return RAD_OP(b, name,
                      RAD_PARAMS(RAD_RANGE("M", 1, 1), RAD_INT("n", n), RAD_STR("dtype", dt)),
                      RAD_NOWEIGHTS);
    };
    auto fill = [&](int64_t n, double value) {
        return RAD_OP(b, "fill",
                      RAD_PARAMS(RAD_RANGE("M", 1, 1), RAD_INT("n", n), RAD_STR("dtype", dt),
                                 RAD_F64("value", value)),
                      RAD_NOWEIGHTS);
    };
    auto fill_f32 = [&](int64_t n, double value) {
        return RAD_OP(b, "fill",
                      RAD_PARAMS(RAD_RANGE("M", 1, 1), RAD_INT("n", n), RAD_STR("dtype", "f32"),
                                 RAD_F64("value", value)),
                      RAD_NOWEIGHTS);
    };
    auto scalar = [&](const char* name, int64_t n) {
        return RAD_OP(b, name,
                      RAD_PARAMS(RAD_RANGE("M", 1, 1), RAD_INT("n", n), RAD_STR("dtype", "f32")),
                      RAD_NOWEIGHTS);
    };

    HEAD_OP(op_cond, gemm(H, H, cond_proj));
    HEAD_OP(op_latent, gemm(H, D, latent_proj));
    HEAD_OP(op_t_fc1, gemm(H, kHaruScalarFreqs, t_fc1));
    HEAD_OP(op_t_fc2, gemm(H, H, t_fc2));
    HEAD_OP(op_dt_fc1, gemm(H, kHaruScalarFreqs, dt_fc1));
    HEAD_OP(op_dt_fc2, gemm(H, H, dt_fc2));
    HEAD_OP(op_h_fc1, gemm(H, kHaruScalarFreqs, h_fc1));
    HEAD_OP(op_h_fc2, gemm(H, H, h_fc2));
    HEAD_OP(op_omega_fc1, gemm(H, kHaruScalarFreqs, omega_fc1));
    HEAD_OP(op_omega_fc2, gemm(H, H, omega_fc2));

    for (int64_t l = 0; l < g.layers; ++l) {
        HaruHeadLayerOp& o = lop[(size_t)l];
        const HaruHeadLayerW& w = lw[(size_t)l];
        HEAD_OP(o.norm,
                RAD_OP(b, "rmsnorm",
                       RAD_PARAMS(RAD_RANGE("M", 1, 1), RAD_INT("n", H),
                                  RAD_F64("eps", g.norm_eps), RAD_STR("dtype", dt)),
                       RAD_WEIGHTS(w.norm)));
        HEAD_OP(o.ada, gemm(3 * H, H, w.ada));
        HEAD_OP(o.gate, gemm(F, H, w.gate));
        HEAD_OP(o.up, gemm(F, H, w.up));
        HEAD_OP(o.down, gemm(H, F, w.down));
    }
    HEAD_OP(op_final_ada, gemm(2 * H, H, final_ada));
    HEAD_OP(op_final_proj, gemm(D, H, final_proj));

    HEAD_OP(op_timestep,
            RAD_OP(b, "timestep_embed",
                   RAD_PARAMS(RAD_RANGE("M", 1, 1), RAD_INT("n", kHaruScalarFreqs),
                              RAD_F64("scale", 1.0), RAD_F64("max_period", 10000.0),
                              RAD_STR("dtype", dt)),
                   RAD_NOWEIGHTS));

    HEAD_OP(op_silu_h, unary("silu", H));
    HEAD_OP(op_silu_f, unary("silu", F));
    HEAD_OP(op_add_h, binary("add", H));
    HEAD_OP(op_mul_h, binary("mul", H));
    HEAD_OP(op_mul_f, binary("mul", F));
    HEAD_OP(op_add_lat, binary("add", D));
    HEAD_OP(op_add_1, scalar("add", 1));
    HEAD_OP(op_mul_1, scalar("mul", 1));
    HEAD_OP(op_rmsnorm_p,
            RAD_OP(b, "rmsnorm_p",
                   RAD_PARAMS(RAD_RANGE("M", 1, 1), RAD_INT("n", H),
                              RAD_F64("eps", g.norm_eps), RAD_STR("dtype", dt)),
                   RAD_NOWEIGHTS));

    /* The three checkpoint scalars and the Newton constants. `interval_embed` is the checkpoint's
     * own value, so a different release gets its scalar without a source change. */
    HEAD_OP(op_fill_zero, fill_f32(1, 0.0));
    HEAD_OP(op_fill_interval, fill_f32(1, g.interval_embed));
    HEAD_OP(op_fill_one, fill_f32(1, 1.0));
    HEAD_OP(op_fill_two, fill_f32(1, 2.0));
    HEAD_OP(op_fill_minus1, fill_f32(1, -1.0));
    HEAD_OP(op_fill_r0, fill_f32(1, 0.75));
    HEAD_OP(op_fill_one_h, fill(H, 1.0));

    /* The two boundary casts: the host's f32 noise into the serving dtype, and the normalized
     * latent back to f32 for the caller's unscale. In the f32 tiny model both are copies. */
    HEAD_OP(op_cast_noise,
            RAD_OP(b, "cast",
                   RAD_PARAMS(RAD_RANGE("M", 1, 1), RAD_INT("n", D),
                              RAD_STR("from", "f32"), RAD_STR("to", dt)),
                   RAD_NOWEIGHTS));
    HEAD_OP(op_cast_out,
            RAD_OP(b, "cast",
                   RAD_PARAMS(RAD_RANGE("M", 1, 1), RAD_INT("n", D),
                              RAD_STR("from", dt), RAD_STR("to", "f32")),
                   RAD_NOWEIGHTS));
    return RAD_OK;
}

void HaruHead::issue(RadCtx* c, RadOperand hidden, RadOperand noise, RadOperand guidance,
                     RadOperand out) const {
    if (!c) return;
    const int64_t H = g.hidden;

    /* Constants used by every scalar embedding and the layer arithmetic. */
    RAD_ISSUE_N(c, op_fill_one_h, 1, RAD_B(b_oneh));
    RAD_ISSUE_N(c, op_fill_two, 1, RAD_B(b_two));
    RAD_ISSUE_N(c, op_fill_minus1, 1, RAD_B(b_minus1));
    RAD_ISSUE_N(c, op_fill_one, 1, RAD_B(b_one));
    RAD_ISSUE_N(c, op_fill_r0, 1, RAD_B(b_r0));

    /* The noise enters f32; land it in the serving dtype once, for the latent projection and the
     * final velocity sum. */
    RAD_ISSUE_N(c, op_cast_noise, 1, noise, RAD_B(b_noise));

    RAD_ISSUE_N(c, op_cond, 1, hidden, RAD_W(cond_proj), RAD_B(b_cond));

    /* One ScalarEmbed. The features come from timestep_embed with scale 1 and max_period 10000,
     * which is exactly ScalarEmbed's [-ln(10000) i / 128] ladder; the order [cos | sin] matches
     * the reference's torch.cat([cos, sin]). */
    auto scalar_embed = [&](rad_op fill_op, rad_op fc1_op, rad_op fc2_op, rad_weight fc1_w,
                            rad_weight fc2_w, RadOperand before, rad_buf dst) {
        RAD_ISSUE_N(c, fill_op, 1, RAD_B(b_scalar));
        RAD_ISSUE_N(c, op_timestep, 1, RAD_B(b_scalar), RAD_B(b_feat));
        RAD_ISSUE_N(c, fc1_op, 1, RAD_B(b_feat), RAD_W(fc1_w), RAD_B(b_emb));
        RAD_ISSUE_N(c, op_silu_h, 1, RAD_B(b_emb), RAD_B(b_emb2));
        RAD_ISSUE_N(c, fc2_op, 1, RAD_B(b_emb2), RAD_W(fc2_w), RAD_B(b_emb));
        RAD_ISSUE_N(c, op_add_h, 1, before, RAD_B(b_emb), RAD_B(dst));
    };

    /* The four sums alternate between b_c and b_cond, so no add reads and writes one buffer. */
    scalar_embed(op_fill_zero, op_t_fc1, op_t_fc2, t_fc1, t_fc2, RAD_B(b_cond), b_c);
    scalar_embed(op_fill_interval, op_dt_fc1, op_dt_fc2, dt_fc1, dt_fc2, RAD_B(b_c), b_cond);

    /* h = 1.0, and the constant 1 the omega scalar needs. */
    scalar_embed(op_fill_one, op_h_fc1, op_h_fc2, h_fc1, h_fc2, RAD_B(b_cond), b_c);

    /* omega = 1 - 1/guidance. Five Newton steps from r0 = 0.75: for g in [1, 2] the relative
     * error after five steps is below 3e-10, an order of magnitude finer than the 1e-4 test
     * tolerance, and no host read of a device buffer is needed. */
    for (int i = 0; i < 5; ++i) {
        const rad_buf cur = (i & 1) ? b_r1 : b_r0;
        const rad_buf nxt = (i & 1) ? b_r0 : b_r1;
        RAD_ISSUE_N(c, op_mul_1, 1, guidance, RAD_B(cur), RAD_B(b_u));       /* g r */
        RAD_ISSUE_N(c, op_mul_1, 1, RAD_B(b_u), RAD_B(b_minus1), RAD_B(b_v));/* -g r */
        RAD_ISSUE_N(c, op_add_1, 1, RAD_B(b_two), RAD_B(b_v), RAD_B(b_s));   /* 2 - g r */
        RAD_ISSUE_N(c, op_mul_1, 1, RAD_B(cur), RAD_B(b_s), RAD_B(nxt));     /* r (2 - g r) */
    }
    /* Five iterations end in b_r1. */
    RAD_ISSUE_N(c, op_mul_1, 1, RAD_B(b_r1), RAD_B(b_minus1), RAD_B(b_u));
    RAD_ISSUE_N(c, op_add_1, 1, RAD_B(b_one), RAD_B(b_u), RAD_B(b_scalar));

    RAD_ISSUE_N(c, op_timestep, 1, RAD_B(b_scalar), RAD_B(b_feat));
    RAD_ISSUE_N(c, op_omega_fc1, 1, RAD_B(b_feat), RAD_W(omega_fc1), RAD_B(b_emb));
    RAD_ISSUE_N(c, op_silu_h, 1, RAD_B(b_emb), RAD_B(b_emb2));
    RAD_ISSUE_N(c, op_omega_fc2, 1, RAD_B(b_emb2), RAD_W(omega_fc2), RAD_B(b_emb));
    RAD_ISSUE_N(c, op_add_h, 1, RAD_B(b_c), RAD_B(b_emb), RAD_B(b_cond));

    /* c = silu(c); b_emb becomes the condition every layer and the final block read. */
    RAD_ISSUE_N(c, op_silu_h, 1, RAD_B(b_cond), RAD_B(b_emb));

    RAD_ISSUE_N(c, op_latent, 1, RAD_B(b_noise), RAD_W(latent_proj), RAD_B(b_x0));
    rad_buf cur = b_x0, nxt = b_x1;
    for (int64_t l = 0; l < g.layers; ++l) {
        const HaruHeadLayerW& w = lw[(size_t)l];
        const HaruHeadLayerOp& o = lop[(size_t)l];
        RAD_ISSUE_N(c, o.ada, 1, RAD_B(b_emb), RAD_W(w.ada), RAD_B(b_ada));
        RAD_ISSUE_N(c, o.norm, 1, RAD_B(cur), RAD_W(w.norm), RAD_B(b_hn));
        /* chunk order is (shift, scale, gate): h * (1 + scale) + shift */
        RAD_ISSUE_N(c, op_add_h, 1, RAD_B(b_oneh), RAD_B_N(b_ada, H, H), RAD_B(b_t1));
        RAD_ISSUE_N(c, op_mul_h, 1, RAD_B(b_hn), RAD_B(b_t1), RAD_B(b_t2));
        RAD_ISSUE_N(c, op_add_h, 1, RAD_B(b_t2), RAD_B_N(b_ada, 0, H), RAD_B(b_t1));
        RAD_ISSUE_N(c, o.gate, 1, RAD_B(b_t1), RAD_W(w.gate), RAD_B(b_g));
        RAD_ISSUE_N(c, o.up, 1, RAD_B(b_t1), RAD_W(w.up), RAD_B(b_up));
        RAD_ISSUE_N(c, op_silu_f, 1, RAD_B(b_g), RAD_B(b_sg));
        RAD_ISSUE_N(c, op_mul_f, 1, RAD_B(b_sg), RAD_B(b_up), RAD_B(b_gu));
        RAD_ISSUE_N(c, o.down, 1, RAD_B(b_gu), RAD_W(w.down), RAD_B(b_fn));
        RAD_ISSUE_N(c, op_mul_h, 1, RAD_B_N(b_ada, 2 * H, H), RAD_B(b_fn), RAD_B(b_gated));
        RAD_ISSUE_N(c, op_add_h, 1, RAD_B(cur), RAD_B(b_gated), RAD_B(nxt));
        std::swap(cur, nxt);
    }

    RAD_ISSUE_N(c, op_final_ada, 1, RAD_B(b_emb), RAD_W(final_ada), RAD_B(b_fada));
    RAD_ISSUE_N(c, op_rmsnorm_p, 1, RAD_B(cur), RAD_B(b_hn));
    /* chunk order is (shift, scale) */
    RAD_ISSUE_N(c, op_add_h, 1, RAD_B(b_oneh), RAD_B_N(b_fada, H, H), RAD_B(b_t1));
    RAD_ISSUE_N(c, op_mul_h, 1, RAD_B(b_hn), RAD_B(b_t1), RAD_B(b_t2));
    RAD_ISSUE_N(c, op_add_h, 1, RAD_B(b_t2), RAD_B_N(b_fada, 0, H), RAD_B(b_t1));
    RAD_ISSUE_N(c, op_final_proj, 1, RAD_B(b_t1), RAD_W(final_proj), RAD_B(b_vel));

    /* One MeanFlow step from t = 0 at h = 1: z = noise + velocity, then back to f32. */
    RAD_ISSUE_N(c, op_add_lat, 1, RAD_B(b_noise), RAD_B(b_vel), RAD_B(b_z));
    RAD_ISSUE_N(c, op_cast_out, 1, RAD_B(b_z), out);
}

#undef HEAD_OP
#undef HEAD_BUF
#undef HEAD_BUF_F32

}  /* namespace arch */
}  /* namespace rad */
