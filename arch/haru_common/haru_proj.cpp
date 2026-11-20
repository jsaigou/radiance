/* haru_proj.cpp -- declare and issue the audio-input projectors (haru_proj.h has the arithmetic).
 *
 * The projectors serve the checkpoint's own dtype (bf16 in the real model, f32 in the tiny one);
 * `latent_scale` and `latent_bias` stay f32 (they are buffers, not parameters, and the unscale
 * Newton reciprocal needs f32 exactness), so they load as ordinary f32 weights and the scaled
 * latent is cast once before fc1.
 */
#include "haru_proj.h"

#include <cstdio>
#include <utility>

namespace rad {
namespace arch {

namespace {

/* The identity name map that finds a weight; the entry must precede the weight. */
int map_weight(RadBuilder* b, const char* name) { return map_copy(b, name, name); }

/* A weight over its already-mapped name, at the given dtype. */
int plain_weight(RadBuilder* b, const char* name, std::initializer_list<int64_t> shape,
                 uint32_t dtype, rad_weight* out) {
    *out = decl_w(b, name, dtype, shape, RAD_ACCESS_PER_TOKEN, RAD_SHARD_NONE, grp_model());
    return *out ? RAD_OK : RAD_E_INVAL;
}

#define PROJ_OP(field, expr)       \
    do {                           \
        (field) = (expr);          \
        if (!(field)) return RAD_E_NOKERNEL; \
    } while (0)

/* The embedding rows are the serving dtype; the latent scalars and the unscale workspace are f32. */
#define PROJ_BUF(field, name, n)                                                   \
    do {                                                                           \
        (field) = decl_b(b, name, wd, { (n) }, RAD_BUF_PERSIST);                   \
        if (!(field)) return RAD_E_INVAL;                                          \
    } while (0)

#define PROJ_BUF_F32(field, name, n)                                               \
    do {                                                                           \
        (field) = decl_b(b, name, RAD_F32, { (n) }, RAD_BUF_PERSIST);              \
        if (!(field)) return RAD_E_INVAL;                                          \
    } while (0)

}  /* namespace */

int HaruProj::declare(RadBuilder* b, rad_weight tokens) {
    if (!b || !tokens) return RAD_E_INVAL;
    const int64_t H = g.hidden;
    const int64_t D = g.latent;
    const int64_t V = g.n_vocab;
    if (H <= 0 || D <= 0 || V <= 0) return RAD_E_INVAL;
    embed_tokens = tokens;

    /* The serving dtype, from the checkpoint's own encoding (bf16 real / f32 tiny). latent_scale
     * stays f32 itself, but the container holds every tensor at the checkpoint's dtype, so it is
     * a reliable probe. */
    RAD_ARCH_TRY(map_weight(b, "latent_scale"));
    wd = weight_codes_dtype(b, "latent_scale", RAD_F32);
    if (wd != RAD_F32 && wd != RAD_BF16) {
        fprintf(stderr, "radiance: haru_proj: latent_scale is %s; this block serves f32 or bf16 "
                        "weights\n", rad_dtype_name(wd));
        return RAD_E_UNSUPPORTED;
    }
    dt = rad_dtype_name(wd);

    PROJ_BUF(b_x, "proj.x", H);
    PROJ_BUF(b_x2, "proj.x2", H);
    PROJ_BUF(b_y, "proj.y", H);
    PROJ_BUF(b_mask, "proj.mask", H);
    PROJ_BUF(b_fc1, "proj.fc1", H);
    PROJ_BUF(b_norm, "proj.norm", H);
    PROJ_BUF_F32(b_bias, "proj.bias", D);
    PROJ_BUF_F32(b_scaled, "proj.scaled", D);
    PROJ_BUF_F32(b_scaled2, "proj.scaled2", D);
    PROJ_BUF_F32(b_ones, "proj.ones", D);
    PROJ_BUF_F32(b_r0, "proj.r0", 1);
    PROJ_BUF_F32(b_r1, "proj.r1", 1);
    PROJ_BUF_F32(b_u, "proj.u", 1);
    PROJ_BUF_F32(b_v, "proj.v", 1);
    PROJ_BUF_F32(b_s, "proj.s", 1);
    PROJ_BUF_F32(b_two, "proj.two", 1);
    PROJ_BUF_F32(b_minus1, "proj.minus1", 1);
    PROJ_BUF_F32(b_bias_d, "proj.bias_d", D);
    PROJ_BUF_F32(b_neg_bias_d, "proj.neg_bias_d", D);
    PROJ_BUF_F32(b_recip_d, "proj.recip_d", D);
    PROJ_BUF_F32(b_unscaled, "proj.unscaled", D);
    PROJ_BUF(b_scaled_bf16, "proj.scaled_bf16", D);

    /* latent_scale is already mapped; declare it f32, then latent_bias. */
    RAD_ARCH_TRY(plain_weight(b, "latent_scale", { 1 }, RAD_F32, &latent_scale));
    RAD_ARCH_TRY(map_weight(b, "latent_bias"));
    RAD_ARCH_TRY(plain_weight(b, "latent_bias", { 1 }, RAD_F32, &latent_bias));

    auto audio = [&](const char* prefix, HaruAudioProjW* w) -> int {
        char n[128];
        std::snprintf(n, sizeof n, "%s.fc1.weight", prefix);
        RAD_ARCH_TRY(map_weight(b, n));
        RAD_ARCH_TRY(plain_weight(b, n, { H, D }, wd, &w->fc1));
        std::snprintf(n, sizeof n, "%s.fc1.bias", prefix);
        RAD_ARCH_TRY(map_weight(b, n));
        RAD_ARCH_TRY(plain_weight(b, n, { H }, wd, &w->fc1_b));
        std::snprintf(n, sizeof n, "%s.norm.weight", prefix);
        RAD_ARCH_TRY(map_weight(b, n));
        RAD_ARCH_TRY(plain_weight(b, n, { H }, wd, &w->norm));
        std::snprintf(n, sizeof n, "%s.fc2.weight", prefix);
        RAD_ARCH_TRY(map_weight(b, n));
        RAD_ARCH_TRY(plain_weight(b, n, { H, H }, wd, &w->fc2));
        std::snprintf(n, sizeof n, "%s.fc2.bias", prefix);
        RAD_ARCH_TRY(map_weight(b, n));
        RAD_ARCH_TRY(plain_weight(b, n, { H }, wd, &w->fc2_b));
        return RAD_OK;
    };
    RAD_ARCH_TRY(audio("model_audio_proj", &model));
    RAD_ARCH_TRY(audio("user_audio_proj", &user));

    auto gemm_bias = [&](int64_t N, int64_t K, rad_weight w, rad_weight bias) {
        return RAD_OP(b, "gemm_nt_bias",
                      RAD_PARAMS(RAD_RANGE("M", 1, 1), RAD_INT("N", N), RAD_INT("K", K),
                                 RAD_STR("dtype", dt)),
                      RAD_WEIGHTS(w, bias));
    };
    auto norm = [&](rad_weight w) {
        return RAD_OP(b, "rmsnorm",
                      RAD_PARAMS(RAD_RANGE("M", 1, 1), RAD_INT("n", H),
                                 RAD_F64("eps", g.norm_eps), RAD_STR("dtype", dt)),
                      RAD_WEIGHTS(w));
    };

    PROJ_OP(op_embed,
            RAD_OP(b, "embed_lookup",
                   RAD_PARAMS(RAD_RANGE("M", 1, 1), RAD_INT("n_embd", H), RAD_INT("n_vocab", V),
                              RAD_STR("dtype", dt), RAD_INT("vocab_offset", 0),
                              RAD_F64("mult", 1.0)),
                   RAD_WEIGHTS(embed_tokens)));
    PROJ_OP(op_fill_d,
            RAD_OP(b, "fill",
                   RAD_PARAMS(RAD_RANGE("M", 1, 1), RAD_INT("n", D), RAD_STR("dtype", "f32"),
                              RAD_F64("value", 1.0)),
                   RAD_NOWEIGHTS));
    /* One handle per scalar: mul_weight's one weight operand is required at declare. */
    PROJ_OP(op_mulw_bias,
            RAD_OP(b, "mul_weight",
                   RAD_PARAMS(RAD_RANGE("M", 1, 1), RAD_INT("n", D), RAD_STR("dtype", "f32"),
                              RAD_INT("scalar", 1)),
                   RAD_WEIGHTS(latent_bias)));
    PROJ_OP(op_mulw_scale,
            RAD_OP(b, "mul_weight",
                   RAD_PARAMS(RAD_RANGE("M", 1, 1), RAD_INT("n", D), RAD_STR("dtype", "f32"),
                              RAD_INT("scalar", 1)),
                   RAD_WEIGHTS(latent_scale)));
    PROJ_OP(op_add_d,
            RAD_OP(b, "add",
                   RAD_PARAMS(RAD_RANGE("M", 1, 1), RAD_INT("n", D), RAD_STR("dtype", "f32")),
                   RAD_NOWEIGHTS));

    PROJ_OP(op_gemm1_m, gemm_bias(H, D, model.fc1, model.fc1_b));
    PROJ_OP(op_gemm2_m, gemm_bias(H, H, model.fc2, model.fc2_b));
    PROJ_OP(op_norm_m, norm(model.norm));
    PROJ_OP(op_gemm1_u, gemm_bias(H, D, user.fc1, user.fc1_b));
    PROJ_OP(op_gemm2_u, gemm_bias(H, H, user.fc2, user.fc2_b));
    PROJ_OP(op_norm_u, norm(user.norm));

    PROJ_OP(op_mask,
            RAD_OP(b, "scale_rows",
                   RAD_PARAMS(RAD_RANGE("M", 1, 1), RAD_INT("n", H), RAD_STR("act", "none"),
                              RAD_STR("dtype", dt)),
                   RAD_NOWEIGHTS));
    PROJ_OP(op_add_h,
            RAD_OP(b, "add",
                   RAD_PARAMS(RAD_RANGE("M", 1, 1), RAD_INT("n", H), RAD_STR("dtype", dt)),
                   RAD_NOWEIGHTS));
    PROJ_OP(op_cast_lat,
            RAD_OP(b, "cast",
                   RAD_PARAMS(RAD_RANGE("M", 1, 1), RAD_INT("n", D),
                              RAD_STR("from", "f32"), RAD_STR("to", dt)),
                   RAD_NOWEIGHTS));

    PROJ_OP(op_mul_1,
            RAD_OP(b, "mul",
                   RAD_PARAMS(RAD_RANGE("M", 1, 1), RAD_INT("n", 1), RAD_STR("dtype", "f32")),
                   RAD_NOWEIGHTS));
    PROJ_OP(op_add_1,
            RAD_OP(b, "add",
                   RAD_PARAMS(RAD_RANGE("M", 1, 1), RAD_INT("n", 1), RAD_STR("dtype", "f32")),
                   RAD_NOWEIGHTS));
    PROJ_OP(op_mulw_scale1,
            RAD_OP(b, "mul_weight",
                   RAD_PARAMS(RAD_RANGE("M", 1, 1), RAD_INT("n", 1), RAD_STR("dtype", "f32"),
                              RAD_INT("scalar", 1)),
                   RAD_WEIGHTS(latent_scale)));
    PROJ_OP(op_scale_row,
            RAD_OP(b, "scale_rows",
                   RAD_PARAMS(RAD_RANGE("M", 1, 1), RAD_INT("n", D), RAD_STR("act", "none"),
                              RAD_STR("dtype", "f32")),
                   RAD_NOWEIGHTS));
    PROJ_OP(op_mul_d,
            RAD_OP(b, "mul",
                   RAD_PARAMS(RAD_RANGE("M", 1, 1), RAD_INT("n", D), RAD_STR("dtype", "f32")),
                   RAD_NOWEIGHTS));
    PROJ_OP(op_fill_r0,
            RAD_OP(b, "fill",
                   RAD_PARAMS(RAD_RANGE("M", 1, 1), RAD_INT("n", 1), RAD_STR("dtype", "f32"),
                              RAD_F64("value", 0.75)),
                   RAD_NOWEIGHTS));
    PROJ_OP(op_fill_two,
            RAD_OP(b, "fill",
                   RAD_PARAMS(RAD_RANGE("M", 1, 1), RAD_INT("n", 1), RAD_STR("dtype", "f32"),
                              RAD_F64("value", 2.0)),
                   RAD_NOWEIGHTS));
    PROJ_OP(op_fill_minus1,
            RAD_OP(b, "fill",
                   RAD_PARAMS(RAD_RANGE("M", 1, 1), RAD_INT("n", 1), RAD_STR("dtype", "f32"),
                              RAD_F64("value", -1.0)),
                   RAD_NOWEIGHTS));
    return RAD_OK;
}

void HaruProj::issue(RadCtx* c, RadOperand token, RadOperand model_latent, RadOperand user_latent,
                     RadOperand model_mask, RadOperand user_mask, RadOperand out) const {
    if (!c) return;
    const int64_t H = g.hidden;

    /* The token stream. b_x is the running sum; the two projector adds alternate between b_x and
     * b_x2 so neither add reads and writes one buffer. */
    RAD_ISSUE_N(c, op_embed, 1, token, RAD_W(embed_tokens), RAD_B(b_x));

    auto projector = [&](const HaruAudioProjW& w, rad_op g1, rad_op nr, rad_op g2,
                         RadOperand lat, RadOperand mask, RadOperand src, RadOperand dst) {
        /* scale_latents: (raw + bias) * scale, in f32 (latent_bias is a one-element weight, so a
         * filled row broadcasts it to the latent width before the add; doing it in this order
         * keeps the reference's two roundings). The scaled latent is then cast to the serving
         * dtype once, for fc1. */
        RAD_ISSUE_N(c, op_fill_d, 1, RAD_B(b_ones));
        RAD_ISSUE_N(c, op_mulw_bias, 1, RAD_B(b_ones), RAD_W(latent_bias), RAD_B(b_bias));
        RAD_ISSUE_N(c, op_add_d, 1, lat, RAD_B(b_bias), RAD_B(b_scaled));
        RAD_ISSUE_N(c, op_mulw_scale, 1, RAD_B(b_scaled), RAD_W(latent_scale), RAD_B(b_scaled2));
        RAD_ISSUE_N(c, op_cast_lat, 1, RAD_B(b_scaled2), RAD_B(b_scaled_bf16));

        RAD_ISSUE_N(c, g1, 1, RAD_B(b_scaled_bf16), RAD_W(w.fc1), RAD_W(w.fc1_b), RAD_NONE,
                    RAD_B(b_fc1));
        RAD_ISSUE_N(c, nr, 1, RAD_B(b_fc1), RAD_W(w.norm), RAD_B(b_norm));
        RAD_ISSUE_N(c, g2, 1, RAD_B(b_norm), RAD_W(w.fc2), RAD_W(w.fc2_b), RAD_NONE,
                    RAD_B(b_y));

        /* y * mask, the mask a row scalar: scale_rows reads one scalar per row. */
        RAD_ISSUE_N(c, op_mask, 1, RAD_B(b_y), mask, RAD_NONE, RAD_B(b_mask));
        RAD_ISSUE_N(c, op_add_h, 1, src, RAD_B(b_mask), dst);
    };

    projector(model, op_gemm1_m, op_norm_m, op_gemm2_m, model_latent, model_mask,
              RAD_B(b_x), RAD_B(b_x2));
    projector(user, op_gemm1_u, op_norm_u, op_gemm2_u, user_latent, user_mask,
              RAD_B(b_x2), out);

    (void)H;
}

void HaruProj::unscale(RadCtx* c, RadOperand normalized, RadOperand out) const {
    if (!c) return;
    /* r = 1 / latent_scale, by Newton's iteration from 0.75. It converges for 0 < s < 2.67 and,
     * with twelve steps, reaches f32 exactness for s in [0.05, 2.6]: the error e = 1 - s r0 is
     * squared each step, and the worst case (s = 0.05, e = 0.9625) is below 1e-30 at step 12.
     * The real Haru-4B checkpoint has latent_scale = 0.1963 (bf16), which six steps left at a
     * relative error of 4e-5 (found in Phase 0 on the real checkpoint, 2026-11-19). */
    RAD_ISSUE_N(c, op_fill_r0, 1, RAD_B(b_r0));
    RAD_ISSUE_N(c, op_fill_two, 1, RAD_B(b_two));
    RAD_ISSUE_N(c, op_fill_minus1, 1, RAD_B(b_minus1));
    rad_buf cur = b_r0, nxt = b_r1;
    for (int i = 0; i < kUnscaleSteps; ++i) {
        RAD_ISSUE_N(c, op_mulw_scale1, 1, RAD_B(cur), RAD_W(latent_scale), RAD_B(b_u)); /* s r */
        RAD_ISSUE_N(c, op_mul_1, 1, RAD_B(b_u), RAD_B(b_minus1), RAD_B(b_v));      /* -s r */
        RAD_ISSUE_N(c, op_add_1, 1, RAD_B(b_two), RAD_B(b_v), RAD_B(b_s));          /* 2 - s r */
        RAD_ISSUE_N(c, op_mul_1, 1, RAD_B(cur), RAD_B(b_s), RAD_B(nxt));            /* r (2-sr) */
        std::swap(cur, nxt);
    }

    /* Broadcast the scalar reciprocal across the latent; broadcast -bias the same way. */
    RAD_ISSUE_N(c, op_fill_d, 1, RAD_B(b_ones));
    RAD_ISSUE_N(c, op_scale_row, 1, RAD_B(b_ones), RAD_B(cur), RAD_NONE, RAD_B(b_recip_d));
    RAD_ISSUE_N(c, op_mulw_bias, 1, RAD_B(b_ones), RAD_W(latent_bias), RAD_B(b_bias_d));
    RAD_ISSUE_N(c, op_scale_row, 1, RAD_B(b_bias_d), RAD_B(b_minus1), RAD_NONE,
                RAD_B(b_neg_bias_d));
    RAD_ISSUE_N(c, op_mul_d, 1, normalized, RAD_B(b_recip_d), RAD_B(b_unscaled));
    RAD_ISSUE_N(c, op_add_d, 1, RAD_B(b_unscaled), RAD_B(b_neg_bias_d), out);
}

#undef PROJ_OP
#undef PROJ_BUF
#undef PROJ_BUF_F32

}  /* namespace arch */
}  /* namespace rad */
