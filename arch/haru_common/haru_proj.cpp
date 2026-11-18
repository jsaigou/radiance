/* haru_proj.cpp -- declare and issue the audio-input projectors (haru_proj.h has the arithmetic).
 *
 * The checkpoint's projector tensors are float32 in the tiny model. `latent_scale` and
 * `latent_bias` are buffers, not parameters, so they load as ordinary weights.
 */
#include "haru_proj.h"

#include <cstdio>

namespace rad {
namespace arch {

namespace {

int plain_weight(RadBuilder* b, const char* name, std::initializer_list<int64_t> shape,
                 rad_weight* out) {
    const int st = map_copy(b, name, name);
    if (st < 0) return st;
    *out = decl_w(b, name, RAD_F32, shape, RAD_ACCESS_PER_TOKEN, RAD_SHARD_NONE, grp_model());
    return *out ? RAD_OK : RAD_E_INVAL;
}

#define PROJ_OP(field, expr)       \
    do {                           \
        (field) = (expr);          \
        if (!(field)) return RAD_E_NOKERNEL; \
    } while (0)

#define PROJ_BUF(field, name, n)                                                   \
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

    PROJ_BUF(b_x, "proj.x", H);
    PROJ_BUF(b_x2, "proj.x2", H);
    PROJ_BUF(b_y, "proj.y", H);
    PROJ_BUF(b_mask, "proj.mask", H);
    PROJ_BUF(b_fc1, "proj.fc1", H);
    PROJ_BUF(b_norm, "proj.norm", H);
    PROJ_BUF(b_bias, "proj.bias", D);
    PROJ_BUF(b_scaled, "proj.scaled", D);
    PROJ_BUF(b_scaled2, "proj.scaled2", D);
    PROJ_BUF(b_ones, "proj.ones", D);

    RAD_ARCH_TRY(plain_weight(b, "latent_scale", { 1 }, &latent_scale));
    RAD_ARCH_TRY(plain_weight(b, "latent_bias", { 1 }, &latent_bias));

    auto audio = [&](const char* prefix, HaruAudioProjW* w) -> int {
        char n[128];
        std::snprintf(n, sizeof n, "%s.fc1.weight", prefix);
        RAD_ARCH_TRY(plain_weight(b, n, { H, D }, &w->fc1));
        std::snprintf(n, sizeof n, "%s.fc1.bias", prefix);
        RAD_ARCH_TRY(plain_weight(b, n, { H }, &w->fc1_b));
        std::snprintf(n, sizeof n, "%s.norm.weight", prefix);
        RAD_ARCH_TRY(plain_weight(b, n, { H }, &w->norm));
        std::snprintf(n, sizeof n, "%s.fc2.weight", prefix);
        RAD_ARCH_TRY(plain_weight(b, n, { H, H }, &w->fc2));
        std::snprintf(n, sizeof n, "%s.fc2.bias", prefix);
        RAD_ARCH_TRY(plain_weight(b, n, { H }, &w->fc2_b));
        return RAD_OK;
    };
    RAD_ARCH_TRY(audio("model_audio_proj", &model));
    RAD_ARCH_TRY(audio("user_audio_proj", &user));

    auto gemm_bias = [&](int64_t N, int64_t K, rad_weight w, rad_weight bias) {
        return RAD_OP(b, "gemm_nt_bias",
                      RAD_PARAMS(RAD_RANGE("M", 1, 1), RAD_INT("N", N), RAD_INT("K", K),
                                 RAD_STR("dtype", "f32")),
                      RAD_WEIGHTS(w, bias));
    };
    auto norm = [&](rad_weight w) {
        return RAD_OP(b, "rmsnorm",
                      RAD_PARAMS(RAD_RANGE("M", 1, 1), RAD_INT("n", H),
                                 RAD_F64("eps", g.norm_eps), RAD_STR("dtype", "f32")),
                      RAD_WEIGHTS(w));
    };

    PROJ_OP(op_embed,
            RAD_OP(b, "embed_lookup",
                   RAD_PARAMS(RAD_RANGE("M", 1, 1), RAD_INT("n_embd", H), RAD_INT("n_vocab", V),
                              RAD_STR("dtype", "f32"), RAD_INT("vocab_offset", 0),
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
                              RAD_STR("dtype", "f32")),
                   RAD_NOWEIGHTS));
    PROJ_OP(op_add_h,
            RAD_OP(b, "add",
                   RAD_PARAMS(RAD_RANGE("M", 1, 1), RAD_INT("n", H), RAD_STR("dtype", "f32")),
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
        /* scale_latents: (raw + bias) * scale. latent_bias is a one-element weight, so a filled
         * row broadcasts it to the latent width before the add; doing it in this order keeps the
         * reference's two roundings (add first, then multiply). */
        RAD_ISSUE_N(c, op_fill_d, 1, RAD_B(b_ones));
        RAD_ISSUE_N(c, op_mulw_bias, 1, RAD_B(b_ones), RAD_W(latent_bias), RAD_B(b_bias));
        RAD_ISSUE_N(c, op_add_d, 1, lat, RAD_B(b_bias), RAD_B(b_scaled));
        RAD_ISSUE_N(c, op_mulw_scale, 1, RAD_B(b_scaled), RAD_W(latent_scale), RAD_B(b_scaled2));

        RAD_ISSUE_N(c, g1, 1, RAD_B(b_scaled2), RAD_W(w.fc1), RAD_W(w.fc1_b), RAD_NONE,
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

#undef PROJ_OP
#undef PROJ_BUF

}  /* namespace arch */
}  /* namespace rad */
