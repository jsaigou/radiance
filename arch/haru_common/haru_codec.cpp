/* haru_codec.cpp -- see haru_codec.h for the design and the launch count. */
#include "haru_codec.h"

#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <cstring>

namespace haru {

using namespace rad::arch;

namespace {

bool good(uint32_t h) { return h != RAD_NULL_HANDLE; }

int64_t ceil_div(int64_t a, int64_t b) { return (a + b - 1) / b; }

uint32_t dt_of(const CodecGeometry& g) { return g.dtype; }
const char* dt_name(const CodecGeometry& g) { return rad_dtype_name(g.dtype); }

/* A checkpoint tensor read through its own name. The map entry is declared first: a weight without
 * one is "absent from the container" and the program cannot run. */
int declare_weight(RadBuilder* b, const char* name, const CodecGeometry& g,
                   std::initializer_list<int64_t> shape, rad_weight* out) {
    if (map_copy(b, name, name) < 0) return RAD_E_INVAL;
    *out = decl_w(b, name, dt_of(g), shape, RAD_ACCESS_PER_TOKEN, RAD_SHARD_NONE,
                  grp_layer(kCodecGroup));
    return good(*out) ? RAD_OK : RAD_E_INVAL;
}

rad_op fill_op(RadBuilder* b, int64_t lo, int64_t hi, int64_t n, const char* dtype, double value,
               rad_buf y) {
    return rw(b, RAD_OP(b, "fill",
                        RAD_PARAMS(RAD_RANGE("M", lo, hi), RAD_INT("n", n),
                                   RAD_STR("dtype", dtype), RAD_F64("value", value)),
                        RAD_NOWEIGHTS),
              {}, { y });
}

rad_op cast_op(RadBuilder* b, int64_t lo, int64_t hi, int64_t n, const char* from,
               const char* to, rad_buf x, rad_buf y) {
    return rw(b, RAD_OP(b, "cast",
                        RAD_PARAMS(RAD_RANGE("M", lo, hi), RAD_INT("n", n),
                                   RAD_STR("from", from), RAD_STR("to", to)),
                        RAD_NOWEIGHTS),
              { x }, { y });
}

rad_op add_op(RadBuilder* b, int64_t rows, int64_t n, const char* dtype, rad_buf a, rad_buf c,
              rad_buf y) {
    return rw(b, RAD_OP(b, "add",
                        RAD_PARAMS(RAD_RANGE("M", 1, rows), RAD_INT("n", n),
                                   RAD_STR("dtype", dtype)),
                        RAD_NOWEIGHTS),
              { a, c }, { y });
}

rad_op mul_weight_op(RadBuilder* b, int64_t rows, int64_t n, const char* dtype, rad_weight w,
                     rad_buf x, rad_buf y) {
    return rw(b, RAD_OP(b, "mul_weight",
                        RAD_PARAMS(RAD_RANGE("M", 1, rows), RAD_INT("n", n),
                                   RAD_STR("dtype", dtype), RAD_INT("scalar", 0)),
                        RAD_WEIGHTS(w)),
              { x }, { y });
}

rad_op rmsnorm_op(RadBuilder* b, int64_t rows, int64_t n, double eps, const char* dtype,
                  rad_weight w, rad_buf x, rad_buf y) {
    return rw(b, RAD_OP(b, "rmsnorm",
                        RAD_PARAMS(RAD_RANGE("M", 1, rows), RAD_INT("n", n),
                                   RAD_F64("eps", eps), RAD_STR("dtype", dtype)),
                        RAD_WEIGHTS(w)),
              { x }, { y });
}

rad_op gemm_bias_op(RadBuilder* b, int64_t rows, int64_t N, int64_t K, const char* dtype,
                    rad_weight w, rad_weight bias, rad_buf x, rad_buf y) {
    return rw(b, RAD_OP(b, "gemm_nt_bias",
                        RAD_PARAMS(RAD_RANGE("M", 1, rows), RAD_INT("N", N), RAD_INT("K", K),
                                   RAD_STR("dtype", dtype)),
                        RAD_WEIGHTS(w, bias)),
              { x }, { y });
}

rad_op gelu_op(RadBuilder* b, int64_t rows, int64_t n, const char* dtype, rad_buf x, rad_buf y) {
    return rw(b, RAD_OP(b, "gelu",
                        RAD_PARAMS(RAD_RANGE("M", 1, rows), RAD_INT("n", n),
                                   RAD_STR("dtype", dtype)),
                        RAD_NOWEIGHTS),
              { x }, { y });
}

/* Declares one conv layer. `copy_src` is the buffer the layer's new input comes from. */
int declare_conv(RadBuilder* b, const CodecGeometry& g, ConvLayer* L, const char* prefix,
                 rad_buf copy_src, int64_t cin, int64_t cout, int64_t k, int64_t stride,
                 int64_t max_in, int64_t max_out, bool transposed, Names& nm) {
    const char* wname = nm.f("%s.weight", prefix);
    const char* bname = nm.f("%s.bias", prefix);
    /* Conv1d stores [cout, cin, k]; ConvTranspose1d stores [cin, cout, k], and conv_t1d reads
     * PyTorch's layout unchanged. */
    if (transposed)
        RAD_ARCH_TRY(declare_weight(b, wname, g, { cin, cout, k }, &L->w));
    else
        RAD_ARCH_TRY(declare_weight(b, wname, g, { cout, cin, k }, &L->w));
    RAD_ARCH_TRY(declare_weight(b, bname, g, { cout }, &L->b));

    L->cin = cin;
    L->cout = cout;
    L->k = k;
    L->stride = stride;
    L->transposed = transposed;
    L->context = transposed ? 1 : k - stride;
    L->cu_rows = (transposed ? 1 : L->context) + max_in;
    L->drop = transposed ? stride : (L->context / stride);
    L->max_in = max_in;
    L->max_out = max_out;

    const int64_t state_rows = L->context;
    const int64_t out_rows = L->drop + max_out;
    L->combined = decl_b(b, nm.f("%s.combined", prefix), dt_of(g), { L->cu_rows, cin });
    L->out = decl_b(b, nm.f("%s.out", prefix), dt_of(g), { out_rows, cout });
    L->state = decl_b(b, nm.f("%s.state", prefix), dt_of(g),
                      { g.max_lanes, state_rows, cin }, RAD_BUF_PERSIST);
    L->cu = decl_b(b, nm.f("%s.cu", prefix), RAD_I32, { 2 }, RAD_BUF_PERSIST);
    if (!good(L->combined) || !good(L->out) || !good(L->state) || !good(L->cu))
        return RAD_E_INVAL;

    L->fill_state = fill_op(b, 1, state_rows, cin, dt_name(g), 0.0, L->state);
    L->copy_state_in = cast_op(b, 1, state_rows, cin, dt_name(g), dt_name(g), L->state,
                               L->combined);
    L->copy_input = cast_op(b, 1, max_in, cin, dt_name(g), dt_name(g), copy_src, L->combined);
    L->fill_pad = fill_op(b, 1, max_in, cin, dt_name(g), 0.0, L->combined);
    L->copy_state_out = cast_op(b, 1, state_rows, cin, dt_name(g), dt_name(g), L->combined,
                                L->state);
    L->iota_cu = rw(b, RAD_OP(b, "iota",
                              RAD_PARAMS(RAD_INT("M", 2), RAD_INT("start", 0),
                                         RAD_INT("step", L->cu_rows)),
                              RAD_NOWEIGHTS),
                    {}, { L->cu });

    if (transposed) {
        L->op = rw(b, RAD_OP(b, "conv_t1d",
                             RAD_PARAMS(RAD_RANGE("M", 1, L->cu_rows), RAD_INT("cin", cin),
                                        RAD_INT("cout", cout), RAD_INT("stride", stride),
                                        RAD_STR("dtype", dt_name(g)), RAD_INT("pad", 0),
                                        RAD_INT("k", k)),
                             RAD_WEIGHTS(L->w, L->b)),
                   { L->combined, L->cu }, { L->out });
    } else {
        L->op = rw(b, RAD_OP(b, "conv1d",
                             RAD_PARAMS(RAD_RANGE("M", 1, L->cu_rows), RAD_INT("cin", cin),
                                        RAD_INT("cout", cout), RAD_INT("k", k), RAD_INT("dil", 1),
                                        RAD_STR("act", "none"), RAD_STR("dtype", dt_name(g)),
                                        RAD_INT("causal", 1), RAD_INT("stride", stride)),
                             RAD_WEIGHTS(L->w, L->b)),
                   { L->combined, L->cu }, { L->out });
    }
    if (!good(L->op) || !good(L->fill_state) || !good(L->copy_state_in) ||
        !good(L->copy_input) || !good(L->fill_pad) || !good(L->copy_state_out) ||
        !good(L->iota_cu))
        return RAD_E_NOKERNEL;
    return RAD_OK;
}

/* Declares one Block. `copy_src` is the unnormed x the first RMSNorm and the first residual read;
 * the depthwise conv's history is the normed x (n1), as in the reference. */
int declare_block(RadBuilder* b, const CodecGeometry& g, BlockLayer* B, const char* prefix,
                  rad_buf copy_src, int64_t dim, int64_t max_rows, Names& nm) {
    RAD_ARCH_TRY(declare_weight(b, nm.f("%s.norm.weight", prefix), g, { dim }, &B->norm));
    RAD_ARCH_TRY(declare_weight(b, nm.f("%s.conv.weight", prefix), g, { dim, 1, 7 }, &B->conv_w));
    RAD_ARCH_TRY(declare_weight(b, nm.f("%s.conv.bias", prefix), g, { dim }, &B->conv_b));
    RAD_ARCH_TRY(declare_weight(b, nm.f("%s.gamma", prefix), g, { dim }, &B->gamma));
    RAD_ARCH_TRY(declare_weight(b, nm.f("%s.ffn_norm.weight", prefix), g, { dim }, &B->ffn_norm));
    RAD_ARCH_TRY(declare_weight(b, nm.f("%s.fc1.weight", prefix), g, { 4 * dim, dim },
                                &B->fc1_w));
    RAD_ARCH_TRY(declare_weight(b, nm.f("%s.fc1.bias", prefix), g, { 4 * dim }, &B->fc1_b));
    RAD_ARCH_TRY(declare_weight(b, nm.f("%s.fc2.weight", prefix), g, { dim, 4 * dim },
                                &B->fc2_w));
    RAD_ARCH_TRY(declare_weight(b, nm.f("%s.fc2.bias", prefix), g, { dim }, &B->fc2_b));
    RAD_ARCH_TRY(declare_weight(b, nm.f("%s.ffn_gamma", prefix), g, { dim }, &B->ffn_gamma));

    B->dim = dim;
    B->max_rows = max_rows;
    B->context = 6;
    B->cu_rows = B->context + max_rows;

    B->combined = decl_b(b, nm.f("%s.combined", prefix), dt_of(g), { B->cu_rows, dim });
    B->dw_out = decl_b(b, nm.f("%s.dw_out", prefix), dt_of(g), { B->cu_rows, dim });
    B->state = decl_b(b, nm.f("%s.state", prefix), dt_of(g), { g.max_lanes, B->context, dim },
                      RAD_BUF_PERSIST);
    B->cu = decl_b(b, nm.f("%s.cu", prefix), RAD_I32, { 2 }, RAD_BUF_PERSIST);
    B->n1 = decl_b(b, nm.f("%s.n1", prefix), dt_of(g), { max_rows, dim });
    B->scaled1 = decl_b(b, nm.f("%s.scaled1", prefix), dt_of(g), { max_rows, dim });
    B->x1 = decl_b(b, nm.f("%s.x1", prefix), dt_of(g), { max_rows, dim });
    B->n2 = decl_b(b, nm.f("%s.n2", prefix), dt_of(g), { max_rows, dim });
    B->h1 = decl_b(b, nm.f("%s.h1", prefix), dt_of(g), { max_rows, 4 * dim });
    B->h2 = decl_b(b, nm.f("%s.h2", prefix), dt_of(g), { max_rows, 4 * dim });
    B->h3 = decl_b(b, nm.f("%s.h3", prefix), dt_of(g), { max_rows, dim });
    B->scaled2 = decl_b(b, nm.f("%s.scaled2", prefix), dt_of(g), { max_rows, dim });
    B->x2 = decl_b(b, nm.f("%s.x2", prefix), dt_of(g), { max_rows, dim });
    if (!good(B->combined) || !good(B->dw_out) || !good(B->state) || !good(B->cu) ||
        !good(B->n1) || !good(B->scaled1) || !good(B->x1) || !good(B->n2) || !good(B->h1) ||
        !good(B->h2) || !good(B->h3) || !good(B->scaled2) || !good(B->x2))
        return RAD_E_INVAL;

    /* rms1 before the copies: the depthwise conv's input is n1, so its copy must follow the norm
     * that writes n1 in the graph order the liveness plan sees. */
    B->rms1 = rmsnorm_op(b, max_rows, dim, g.norm_eps, dt_name(g), B->norm, copy_src, B->n1);
    B->fill_state = fill_op(b, 1, B->context, dim, dt_name(g), 0.0, B->state);
    B->copy_state_in = cast_op(b, 1, B->context, dim, dt_name(g), dt_name(g), B->state,
                               B->combined);
    B->copy_input = cast_op(b, 1, max_rows, dim, dt_name(g), dt_name(g), B->n1, B->combined);
    B->fill_pad = fill_op(b, 1, max_rows, dim, dt_name(g), 0.0, B->combined);
    B->copy_state_out = cast_op(b, 1, B->context, dim, dt_name(g), dt_name(g), B->combined,
                                B->state);
    B->iota_cu = rw(b, RAD_OP(b, "iota",
                              RAD_PARAMS(RAD_INT("M", 2), RAD_INT("start", 0),
                                         RAD_INT("step", B->cu_rows)),
                              RAD_NOWEIGHTS),
                    {}, { B->cu });

    B->dw = rw(b, RAD_OP(b, "conv1d_dw",
                         RAD_PARAMS(RAD_RANGE("M", 1, B->cu_rows), RAD_INT("n", dim),
                                    RAD_INT("k", 7), RAD_INT("cu_div", 1),
                                    RAD_STR("dtype", dt_name(g))),
                         RAD_WEIGHTS(B->conv_w, B->conv_b)),
               { B->combined, B->cu }, { B->dw_out });
    B->scale1 = mul_weight_op(b, max_rows, dim, dt_name(g), B->gamma, B->dw_out, B->scaled1);
    B->add1 = add_op(b, max_rows, dim, dt_name(g), copy_src, B->scaled1, B->x1);
    B->rms2 = rmsnorm_op(b, max_rows, dim, g.norm_eps, dt_name(g), B->ffn_norm, B->x1, B->n2);
    B->fc1 = gemm_bias_op(b, max_rows, 4 * dim, dim, dt_name(g), B->fc1_w, B->fc1_b, B->n2,
                          B->h1);
    B->gelu = gelu_op(b, max_rows, 4 * dim, dt_name(g), B->h1, B->h2);
    B->fc2 = gemm_bias_op(b, max_rows, dim, 4 * dim, dt_name(g), B->fc2_w, B->fc2_b, B->h2,
                          B->h3);
    B->scale2 = mul_weight_op(b, max_rows, dim, dt_name(g), B->ffn_gamma, B->h3, B->scaled2);
    B->add2 = add_op(b, max_rows, dim, dt_name(g), B->x1, B->scaled2, B->x2);

    if (!good(B->rms1) || !good(B->dw) || !good(B->scale1) || !good(B->add1) ||
        !good(B->rms2) || !good(B->fc1) || !good(B->gelu) || !good(B->fc2) ||
        !good(B->scale2) || !good(B->add2) || !good(B->fill_state) || !good(B->copy_state_in) ||
        !good(B->copy_input) || !good(B->fill_pad) || !good(B->copy_state_out) ||
        !good(B->iota_cu))
        return RAD_E_NOKERNEL;
    return RAD_OK;
}

/* A valid output window: the operand is a row slice of a layer buffer. */
struct View {
    RadOperand op{};
    int64_t rows = 0;
};

/* One conv layer's step. `in` is the previous layer's valid output (or the staged input). */
View run_conv(RadCtx* c, const ConvLayer& L, int64_t lane, RadOperand in, int64_t rows) {
    const int64_t context = L.context;
    const RadOperand state = brow_slice(L.state, lane * context, context, L.cin);
    RAD_ISSUE_N(c, L.copy_state_in, context, state,
                brow_slice(L.combined, 0, context, L.cin));
    RAD_ISSUE_N(c, L.copy_input, rows, in,
                brow_slice(L.combined, context, rows, L.cin));
    if (rows < L.max_in)
        RAD_ISSUE_N(c, L.fill_pad, L.max_in - rows,
                    brow_slice(L.combined, context + rows, L.max_in - rows, L.cin));
    if (L.transposed)
        RAD_ISSUE_N(c, L.op, L.cu_rows, RAD_B(L.combined), RAD_W(L.w), RAD_W(L.b),
                    RAD_B(L.cu), RAD_B(L.out));
    else
        RAD_ISSUE_N(c, L.op, L.cu_rows, RAD_B(L.combined), RAD_W(L.w), RAD_B(L.cu), RAD_NONE,
                    RAD_B(L.out), RAD_W(L.b));
    RAD_ISSUE_N(c, L.copy_state_out, context,
                brow_slice(L.combined, L.max_in, context, L.cin), state);
    const int64_t valid = L.transposed ? rows * L.stride : ceil_div(rows, L.stride);
    return View{ brow_slice(L.out, L.drop, valid, L.cout), valid };
}

/* One Block's step. */
View run_block(RadCtx* c, const BlockLayer& B, int64_t lane, RadOperand in, int64_t rows) {
    const RadOperand state = brow_slice(B.state, lane * B.context, B.context, B.dim);
    /* Every row-wise operand is narrowed to this step's valid rows. A full buffer left at its
     * declared max_rows would make the kernel process (and read stale bytes from) the rows past
     * the partial frame's valid ones, even though the next layer would not read them. */
    const RadOperand n1 = brow_slice(B.n1, 0, rows, B.dim);
    const RadOperand scaled1 = brow_slice(B.scaled1, 0, rows, B.dim);
    const RadOperand x1 = brow_slice(B.x1, 0, rows, B.dim);
    const RadOperand n2 = brow_slice(B.n2, 0, rows, B.dim);
    const RadOperand h1 = brow_slice(B.h1, 0, rows, 4 * B.dim);
    const RadOperand h2 = brow_slice(B.h2, 0, rows, 4 * B.dim);
    const RadOperand h3 = brow_slice(B.h3, 0, rows, B.dim);
    const RadOperand scaled2 = brow_slice(B.scaled2, 0, rows, B.dim);
    const RadOperand x2 = brow_slice(B.x2, 0, rows, B.dim);
    RAD_ISSUE_N(c, B.rms1, rows, in, RAD_W(B.norm), n1);
    RAD_ISSUE_N(c, B.copy_state_in, B.context, state,
                brow_slice(B.combined, 0, B.context, B.dim));
    RAD_ISSUE_N(c, B.copy_input, rows, n1, brow_slice(B.combined, B.context, rows, B.dim));
    if (rows < B.max_rows)
        RAD_ISSUE_N(c, B.fill_pad, B.max_rows - rows,
                    brow_slice(B.combined, B.context + rows, B.max_rows - rows, B.dim));
    RAD_ISSUE_N(c, B.dw, B.cu_rows, RAD_B(B.combined), RAD_W(B.conv_w), RAD_B(B.cu),
                RAD_B(B.dw_out), RAD_W(B.conv_b));
    RAD_ISSUE_N(c, B.copy_state_out, B.context,
                brow_slice(B.combined, B.max_rows, B.context, B.dim), state);
    RAD_ISSUE_N(c, B.scale1, rows, brow_slice(B.dw_out, B.context, rows, B.dim), RAD_W(B.gamma),
                scaled1);
    RAD_ISSUE_N(c, B.add1, rows, in, scaled1, x1);
    RAD_ISSUE_N(c, B.rms2, rows, x1, RAD_W(B.ffn_norm), n2);
    RAD_ISSUE_N(c, B.fc1, rows, n2, RAD_W(B.fc1_w), RAD_W(B.fc1_b), RAD_NONE, h1);
    RAD_ISSUE_N(c, B.gelu, rows, h1, h2);
    RAD_ISSUE_N(c, B.fc2, rows, h2, RAD_W(B.fc2_w), RAD_W(B.fc2_b), RAD_NONE, h3);
    RAD_ISSUE_N(c, B.scale2, rows, h3, RAD_W(B.ffn_gamma), scaled2);
    RAD_ISSUE_N(c, B.add2, rows, x1, scaled2, x2);
    return View{ brow_slice(B.x2, 0, rows, B.dim), rows };
}

/* Runs one stack. `capture_stage` >= 0 writes that stage's last-block output to audio_out and
 * returns before the head; otherwise the head's valid output goes to out_latent/out_audio. */
void run_stack(RadCtx* c, const Codec& k, const Stack& S, int64_t lane, RadOperand input,
               int64_t rows, int64_t capture_stage, rad_buf audio_out) {
    View v = run_conv(c, S.stem, lane, input, rows);
    for (int stage = 0; stage < kCodecStages; ++stage) {
        for (int64_t j = 0; j < S.depths[(size_t)stage]; ++j)
            v = run_block(c, S.blocks[(size_t)stage][(size_t)j], lane, v.op, v.rows);
        if (capture_stage == stage) {
            const int64_t dim = S.blocks[(size_t)stage][0].dim;
            const int64_t n = v.rows * dim;
            RAD_ISSUE_N(c, S.debug_copy[(size_t)stage], n, v.op,
                        bcol_at(audio_out, lane, k.g.frame_samples, 0, n, 1));
            return;
        }
        if (stage < kCodecRatios) {
            const ConvLayer& T = S.trans[(size_t)stage];
            v = run_conv(c, T, lane, v.op, v.rows);
        }
    }
    v = run_conv(c, S.head, lane, v.op, v.rows);
    if (S.encoder)
        RAD_ISSUE_N(c, S.head_out, 1, v.op, brow_slice(k.out_latent, 0, 1, k.g.latent_dim));
    else
        RAD_ISSUE_N(c, S.head_out, k.g.frame_samples, v.op,
                    brow_slice(k.out_audio, 0, k.g.frame_samples, 1));
}

}  /* namespace */

int codec_geometry_from_meta(const RadModelMeta* m, CodecGeometry* g, std::string* why) {
    auto fail = [&](const std::string& w) {
        if (why) *why = w;
        return RAD_E_INVAL;
    };
    if (!m || !g) return fail("no metadata");
    g->latent_dim = rad_meta_geti(m, "codec.latent_dim", g->latent_dim);
    g->base_channels = rad_meta_geti(m, "codec.base_channels", g->base_channels);
    g->frame_samples = rad_meta_geti(m, "frame_samples", g->frame_samples);
    g->norm_eps = rad_meta_getf(m, "codec.norm_eps", g->norm_eps);
    g->latent_std = rad_meta_getf(m, "codec.latent_std", g->latent_std);

    auto parse = [&](const char* key, int64_t* out, int n) {
        const char* text = rad_meta_gets(m, key, nullptr);
        if (!text) return fail(std::string("the container has no ") + key);
        int i = 0;
        bool bad = false;
        for_each_token(text, [&](const char* tok, size_t) {
            if (i >= n) { bad = true; return false; }
            char* end = nullptr;
            const long long v = std::strtoll(tok, &end, 10);
            if (end == tok) { bad = true; return false; }
            out[i++] = (int64_t)v;
            return true;
        });
        if (bad || i != n)
            return fail(std::string(key) + " must hold exactly " + std::to_string(n) +
                        " integers");
        return RAD_OK;
    };
    int rc = parse("codec.ratios", g->ratios.data(), kCodecRatios);
    if (rc < 0) return rc;
    rc = parse("codec.encoder_depths", g->encoder_depths.data(), kCodecStages);
    if (rc < 0) return rc;
    rc = parse("codec.decoder_depths", g->decoder_depths.data(), kCodecStages);
    if (rc < 0) return rc;

    if (g->latent_dim <= 0 || g->base_channels <= 0 || g->frame_samples <= 0)
        return fail("the codec's latent_dim, base_channels and frame_samples must be positive");
    int64_t samples = 1;
    for (int i = 0; i < kCodecRatios; ++i) {
        if (g->ratios[(size_t)i] <= 0) return fail("a codec ratio is not positive");
        samples *= g->ratios[(size_t)i];
    }
    if (samples != g->frame_samples)
        return fail("the codec ratios multiply to " + std::to_string(samples) +
                    ", not frame_samples " + std::to_string(g->frame_samples));
    for (int i = 0; i < kCodecStages; ++i) {
        if (g->encoder_depths[(size_t)i] < 1 || g->encoder_depths[(size_t)i] > kCodecMaxDepth)
            return fail("an encoder depth is outside 1.." + std::to_string(kCodecMaxDepth));
        if (g->decoder_depths[(size_t)i] < 1 || g->decoder_depths[(size_t)i] > kCodecMaxDepth)
            return fail("a decoder depth is outside 1.." + std::to_string(kCodecMaxDepth));
    }
    const char* dtype = rad_meta_gets(m, "dtype", "float32");
    if (!std::strcmp(dtype, "float32") || !std::strcmp(dtype, "f32"))
        g->dtype = RAD_F32;
    else if (!std::strcmp(dtype, "bfloat16") || !std::strcmp(dtype, "bf16"))
        g->dtype = RAD_BF16;
    else
        return fail(std::string("the codec does not serve dtype '") + dtype + "'");
    if (why) why->clear();
    return RAD_OK;
}

int codec_declare(RadBuilder* b, const CodecGeometry& g, rad_buf audio_in, rad_buf audio_out,
                  Codec* out) {
    if (!b || !out || !good(audio_in) || !good(audio_out)) return RAD_E_INVAL;
    if (g.max_lanes <= 0)
        return RAD_E_INVAL;
    out->g = g;
    Names nm;

    std::array<int64_t, kCodecStages> enc_rows{};
    std::array<int64_t, kCodecStages> dec_rows{};
    enc_rows[0] = g.frame_samples;
    for (int i = 0; i < kCodecRatios; ++i)
        enc_rows[(size_t)i + 1] = ceil_div(enc_rows[(size_t)i], g.ratios[(size_t)(kCodecRatios - 1 - i)]);
    dec_rows[0] = 1;
    for (int i = 0; i < kCodecRatios; ++i)
        dec_rows[(size_t)i + 1] = dec_rows[(size_t)i] * g.ratios[(size_t)i];

    out->in_audio = decl_b(b, nm.f("codec.in_audio"), g.dtype, { g.frame_samples, 1 });
    out->in_latent = decl_b(b, nm.f("codec.in_latent"), g.dtype, { 1, g.latent_dim });
    out->out_latent = decl_b(b, nm.f("codec.out_latent"), g.dtype, { 1, g.latent_dim });
    out->out_audio = decl_b(b, nm.f("codec.out_audio"), g.dtype, { g.frame_samples, 1 });
    out->sample_mean = decl_b(b, nm.f("codec.sample_mean"), g.dtype, { 1, g.latent_dim });
    out->sample_noise = decl_b(b, nm.f("codec.sample_noise"), g.dtype, { 1, g.latent_dim });
    if (!good(out->in_audio) || !good(out->in_latent) || !good(out->out_latent) ||
        !good(out->out_audio) || !good(out->sample_mean) || !good(out->sample_noise))
        return RAD_E_INVAL;

    out->copy_frame_in = cast_op(b, 1, g.frame_samples, 1, "f32", dt_name(g), audio_in,
                                 out->in_audio);
    out->copy_latent_in = cast_op(b, 1, 1, g.latent_dim, "f32", dt_name(g), audio_in,
                                  out->in_latent);
    out->copy_sample_mean = cast_op(b, 1, 1, g.latent_dim, "f32", dt_name(g), audio_in,
                                    out->sample_mean);
    out->copy_sample_noise = cast_op(b, 1, 1, g.latent_dim, "f32", dt_name(g), audio_in,
                                     out->sample_noise);
    out->sample_add = add_op(b, 1, g.latent_dim, dt_name(g), out->sample_mean,
                             out->sample_noise, out->out_latent);
    if (!good(out->copy_frame_in) || !good(out->copy_latent_in) ||
        !good(out->copy_sample_mean) || !good(out->copy_sample_noise) || !good(out->sample_add))
        return RAD_E_NOKERNEL;

    /* Encoder: stem, stage blocks, down, ..., stage 6 blocks, head. */
    out->enc.encoder = true;
    out->enc.depths = g.encoder_depths;
    RAD_ARCH_TRY(declare_conv(b, g, &out->enc.stem, "codec.encoder.stem", out->in_audio, 1,
                              g.base_channels, 7, 1, g.frame_samples, g.frame_samples, false, nm));
    rad_buf prev = out->enc.stem.out;
    for (int stage = 0; stage < kCodecStages; ++stage) {
        for (int64_t j = 0; j < g.encoder_depths[(size_t)stage]; ++j) {
            BlockLayer& B = out->enc.blocks[(size_t)stage][(size_t)j];
            const char* prefix = nm.f("codec.encoder.stages.%d.%lld", stage, (long long)j);
            RAD_ARCH_TRY(declare_block(b, g, &B, prefix, prev, g.base_channels << stage,
                                       enc_rows[(size_t)stage], nm));
            prev = B.x2;
        }
        /* Declared here, not after the head: the debug copy reads this stage's last x2, and its
         * position keeps the x2's live range from stretching across the rest of the graph. */
        {
            const BlockLayer& last =
                out->enc.blocks[(size_t)stage][(size_t)(g.encoder_depths[(size_t)stage] - 1)];
            out->enc.debug_copy[(size_t)stage] =
                rw(b, RAD_OP(b, "cast",
                             RAD_PARAMS(RAD_RANGE("M", 1, g.frame_samples), RAD_INT("n", 1),
                                        RAD_STR("from", dt_name(g)), RAD_STR("to", "f32")),
                             RAD_NOWEIGHTS),
                   { last.x2 }, { audio_out });
        }
        if (stage < kCodecRatios) {
            ConvLayer& T = out->enc.trans[(size_t)stage];
            const int64_t s = g.ratios[(size_t)(kCodecRatios - 1 - stage)];
            const char* prefix = nm.f("codec.encoder.down.%d", stage);
            RAD_ARCH_TRY(declare_conv(b, g, &T, prefix, prev, g.base_channels << stage,
                                      g.base_channels << (stage + 1), 2 * s, s,
                                      enc_rows[(size_t)stage], enc_rows[(size_t)stage + 1],
                                      false, nm));
            prev = T.out;
        }
    }
    RAD_ARCH_TRY(declare_conv(b, g, &out->enc.head, "codec.encoder.head", prev,
                              g.base_channels << kCodecRatios, g.latent_dim, 7, 1,
                              enc_rows[(size_t)kCodecStages - 1], 1, false, nm));
    out->enc.head_out = rw(b, RAD_OP(b, "cast",
                                     RAD_PARAMS(RAD_RANGE("M", 1, 1),
                                                RAD_INT("n", g.latent_dim),
                                                RAD_STR("from", dt_name(g)),
                                                RAD_STR("to", dt_name(g))),
                                     RAD_NOWEIGHTS),
                           { out->enc.head.out }, { out->out_latent });
    if (!good(out->enc.head_out)) return RAD_E_NOKERNEL;
    for (int stage = 0; stage < kCodecStages; ++stage)
        if (!good(out->enc.debug_copy[(size_t)stage])) return RAD_E_NOKERNEL;
    /* Declared after the encoder so its read of out_latent follows the head's write in the graph
     * order (the sample add writes out_latent earlier still). */
    out->copy_latent_out = cast_op(b, 1, 1, g.latent_dim, dt_name(g), "f32", out->out_latent,
                                   audio_out);
    if (!good(out->copy_latent_out)) return RAD_E_NOKERNEL;

    /* Decoder: stem, stage blocks, up, ..., stage 6 blocks, head. */
    out->dec.encoder = false;
    out->dec.depths = g.decoder_depths;
    RAD_ARCH_TRY(declare_conv(b, g, &out->dec.stem, "codec.decoder.stem", out->in_latent,
                              g.latent_dim, g.base_channels << kCodecRatios, 7, 1, 1, 1, false,
                              nm));
    prev = out->dec.stem.out;
    for (int stage = 0; stage < kCodecStages; ++stage) {
        const int64_t dim = g.base_channels << (kCodecRatios - stage);
        for (int64_t j = 0; j < g.decoder_depths[(size_t)stage]; ++j) {
            BlockLayer& B = out->dec.blocks[(size_t)stage][(size_t)j];
            const char* prefix = nm.f("codec.decoder.stages.%d.%lld", stage, (long long)j);
            RAD_ARCH_TRY(declare_block(b, g, &B, prefix, prev, dim, dec_rows[(size_t)stage], nm));
            prev = B.x2;
        }
        {
            const BlockLayer& last =
                out->dec.blocks[(size_t)stage][(size_t)(g.decoder_depths[(size_t)stage] - 1)];
            out->dec.debug_copy[(size_t)stage] =
                rw(b, RAD_OP(b, "cast",
                             RAD_PARAMS(RAD_RANGE("M", 1, g.frame_samples), RAD_INT("n", 1),
                                        RAD_STR("from", dt_name(g)), RAD_STR("to", "f32")),
                             RAD_NOWEIGHTS),
                   { last.x2 }, { audio_out });
        }
        if (stage < kCodecRatios) {
            ConvLayer& T = out->dec.trans[(size_t)stage];
            const int64_t s = g.ratios[(size_t)stage];
            const char* prefix = nm.f("codec.decoder.up.%d", stage);
            RAD_ARCH_TRY(declare_conv(b, g, &T, prefix, prev, dim,
                                      g.base_channels << (kCodecRatios - stage - 1), 2 * s, s,
                                      dec_rows[(size_t)stage], dec_rows[(size_t)stage + 1],
                                      true, nm));
            prev = T.out;
        }
    }
    RAD_ARCH_TRY(declare_conv(b, g, &out->dec.head, "codec.decoder.head", prev,
                              g.base_channels, 1, 7, 1, dec_rows[(size_t)kCodecStages - 1],
                              dec_rows[(size_t)kCodecStages - 1], false, nm));
    out->dec.head_out = rw(b, RAD_OP(b, "cast",
                                     RAD_PARAMS(RAD_RANGE("M", 1, g.frame_samples), RAD_INT("n", 1),
                                                RAD_STR("from", dt_name(g)),
                                                RAD_STR("to", dt_name(g))),
                                     RAD_NOWEIGHTS),
                           { out->dec.head.out }, { out->out_audio });
    if (!good(out->dec.head_out)) return RAD_E_NOKERNEL;
    for (int stage = 0; stage < kCodecStages; ++stage)
        if (!good(out->dec.debug_copy[(size_t)stage])) return RAD_E_NOKERNEL;
    out->copy_frame_out = cast_op(b, 1, g.frame_samples, 1, dt_name(g), "f32", out->out_audio,
                                  audio_out);
    if (!good(out->copy_frame_out)) return RAD_E_NOKERNEL;

    rad_note(b, "haru codec: %lld stages, base %lld, latent %lld, %s, %lld lane(s)",
             (long long)kCodecStages, (long long)g.base_channels, (long long)g.latent_dim,
             dt_name(g), (long long)g.max_lanes);
    return RAD_OK;
}

int codec_prepare(RadCtx* c, const Codec& k) {
    if (!c) return RAD_E_INVAL;
    auto prep = [&](const ConvLayer& L) {
        if (good(L.iota_cu)) RAD_ISSUE(c, L.iota_cu, RAD_B(L.cu));
    };
    auto prep_block = [&](const BlockLayer& B) {
        if (good(B.iota_cu)) RAD_ISSUE(c, B.iota_cu, RAD_B(B.cu));
    };
    for (const Stack* S : { &k.enc, &k.dec }) {
        prep(S->stem);
        for (int i = 0; i < kCodecRatios; ++i) prep(S->trans[(size_t)i]);
        for (int stage = 0; stage < kCodecStages; ++stage)
            for (int64_t j = 0; j < S->depths[(size_t)stage]; ++j)
                prep_block(S->blocks[(size_t)stage][(size_t)j]);
        prep(S->head);
    }
    return RAD_OK;
}

int codec_reset_lane(RadCtx* c, const Codec& k, int64_t lane) {
    if (!c || lane < 0 || lane >= k.g.max_lanes) return RAD_E_INVAL;
    auto reset_conv = [&](const ConvLayer& L) {
        if (!good(L.fill_state)) return;
        RAD_ISSUE_N(c, L.fill_state, L.context,
                    brow_slice(L.state, lane * L.context, L.context, L.cin));
    };
    auto reset_block = [&](const BlockLayer& B) {
        if (!good(B.fill_state)) return;
        RAD_ISSUE_N(c, B.fill_state, B.context,
                    brow_slice(B.state, lane * B.context, B.context, B.dim));
    };
    for (const Stack* S : { &k.enc, &k.dec }) {
        reset_conv(S->stem);
        for (int i = 0; i < kCodecRatios; ++i) reset_conv(S->trans[(size_t)i]);
        for (int stage = 0; stage < kCodecStages; ++stage)
            for (int64_t j = 0; j < S->depths[(size_t)stage]; ++j)
                reset_block(S->blocks[(size_t)stage][(size_t)j]);
        reset_conv(S->head);
    }
    return RAD_OK;
}

int codec_encode(RadCtx* c, const Codec& k, int64_t lane, bool first, rad_buf audio_in,
                 rad_buf audio_out, int64_t n_samples, int64_t capture_stage) {
    if (!c || lane < 0 || lane >= k.g.max_lanes || n_samples <= 0 ||
        n_samples > k.g.frame_samples)
        return RAD_E_INVAL;
    if (first) RAD_ARCH_TRY(codec_reset_lane(c, k, lane));
    RAD_ISSUE_N(c, k.copy_frame_in, n_samples,
                bcol_at(audio_in, lane, k.g.frame_samples, 0, n_samples, 1),
                brow_slice(k.in_audio, 0, n_samples, 1));
    run_stack(c, k, k.enc, lane, brow_slice(k.in_audio, 0, n_samples, 1), n_samples,
              capture_stage, audio_out);
    if (capture_stage < 0)
        RAD_ISSUE_N(c, k.copy_latent_out, 1, brow_slice(k.out_latent, 0, 1, k.g.latent_dim),
                    bcol_at(audio_out, lane, k.g.frame_samples, 0, k.g.latent_dim, 1));
    return RAD_OK;
}

int codec_decode(RadCtx* c, const Codec& k, int64_t lane, bool first, rad_buf audio_in,
                 rad_buf audio_out, int64_t capture_stage) {
    if (!c || lane < 0 || lane >= k.g.max_lanes) return RAD_E_INVAL;
    if (first) RAD_ARCH_TRY(codec_reset_lane(c, k, lane));
    RAD_ISSUE_N(c, k.copy_latent_in, 1,
                bcol_at(audio_in, lane, k.g.frame_samples, 0, k.g.latent_dim, 1),
                brow_slice(k.in_latent, 0, 1, k.g.latent_dim));
    run_stack(c, k, k.dec, lane, brow_slice(k.in_latent, 0, 1, k.g.latent_dim), 1, capture_stage,
              audio_out);
    if (capture_stage < 0)
        RAD_ISSUE_N(c, k.copy_frame_out, k.g.frame_samples,
                    brow_slice(k.out_audio, 0, k.g.frame_samples, 1),
                    bcol_at(audio_out, lane, k.g.frame_samples, 0, k.g.frame_samples, 1));
    return RAD_OK;
}

int codec_sample(RadCtx* c, const Codec& k, int64_t lane, rad_buf audio_in, rad_buf audio_out) {
    if (!c || lane < 0 || lane >= k.g.max_lanes) return RAD_E_INVAL;
    RAD_ISSUE_N(c, k.copy_sample_mean, 1,
                bcol_at(audio_in, lane, k.g.frame_samples, 0, k.g.latent_dim, 1),
                brow_slice(k.sample_mean, 0, 1, k.g.latent_dim));
    RAD_ISSUE_N(c, k.copy_sample_noise, 1,
                bcol_at(audio_in, lane, k.g.frame_samples, k.g.latent_dim, k.g.latent_dim, 1),
                brow_slice(k.sample_noise, 0, 1, k.g.latent_dim));
    RAD_ISSUE_N(c, k.sample_add, 1, RAD_B(k.sample_mean), RAD_B(k.sample_noise),
                RAD_B(k.out_latent));
    RAD_ISSUE_N(c, k.copy_latent_out, 1, brow_slice(k.out_latent, 0, 1, k.g.latent_dim),
                bcol_at(audio_out, lane, k.g.frame_samples, 0, k.g.latent_dim, 1));
    return RAD_OK;
}

}  /* namespace haru */
