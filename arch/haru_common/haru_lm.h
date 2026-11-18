/* haru_lm.h -- Haru's language-model trunk: the Qwen3.5 hybrid (gated delta net and full
 * attention) run on a caller-supplied embedding row, one position a frame.
 *
 * ============================== WHAT IT IS ==============================
 *
 * A dtype-parameterised mirror of the declaration and issue order of three qwen35 blocks:
 *
 *   arch/common/rad_block_attn_gated.h   AttnGatedBF16::declare/step   (qwen35 revision at
 *                                        commit 47a045b; the block is unchanged since)
 *   arch/common/rad_block_gdn.h         GdnBF16::declare/step
 *   arch/common/rad_block_mlp.h         MlpBF16::declare/step
 *
 * It is a copy rather than a call because those blocks declare every linear weight RAD_BF16
 * (rad_block_attn_gated.h:121-133, rad_block_gdn.h:158-178, rad_block_mlp.h:57-60), and the H4
 * tiny checkpoint is float32.  The shared headers are not edited: four shipped plugins include
 * them.  If a block's arithmetic or operand order changes, this file must follow; the line ranges
 * above are what to diff against.
 *
 * What Haru changes from qwen35:
 *   - the input is an embedding row, not a token id.  `b_x` is the caller's to fill: op_embed for
 *     the one prompt row the test drives, plus the projectors in the assembled plugin;
 *   - logits are the block's own f32 buffer (a duplex tenant cannot call rad_declare_logits,
 *     core/build/rad_builder.cpp:2065-2082), computed for the last row of a prompt pass or one row
 *     a lane of a step;
 *   - two optional debug products: the final normed hidden is always `b_h`; a `tap` buffer holds
 *     the residual stream after each layer of a one-row pass, and a two-row f32 view of the
 *     embedding table gives the PAD/EOP logits the reference recomputes in f32.
 *
 * Positions, the KV groups and the qwen35 geometry keys are the ones the design note pins down:
 * docs/HARU-H4-DESIGN.md sections 1.4, 3 and 4.
 *
 * ============================== THE INTERFACE ==============================
 *
 * `declare` names the model once, against the metadata and the build context.  `step_layers` runs
 * the trunk over `batch->n_tok` rows of `b_x` and leaves:
 *
 *   w.h        [n_tok, n_embd]  the final normed hidden, one row an input row
 *   w.logits   [n_lanes, n_vocab] f32, when `Options::logits`, for the last row of a prompt pass
 *              or one row a lane of a step
 *   w.pad_eop  [n_lanes, 2] f32, the same rows through the PAD/EOP rows of the tied table
 *   w.tap      [1, n_layer * n_embd] f32, when `Options::taps`, layer 0 first, of a one-row pass
 *
 * The caller issues its own embed/projector/copy ops into `b_x` before `step_layers`; the block
 * issues no token lookup itself and holds no global state (the plugin owns one `HaruLm` a rank).
 */
#ifndef HARU_LM_H
#define HARU_LM_H

#include <arch/rad_arch.h>

#include <vector>

namespace haru_lm {

using namespace rad::arch;

/* The GDN geometry, as GdnBF16::Config (rad_block_gdn.h:52-64) reads it. */
struct GdnCfg {
    int64_t n_head_k   = 0;
    int64_t head_k     = 0;
    int64_t n_head_v   = 0;
    int64_t head_v     = 0;
    int64_t conv_width = 4;
    int64_t chunk      = 64;

    int64_t k_dim() const { return n_head_k * head_k; }
    int64_t v_dim() const { return n_head_v * head_v; }
    int64_t conv_dim() const { return 2 * k_dim() + v_dim(); }
    int64_t in_dim() const { return conv_dim() + 2 * n_head_v; }
};

struct Options {
    /* Compute the tied logits and the f32 PAD/EOP pair (the H4 test and the assembled plugin). */
    bool logits = false;
    /* Keep the per-layer residual stream of a one-row pass for the tunnel (test only). */
    bool taps = false;
};

struct HaruLm {
    /* ---- geometry */
    Names   nm{""};
    Geom    g{};
    GdnCfg  gc{};
    int64_t n_rot = 0;
    uint32_t wd = RAD_BF16;          /* the checkpoint's own dtype: linears, embedding, conv */
    const char* dt = "bf16";         /* the operand-combination string: "bf16" or "f32" */
    int64_t pad_id = -1, eop_id = -1;

    bool want_logits = false, want_taps = false;
    int64_t tap_cols = 0;            /* n_layer * n_embd when taps are on */

    /* ---- state groups: one paged full-attention group and the GDN's two */
    rad_kvgroup kv_full = 0, kv_state = 0, kv_conv = 0;

    /* ---- the residual stream, the vocabulary edges and the debug products */
    rad_buf b_x = 0, b_h = 0;
    rad_buf b_hf32 = 0, b_logits = 0, b_pad_eop = 0, b_tap = 0;

    /* ---- full attention and gated delta net activations, shared by every layer of a kind */
    rad_buf b_qg = 0, b_k = 0, b_v = 0, b_q = 0, b_gate = 0, b_attn = 0;
    rad_buf b_in = 0, b_z = 0, b_gq = 0, b_gk = 0, b_gv = 0;
    rad_buf b_gdec = 0, b_beta = 0, b_kkt = 0, b_go = 0;
    rad_buf b_gate_up = 0, b_ffn = 0;

    rad_weight w_tok = 0, w_out_norm = 0, w_pad_eop = 0;
    rad_op op_embed = 0, op_embed_ar = 0, op_final_norm = 0;
    rad_op op_hf32 = 0, op_logits = 0, op_pad_eop = 0, op_tap = 0;

    struct Attn {
        rad_weight w_norm = 0, w_qg = 0, w_k = 0, w_v = 0, w_q_norm = 0, w_k_norm = 0, w_o = 0;
        rad_op op_norm = 0, op_qg = 0, op_kp = 0, op_vp = 0;
        rad_op op_q_norm = 0, op_k_norm = 0, op_rope_q = 0, op_rope_k = 0;
        rad_op op_attn = 0, op_kv_store = 0, op_sigmoid = 0, op_gate = 0;
        rad_op op_o = 0, op_ar = 0, op_add = 0;
    };
    struct Gdn {
        rad_weight w_norm = 0, w_in = 0, w_z = 0, w_conv = 0;
        rad_weight w_a_log = 0, w_dt_bias = 0, w_out_norm = 0, w_out = 0;
        rad_op op_norm = 0, op_in = 0, op_z = 0;
        rad_op op_conv_prep = 0, op_kkt = 0, op_scan = 0, op_gnorm = 0;
        rad_op op_conv_update = 0, op_recur = 0;
        rad_op op_out = 0, op_ar = 0, op_add = 0;
    };
    struct Mlp {
        rad_weight w_norm = 0, w_gate_up = 0, w_down = 0;
        rad_op op_norm = 0, op_gate_up = 0, op_act = 0, op_down = 0, op_ar = 0, op_add = 0;
    };
    struct Layer {
        int full = 0;
        int layer = 0;   /* the model layer index, for rad_bind_layer_kv and kv_cache */
        Attn a;
        Gdn  d;
        Mlp  m;
    };
    std::vector<Layer> layers;

    /* Name the model once.  The caller must have declared no weight or op that shares these names;
     * the block declares its own name maps before every weight it names. */
    int declare(RadBuilder* b, const RadModelMeta* meta, const RadBuildCtx* ctx,
                const Options& opts);

    /* Run the trunk over `batch->n_tok` rows already in `b_x`, and fill `b_h` (and the logits/tap
     * products when declared).  No allocation, no host work. */
    void step_layers(RadCtx* c, const RadBatch* batch) const;

private:
    int declare_attn(RadBuilder* b, int l, Layer& lay);
    int declare_gdn(RadBuilder* b, int l, Layer& lay);
    int declare_mlp(RadBuilder* b, int l, Layer& lay);

    void step_attn(RadCtx* c, const RadBatch* batch, const Layer& lay) const;
    void step_gdn(RadCtx* c, const RadBatch* batch, const Layer& lay) const;
    void step_mlp(RadCtx* c, const RadBatch* batch, const Layer& lay) const;
};

}  /* namespace haru_lm */

#endif /* HARU_LM_H */
