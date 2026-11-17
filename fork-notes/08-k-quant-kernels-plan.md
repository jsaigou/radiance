# K-quant kernels (Q4_K, Q5_K, and what comes with them): research and plan

Status: **research and a gated plan; no kernel code written.** Written 2026-11-17 against `strix-halo-r2`. Labels: **Measured** (a number from our records, source named), **Documented** (a cited file),
**Estimated** (our arithmetic), **Hypothesis** (to be falsified by a measurement the plan schedules). Related: [`../docs/FORK-KERNELS.md`](../docs/FORK-KERNELS.md), [`06-moe-unrot-bench.md`](06-moe-unrot-bench.md),
[`../docs/RAD-FILES.md`](../docs/RAD-FILES.md).

## 1. What K-quants buy, and what they do not

- **They unlock the exact Unsloth Dynamic file for any model that uses them.** Nearly every Unsloth GGUF is built from Q4_K, Q5_K, Q6_K and Q8_0. Today the `mirror` policy of `tools/ud_map` stops at the first format
  without a served kernel, so for those models we can only map to the nearest rotated formats (`w4nl64a8h`, `w5nl64a8h`): a different quantisation, not Unsloth's.
- **They are not a speed win for Qwen3.6.** Our W4/W5 container is already smaller than Unsloth's UD file (21.01 against 21.65 GiB) and decodes within 3 to 8% of llama.cpp while prefilling 2.2x faster (measured today through a router;
  [cards](../docs/hf-cards/)). A K-quant mirror of the same model buys *fidelity to Unsloth's exact weights* and the ability to do the same for other models; it will not beat the rotated container on speed.
- **Cost in bits** (Documented, `radiance-ud-lossless-repack`): Q4_K 4.5 bits per weight against `w4nl64a8h` 4.125 (+9%); Q5_K 5.5 against `w5nl64a8h` 5.125 (+7%). A bandwidth-bound decode reads proportionally more.

## 2. The formats and what is already built

| | Q4_K | Q5_K |
|---|---|---|
| block | 256 values, 144 bytes | 256 values, 176 bytes |
| fields | `d`, `dmin` f16; 8 x (6-bit scale, 6-bit min), one pair per 32 values; 4-bit codes | the same plus `qh`, the fifth bit of each code |
| dequant (f32) | `w = d*sc*q - dmin*mn` | the same, `q` is 5 bits |

- **Container side: done.** `core/format/ggml_planes.h` splits both into planes (codes `u4`/`u5`, `scale` and `min` `u6[1x32]`, `scale.1` and `min.1` f16 per 256); `libquant/lq_ggml.cpp` has the encoder and the plane description;
  `tools/ud_repack` reproduces llama.cpp's own dequantisation bit for bit on real Qwen3.6 tensors (Measured, 2,304 expert slices). So **only the kernels (and the architecture's willingness to declare them) are missing**.
- **Nearest existing kernels (Documented):** the unrotated expert GEMMs for IQ4_XS (two-level scales: `i6` per 32 plus an f16 per 256, third table), IQ4_NL, Q5_0 and Q2_0, and the dense Q6_K and Q8_0 GEMMs.
  A Q4_K expert GEMM is the IQ4_XS one **without the table lookup** (codes are used directly) **plus a min plane**.
- **The min term is cheap.** `sum(w*a) = (d*sc) * sum(q*a) - (dmin*mn) * sum(a)` per 32-value sub-block. The activation block sums are computed once per token, shared by every row and expert. This is the same trick llama.cpp's kernels use.

## 3. Where Radiance's speed comes from, and what could be lost

Measured facts that bound the design:

1. **Decode is bandwidth-bound and the existing A16 expert forms already reach rotated-class bandwidth.** `06-moe-unrot-bench.md`, two-rank slice, decode M=1: `iq4nl32a16` gate_up 107.5 GB/s and `q50a16` 113.5 against
   `w4nl64a8h` 116.2 and `w5nl64a8h` 131.7; on the spread case 130 and 138 GB/s. Streaming ceiling in that run: 204.8 GB/s. **Estimate: a Q4_K decode kernel lands in the `iq4nl32a16` class (same 4.5 bits), so its decode is about
   9% slower per token than `w4nl64a8h` from bytes alone.**
2. **Prefill is where unrotated forms lose.** Same file, M=512: `iq4nl32a16` is +17.8% (gate_up) and +17.9% (down) slower than `w4nl64a8h`; `q50a16` +36.4% and +34.1%. Prefill is the largest Radiance advantage over llama.cpp (2.2x measured), so
   **a naive K-quant port that inherits this would give back a third to half of it.**
3. **Why the A16 forms are slower at prefill (Documented, `halo_moe_gemm_q.hip`):** gfx1151 has **no fp8 matrix instruction**. Every expert kernel dequantises both operands to f16 into LDS and runs an f16 WMMA. The rotated forms stage
   activations as 8-bit E4M3 bytes; the A16 forms stage bf16, **twice the LDS bytes**.
4. **An integer path already exists in libhalo (Documented).** `v_wmma_i32_16x16x16_iu8` is used for the int8 attention scores (`halo_attn_gen.hip`) and by the dense `halo_gemm_w4a8_*` and `halo_gemm_w8a8_*` families (4-bit codes unpacked to bytes at
   staging, int8 activations, an f32 fold per group). It is **not** used by any expert GEMM.

**Design implication.** Two forms, built in this order and kept only if they earn it:
- **A16** (bf16 activations, f16 WMMA): the quick path, reuses the IQ4_XS expert skeleton. Expected: decode in class, prefill 20 to 35% behind the rotated forms.
- **A8** (int8 activations, `iu8` WMMA, int32 accumulate per 32-value sub-block, f32 fold with `d*sc` and the min term): the llama.cpp numerics (it quantises activations to int8 as well, so this is *closer* to the reference than bf16) and
  half the staged bytes. **Hypothesis: it recovers the prefill gap and may beat the f16 forms**, because integer WMMA has twice the f16 matrix throughput on RDNA3-class parts. That is unmeasured on gfx1151 and decides whether A8 is worth building.
  The cost is a more frequent fold: a group of 32 folds four times as often as the existing groups of 128.

## 4. Scope is wider than two kernels (Qwen3.6 as the worked case)

Documented (`radiance-ud-lossless-repack`), the Qwen3.6 UD-Q4_K_XL expert tensors: gate/up Q4_K in layers 0 and 2 to 39 (50.7% of the file's bytes, 78 tensors); down Q5_K in 36 layers (30.2%, 38 tensors); **down Q6_K in layers 1, 34, 38, 39**,
gate/up Q5_K in layer 1, and **Q8_0 experts in the MTP layer**. A full mirror therefore needs **four expert forms** (Q4_K, Q5_K, Q6_K, Q8_0 expert GEMMs), not two, and **per-layer format mixing in the `qwen35moe` plugin** (today only `qwen4exp_fp8`
mixes formats by layer; `qwen35moe_gguf` imports the planes but the engine refuses the unserved encodings by name). The shared MoE block already chooses each projection's format independently; the per-layer configuration would have to be carried over from `qwen4exp_fp8`.

## 5. Plan, with gates

Each phase ends in a measurement; a failed gate stops the phase. All GPU runs under the shared lock, in a scratch build, never on a loaded production slot.

| Phase | Work | Gate (pass to continue) |
|---|---|---|
| 0 | Microbenchmark `iu8` against f16 WMMA on gfx1151 at the expert tile shapes | iu8 sustained throughput >= 1.5x f16. Decides whether A8 is attempted. |
| 1 | **Q4_K expert GEMM, A16 form**; host reference in `libref`/`libavx`; layout hooks; bench in `halo_moe_unrot_bench` | Bit-for-bit across launch forms (as the existing selftests); within 0.4% + 1e-4 of the host decode; **decode >= 90% of `iq4nl32a16` GB/s** at M=1/4/32; prefill within 25% of `w4nl64a8h` |
| 2 | **Q5_K** (Q4_K plus the fifth-bit plane, the `w5` shape) | same as phase 1 against `q50a16` |
| 3 | **A8 form** for Q4_K and Q5_K, if phase 0 passed and A16 prefill missed | **prefill >= `w4nl64a8h` minus 10%**; otherwise drop A8 and keep A16 |
| 4 | Q6_K and Q8_0 expert forms; per-layer mixing in `qwen35moe_gguf`; `ud_map` `mirror` for `qwen35moe` | each form passes the phase 1 checks; a Qwen3.6 mirror container loads |
| 5 | End to end on the Qwen3.6 UD mirror | **KL against llama.cpp running the same GGUF** (the only valid reference for "lossless") mean <= 0.005; decode >= 90% of llama.cpp; prefill >= 1.5x llama.cpp on the 38K-token prompt used for the container cards |

The end-to-end bar is lower than the W4/W5 container's (2.2x prefill) on purpose: the mirror is the *exact-Unsloth option*, and the W4/W5 container stays the fast one.

**Effort (Estimated, focused work, GPU benches included):** phase 0 hours; phase 1 two to four days; phase 2 one to two; phase 3 about a week; phase 4 a week (mixing and two more expert forms); phase 5 two days. Roughly three weeks for the whole mirror, but
**phases 0 to 2 are the useful decision point**: they answer whether K-quants can run at Radiance speed at all, in under a week.

## 6. Decision requested

Approve phases 0 to 2 (kernel and bench work in the fork, GPU time on the reference machine under the shared lock, no host changes). Everything after phase 2 is decided from those measurements.
