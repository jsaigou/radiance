# ud_map end-to-end on Qwen3.6-35B-A3B (measured)

Repo: `jsaigou/radiance` (PUBLIC, unofficial fork), branch `ud-map-eval`, local only.
Inputs: the UD map `/opt/tohil/models/Qwen3.6-35B-A3B-MTP-UD-Q4_K_XL.gguf`, the bf16
source `/opt/tohil/models/_safetensors-src/Qwen3.6-35B-A3B`, the calibration
directory `~/radiance-test/calib/hessians`, and the existing calibration corpus.

The map is `qwen35moe`; `qwen35moe_fp8` does not serve the GGUF's own K-quant planes,
so `build` uses the `fast` policy (the Route B formats the hand-built `q36-w4w5`
recipes use): routed experts Q4_K -> `w4nl64`, Q5_K/Q6_K/Q8_0 -> `w5nl64`, the dense
Q8_0 trunk -> block FP8, the MTP draft head u2 from `output.weight`.

## Pipeline

```sh
# plan only; writes the recipe, the declared plan and the machine-readable plan JSON
python3 tools/ud_map/ud_map.py build \
  /opt/tohil/models/Qwen3.6-35B-A3B-MTP-UD-Q4_K_XL.gguf \
  --source /opt/tohil/models/_safetensors-src/Qwen3.6-35B-A3B \
  --out /opt/tohil/models/Qwen3.6-35B-A3B-UD-MAP/qwen3.6-35b-a3b-ud-map-rtn.rad \
  --rad-convert build/bin/rad-convert --home build/radiance_home --kernels libhalo --plan-only

# RTN build (CPU conversion)
python3 tools/ud_map/ud_map.py build ... --out ...-rtn.rad --rad-convert ... --home ... --kernels libhalo

# a small calibration run (the arch tap reads RADIANCE_CALIB_DIR, not just --calib-dir)
python3 tools/ud_map/ud_map.py calibrate \
  --model ...-rtn.rad --corpus ~/radiance-test/calib/calib.jsonl \
  --calib-dir /tmp/ud-map-calib --tokens 8192 --calibrate-only \
  --radiance build/bin/radiance --home build/radiance_home --kernels libhalo \
  --max-model-len 8192 --tp 1 --kv-cache-dtype fp8 --max-num-batched-tokens 8192
# radiance: calibration wrote layer Hessians after 10225 tokens; 10320 Gram files

# GPTQ build, reusing the RTN trunk (the provided 1.5M-token Hessians)
python3 tools/ud_map/ud_map.py calibrate --convert-only \
  --model ...-rtn.rad --calib-dir ~/radiance-test/calib/hessians --source <bf16> \
  --gguf <UD.gguf> --out ...-gptq.rad --rad-convert ... --home ... --kernels libhalo

# one command from plan to a verified container (reuse makes the convert a copy)
python3 tools/ud_map/ud_map.py build <UD.gguf> --source <bf16> --out ...-pipeline.rad \
  --reuse ...-rtn.rad --verify --max-experts 8 --rad-convert ... --home ... --kernels libhalo

# verify
python3 tools/ud_map/ud_map.py verify --gguf <UD.gguf> --rad ...-gptq.rad --source <bf16> \
  --plan-json ...-rtn.rad.plan.json --max-experts 8 --out ...-gptq.rad.verify.json

# KL, the existing quality set, and the variations report
python3 tools/ud_map/ud_map.py score --mode kld --reference <FP8.rad> \
  --ref-dir ~/radiance-test/calib/kld-ref --corpus ~/radiance-test/calib/score_prose_code.jsonl \
  --top 20 --candidate rtn=...-rtn.rad --candidate gptq=...-gptq.rad --out-dir /tmp/ud-map-score ...
python3 tools/ud_map/ud_map.py score --mode quality --model ...-gptq.rad \
  --evalset ~/radiance-test/calib/eval60.jsonl --out-dir /tmp/ud-map-quality ...
python3 tools/ud_map/ud_map.py variations <UD.gguf> --declared ...-rtn.rad.declared.json \
  --container rtn=...-rtn.rad --container gptq=...-gptq.rad --measure --out-dir /tmp/ud-map-variations ...
```

## Measured

| | RTN | GPTQ |
|---|---|---|
| container | 22,698,414,080 B = **21.140 GiB** | same size |
| sha256 | `0bcd8fec...aec8` | `7894521d...11c5` |
| worst weight error (`rad-convert --report`) | gate/up **8.65e-2** | gate/up **1.06e-1** |
| verify sample, RMSE vs bf16 source | 0.0292 | 0.0304 |
| verify sample, RMSE vs GGUF dequant | 0.0620 | 0.0642 |
| KLD mean / median | 0.0633 / 0.0104 | **0.0567 / 0.0089** |
| top-1 agreement | 90.68% | **91.27%** |
| PPL ratio (vs FP8 ref) | 1.0167 | **1.0103** |

Plan totals: 611/611 distinct declared targets present; 21.650 GiB in the map's
formats + 0.832 GiB kept from bf16 (the vision tower, which the text GGUF does not
carry) + 0.130 GiB derived (`mtp.draft_head`, u2 from `output.weight`) = 22.612 GiB
of map-size total. MTP layer 40 experts 0.797 GiB at the map's Q8_0, 0.387 at
w4nl64, 0.480 at w5nl64, 1.500 at bf16.

Verify (GPTQ, 1519 weights: every dense/vision/MTP-head weight and 8 experts per
(layer, projection)): policy **PASS**; 675/1519 hash-equal to the bf16 source
(the bf16 1-D/norm/vision/token weights and the 2-bit draft head, which is sampled);
the 333 vision weights are bf16-exact (333/333), the 2-bit draft head is exempt.
The dense trunk is block FP8, the experts `w4nl64a8h`/`w5nl64a8h` rotated.

Quality: `eval60.jsonl` 59/60 correct (24/25 GSM8K, 25/25 MC, 10/10 tool calls),
0 errors.

Variations (2 streams, 200 output tokens; ~6K-token prefill; depth 2):

| | RTN | GPTQ |
|---|---|---|
| decode, one stream | 52.8 tok/s | **59.4 tok/s** |
| decode, aggregate 2 streams | 101.5 tok/s | **104.1 tok/s** |
| 6K prefill TTFT | 1.52 s | 1.51 s |
| tokens/step / acceptance | 3.69 / 0.470 | 3.59 / 0.493 |

The MTP layer's size lever is small at this model: w4nl64 saves 0.410 GiB of the
22.61 GiB map-size total. The vision tower saves 0.414 GiB at fp8 and 0.614 GiB at
w4nl64, and is kept bf16 until an image slice can judge it.

## Flash-Next fast plan (L5; plan-only, no container)

The `fast` emitter now covers the Qwen3.8-Flash-Next UD v3.1 map (base `2a3200f`). Inputs:
the four `Qwen3.8-Flash-Next-UD-Q3_K_XL-*.gguf` shards, the bf16 source, and the declared plan
`fn-bf16.plan.json` (51,488 weights). `plan` reports **1044/1044 distinct declared targets** with
a selected fast dtype; no target falls back to “has no fast format”.

| UD type | fast dtype | UD GiB | fast GiB | served now | dependency |
|---|---|---:|---:|---|---|
| IQ4_NL (PLE n-gram table, 1 row) | `iq4_nl` native | 26.822 | 26.822 | yes (PLE gather) | — |
| IQ4_NL (routed down, 26) | `w4nl64a8h` | 11.426 | 10.474 | yes | — |
| IQ4_XS (routed gate/up, 12) | `w4nl64a8h` | 4.980 | 4.834 | yes | — |
| Q5_0 (routed down, 20) | `w5nl64a8h` | 10.742 | 10.010 | yes | — |
| IQ3_XXS (routed gate/up, 76) | `w3nl64a8h` | 22.729 | 23.193 | **no** | arch: `w3nl64a8h` format (kernel exists) |
| IQ2_S (routed gate/up, 8) | `w3nl64a8h` | 2.002 | 2.441 | **no** | same |
| Q2_0 / TYPE42 (routed down, 2) | `w3nl64a8h` | 0.439 | 0.610 | **no** | same; see family note |
| Q6_K (dense trunk, 213) | `q6_k` native | 2.181 | 2.181 | **no** | L3 dense wiring |
| Q8_0 (dense trunk, 284) | `q8_0` native | 1.344 | 1.344 | **no** | L3 dense wiring |
| IQ4_XS (`token_embd`, 1) | `bf16` (kept) | 0.315 | 1.184 | yes | q8_0 alt = 0.630 GiB after L3 |
| F32 / BF16 (norms, router, qsa_qk) | kept | 0.329 | 0.329 | yes | — |

Totals: map **83.423 GiB** + kept **5.692 GiB** = **89.116 GiB** with the MTP layer kept bf16.
The fast recipe writes the kept MTP layer (`blk.48`) experts at `w4nl64a8h` — a whole bf16 expert
layer is refused by the arch — so the build size is **85.637 GiB** (`-3.479 GiB` for the MTP
experts). That is **3.366 GiB below the 89.003 GiB UD plan** (-3.8%). The MTP layer is at w4 only
on the experts; its dense weights stay bf16.

`fast servable now`: **55.866 GiB (65.2% of the 85.637 GiB build)**. Blocked: **29.771 GiB** =
26.244 (`w3nl64a8h` arch format) + 3.525 (L3 dense Q6_K/Q8_0).

Decisions and deltas:

- **IQ2_S -> `w3nl64a8h`** is the smallest served form that improves on the 2-bit grid: +0.439 GiB
  over the UD IQ2_S. `w4nl64a8h` would be +1.22 GiB; keeping IQ2_S is not servable.
- **IQ3_XXS -> `w3nl64a8h`** costs +0.464 GiB; `w4nl64a8h` would add ~7.9 GiB and miss the
  “at or below UD” target.
- **Q2_0 -> `w3nl64a8h`, not native `q20a16`.** Native is 0.439 GiB (2.25 bpw) and `q20a16` is
  served, but its two tensors are `blk.8`/`blk.10` down projections whose gate/up are IQ2_S.
  `MoeFP8` requires one encoding family per layer, and `q20a16` is unrotated/bf16-activation while
  the codebooks are rotated; the layer is refused. `w3nl64a8h` keeps the layer in the rotated
  family (+0.171 GiB).
- **`token_embd` (IQ4_XS) stays bf16** (1.184 GiB) for the early path; q8_0 is 0.630 GiB once L3
  wires the embedding (-0.554 GiB).
- **PLE table stays native IQ4_NL** (26.822 GiB): the PLE gather kernel reads IQ4_NL. A `w4nl64`
  gather does not exist; if it did, the table would be ~24.6 GiB (-2.2 GiB). Block-FP8 would be
  51.2 GiB, so native IQ4_NL is both the served and the sensible early choice.
- **Q6_K/Q8_0 dense** keep their own formats; the kernels exist but `qwen4exp` does not wire them
  (L3). Mapping them to the production int8 W8A8 trunk would build today at a quality cost; that
  is a lead decision, not the fast recipe's default.

Sanity on Qwen3.6 (the same fast emitter, `qwen35moe`): 20.613 GiB (map 19.652 + vision 0.832 +
derived 0.130), within ~2% of the existing 21.01 GiB `w4nl64-i8lm.rad`; the 0.4 GiB gap is the
MTP layer mapped Q8_0 -> `w5nl64a8h` (the measured L40 Q8_0 0.797 GiB vs w5nl64 0.480 GiB).

Reproduce on Tohil (CPU, read-only):

```sh
python3 tools/ud_map/ud_map.py plan /opt/tohil/models/qwen3.8-flash-next/UD-Q3_K_XL/*.gguf \
  --declared ~/radiance-test/fn-bf16.plan.json \
  --source /opt/tohil/models/_safetensors-src/Qwen3.8-Flash-Next \
  --policy fast --recipe-out fn-fast.recipe --json fn-fast.plan.json
python3 tools/ud_map/ud_map.py build <the same four shards> \
  --source /opt/tohil/models/_safetensors-src/Qwen3.8-Flash-Next \
  --declared ~/radiance-test/fn-bf16.plan.json --policy fast \
  --out fn-fast.rad --dry-run --rad-convert build/bin/rad-convert \
  --home build/radiance_home --kernels libhalo
```

The dry-run emits `fn-fast.rad.recipe` and `fn-fast.rad.plan.json` (per-tensor `fast_dtype`,
`fast_bytes`, `fast_dep`) and prints the convert command. The recipe parses under
`rad-convert --plan-only` on a tiny llama fixture (516 rules; 8 weights quantised, 7 kept;
nothing written).

Tests: `python3 -m unittest tools/ud_map/test_ud_map.py` — 36 tests (2 integration gated on
`UD_MAP_GGUF`/`UD_MAP_DECLARED`); the two integration tests pass on Tohil against the Flash-Next
GGUF + declared plan.

## Flash-Next mirror (UD-faithful) plan (L8; plan-only, no container)

The mirror policy keeps every GGUF tensor's own type; it is the 89.0 GiB UD-faithful target, not
the fast remap. Measured on Tohil (2026-11-15, CPU, ~11 s):

```sh
python3 tools/ud_map/ud_map.py plan /opt/tohil/models/qwen3.8-flash-next/UD-Q3_K_XL/*.gguf \
  --declared ~/radiance-test/fn-bf16.plan.json \
  --source /opt/tohil/models/_safetensors-src/Qwen3.8-Flash-Next \
  --policy mirror --recipe-out fn-mirror.recipe --json fn-mirror.plan.json
```

**1044/1044 declared targets, 1224 mapped tensors, 0 fusion problems, 0 unmapped.** Mapped
83.310 GiB + kept 5.692 GiB = **89.003 GiB** (95,566,087,928 B). By type (GiB): IQ4_NL 38.248,
IQ3_XXS 22.730, Q5_0 10.742, IQ4_XS 5.295, Q6_K 2.181, IQ2_S 2.002, Q8_0 1.344, Q2_0 0.439,
F32 0.292, BF16 0.037. Kept: MTP layer `blk.48` 4.795, MTP head `mtp.*` 0.061, vision `v.*`
0.836, i64 metadata; derived 0. The base build serves 49.758 GiB by format; the rest waits on
L1/L2/L3.

`rad-convert --plan-only` parses the 511 rules and stops on the first missing format:
`blk.0.ffn_gate_up_exps.0.weight` is IQ2_S (`u10`, 1024x8 grid, signs, `u5`) — L1
(`acdc63f`). L2 (`ead15cc`) and L3 (`13ab816`) are also unmerged. After those, the map still
has two non-lane blockers: the IQ4_XS routed experts (4.98 GiB) have no kernel (L2 only adds
the token-embedding gather), and `MoeFP8` serves one encoding per layer except the rotated
w4/w5 family (the fast path's w3 adds a fourth), while the UD map mixes gate/up and down in all
48 layers. The kept bf16 MTP layer (512 experts) is also refused (protected-expert cap 8). The
fast recipe exists to avoid exactly those two constraints. The fit model, levers and lead
runbook are in `fork-notes/flashnext-ud-fit-runbook.md`.

### For the lead: the one arch change this lane needs

`arch/common/rad_block_moe_fp8.h` must declare `w3nl64a8h` (the existing libhalo kernel, W4=5):

- `format_of`: add the case with `e.plane[0].dtype == RAD_U3`, a `table` plane of
  `f32{1x8}` (`w3nl`), `scale` `fp8_e4m3[1x64]`, `scale.1` f32 `[*x*]`, `transform=fwht128`;
  return a new code 13.
- `kServed[13]` -> 14 entries, appending `"w3nl64a8h"`.
- `probe()`: code 13 sets `expert_w4`, `expert_rot`, `expert_nl`, `expert_g64 = 2`,
  `expert_w5 = 0`, and a new `expert_w3 = 1` (without it `declare()`'s `want` expression reads
  `expert_g64 = 2` as w4nl64, code 5).
- `declare()`: the `want` chain tests `expert_w3 ? 13 : ...` before `expert_w5`, and
  `family()` becomes `f == 5 || f == 6 || f == 7 || f == 13` so a layer may mix w3/w4/w5
  codebooks per projection.
- The layout hook and `halo_rows.cpp` already accept `w3nl64a8h`; `halo_layout.cpp`'s
  `w4nl64a8h`-family checks already include it, and `moe_gemm_launch<5>` already dispatches it.

With that and L3 merged, the fast container is buildable; until both land, a conversion stops on
the first `w3nl64a8h` expert or dense `Q6_K`/`Q8_0` weight (the plan's blocked list is 26.244 +
3.525 GiB). No kernel, `tools/ud_repack` or other arch change is needed for this lane.
