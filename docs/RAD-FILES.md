# Building smaller `.rad` containers with Unsloth Dynamic as the map

> **Unofficial fork.** See [`../UNOFFICIAL-FORK.md`](../UNOFFICIAL-FORK.md). The formats and kernels this relies on are in
> [`FORK-KERNELS.md`](FORK-KERNELS.md); hardware and measured results are in [`STRIX-HALO.md`](STRIX-HALO.md).

A `.rad` container needs every tensor stored in a format the engine has a kernel for. Unsloth's Dynamic ("UD") GGUFs show which tensors
tolerate fewer bits. `tools/ud_map` uses a UD GGUF as a **map** (which format each tensor gets) and builds the `.rad` from the model's
**bf16 weights**, so nothing is dequantised from an already-quantised file unless you choose the exact-mirror policy.

## Policies

`ud_map plan|build --policy ...`:

| Policy | What it does | Lossless w.r.t. the UD file? |
|---|---|---|
| `mirror` | Each tensor in the UD file's own format, as unrotated planes. Needs a kernel for every format the file uses; the build stops on the first one the engine cannot serve. | yes (byte-identical planes; decoded values bit-identical to llama.cpp) |
| `fast` | Maps UD formats onto the fastest served rotated codebooks (`w3nl64a8h`, `w4nl64a8h`, `w5nl64a8h`), all experts rotated. | no |
| `size` | Per-layer: the smallest served format per tensor, keeping the native unrotated form for the few layers where UD's own format is already smaller (IQ2_S/Q2_0). | no |
| `auto` | The tool's default for `build`: `fast` for the `qwen35moe` architecture (Qwen3.6), `mirror` for any other. | depends |

## Pipeline

```sh
python3 tools/ud_map/ud_map.py plan   UD/*.gguf --declared PLAN.json --source BF16_DIR --policy size --recipe-out R.recipe --json P.json
python3 tools/ud_map/ud_map.py build  ...        # plan -> recipe -> rad-convert
python3 tools/ud_map/ud_map.py calibrate ...     # optional GPTQ pass over a calibration batch
python3 tools/ud_map/ud_map.py verify ...        # hashes, per-tensor RMSE, size table, policy verdict
python3 tools/ud_map/ud_map.py score  --mode kld|quality ...
```

`PLAN.json` is the architecture plugin's declared weight list (`rad-convert --plan-only`). `--mtp-variant` chooses how the kept MTP
layer is stored (`keep`, `bf16`, `fp8`, `q8_0`, `w4nl64`, `w5nl64`). Recipes for the Qwen3.6 pilots are in `data/recipes/q36-*.recipe`; the
Flash-Next plan, recipe and fit model are in `fork-notes/flashnext-ud-mirror.*` and `fork-notes/flashnext-ud-fit-runbook.md`.

## Containers we built

### Qwen3.8-Flash-Next (UD v3.1 map), `qwen4exp_fp8`

| Container | Policy | Size | eval60 | 835-prompt English | Notes |
|---|---|---:|---:|---:|---|
| `flash-mirror.rad` | mirror | 86.862 GiB | 59/60 | 777 (93.1%) | UD-faithful; serves one 262144-token sequence, not four |
| `flash-fast.rad` | fast | 85.716 GiB | 59/60 | 781 (93.5%) | all experts rotated |
| `flash-size.rad` | size | **84.37 GiB** | 59/60 | 784 (93.9%) | 44 rotated + 4 native layers; **our recommended one** |

The size policy: IQ3_XXS to `w3nl64a8h`; IQ4_NL and IQ4_XS to `w4nl64a8h`; Q5_0 to `w5nl64a8h`; native IQ2_S/Q2_0 on layers 0, 8, 9, 10; the
token embedding in IQ4_XS; the MTP layer in `w4nl64`; the vision tower in bf16.

KL against the exact container over 1,575,352 positions (8K context, one sequence): fast mean 0.2240 / median 0.0588 / top-1 agreement 83.08%;
size mean 0.2090 / median 0.0547 / top-1 83.53%. The mean and p99 carry a long tail; no paired task loss was observed. This is added loss
**against Unsloth's file**, not against bf16 or FP8, and no FP8 Flash-Next reference was run.

### Qwen3.6-35B-A3B

| Container | Size | Experts | Notes |
|---|---:|---|---|
| W4/W5 RTN (`q36-w4w5-rtn.recipe`) | 21.01 GiB | gate/up `w4nl64a8h`, down `w5nl64a8h`, block-FP8 trunk | smaller than Unsloth's UD (21.65 GiB) and the FP8 container (34.63 GiB) |
| W4/W5 GPTQ (`q36-w4w5-gptq.recipe`) | built, loaded and measured | same, calibrated | |

## Verifier caveat

`ud_map verify` applies one generic RMSE gate. Unsloth's own IQ3_XXS tensors sit at 0.283 RMSE against bf16 and fail it, so the exact container reports
`policy quantised: FAIL`; eight `mtp.attn_hc_*` tensors have no source reference at all. Both are verifier-policy gaps (per-format tolerances, an
"unverifiable" class), not decoder failures. Judge fast/size containers by evaluation, as above.

## Reproducibility and sharing

- Containers are large: 84-87 GiB for Flash-Next, about 21 GiB for Qwen3.6.
- The source weights keep their own licences, and so do Unsloth's quantisation maps and importance matrices; a published `.rad` is a derivative of
  both. Check each upstream licence and attribute it before redistributing a container. `core/server/systemone.cpp` and `data/vectors/` in this
  repository carry their own terms (see `NOTICE`).
- A `.rad` embeds its recipe verbatim, so a recipe option that names a local path (the GPTQ `calib=` directory, for one) travels with the file. The engine ignores it at load. Before sharing,
  search the first 20 MiB (everything before the weight data) for local paths and user names, then replace each occurrence **in place with a string of exactly the same length**: the recipe lives in the
  container's string blob, addressed by offset, so equal-length edits move nothing. Work on a copy, confirm with `cmp -l` that the only differences are those bytes, and load the copy and compare its
  output with the original's at temperature 0. We did this for the Qwen3.6 GPTQ container (4 occurrences, 140 differing bytes, identical output on six prompts including a 10K-token one). The model
  cards for the containers we publish are in [`hf-cards/`](hf-cards/).
- Record, for every published container, the fork commit, the recipe, the source model revision, the UD file revision and the SHA-256 of the `.rad`.
  `ud_map verify` prints the hashes.
