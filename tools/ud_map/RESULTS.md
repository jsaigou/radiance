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
