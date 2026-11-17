# Model cards for the published containers

These are the cards (as `README.md`) for the Radiance containers published under the Hugging Face account `jsaigou`. Each card records the container's SHA-256, the source model revision, the
quantisation, the measurements behind it, how to run it, and the license obligations. They are generated from measured facts in this repository's documentation; the hashes are of the exact files uploaded.

| Card | Container | License |
|---|---|---|
| [`Qwen3.6-35B-A3B-W4W5-RTN-radiance.md`](https://huggingface.co/jsaigou/Qwen3.6-35B-A3B-W4W5-RTN-radiance) | 4/5-bit experts, round to nearest, 21.01 GiB | Apache-2.0 |
| [`Qwen3.6-35B-A3B-W4W5-GPTQ-radiance.md`](https://huggingface.co/jsaigou/Qwen3.6-35B-A3B-W4W5-GPTQ-radiance) | 4/5-bit experts, GPTQ, 21.01 GiB (recipe path scrubbed; see `../RAD-FILES.md`) | Apache-2.0 |
| [`Qwen3.8-Flash-Next-UD-Size-radiance.md`](https://huggingface.co/jsaigou/Qwen3.8-Flash-Next-UD-Size-radiance) | Unsloth-Dynamic size policy, 84.37 GiB | Qwen Community License 1.0 |

Published 2026-11-17. The SHA-256 of each `.rad` on the Hub was checked against the hash recorded here and on its card.
