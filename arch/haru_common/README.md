# arch/haru_common: the Haru plugin's building blocks

Shared sources for the `haru` architecture plugin (a full-duplex speech-to-speech model: Qwen3.5-4B hybrid LM,
two audio projectors, a MeanFlow head and a VibeVoice causal codec). **No `CMakeLists.txt` here on purpose**: every
`arch/*` directory that has one is built and installed as a plugin and scanned by every engine start; this directory
follows `arch/hyv3_common`, which is compiled into other targets by path. A block is compiled into the test plugin
that checks it (`tests/haru_plugins/<name>/`) and, once assembled, into `arch/haru/` (not created until the blocks
pass). Plan: `docs/DUPLEX-PLUGIN-ABI.md` here, and the Haru plan in the operator's Forge repo.

One package owns each file, so parallel work does not conflict:

| Files | Block |
|---|---|
| `haru_codec.{h,cpp}` | VibeVoice causal codec, encoder and decoder, streaming per-lane state |
| `haru_head.{h,cpp}`, `haru_proj.{h,cpp}` | MeanFlow head; the two audio-input projectors |
| `haru_map.{h,cpp}` | checkpoint tensor name map and conversion recipe helpers |
| `haru_lm.{h,cpp}` | language model trunk (reuses the qwen35 hybrid blocks) |

Goldens for all of them are in `tests/data/haru/` (a tiny random-weight Haru run on CPU by the public reference).
