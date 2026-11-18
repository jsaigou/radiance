# Haru H5: the assembled `haru` duplex architecture plugin

Status: design note for STEP 1. **No code is written in this step.** This note describes the plugin in
`arch/haru/` that assembles the verified Haru blocks behind one duplex tenant, the per-pass dataflow,
the one-ahead frame pairing, the host-sampling ABI addition (ABI 39), the state and lane lifecycle,
the block/engine interfaces that do not fit the assembly, and the test plan. Every claim about the
reference has a citation into the read-only copy under `.haru-ref/haru/`; every claim about this tree
cites a file and line. I cannot build or run the Linux C++ tree on this host, so nothing below is a
test result.

Reference and trees:

* reference package: `.haru-ref/haru/` (the pinned `haru-speech-v0.1.1` copy; read-only);
* real geometry: `.haru-ref/real-config.json`;
* tiny checkpoint and goldens: `tests/data/haru/` (`golden.safetensors`, `golden_meta.json`,
  `haru_tiny/`);
* blocks: `arch/haru_common/haru_{lm,proj,head,codec}.{h,cpp}`;
* duplex ABI and driver: `abi/rad_builder.h`, `abi/rad_runtime.h`, `core/gen/duplex_driver.*`,
  `core/server/duplex.cpp`;
* contract: `docs/DUPLEX-ENGINE.md`; ABI rationale: `docs/DUPLEX-PLUGIN-ABI.md`.

The decisions D1-D3 in the package brief are taken as given: host-side text sampling, drop model
frame 0, and the plugin's ChatML system template. This note does not reopen them; it says how the
assembly implements them.

---

## 0. Summary

The plugin is one duplex tenant named `haru`. A lane is one conversation. `open` runs a PROMPT pass
over the character-card tokens and the reference-frame rows and computes model frame 0; frame 0 is
computed and decoded (so the causal decoder state is correct) but is **not sent**. Every wire STEP
consumes one user frame and emits the next model frame, so the reply to wire step `s` (zero-based)
is model frame `s + 1`. The plugin writes the current model row's text logits and the f32 PAD/EOP
pair; the host applies DRY, PAD/EOP replacement, `eop_bias`, temperature/top-k/top-p, and feeds the
sampled token back as the next step's text input. The plugin never samples text and never owns an
RNG.

The first code iteration may use a test-only teacher feed for the token and the two noise streams;
the plugin reads the same ABI buffers in both modes. The production host replaces that feed with its
sampler and a deterministic, seed-derived noise stream.

---

## 1. What is assembled

`arch/haru/` is a normal architecture plugin (`RAD_ARCH_PLUGIN`/`RAD_ARCH_PROBE`, as
`tests/duplex_plugin/radtest_duplex.cpp:198-201`), compiled with the four block sources by path (the
`arch/hyv3_common` pattern; `arch/haru_common/README.md:1-8`). It declares, in this order:

1. `haru_lm::HaruLm` (`arch/haru_common/haru_lm.h:79-154`) with `Options::logits = true` and
   `Options::taps = false` in production. The block declares the tied embedding
   (`lm.w_tok`), the final norm, the PAD/EOP two-row view, the three KV groups and the three
   hybrid block kinds (header `:92-135`; issue `haru_lm.cpp:761-796`). Its `b_x` is the caller-filled
   input row set (`haru_lm.h:35-45`).
2. `rad::arch::HaruProj` (`haru_proj.h:40-68`) with the LM's `lm.w_tok` handle passed in
   (`declare(b, tokens)`), so there is one embedding table. It owns `latent_scale`, `latent_bias`,
   the two projectors and the serial scaling/embedding-sum.
3. `rad::arch::HaruHead` (`haru_head.h:63-108`), one MeanFlow step per model frame.
4. `haru::Codec` (`haru_codec.h:102-145`), the causal encoder/decoder and its per-lane state.

The declaration (`RadDuplexDecl`, `abi/rad_builder.h:1619-1661`) states:

* `name = "haru"`, `text = 1`;
* `sample_rate = 24000`, `channels = 1`, `samples_per_frame = 3200`, `latent_dim = 64`,
  `frame_rate_num/den = 15/2` (`.haru-ref/real-config.json:6-12`);
* `max_lanes` from the deployment; `max_prompt` from the card plus the reference frame bound;
* `max_ref_frames`, `ref_rate = 24000`, `ref_hop = 3200`;
* `n_audio_in = 2` with `source = RAD_DUPLEX_AUDIO_USER` and `RAD_DUPLEX_AUDIO_MODEL`,
  `in_dim = 64`, `hidden = out_dim =` the LM hidden (2560 real, 64 tiny), `rms_norm = 1` (the
  declaration is descriptive; the projector is the `HaruProj` block);
* `audio_in`, `audio_out` f32 `[max_lanes, 3200]`, `text_out` i32 `[max_lanes]`,
  `status` i32 `[max_lanes]`, all `RAD_BUF_PERSIST` device buffers, plus the ABI 39 host-sampling
  buffers of section 6.

The probe sets `gen_duplex = 1`, `gen_rows` to the largest pass (`max_prompt` or one row a lane),
and `gen_ctx` to the lane bound (the tiny test uses 128; the real 30-minute lane is 13,500 rows).
`gen_audio` stays 0: this is a tenant, not an audio generator (`abi/rad_builder.h:1807`).

The tiny geometry comes entirely from `haru_tiny/config.json`: LM hidden 64, vocab 512, 4 layers
(3 linear + 1 full), head hidden 64/64 with 2 layers, codec base 1 with all seven stages
(`tests/data/haru/README.md:73-113`). The real geometry comes from `.haru-ref/real-config.json`.

---

## 2. Timing: one model frame ahead, and what "drop frame 0" means

Use zero-based wire/index notation:

* `prompt_length = P` (`P = 37` in the golden: 23 card ids + 14 reference rows).
* The PROMPT pass runs positions `0 .. P-1` and ends with hidden `H[P-1]`.
* From `H[P-1]` the reference samples **token 0** and the head produces **latent 0**, then decodes
  **audio 0**. `start()` returns frame 0 (`session.py:97-126`, `_emit` at `:145-178`).
* Wire step `s` (`s = 0, 1, ...`) carries user frame `s`. Its LM row is position `P + s` and uses
  token `s`, model latent `s` and user latent `s`; it produces model frame `s + 1`
  (`session.py:131-143`, `:145-178`; `tests/data/haru/README.md:133-139`).
* The reply to wire step `s` is model frame `s + 1`. Model frame 0 is never on the wire. D2's
  "user frame `t` is answered by model frame `t+1`" is the same statement with zero-based user
  frames.

Frame 0 is **computed, not skipped**. The codec is causal: the reference decodes audio 0 before
audio 1, and the decoder state after frame 0 is part of the input to frame 1
(`session.py:159-163`; `haru_codec.h:111-135`). The PROMPT pass therefore:

1. runs the head on `H[P-1]` with frame 0's head noise to get normalized latent 0;
2. unscales it to raw codec space;
3. decodes it through the streaming decoder into a discard buffer, advancing the decoder state;
4. does **not** write it to `audio_out` and does not send it.

The first STEP then decodes frame 1 with the decoder state left by frame 0. Skipping the frame 0
decode would make every reply's audio wrong even though the LM rows were right.

| quantity | PROMPT | STEP `s` |
|---|---|---|
| LM positions | `0 .. P-1` | `P+s` |
| text token used | `prompt.tokens[t]` (PAD/EOP on reference rows) | `text_in[lane]` (token `s`) |
| model latent used | reference latents on reference rows, zero on card rows | raw latent `s` (plugin state) |
| user latent used | none (mask 0) | codec-sampled user frame `s` |
| model frame produced | frame 0 (discarded, decoded) | frame `s+1` (sent) |
| text token sampled by host | token 0 (not sent) | token `s+1` (sent as this reply's text) |

---

## 3. PROMPT pass

### 3.1 Inputs and row construction

The driver builds one lane's PROMPT (`core/gen/duplex_driver.cpp:733-807`):

* `batch->n_tok = T`, `token_ids[0..T)`, `positions[i] = i`, `cu_seqlens = {0, T}`, the full-KV
  slot mapping and block table;
* `dup_b->lane_h[0]`, `dup_b->n_tok = T`, `dup_b->last_row_h[0] = T-1`;
* `dup_b->ref_audio` (flat device f32), `ref_cu` (clip bounds), `ref_frame[t]` per row (`-1` card,
  `>= 0` reference frame), `n_ref` clip count, `ref_samples`;
* ABI 39: `codec_noise` as `T` rows for the reference rows, `head_noise` as one row for frame 0.

The server renders the plugin template (`core/server/duplex.cpp:262-283`). For Haru:

```
tmpl = "<|im_start|>system\n{card}<|im_end|>\n{references}"
```

The literal template text is tokenised with special-token parsing on, and the card text with it off
(`core/server/duplex.cpp:262-266, :280`). That is exactly D3 and the reference's
`system_prompt_ids`: the wrappers become the tokenizer's special ids and the card is ordinary BPE
(`prompt.py:46-50`). For the golden card `"You are Haru."` this is the 23 ids at
`prompt.tokens[0..23)`. The server appends one placeholder row per reference frame with token 0
(`core/server/duplex.cpp:273-278`); the plugin must replace those with the reference's PAD/EOP
tokens:

* `n_frames = ceil(ref_samples / ref_hop)`, one clip assumed;
* `token = (ref_frame[t] == n_frames - 1) ? eop_id : pad_id`; for the golden, 13 PAD and one EOP
  (`prompt.py:53-54`, `tests/data/haru/README.md:126-128`).

The reference frames are stored in one flat `ref_audio` buffer with `ref_cu` bounds. The codec
encode entry point expects a lane-indexed audio buffer (`haru_codec.cpp:607-617`), so the plugin
copies each frame into a lane-indexed work buffer as it processes the row. Frame `f` length is
`min(ref_hop, ref_cu.back() - f*ref_hop)`; the final golden frame is 2,400 samples of the 44,000
sample clip, not 3,200 with an audio-zero tail. The engine/server reference staging currently pads
the decoded clip to a whole hop and the driver requires `ref_cu` bounds to be multiples of `hop`
(`core/server/duplex.cpp:248-256`, `core/gen/duplex_driver.cpp:424-429`). That is a blocker for the
44,000-sample golden and is named in section 8.

### 3.2 Per-row embedding and the LM

For each row `t` in order:

* `ref_frame[t] == -1`: `model_latent = 0`, `user_latent = 0`, `model_mask = 0`,
  `user_mask = 0`, `token = batch->token_ids[t]`.
* `ref_frame[t] >= 0`:
  * encode the reference frame with the streaming encoder (`codec_encode`, `first` only on the first
    reference frame) into the latent means;
  * add the host-supplied combined codec noise (`codec_noise[t]`) with `codec_sample` to get the
    raw reference latent (`session.py:140`; `prompt.py:97-99`; noise meaning below);
  * `token = PAD/EOP` as above, `model_latent = raw reference latent`, `model_mask = 1`,
    `user_latent = 0`, `user_mask = 0`.

`HaruProj::issue` is one position (`haru_proj.h:63-67`, `haru_proj.cpp:135-175`). The plugin issues it
once per prompt row, writing into row `t` of `lm.b_x`. This is the reference's per-position sum
(`haru.py:83-107`; `prompt.py:119-123`). Then one `lm.step_layers(c, batch)` runs the whole PROMPT
(`haru_lm.h:143-144`, `haru_lm.cpp:761-796`), filling `lm.b_h` and, for the last row only, `lm.b_logits`
and `lm.b_pad_eop` (`haru_lm.cpp:780-793`).

The masks matter: a card row must not receive either audio projection, and a reference row receives
only the model projection (`prompt.py:118-124`). The plugin needs persistent `[1]` f32 zero/one
buffers for the mask operands (`HaruProj::issue` takes one-scalar views).

### 3.3 End of PROMPT: frame 0 and the host token

After `step_layers` the plugin:

1. copies `lm.b_logits` row 0 to `logits_out[lane]` and `lm.b_pad_eop` row 0 to
   `pad_eop_out[lane]`; these are frame 0's text row and PAD/EOP pair
   (`session.py:148-151`);
2. runs `HaruHead::issue` on `lm.b_h[T-1]` with `head_noise[0]` and the guidance scalar, getting
   normalized latent 0;
3. unscales it (`z / latent_scale - latent_bias`, `haru.py:71-76`) into
   `prev_latent[lane]` (raw codec space);
4. resets the codec lane and decodes `prev_latent[lane]` into a discard buffer, advancing the
   decoder state (section 2).

The host reads `logits_out[lane]`/`pad_eop_out[lane]` after the pass, samples token 0, and writes
it to `text_in[lane]` before the first STEP. The PROMPT pass writes no `audio_out`, `text_out` or
`status`; the driver does not read them on PROMPT (`core/gen/duplex_driver.cpp:967-978`).

Codec reset order is important. The reference encodes the reference clip on a separate whole-clip
call and decodes live audio on a streaming codec (`session.py:105-107`, `prompt.py:97-99`). The
assembled block has one state arena, so the plugin resets it twice in PROMPT: once before encoding
the reference frames, and once after them, before decoding frame 0. The second reset discards the
reference encoder history and starts the live encoder/decoder from zero, exactly as the reference's
streaming codec starts. There is no reset on `RAD_DUPLEX_F_FIRST`; see section 5.

---

## 4. STEP pass

The driver stages one row per ready lane (`core/gen/duplex_driver.cpp:809-895`). It zeroes the whole
declaration `audio_in`, `audio_out`, `text_out` and `status` buffers, copies each lane's 3,200 wire
samples to `audio_in[lane]`, sets `positions[i] = j.pos`, `cu_seqlens[i] = i`, and sets
`frame_h[i] = j.frame` and `RAD_DUPLEX_F_FIRST` when `j.frame == 0`
(`core/gen/duplex_driver.cpp:828-863`). ABI 39 adds the text/noise staging.

For each entry `i` in the pass, with `lane = dup_b->lane_h[i]` (never assume `i == lane`):

1. **Codec encode user frame.** `codec_encode(c, codec, lane, /*first=*/false, audio_in,
   audio_out, 3200, -1)` reads `audio_in[lane]`, writes the 64 means to the first 64 floats of
   `audio_out[lane]`, and advances the encoder state (`haru_codec.cpp:607-622`).
2. **User latent sample.** Put the mean and `codec_noise[i]` where `codec_sample` expects them and
   add: the block's `codec_sample` reads mean at offset 0 and noise at offset `latent_dim` of one
   lane-indexed buffer and writes the raw user latent to `audio_out[lane]`
   (`haru_codec.cpp:640-655`). The combined noise is `scale * (latent_std/0.8) * raw`
   (`vibevoice_codec.py:196-200`; `tests/data/haru/README.md:140-145`).
3. **Embed.** Issue `HaruProj::issue` for this lane into row `i` of `lm.b_x`:
   `token = text_in[lane]`, `model_latent = prev_latent[lane]`, `user_latent = user latent`,
   `model_mask = user_mask = 1` (`haru.py:83-107`; `session.py:140-142`).
4. **LM step.** Call `lm.step_layers(c, batch)` over the pass's `L` rows. It writes `lm.b_h` and,
   for a STEP, one logits row per lane (`haru_lm.cpp:780-793`).
5. **Logits out.** Copy `lm.b_logits` row `i` to `logits_out[lane]` and `lm.b_pad_eop` row `i` to
   `pad_eop_out[lane]`. The host samples token `s+1` from them.
6. **Head.** Issue `HaruHead::issue` on `lm.b_h` row `i` with `head_noise[i]` and guidance, getting
   normalized latent `s+1`; unscale to raw and write `prev_latent[lane]`.
7. **Codec decode output frame.** Write the raw latent into the first 64 floats of a lane-indexed
   work buffer and `codec_decode` it with `first=false` into `audio_out[lane]`, advancing the
   decoder state (`haru_codec.cpp:624-638`).
8. **Clamp and status.** Clamp the 3,200 samples to `[-1, 1]` (`session.py:163`) and write
   `status[lane] = 0` (audio present). A silent frame is still a frame here; status 1 is not used.

The plugin does not write `text_out` when `host_sampling = 1`; the host's sampled token is the
reply's text (and the next step's `text_in`). With plugin-side sampling (not used by Haru), the old
`text_out` path remains.

### 4.1 Exact row layout in the declaration buffers

All declaration buffers are lane-indexed, not entry-indexed:

| buffer | PROMPT | STEP |
|---|---|---|
| `audio_in` | not staged; the plugin must not require it (the driver does not upload it) | driver copies the wire frame to `audio_in[lane*3200 .. lane*3200+3200)` |
| `audio_out` | not read; do not rely on it | plugin writes model frame `s+1` to `audio_out[lane*3200 .. +3200)`; unused lanes stay zero |
| `text_in` | not used | host writes token `s` at `text_in[lane]` before the pass |
| `logits_out` | plugin writes frame 0's row at `logits_out[lane*n_vocab .. +n_vocab)` | plugin writes frame `s+1`'s row at the same place |
| `pad_eop_out` | plugin writes PAD/EOP at `pad_eop_out[2*lane .. 2*lane+2)` | same |
| `text_out` | not used | ignored when host sampling is on |
| `status` | not read | plugin writes `status[lane] = 0` |

The batch's `codec_noise` and `head_noise` are entry-indexed: `codec_noise` has `n_tok` rows on
PROMPT (only rows with `ref_frame[t] >= 0` are read) and `n_lane` rows on STEP; `head_noise` has
`n_lane` rows on both passes. The plugin uses `i` for these and `lane_h[i]` for the declaration
buffers.

The PROMPT pass may use a private lane-indexed work buffer for reference audio and the frame 0
decode. Using `audio_out` as PROMPT scratch would also work because the driver does not read it
then, but a private `[max_lanes, 3200]` f32 buffer is clearer and keeps the wire buffer semantics
unambiguous.

---

## 5. State, lane lifecycle, and sizes

### 5.1 Engine state

The plugin declares the three hybrid groups exactly as `qwen35` does (`docs/HARU-H4-DESIGN.md:208-254`):

| group | tiny | real |
|---|---|---|
| `RAD_KV_FULL` | 1 full layer, f32, 2 KV heads, head 64 | 8 full layers, bf16, 4 KV heads, head 256 |
| `RAD_KV_LINEAR` | 3 GDN layers, f32, 4 value heads, state `{16,16}` | 24 GDN layers, f32, 32 value heads, state `{128,128}` |
| `RAD_KV_CONV` | 3 GDN layers, f32, 8 heads, width 4 | 24 GDN layers, bf16, 64 heads, width 4 |

The driver accepts exactly one full-attention group plus any number of linear/conv groups
(`docs/HARU-H4-DESIGN.md:220-230`; `core/gen/duplex_driver.cpp:198-220`). It stages one block-table
word and one state-index row per lane (`core/gen/duplex_driver.cpp:708-731`).

### 5.2 Per-lane sizes

From `docs/HARU-H4-DESIGN.md:232-249` and `docs/HARU-H1-DESIGN.md:200-202, 448-454`:

| item | tiny / lane | real / lane |
|---|---:|---:|
| full-attention KV at the bound | ~131 KB at 128 rows | ~422 MiB at 13,500 rows (844 MiB at 27,000) |
| GDN recurrent state | 12,288 B | 48 MiB |
| GDN conv state | 4,608 B | ~1.1 MiB |
| codec persistent state | 2,828 floats = ~11 KB f32 | 298,822 floats = 1.20 MB f32 / 0.60 MB bf16 |
| plugin `prev_latent` + guidance | 260 B | 260 B |
| plugin work buffer `[max_lanes, 3200]` | 25.6 KB for 2 lanes | 25.6 KB for 2 lanes |
| host DRY history (`lookback 1024 + 32 + 1` ints) | ~4.2 KB | ~4.2 KB |

The real lane is dominated by the full-attention KV and the codec, not by the assembly glue. The
engine reserves the whole `max_ctx` at open (`core/gen/duplex_driver.cpp:455-459`), and the state
pool is sized by `max_seqs`, not `max_lanes` (`docs/HARU-H4-DESIGN.md:244-249`), so a deployment
must set both the lane count and the context bound deliberately.

### 5.3 Open, close, reset, flush

* **open.** The driver allocates one KV sequence for `max_ctx`, the state slots for the three
  groups, and a free lane index (`core/gen/duplex_driver.cpp:400-475`). The PROMPT pass then resets
  the codec lane, encodes the reference frames, resets again, computes frame 0, decodes and discards
  it, and writes frame 0's logits/PAD-EOP. The host samples token 0 and the open ticket becomes
  ready only after the whole PROMPT pass (`core/gen/duplex_driver.cpp:967-978`).
* **reset.** PROMPT is the reset point, not `RAD_DUPLEX_F_FIRST`. The engine has no plugin close
  callback (`core/gen/duplex_driver.cpp:557-589`), and a lane index is reused after close
  (`:462-463`). Resetting in PROMPT means a reopened index never sees the previous conversation's
  codec history. `FIRST` only marks the first live step; the plugin must not call
  `codec_reset_lane` there, or it would erase the decoder state advanced by frame 0.
* **close.** The driver refuses or cancels outstanding work, then frees the sequence, state slots
  and lane index; plugin-private buffers are not touched. A close during a pass is deferred to
  `done()`, so no pass sees a half-freed lane (`core/gen/duplex_driver.cpp:557-589, 960-1027`).
* **flush.** The current driver returns one reply with `flags = 0` and changes no state
  (`core/gen/duplex_driver.cpp:521-555`). It is not a reset and not a step. If a later product
  requires flush to return buffered audio, that is a driver/pass change, not an assembly change.

---

## 6. Proposed ABI 39: host-side text sampling and host noise

### 6.1 Why a new ABI version

The current ABI is 38 (`abi/rad_abi.h:22`; `tests/abi_layout_test.cpp:62`). The extra state cannot
live in a plugin-private buffer alone: the host must read the logits/PAD-EOP row and write the
sampled token and noise, and the plugin must read them through the same declaration the driver
stages. Appending fields to `RadDuplexDecl` and `RadDuplexBatch` changes their size and the layout
fingerprint, so `RAD_ABI_VERSION` must move to 39 (`abi/rad_abi.h:15-22`; `docs/PLUGIN.md:45-56`).
Existing structs' offsets do not move because every field is appended.

### 6.2 `RadDuplexDecl` additions

Append after `status` (`abi/rad_builder.h:1658`), in this exact order:

```c
typedef struct RadDuplexDecl {
    /* ... existing fields through `status` ... */

    /* ---- ABI 39: host-side text sampling ---- */
    int64_t     n_vocab;        /* > 0 when logits_out is declared */
    int32_t     host_sampling;  /* 1: the host samples logits_out and writes text_in */
    int32_t     pad_id;         /* the PAD token id, or -1 */
    int32_t     eop_id;         /* the EOP token id, or -1 */
    int32_t     eos_id;         /* the EOS token id, or -1 (DRY-exempt; banned only for force-text) */
    rad_buf     logits_out;     /* f32 [max_lanes, n_vocab], device */
    rad_buf     pad_eop_out;    /* f32 [max_lanes, 2], device */
    rad_buf     text_in;        /* i32 [max_lanes], device; host writes */
} RadDuplexDecl;
```

Ownership and sizes:

* `logits_out`, `pad_eop_out` and `text_in` are declared by the plugin with `decl_b` as
  `RAD_BUF_PERSIST` device buffers, exactly like `audio_out`/`status`. The program owns the
  allocation; the plugin keeps the handles. The driver writes `text_in` and reads the other two; the
  plugin writes `logits_out`/`pad_eop_out` and reads `text_in`.
* `logits_out` is `max_lanes * n_vocab * 4` bytes (real: ~0.99 MB a lane at 248,320 ids; ~2.0 MB
  for two lanes). `pad_eop_out` is `max_lanes * 2 * 4` bytes. `text_in` is `max_lanes * 4` bytes.
* For Haru, `pad_id = 248076`, `eop_id = 248071`, `eos_id = 248044`
  (`.haru-ref/real-config.json:11-12, 83`); the tiny ids are 480, 481, 500
  (`tests/data/haru/README.md:80-81`).

Validation in `Builder::declare_duplex_tenant` follows the existing buffer check
(`core/build/rad_builder.cpp:2169-2190`) and adds:

* `host_sampling` is 0 or 1; 1 requires `text = 1`, `n_vocab > 0`, all three handles valid,
  device, and big enough; `pad_id`, `eop_id`, `eos_id` are each `-1` or in `[0, n_vocab)`;
* `n_vocab` is the model's vocabulary and must equal the `n_vocab` the plugin declares for its LM;
  the builder can only check positivity and the buffer size, so the plugin checks equality at
  declare and refuses a mismatch;
* `logits_out` must be f32, `pad_eop_out` f32, `text_in` i32.

### 6.3 `RadDuplexBatch` additions

Append after `flags_h` (`abi/rad_runtime.h:689`), in this exact order:

```c
typedef struct RadDuplexBatch {
    /* ... existing fields through `flags_h` ... */

    /* ---- ABI 39: host-provided noise ---- *
     * PROMPT: codec_noise is [n_tok, latent_dim]; row t is read when ref_frame[t] >= 0.
     *         head_noise is [n_lane, latent_dim]; row i is frame 0's head noise.
     * STEP:   codec_noise is [n_lane, latent_dim]; head_noise is [n_lane, latent_dim];
     *         row i belongs to lane lane_h[i].
     * Both are device pointers; the _h forms are host staging valid only inside next()/the pass. */
    const float* codec_noise;
    const float* codec_noise_h;
    const float* head_noise;
    const float* head_noise_h;
} RadDuplexBatch;
```

`codec_noise` is the **combined** added term `scale * (latent_std/0.8) * raw`
(`vibevoice_codec.py:196-200`; `tests/data/haru/README.md:140-145`), because `codec_sample` adds it
directly (`haru_codec.cpp:640-655`). `head_noise` is the raw noise vector the head starts from
(`meanflow_head.py:117-134`; `session.py:159-162`). The plugin never draws either.

### 6.4 Host sampling algorithm

This is the reference's `_emit` (`session.py:145-178`) moved to the host. For lane `lane`, after the
pass:

1. Copy the f32 row from `logits_out[lane]` (readback).
2. Apply the lane's DRY penalties (`sampling.py:53-77`). The exempt set is `{pad_id, eop_id,
   eos_id}`; exempt tokens never enter the history and are never penalised.
3. Replace the PAD and EOP entries with `pad_eop_out[lane][0]` and `[1]` (the reference's f32
   recomputation; `haru.py:117-123`; `session.py:151`). This happens after DRY.
4. Add `eop_bias` to the EOP entry (`session.py:152-153`).
5. If a `speak_first` opening turn is forcing text, set the banned rows `{pad, eop, eos}` to
   `-inf` and decrement the force counter (`session.py:154-156`). Haru's deployment is
   `speak_first = false`, so this is normally inactive.
6. Sample: `temperature <= 0` is greedy; otherwise divide by temperature, keep top-k (25), softmax,
   keep the top-p mass (1.0), and draw from the lane's RNG (`sampling.py:80-100`).
7. Write the token to the reply and to `text_in[lane]` for the next step. Update the lane's DRY
   history unless the token is exempt.
8. Detokenise host-side. The wire carries UTF-8 text, not token ids, and the reference suppresses
   a run of PAD/EOP into one space and holds a multi-byte character until it is complete
   (`sampling.py:103-128`). A sampled `pad_id`/`eop_id` must not be sent as a visible token; the
   per-lane stream detokenizer decides whether `flags` bit 1 is set. `eos_id` is only banned during
   a forced opening turn; otherwise it detokenises like any other token. The current server decodes
   every token id it is handed (`core/server/duplex.cpp:177-183`), so the host sampler must own this
   stream state or the server must grow it.

The token sampled at PROMPT is token 0 and is only written to `text_in`; it is not a wire reply
(frame 0 is dropped). The token sampled at STEP `s` is token `s+1`; it is both the reply's text and
the next step's input.

`text_out` becomes unused when `host_sampling = 1`. The driver must not read it in that mode, and
the plugin must not write it (or writes `-1`, which the driver ignores). This keeps the existing
plugin-sampling tenants unchanged.

### 6.5 Driver readback and feedback

The driver changes are mechanical but are part of ABI 39's behaviour:

* `build_prompt` also registers readbacks of `logits_out[lane]` (`n_vocab*4` bytes) and
  `pad_eop_out[lane]` (`2*4` bytes), and `done()` samples token 0 and stages `text_in[lane]`.
* `build_step` registers the same readbacks for each participating lane and, in `done()`, runs the
  sampler/teacher feed, fills the reply's token and stages `text_in[lane]` for the next step.
* `upload()` copies `text_in` and the pass's `codec_noise`/`head_noise` to device memory with
  `audio_in` (`core/gen/duplex_driver.cpp:641-654`).
* To avoid copying all `max_lanes` logits rows every step, each readback is per participating lane
  at offset `lane*n_vocab`; `Readback` needs an offset field (or the driver uses a per-lane view).
  This is a driver-internal change, not another struct field.
* `RadDuplexBatch::seed` is staged as two 32-bit words per lane (`core/gen/duplex_driver.cpp:855-856`)
  while the field is `const uint32_t*` (`abi/rad_runtime.h:687`). ABI 39 should either re-document
  `seed[2*i]`/`seed[2*i+1]` as the low/high words or add a `const uint64_t* seed64`; the host sampler
  uses `DuplexRequest.seed` directly, so the plugin need not read either in this design.

### 6.6 Layout table and version files

* `abi/rad_abi.h`: `RAD_ABI_VERSION` 38 -> 39.
* `abi/rad_builder.h`: the `RadDuplexDecl` fields above.
* `abi/rad_runtime.h`: the `RadDuplexBatch` fields above.
* `abi/rad_abi_layout_rows.h`: append `ABI_FIELD(RadDuplexDecl, ...)` rows after `status` and
  `ABI_FIELD(RadDuplexBatch, ...)` rows after `flags_h`; the structs are already listed at
  `:320-338` and `:339-359`.
* `tests/abi_layout_table.h`: regenerated on a machine that can build.
* `tests/abi_layout_test.cpp`: `kTableAbi` 38 -> 39 (`:62`).
* `tests/abi_c_test.c` and `abi/rad_abi_layout.h`: no edit; they follow the rows/version.
* `docs/PLUGIN.md` and `docs/DUPLEX-PLUGIN-ABI.md`: record 39 and the new fields.

No enum value is required for host sampling. The existing `RAD_DUPLEX_F_FIRST`/`FLUSH` bits stay as
they are.

---

## 7. Teacher-forced mode through the same ABI

The first code iteration may not have the host sampler. It still uses the ABI 39 buffers:

* `text_in[lane]` is written from the recorded `frame{t}.token_in` (or `frame0.token_out` for the
  first step) instead of from a sampler.
* `codec_noise` is written from `prompt.ref_noise` (one row per reference row) and
  `frame{t}.user_noise` (one row per step).
* `head_noise` is written from `frame{t}.head_noise` (including frame 0).
* `logits_out`/`pad_eop_out` are still written by the plugin and still read by the test, so the
  host's `logits_final`/`token_out` path can be checked separately.

The plugin reads `text_in`, `codec_noise` and `head_noise` exactly as in production; it has no
teacher flag. The difference is only who fills the buffers: the production driver's sampler and
PRNG, or a test-only feed that replays the golden schedule. The feed is a driver/test seam, not an
ABI field. `tests/haru_harness.h` currently has no way to send tokens or noise over the WebSocket
(`tests/HARU_HARNESS.md:10-16, 58-77`), so the parity test must either drive `DuplexDriver` directly
or use a test-only engine option that reads a schedule file; that is named as a test-harness gap in
section 8.

Production noise, stated explicitly: `DuplexRequest.seed` is the server's
`hash(session_id)` (`core/server/duplex.cpp:290`). The host draws both streams from a per-lane
deterministic PRNG seeded by it, in a documented order (reference rows, then per-frame
head/codec noise). The reference's torch RNG cannot be reproduced, so production audio is not
expected to match the goldens bit-for-bit; the seed exists for replay and determinism, not for
parity.

---

## 8. Block and engine interfaces that do not fit the assembly

These are the concrete gaps. "Blocker" means the tiny golden parity test cannot pass without a
change or a test-only workaround.

1. **Reference partial hop is lost (blocker for the 44,000 golden).** The server pads a decoded
   reference to a whole `ref_hop` and the driver requires `ref_cu` bounds to be multiples of `hop`
   (`core/server/duplex.cpp:248-256`; `core/gen/duplex_driver.cpp:424-429`). The golden reference is
   44,000 samples = 13.75 frames, and zero-padding it as audio is measurably wrong on the last
   latent (`docs/HARU-H1-DESIGN.md:221-251`). Fix: preserve the true sample count (do not pad in the
   server), allow a final partial frame in `ref_cu`, and have the plugin pass
   `n_samples = min(hop, remaining)` for the last frame. This is an engine/server change, not an
   ABI 39 field.
2. **Codec reset semantics (blocker if used as documented).** `codec_reset_lane` resets both the
   encoder and decoder (`haru_codec.cpp:584-605`), and `codec_encode/decode(first=true)` calls it.
   The one-ahead flow needs: reset before reference encoding; reset again before frame 0's decode;
   **no reset on FIRST** (it would erase the decoder state frame 0 produced). The header's "call on
   `RAD_DUPLEX_F_FIRST`" (`haru_codec.h:121-123`) is wrong for the assembly. Fix the documentation
   and/or add encoder-only/decoder-only reset; the assembly can work with the current both-stack
   reset at the two PROMPT points.
3. **`codec_sample` packs mean and noise into one buffer.** It reads mean at offset 0 and noise at
   offset `latent_dim` of one lane-indexed `audio_in` and writes to `audio_out`
   (`haru_codec.h:137-140`; `haru_codec.cpp:640-655`). The assembly must copy the encoder mean and
   the host noise into a lane-indexed scratch for every reference frame and every step. A direct
   `codec_sample(mean, noise, out)` form, or separate mean/noise row arguments, would remove the
   copies. Not a blocker; it costs extra launches.
4. **The codec's internal buffers are one row, the caller's must be lane-indexed.** `in_audio`,
   `out_latent`, `out_audio`, `sample_mean` and `sample_noise` are `[1, ...]` program-wide
   (`haru_codec.cpp:427-432`), while `codec_encode/decode/sample` index the caller's buffer by
   `lane`. The plugin must declare a lane-indexed work buffer `[max_lanes, 3200]`; the internal
   buffers are not a substitute. Documented, not a blocker.
5. **No unscale in the head.** `HaruHead::issue` returns normalized latents
   (`haru_head.h:1-19, 105-108`), and `HaruProj` owns `latent_scale`/`latent_bias`
   (`haru_proj.h:44`) but has no unscale. The assembly needs `raw = z / scale - bias`
   (`haru.py:71-76`). There is no reciprocal/divide op in the current elementwise set
   (`docs/OPS.md:285-292`, `libref/ref_elementwise.cpp:359-380`), so the assembly must either
   derive a reciprocal weight/buffer from the checkpoint or the block must expose an unscale
   helper. The first code iteration can hard-code the tiny/real 0.8/0.1 only as a test workaround;
   production needs a real source. Blocker for a clean production path.
6. **Projector is one position.** `HaruProj::issue` is `M = 1` (`haru_proj.cpp:135-175`), so a
   37-row PROMPT is 37 issue loops (and a real ~1,400-row prompt is worse). It is correct but not
   batched. A `HaruProj::issue_rows` over `T` rows is the missing interface. Not a correctness
   blocker for the tiny test; a performance risk for production.
7. **Reference placeholder token.** The server appends token 0 for every reference row
   (`core/server/duplex.cpp:273-278`); the reference uses PAD for all but the last reference frame
   and EOP for the last (`prompt.py:53-54`). The plugin must synthesise PAD/EOP from `pad_id`/
   `eop_id` and `ref_frame`, ignoring the staged 0. This is an assembly rule; it should be written
   in `docs/DUPLEX-PLUGIN-ABI.md` because a plugin that trusts `token_ids` on reference rows would
   silently train the LM on token 0.
8. **`haru_map.{h,cpp}` is listed but absent.** `arch/haru_common/README.md:15` names it; the
   directory contains only the four block pairs. The blocks declare their own names and name maps
   (`haru_lm.cpp:38-40, 207-229`; `haru_head.cpp:19-23`; `haru_proj.cpp:15-19`), so the assembly
   may not need a separate map, but the README and the file plan disagree. Either add it or remove
   the row.
9. **No plugin close callback.** The engine releases KV/state slots and the lane index, but cannot
   reset plugin-private codec/latent state (`core/gen/duplex_driver.cpp:557-589`). The assembly's
   PROMPT reset is the workaround; this is already the v0 plan (`docs/DUPLEX-PLUGIN-ABI.md:410-429`).
10. **Host sampling does not exist yet.** The driver reads `audio_out/text_out/status` only on STEP
    (`core/gen/duplex_driver.cpp:887-892, 979-1014`), has no PROMPT readback, and has no sampler.
    ABI 39 and the driver changes in section 6 are required for D1. The first iteration's teacher
    feed is the interim.
11. **`text_out` is zero-initialised, not `-1`.** `build_step` memsets it to 0
    (`core/gen/duplex_driver.cpp:830`). A plugin-sampling tenant that forgets to write a lane
    returns token 0, which is a valid id. Host sampling sidesteps this, but the plugin-sampling path
    remains a footgun.
12. **`seed` staging is 64-bit data through a 32-bit pointer.** See section 6.5. The host sampler
    does not need the plugin to read it; a plugin that does would read the wrong lane's high word.
13. **`RadDuplexBatch::last_row` device pointer is not set.** `build_prompt` sets only
    `last_row_h` (`core/gen/duplex_driver.cpp:758-793`); a plugin that reads the device pointer sees
    null. The assembly uses `last_row_h`, so this is a documentation/hygiene gap, not a blocker.
14. **Guidance and head noise have no declaration field.** The head takes a guidance operand
    (`haru_head.h:105-108`); the assembly declares a persistent `[1]` f32 buffer filled with the
    config's 1.5 (`real-config.json:117-129`). A per-lane or runtime guidance is not expressible
    without another ABI field; not required for v0.
15. **Codec state is not engine state.** It is a plugin-private per-lane arena; the engine cannot
    checkpoint, offload or inspect it (`docs/DUPLEX-PLUGIN-ABI.md:425-429`). Acceptable for v0,
    but it is a real limitation for multi-lane memory planning and reconnect/resume.
16. **Flush cannot return a held frame.** The driver returns `flags = 0` and runs no pass
    (`core/gen/duplex_driver.cpp:521-555`). Haru normally emits a frame every step, so this is
    acceptable; a future buffering optimization would need a flush pass.
17. **Text output is not a token stream yet.** The server decodes each `DuplexReply.token`
    independently (`core/server/duplex.cpp:167-184`), while the reference suppresses PAD/EOP runs
    and holds partial UTF-8 (`sampling.py:103-128`). The assembly needs a per-lane host-side stream
    detokenizer, or the server does; otherwise silence tokens and split characters leak into the
    reply. D1 makes the host the sampler, so the host is the natural owner.
18. **Template segments are tokenised separately.** The server tokenises the literal template
    around `{card}` and `{references}` and the card text separately
    (`core/server/duplex.cpp:262-283`), while the reference tokenises the whole
    `<|im_start|>system\n{card}<|im_end|>\n` string in one call (`prompt.py:46-50`). With the tiny
    byte-level tokenizer (empty merge table) the two agree; a real BPE tokenizer could merge across
    the card/literal boundary and produce a different id sequence. Verify against the real
    tokenizer before the real-model gate; if it differs, the server must compose and tokenise the
    card span as one unit while still keeping special-token parsing off inside the card. The
    reference also returns no system wrapper for an empty `system_prompt` (`prompt.py:47-48`),
    while this template always emits the wrapper; the assembled tenant should require a non-empty
    card or special-case it.

---

## 9. Test plan

### 9.1 Declare and graph

Start the real `radiance` binary on `tests/data/haru/haru_tiny` with `--kernels libref
--debug-accept-reference-kernels --http-threads 4`, the assembled `haru` plugin in a temporary
`RADIANCE_HOME/architectures`, and check:

* `/duplex/v1/info` lists exactly one `haru` tenant with `text = true`, the declared `max_lanes`,
  `sample_rate = 24000`, `samples = 3200`, `format = f32le`;
* the process starts and the graph dump (`--debug-graph`) shows one PROMPT pass, one STEP pass, one
  `RAD_KV_FULL` group, three `RAD_KV_LINEAR` and three `RAD_KV_CONV` groups (tiny geometry), the
  tied embedding feeding both `embed_lookup` and `logits_gemm`, the PAD/EOP row view, and one
  tenant declaration;
* no duplicate weight/name-map entry is reported by the loader; all codec/head/LM/projector tensors
  are mapped.

This is the cheapest place to catch a name map or geometry error before any numeric test.

### 9.2 Tiny teacher-forced parity through the real engine

Fixture: `tests/data/haru/golden.safetensors`, all float32 (`tests/data/haru/README.md:15-24`);
card `"You are Haru."`, the 44,000-sample reference, 24 user frames, seed 1234, the recorded
tokens/noise. The test opens a lane, sends the reference and the 24 user frames, and replays the
recorded token/noise through the ABI 39 buffers. Compare, in order (fail early on the first
mismatch so a codec bug is not hidden by the LM):

1. **Prompt assembly.** `prompt.tokens[0..37)` (card 23 + 13 PAD + EOP); the plugin's per-row PAD/EOP
   synthesis; `prompt.model_latents` (card zeros, reference rows `ref_codec_mean + ref_noise`);
   masks. This checks the template, tokenizer, reference staging and reference codec.
2. **Reference codec.** `prompt.ref_codec_mean [14,64]` from the per-frame streaming encode, with
   the last frame 2,400 samples. Then `prompt.model_latents[23..37)`.
3. **Projector.** `prompt.embed [37,64]` and `frame{t}.embed [64]` for t = 1..24. This checks the
   raw-latent scaling, both projectors, masks and the embedding sum.
4. **LM.** `prompt.hidden_last`, `prompt.lm_layer0..3` (if taps are enabled in a debug build),
   `frame{t}.hidden` for t = 1..24, and `frame{1,2,3}.lm_layer0..3`. The tiny prompt is one PROMPT
   pass; the STEPs are one row a lane. This is the row-by-row vs chunked-prefill risk called out in
   `docs/HARU-H4-DESIGN.md:558-575`; compare the last prompt row first.
5. **Logits and sampling.** `frame0.logits`, `frame0.logits_fp32_pad_eop`, and for t = 1..24
   `frame{t}.logits`/`logits_fp32_pad_eop`. In teacher mode, compare `token_out` against
   `frame{t}.logits_final` + the recorded token; also run the host sampler on the same row and
   compare `logits_final`/the sampled token when its RNG is seeded identically (the token is
   otherwise the recorded one).
6. **Head and latent.** `frame0.head_latent_normalized`/`latent_out`, then `frame{t}.head_latent_normalized`
   and `latent_out` for t = 1..24. This checks the head, noise, guidance, unscale and the
   one-ahead latent pairing.
7. **Codec decode and audio.** `frame{t}.audio_out` for t = 1..24, after the same clamp
   (`session.py:163`). Frame 0 is decoded for state but not sent; its audio is not part of the wire
   comparison. The first compared reply is frame 1 (wire seq 0), which proves both the one-ahead
   pairing and that frame 0 advanced the decoder state.
8. **Text pairing.** The first reply carries token 1, not token 0; token 0 is only `text_in` for
   the first step. Assert this explicitly.

Tolerance: start at `Tol{1e-4, 1e-4}` for f32 on libref (`tests/HARU_HARNESS.md:91-92`); record the
measured maximum per family in the test comment. The README's measured thread A/B is exactly zero
(`tests/data/haru/README.md:250-278`), so it gives no C++ tolerance; the float64 sensitivity table
is the starting evidence (`:335-372`). If the row-by-row GDN path is the limiting family, record the
number before loosening. For a bf16 real-model run, measure separately and do not reuse the f32
numbers.

### 9.3 Lifecycle and concurrency

* Two lanes interleaved in one STEP pass: each lane's audio, text, logits and state match its
  single-lane run; `lane_h[i]` indexing is proven (not `i == lane`).
* Close and reopen the same lane index: the second PROMPT resets the codec, `prev_latent` and the
  DRY history; no state from the first conversation appears.
* `flush`: one reply, `flags = 0`, no state change, no extra pass.
* `max_lanes`: the next open is refused with the engine's `session_limit`; no lane sees another
  lane's state.
* Close during/after a pending step: the driver cancels or completes safely; the plugin never sees
  a half-freed lane.
* Wrong frame length and wrong `seq`: `bad_frame`/`bad_sequence`, fatal after `ready`
  (`docs/DUPLEX-ENGINE.md:91-105`).

### 9.4 Forge conformance suite

After the tiny parity test is green, run the Forge conformance suite against the assembled plugin
with the real checkpoint. The suite is the wire/lifecycle gate (info, start/reference, binary
step, flags, seq echo, text delta, flush, close, errors, lane limit, no card/reference/audio in
logs, p95 latency). It is not a bit-level reference comparison: with production sampling the
tokens and audio are stochastic. Keep the tiny teacher-forced test as the numeric gate and the
Forge suite as the conformance gate. The real-model build also needs a conversion/load check that
the `haru` plugin claims the checkpoint and the 32-layer geometry reads correctly.

### 9.5 What I cannot verify here

No C++ build or run is possible on this macOS worktree. I have not measured any of the tolerances,
the launch counts, the p95 step time, the per-lane memory, the graph declaration, the f32/bfloat16
selector bands, the tokenizer rendering through the C++ server, the reference decode path, the
server's partial-hop fix, the ABI layout table regeneration, or the Forge suite. The sizes above are
arithmetic from the published shapes and the H1/H4 notes, not a buffer-plan result. Every numeric
comparison and every conformance result must come from the reviewer's Linux build.

---

## 10. Files the next step will touch

* `arch/haru/CMakeLists.txt` and `arch/haru/haru.cpp`: the plugin, compiled with the four
  `arch/haru_common/*.cpp` by path; `arch/haru` is not created yet.
* `arch/haru_common/haru_map.{h,cpp}` if the loader path needs a single map (section 8.8), or the
  README row removed.
* `abi/rad_abi.h`, `abi/rad_builder.h`, `abi/rad_runtime.h`, `abi/rad_abi_layout_rows.h`,
  `tests/abi_layout_table.h`, `tests/abi_layout_test.cpp`, `docs/PLUGIN.md`,
  `docs/DUPLEX-PLUGIN-ABI.md`: ABI 39.
* `core/gen/duplex_driver.{h,cpp}`: PROMPT readback, text/noise staging, sampler, teacher feed.
* `core/server/duplex.cpp`: keep the true reference sample count; no audio zero-pad before the
  driver.
* `tests/haru_assembly_test.cpp`, `tests/haru_plugins/assembly/` (or the real `haru` target plus a
  test-only driver seam), and the CMake definitions for `RAD_HARU_PLUGIN`/`RAD_HARU_DATA`.
* `DUPLEX-REPORT-H5.md` (untracked): the implementation report after the reviewer's build.
