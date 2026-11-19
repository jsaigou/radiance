# How an architecture plugin declares a duplex tenant

Status: design note for R3a. The wire contract is `docs/DUPLEX-ENGINE.md`; this note is about the
engine/plugin ABI behind that contract. It describes the code as it is in this tree and then
proposes the smallest ABI that lets one plugin serve a lockstep, one-audio-frame-in /
one-audio-frame-out tenant.

Nothing in this note changes code. The proposed symbols are names and shapes to argue about before
R3b; where I was not sure a facility exists I say so and give the check I ran.

---

## 1. Today: an audio-generating plugin

### 1.1 The declaration and its drive shapes

An audio-generating architecture calls `rad_declare_audio_generator(b, &d)` with a
`RadAudioGenDecl` (`abi/rad_builder.h:1189-1289`). The declaration is one of the six mutually
exclusive program kinds: a program declares either logits, a pooled output, an image generator, an
audio generator, a video generator or a shape generator
(`core/build/rad_builder.cpp:1539-1548`).

The core drives four pass kinds (`abi/rad_builder.h:1186-1187`):

| pass | what the plugin does |
|---|---|
| `RAD_GEN_PASS_PROMPT` (`4`) | an ordinary causal prefill of a request's prompt, one or two branches; a long prompt may be chunked (`prompt_whole` disables chunking); the pass that ends a prompt leaves the branch's last hidden state in the lane (`abi/rad_builder.h:1161-1166`). |
| `RAD_GEN_PASS_FRAME` (`5`) | samples the next frame from the lane's last hidden state, embeds it, and runs the language model over the frame's one row a branch (`abi/rad_builder.h:1167-1171`). |
| `RAD_GEN_PASS_WINDOW` (`6`) | one denoising step of a run of frames (a window), usually two entries with classifier-free guidance (`abi/rad_builder.h:1172-1173`). |
| `RAD_GEN_PASS_VOCODE` (`7`) | a finished window's latents to samples in `audio_out`, or, when `voc_chunk > 0`, a chunk of the frames themselves (`abi/rad_builder.h:1174-1177`). |

`RadAudioGenDecl` carries the prompt templates (`tmpl`, `tmpl_plain`, `tmpl_ref`), the geometry
(`max_prompt`, `max_frames`, `frame_rate_num/den`), the window geometry
(`win_frames`, `win_hop`, `win_lead`, `lat_num/lat_den`, latent/channel/sample counts), the
denoiser defaults, `max_lanes`, the `status` and `audio_out` buffers, the branch count and the
reference fields (`abi/rad_builder.h:1189-1287`). The driver refuses a declaration whose
geometry cannot fit the pass it will stage (`core/build/rad_builder.cpp:1619-1657`).

The real in-tree examples are:

- `arch/minimaxmusic3/minimaxmusic3.cpp:1106-1135` declares `RadAudioGenDecl` and dispatches PROMPT
  and FRAME at `:1390-1391`;
- `arch/fishomni/fishomni.cpp:1737-1766` declares a one-branch, chunked-vocode speech model and
  dispatches PROMPT, FRAME and VOCODE at `:2192-2198`.

### 1.2 What a pass carries

`RadAudioGenBatch` (`abi/rad_runtime.h:542-628`) is the audio-specific payload hung off
`RadBatch::agen_b` (`abi/rad_runtime.h:276-280`). `RadBatch` itself already carries
`token_ids`, `positions`, `cu_seqlens`, `seq_ids`, `q_lens`, `ctx_lens`, the KV-group batches
(`RadBatch::kv`) and the multimodal `rope_pos` fields (`abi/rad_runtime.h:77-120`,
`:211-242`).

For an audio pass the important fields are:

- PROMPT/FRAME: `n_lane`, `lane`, `last`, `frame`, `seed`, `slot`, `last_row`, `prefix`
  (`abi/rad_runtime.h:543-565`);
- WINDOW: `t`, `t_next`, `cfg`, `pair`, `win_lane`, flags, `win_index/frame0/frames`,
  `lat_cu`, `lat_src` (`abi/rad_runtime.h:567-592`);
- VOCODE: `voc_lane`, `voc_window`, `voc_latents`, `voc_left/right`, `voc_frame0`
  (`abi/rad_runtime.h:594-603`);
- references: `n_ref`, `ref_samples`, `ref_audio`, `ref_cu`, `ref_cu_h`, `ref_frame`,
  `ref_frame_h` (`abi/rad_runtime.h:615-627`).

The driver builds these fields in `core/gen/audio_driver.cpp`: PROMPT at `:635-736`, FRAME at
`:738-820`, WINDOW at `:822-914`, VOCODE at `:916-946`, and a chunked vocode at `:948-982`. Its
own scheduling order is documented at `core/gen/audio_driver.h:18-34` and implemented in
`core/gen/audio_driver.cpp:566-627`: vocode first, then prompt, then FRAME/WINDOW alternating.
This is piece generation, not a reply to an input frame.

### 1.3 Per-lane state groups

The audio driver supports precisely one paged full-attention group plus any number of
linear-attention and convolution groups:

- `core/gen/audio_driver.cpp:167-197` counts the groups; a second `RAD_KV_FULL` or any other kind
  is refused;
- `StateGroup` is `{group, width}` (`core/gen/audio_driver.h:129-131`);
- each state group is staged as one block-table word per sequence (its slot) and a `state_index`
  row (`core/gen/audio_driver.h:129-131`, `core/gen/audio_driver.cpp:493-502`);
- `stage_state` writes `ctx_lens` and a one-based `num_accepted` and reads the sequence's state
  slot and index row (`core/gen/audio_driver.cpp:535-558`);
- the header states that a hybrid model's linear-attention state and convolution history are one
  slot a sequence each, zeroed by the engine before the first pass that reads them
  (`core/gen/audio_driver.h:12-16`).

The state kinds are `RAD_KV_FULL`, `RAD_KV_WINDOW`, `RAD_KV_LINEAR`, `RAD_KV_CONV` and
`RAD_KV_RING` (`abi/rad_builder.h:461-473`). `RadKVGroupDecl` gives the kind, dtype, window,
head count, head width, linear `state_dim[2]` and `conv_width` (`abi/rad_builder.h:475-486`);
layers are bound with `rad_bind_layer_kv` (`abi/rad_builder.h:497-498`).

Qwen3.5's plugin is the model for a hybrid trunk: it declares a `RAD_KV_FULL` group
(`arch/qwen35_fp8/qwen35_fp8.cpp:646-670`), a `RAD_KV_LINEAR` group
(`:685-692`) and a `RAD_KV_CONV` group (`:694-700`), then binds full-attention layers to the first
and delta-net layers to the other two (`:1113-1130`). `arch/fishomni/fishomni.cpp:542-565` and
`:701-717` do the same for the speech model it hosts.

### 1.4 Reference handling

Reference audio is not a separate pass. It is encoded inside the PROMPT:

- the declaration names a reference template and an item template, the clip geometry, and the
  placeholder slots: `tmpl_ref` has `{references}` and `{input}`, `ref_item` has `{text}` before
  its one `{audio}` (`abi/rad_builder.h:1269-1286`);
- the server renders the references and builds `AudioRequest::ref_audio`, `ref_cu` and
  `ref_frame` (`core/gen/audio.h:62-79`, `core/server/audio_gen.cpp:186-320`);
- the driver stages the clips, their bounds and each prompt row's reference frame index
  (`core/gen/audio_driver.cpp:695-710`);
- a reference-frame row stages token 0 and is identified by `ref_frame[t]`; the plugin embeds the
  frame there instead of the token (`abi/rad_runtime.h:615-627`);
- `fishomni` is the working example: `step_encode` runs its codec encoder over the clips
  (`arch/fishomni/fishomni.cpp:1809-1884`) and `op_pfb` adds the projected code embedding to `b_x`
  after `op_embed` filled it from the token table (`arch/fishomni/fishomni.cpp:1113-1134`,
  `:1986-1997`). `op_fb` does the same for a made frame (`:1113-1119`).

That is the existing "audio input projector" hook: an ordinary op adds a projected audio row into
the token embedding buffer before the trunk runs. The generic `RadBatch::mm_embd` /
`mm_rows` mechanism can also replace a token's embedding with an encoder's row
(`abi/rad_runtime.h:229-242`), but the audio generator does not use it for references.

### 1.5 What is missing for a duplex model

The model in question differs in four ways from what `RadAudioGenDecl` describes:

1. **One frame in, one frame out.** A duplex tenant consumes one user audio frame per step and
   must reply with one model audio frame and optionally one text token
   (`docs/DUPLEX-ENGINE.md:60-80`). `RAD_GEN_PASS_FRAME` consumes no audio: its row embedding is
   whatever the plugin made from the sampled frame (`abi/rad_builder.h:1167-1171`), and
   `RadAudioGenBatch` has no input-audio buffer.
2. **No piece geometry.** There are no windows, no multi-step denoiser, no vocoder pass and no
   `max_frames` endpoint. `RadAudioGenDecl` requires window fields unless `voc_chunk > 0`, and a
   positive `default_steps` unless chunked (`core/build/rad_builder.cpp:1619-1657`); reusing it
   would mean declaring meaningless geometry.
3. **A lane lifecycle.** `AudioDriver::submit` is a request/response ticket for a whole piece
   (`core/gen/audio.h:29-79`, `core/gen/audio_driver.cpp:271-328`), not `open lane`, `step`,
   `flush`, `close`. `max_lanes` exists for concurrent pieces, not for independent conversations.
4. **Text output.** The only per-lane output today is `status` (`>= 0` a frame, `< 0` the piece
   ended) and `audio_out` (`abi/rad_builder.h:1230-1236`, `abi/rad_runtime.h:548-553`,
   `core/gen/audio_driver.cpp:996-1025`). There is no token-id output buffer and no per-step flag
   for "text present".

A duplex tenant also needs a prompt shape the current generator does not express: character-card
tokens, then reference frames (model audio only), then live frames. `tmpl_ref` exists, but it
interleaves a transcript with each clip and is designed for voice cloning (`abi/rad_builder.h:1269-1286`).

---

## 2. Proposal: a separate `RadDuplexDecl`

### 2.1 New declaration struct, not an extension of `RadAudioGenDecl`

I recommend a new declaration struct, `RadDuplexDecl`, and a new entry point,
`rad_declare_duplex_tenant`. The alternatives have these consequences:

**Extend `RadAudioGenDecl` (append fields).** An appended field is source-compatible for an old
plugin only because it zero-initialises `RadAudioGenDecl` (`RadAudioGenDecl d{}` in
`arch/fishomni/fishomni.cpp:1737` and `arch/minimaxmusic3/minimaxmusic3.cpp:1107`), but the
struct's size changes, so `RAD_ABI_VERSION` must still move: `abi/rad_abi.h:15-22` says the version
changes with "a field appended to a struct included", and `docs/PLUGIN.md:45-56` repeats it. Old
engine binaries would then refuse new plugins by version, which is correct. The cost is conceptual
and mechanical: the builder's validation is written for a piece
(`core/build/rad_builder.cpp:1619-1657`), the driver's pass schedule is piece-shaped
(`core/gen/audio_driver.cpp:566-627`), and every new duplex field would coexist with window,
vocoder, guidance and frame-count fields that mean nothing to a tenant. A single `duplex` flag
inside `RadAudioGenDecl` also makes one declaration mean two different drive shapes, contrary to
the current "declare exactly one of these" rule (`abi/rad_builder.h:1185`).

**New `RadDuplexDecl` (recommended).** Existing layouts do not move; `RadAudioGenDecl` keeps its
current meaning and its in-tree users. The new kind is explicit in `Program`, so the engine cannot
mistake a tenant for a piece generator. The cost is a new declaration, a new batch, a new driver
path and a new pass kind, which the drive shape needs anyway. A new required engine export plus
new fields in `RadBatch`/`RadArchProbe` still require an ABI-version bump (section 3.3); this is a
release-coordination cost, not a reason to overload the old struct.

The recommendation is therefore: **new struct, new entry point, new pass kind**, leaving
`RadAudioGenDecl` and every existing plugin byte-for-byte unchanged.

### 2.2 The declaration

The declaration describes only what the core must stage; the plugin declares its own weights,
buffers and ops with the existing builder calls. Proposed shape:

```c
/* abi/rad_builder.h */

enum {
    RAD_GEN_PASS_DUPLEX_PROMPT = 19,  /* card tokens + reference frames, no reply */
    RAD_GEN_PASS_DUPLEX_STEP   = 20   /* one live frame in, one frame/text out */
};

enum {
    RAD_DUPLEX_AUDIO_USER  = 1,       /* the client's live input frame */
    RAD_DUPLEX_AUDIO_MODEL = 2        /* the model's own frame (reference or last output) */
};

typedef struct RadDuplexAudioIn {
    int32_t source;       /* RAD_DUPLEX_AUDIO_* */
    int64_t in_dim;       /* 64 for the model in section 5 */
    int64_t hidden;       /* 2560 */
    int64_t out_dim;      /* 2560 */
    int32_t rms_norm;     /* 1: RMSNorm between the two projections */
} RadDuplexAudioIn;

typedef struct RadDuplexDecl {
    const char* name;              /* the tenant name in /duplex/v1/info */

    /* The preamble. {card} is the start message's card rendered as text; {references} is
     * replaced by one row per reference frame, model audio only. No transcript. */
    const char* tmpl;
    int64_t     max_prompt;        /* card tokens + reference rows */
    int64_t     max_ref_frames;
    int64_t     ref_rate;          /* 24000 */
    int64_t     ref_hop;           /* 3200 samples a frame */

    /* One conversation a lane. */
    int64_t     max_lanes;

    /* The frame contract (docs/DUPLEX-ENGINE.md:33-38, :60-80). */
    int64_t     sample_rate;       /* 24000 */
    int64_t     channels;          /* 1 */
    int64_t     samples_per_frame; /* 3200 */
    int64_t     latent_dim;        /* 64 */
    int64_t     frame_rate_num;    /* 15 */
    int64_t     frame_rate_den;    /* 2  -> 7.5 frames a second */

    /* The audio-input projectors. Each is a network the plugin declares itself; the core only
     * needs to know which source feeds it and the geometry, so the driver and the graph dump can
     * name it. */
    int32_t        n_audio_in;     /* 0..2 */
    RadDuplexAudioIn audio_in[2];

    /* 1: the model may emit a text token beside its audio frame. */
    int32_t     text;

    /* Device buffers. audio_in is written by the driver from the wire; audio_out, text_out and
     * status are read back after STEP. */
    rad_buf     audio_in;          /* f32 [max_lanes, samples_per_frame] */
    rad_buf     audio_out;         /* f32 [max_lanes, samples_per_frame] */
    rad_buf     text_out;          /* i32 [max_lanes], -1 = no token */
    rad_buf     status;            /* i32 [max_lanes], 0 = replied, < 0 = fatal */
} RadDuplexDecl;

int rad_declare_duplex_tenant(RadBuilder* b, const RadDuplexDecl* d);
```

The core copies the strings into `Program`, as it does for the audio generator's templates
(`core/build/rad_builder.cpp:1680-1693`). A `shape_probe` declare must not touch live state; the
existing generators already skip their declaration under `ctx->shape_probe`
(`arch/fishomni/fishomni.cpp:1736`, `arch/minimaxmusic3/minimaxmusic3.cpp:1106`).

The pass kind is appended after the shape generator's `15..18`
(`abi/rad_builder.h:1551-1552`); `19`/`20` are the next free values. The values are part of the
ABI's observable behaviour even though the enum is not a struct.

### 2.3 The passes

**`RAD_GEN_PASS_DUPLEX_PROMPT` (prefill).** Carries the card tokens and reference frames of one or
more lanes in the usual `RadBatch` rows. `batch->token_ids`, `positions`, `cu_seqlens` and the KV
group's slot mapping are the ordinary prefill fields. `RadDuplexBatch` adds the reference frame
index per row and the raw reference audio, exactly as the audio generator does for a reference
PROMPT (`abi/rad_runtime.h:615-627`). A row whose `ref_frame[t] == -1` is a card/text row and uses
the token embedding; a row whose `ref_frame[t] >= 0` is a reference frame and the plugin embeds the
projected frame in place of the placeholder token.

The pass runs the trunk over the rows, fills the lane's `RAD_KV_FULL` cache and advances the
`RAD_KV_LINEAR`/`RAD_KV_CONV` state by the same rows. The last row per lane is stored as the lane's
first live context, as PROMPT's `last_row` does today (`abi/rad_runtime.h:558-565`).

**`RAD_GEN_PASS_DUPLEX_STEP` (one live frame).** One row a lane, at the lane's next position. It
consumes:

- `audio_in[lane]`, 3200 f32 samples (mono, 24 kHz, [-1, 1]) staged from the wire frame after its
  `seq` header (`docs/DUPLEX-ENGINE.md:64-68`);
- the lane's previous model latent (kept by the plugin) or a zero vector on the first live step;
- the lane's `RAD_KV_FULL` cache and its `RAD_KV_LINEAR`/`RAD_KV_CONV` state from the prompt and
  earlier steps.

It produces:

- `audio_out[lane]`, 3200 f32 samples: one model audio frame;
- `text_out[lane]`, one token id or `-1`;
- `status[lane]`, which the driver reads to decide whether the reply has audio present; a
  negative status fails the lane.

The wire reply's `flags` are built by the tenant from `status` and `text_out`
(`docs/DUPLEX-ENGINE.md:69-77`). A lane with nothing to say still gets a reply with `flags = 0`
(`docs/DUPLEX-ENGINE.md:78-80`); the plugin writes `status = 0` and leaves `audio_out` alone.

I have deliberately not added a separate `RAD_GEN_PASS_DUPLEX_VOCODE`. The codec encode and decode
run inside the plugin's STEP, because the wire requires the frame's samples in the same exchange.
If profiling later shows the decode must be a second pass, a `VOCode` pass can be appended without
changing the declaration's frame contract; it is not needed for v0.

### 2.4 How the audio input projector is hooked into the embedding

The hook is the same one the existing speech plugins already use: the trunk's input row is built in
the plugin's step, and the projector output is added to the token embedding before the first block.
`fishomni` does exactly this with `op_embed` followed by `op_fb` / `op_pfb`, which add the projected
code rows into `b_x` (`arch/fishomni/fishomni.cpp:1113-1134`, `:1986-1997`).

For the duplex model the row is:

```
x[row] = token_embedding(placeholder_or_text_token)
       + P_user(user_latent)        // the client's current 64-dim frame
       + P_model(model_latent_prev) // the model's previous 64-dim frame
```

with each `P_*` declared as the model's two-stage 64 -> 2560 -> 2560 network and its RMSNorm.
`RadDuplexAudioIn::{source,in_dim,hidden,out_dim,rms_norm}` is what the driver and graph dump need
to know; the weights and ops are the plugin's ordinary `rad_decl_weight` / `RAD_OP` declarations.

For a prompt row the same expression applies with the appropriate source:

- a card/text row has only the token embedding;
- a reference-frame row has the token embedding of the template's placeholder plus
  `P_model(reference_latent)`, because the brief says references are model audio only.

The two-projector split (`USER` vs `MODEL`) is a declaration choice, not a hard-coded engine rule.
If the checkpoint turns out to use both projectors for one direction, the enum can name the streams
and `source` becomes a small bitmask; that is a field-value change, not a layout change.

### 2.5 `RadDuplexBatch`

```c
/* abi/rad_runtime.h */

enum {
    RAD_DUPLEX_F_FIRST = 1,   /* the lane's first live step */
    RAD_DUPLEX_F_FLUSH = 2    /* a flush; no new input frame */
};

typedef struct RadDuplexBatch {
    /* PROMPT and STEP: the lanes of the pass, in entry order. */
    int64_t         n_lane;
    const int32_t*  lane;
    const int32_t*  lane_h;

    /* PROMPT: the number of prompt rows. */
    int64_t         n_tok;
    /* PROMPT: the last prompt row of each lane. */
    const int32_t*  last_row;
    const int32_t*  last_row_h;

    /* PROMPT: the clips' samples end to end, their bounds in samples, and the frame each row
     * names, as RadAudioGenBatch's reference fields: -1 is a card/text row; >= 0 is the index
     * of the reference frame that row embeds. */
    int64_t         n_ref;
    int64_t         ref_samples;
    const float*    ref_audio;
    const int32_t*  ref_cu;
    const int32_t*  ref_cu_h;
    const int32_t*  ref_frame;
    const int32_t*  ref_frame_h;

    /* STEP: the live frame number, the lane's seed, and RAD_DUPLEX_F_*. */
    const int32_t*  frame;
    const int32_t*  frame_h;
    const uint32_t* seed;
    const int32_t*  flags;
    const int32_t*  flags_h;
} RadDuplexBatch;
```

`RadBatch` gains one trailing member:

```c
const struct RadDuplexBatch* dup_b;   /* null on every other step */
```

The row's token ids, positions, `cu_seqlens`, KV slot mapping, `ctx_lens` and `num_accepted` stay in
`RadBatch`, so the plugin issues the same trunk ops it issues for an ordinary decode row. The
`audio_in`, `audio_out`, `text_out` and `status` buffers are the declaration's handles, not batch
fields: the plugin captured them at declare and the driver writes/reads them around the pass.

### 2.6 Prompt construction: card, references, live frames

At `start` the tenant has the opaque `card` and, if `reference.bytes > 0`, one binary reference
message (`docs/DUPLEX-ENGINE.md:44-58`). A plugin-declared tenant builds the lane in this order:

1. Render `tmpl`: `{card}` becomes the card text (the tenant serialises the JSON object; the plugin
   may rewrite it through `rad_arch_text_prepare`, `abi/rad_builder.h:1910-1922`), and
   `{references}` becomes one placeholder row per reference frame. The card tokens are tokenised
   with their special tokens, as the audio prompt is (`core/server/audio_gen.cpp:267-320`).
2. Decode the reference message to mono f32 at `ref_rate`, pad to whole `ref_hop` frames, and stage
   it as `ref_audio`/`ref_cu`; `ref_frame[t]` names the frame for every reference row. This is the
   existing reference staging path (`core/gen/audio_driver.cpp:695-710`) without a transcript.
3. Run `RAD_GEN_PASS_DUPLEX_PROMPT` over the card rows and reference rows in that order. The plugin
   encodes the reference clips with its codec, projects each 64-dim frame through `P_model`, adds
   it to the placeholder token's embedding, and advances the trunk. The last row's hidden state is
   the live context.
4. Reply `ready` (`docs/DUPLEX-ENGINE.md:55-58`). Every later `step` appends one live row after the
   reference rows; the lane's prompt is never rebuilt.

The order is therefore exactly the brief's: **character-card tokens, then reference frames (model
audio only), then live frames**. A reference frame has no transcript row; the template controls
any marker tokens around `{references}`.

### 2.7 State groups and lifetime

State groups are declared by the plugin with the existing `rad_decl_kv_group` and bound with
`rad_bind_layer_kv`; `RadDuplexDecl` does not repeat them. For a hybrid trunk the declaration is
the same one Qwen3.5 already uses:

- `RAD_KV_FULL` for the full-attention layers (`arch/qwen35_fp8/qwen35_fp8.cpp:646-670`);
- `RAD_KV_LINEAR` for the delta-net recurrent state (`:685-692`);
- `RAD_KV_CONV` for the delta-net convolution history (`:694-700`);
- bindings at `:1113-1130`.

Lifetime is the lane, which is the conversation (`docs/DUPLEX-ENGINE.md:15`):

| event | state |
|---|---|
| `open lane` | allocate one sequence in the full-attention pool and one state slot/row per state group; zero the linear and convolution state before the first read, as the audio driver's contract says (`core/gen/audio_driver.h:12-16`); zero the plugin's codec state for the lane |
| `DUPLEX_PROMPT` | append card and reference rows to the full-attention cache; advance the state groups by those rows |
| each `DUPLEX_STEP` | append exactly one row; read and write the state groups at that row's context |
| `flush` | no state change; return any buffered frame or `flags = 0` (`docs/DUPLEX-ENGINE.md:84-89`) |
| `close` | release the sequence, every state slot/index row and every plugin-private lane buffer |

No lane state may affect a later lane, and a closed WebSocket releases everything
(`docs/DUPLEX-ENGINE.md:100-106`). There is no prefix cache, no checkpoint and no speculation in a
duplex lane. The driver should reuse the audio driver's `stage_state` logic
(`core/gen/audio_driver.cpp:535-558`) rather than invent a second state staging convention.

Codec state is the one new lifetime question. The codec is causal and is not a transformer layer,
so it is not naturally a `RadKVGroupDecl` bound by layer index. The safe v0 is a plugin-private
`RAD_BUF_PERSIST` buffer with a lane axis, zeroed by the plugin on the first step of a lane and on
`close`. If the codec must be state-cached by the engine (for offload or checkpointing), a new
state kind is needed; that is an open question, not part of the v0 ABI below.

---

## 3. Exact ABI additions

### 3.1 Enumerators and constants

| symbol | value | why |
|---|---|---|
| `RAD_GEN_PASS_DUPLEX_PROMPT` | `19` | next free `RAD_GEN_PASS_*` value; shape uses `15..18` (`abi/rad_builder.h:1551-1552`) |
| `RAD_GEN_PASS_DUPLEX_STEP` | `20` | one live frame in, one frame/text out |
| `RAD_DUPLEX_AUDIO_USER` | `1` | the client's frame |
| `RAD_DUPLEX_AUDIO_MODEL` | `2` | the model's reference/last frame |
| `RAD_DUPLEX_F_FIRST` | `1` | first live step of a lane |
| `RAD_DUPLEX_F_FLUSH` | `2` | a flush, no new input frame |
| `RAD_TEXT_CARD` (optional) | `4` | only if the card text must be rewritten by `rad_arch_text_prepare`; today the enum is `1..3` (`abi/rad_builder.h:1921`) |

`RAD_TEXT_CARD` is optional because the tenant can serialise the card to text and have the plugin
parse it. If the card has structure the plugin must interpret, a better addition is an optional
`rad_arch_duplex_card` export with the JSON bytes; I did not invent its memory contract here
because the card shape is not in this tree.

### 3.2 Structs and fields

| location | addition |
|---|---|
| `abi/rad_builder.h` | `RadDuplexAudioIn`, `RadDuplexDecl`, `rad_declare_duplex_tenant`, the two pass enumerators, the source enumerators |
| `abi/rad_runtime.h` | `RadDuplexBatch`, the flag enumerators, `RadBatch::dup_b` appended |
| `abi/rad_builder.h` (`RadArchProbe`) | `int32_t gen_duplex` appended; nonzero means the plugin is a duplex tenant. Reuse `gen_rows` for the largest prompt/step row count and `gen_ctx` for the lane's context bound; `gen_audio` stays 0. Existing fields are at `abi/rad_builder.h:1624-1676` |
| `abi/rad_abi.h` | `RAD_ABI_VERSION` from `37` to `38` |
| `abi/rad_abi_layout_rows.h` | `ABI_STRUCT`/`ABI_FIELD` rows for the two new structs, `RadBatch::dup_b`, `RadArchProbe::gen_duplex` |

No field is added to `RadAudioGenDecl`, `RadAudioGenBatch`, `RadVideoGenDecl` or any other existing
declaration, so their offsets do not move.

The declaration's buffers are validated at declare the way the audio generator's are
(`core/build/rad_builder.cpp:1658-1679`): `audio_in` and `audio_out` must be device f32 of at least
`max_lanes * samples_per_frame`; `text_out` and `status` must be device i32 of at least
`max_lanes`. `max_lanes` must fit `RadBuildCtx::max_seqs`; the largest prompt must fit
`RadBuildCtx::max_tok`; and `max_ctx` must hold `max_prompt` plus the live frames the deployment
allows. The frame geometry is checked as one ratio: `samples_per_frame * frame_rate_num ==
sample_rate * frame_rate_den` (3200 * 15 == 24000 * 2), and `ref_hop == samples_per_frame` when
references are declared. `tmpl` must contain `{card}` once and at most one `{references}`; each
`audio_in` source must be one of the enumerators and its dimensions positive; `text` requires a
valid `text_out` buffer, and `max_ref_frames > 0` requires positive `ref_rate`/`ref_hop`.

### 3.3 Version bump rule and current version

The current version is **37** (`abi/rad_abi.h:22`; the layout table is pinned at 37 in
`tests/abi_layout_test.cpp:61-65`). The rule in `abi/rad_abi.h:15-22` is normative: the version is
bumped whenever anything a plugin compiles against changes shape, including "a field appended to a
struct included", and the release that ships the change assigns the number. `docs/PLUGIN.md:45-56`
adds that the loader refuses any plugin whose reported version differs and prints both numbers, and
that the layout fingerprint (`abi/rad_abi_layout.h:40-52`) catches two different layouts given the
same version.

This proposal must therefore ship as **ABI 38**, not 37:

- `RadBatch` grows by `dup_b` and `RadArchProbe` grows by `gen_duplex`;
- the new engine export `rad_declare_duplex_tenant` is required by a plugin that declares a
  tenant, so a new plugin must not load into a 37 engine that lacks it, even if no existing field
  moved;
- the layout fingerprint changes automatically because `rad_abi_layout_mix` includes the version
  and every struct/field (`abi/rad_abi_layout.h:43-51`).

The `abi_layout_test` table's allowance for a struct to grow past its old end
(`tests/abi_layout_test.cpp:135-186`) is a mechanical check inside one version; it does not waive
the version bump. When `RAD_ABI_VERSION` changes, the test prints the live table and asks for
`abi_layout_table.h` to be regenerated and `kTableAbi` updated (`tests/abi_layout_test.cpp:139-146`).

### 3.4 Files that must change

ABI and layout:

- `abi/rad_abi.h` — version 38.
- `abi/rad_builder.h` — pass/source enums, `RadDuplexAudioIn`, `RadDuplexDecl`,
  `rad_declare_duplex_tenant`, `RadArchProbe::gen_duplex`, optional `RAD_TEXT_CARD`.
- `abi/rad_runtime.h` — flag enums, `RadDuplexBatch`, `RadBatch::dup_b`.
- `abi/rad_abi_layout_rows.h` — rows for the new structs and appended fields.
- `tests/abi_layout_table.h` — regenerated table; header comment says 37 today.
- `tests/abi_layout_test.cpp` — `kTableAbi` from 37 to 38.
- `tests/abi_c_test.c` — no edit; it includes `rad_abi_layout_rows.h` (`tests/abi_c_test.c:24-31`)
  and is compiled into `abi_layout_test` (`tests/CMakeLists.txt:381-384`).
- `abi/rad_abi_layout.h` — no edit; its fingerprint derivation changes with the rows/version.

Engine and builder:

- `core/rad_core.h` — `RadDuplexDecl duplex`, its string storage, `is_duplex_tenant()`, and
  `is_pass_driven()` including it (the current fields are `:439-468`).
- `core/build/rad_build.h` and `core/build/rad_builder.cpp` — `Builder::declare_duplex_tenant`,
  validation, `RadArchProbe::gen_duplex` handling, and the startup log at finish (the audio path is
  `core/build/rad_builder.cpp:1527-1700`; the finish diagnostics are `:1210-1260`).
- `core/gen/duplex_driver.h`, `core/gen/duplex_driver.cpp` — the new pass driver (or a new mode in
  `core/gen/audio_driver.*`; I recommend a new driver so the audio piece path stays untouched).
- `core/gen/pass.h` / `core/engine.h` — only if the driver is registered as a new pass driver;
  `gen::IPassDriver` is the existing seam (`core/gen/pass.h:1-20`, `core/gen/audio_driver.h:1-34`).
- `core/engine.cpp`, `core/engine.h` — construct the driver for `is_duplex_tenant()`, expose the
  tenant info, and route `GET /duplex/v1/info` / the session bridge (the audio equivalents are
  `core/engine.cpp:2048-2088`, `:2260-2268`, `:3379-3440`).
- `core/engine_bringup.cpp` — size `max_tok`, `max_seqs`, `max_ctx` and the KV plan from
  `RadArchProbe::gen_duplex`/`gen_rows`/`gen_ctx` (the audio branch is `:1124-1144`; the KV
  capacity warning is `:3740-3778`).

Server:

- `core/server/duplex.h` / `core/server/duplex.cpp` — being written separately; the engine-side
  adapter below expects only `open lane`, `step`, `flush`, `close`.
- `core/server/server.cpp`, `core/server/http.cpp`, `core/server/iface.h` — register
  `/duplex/v1/info` and `/duplex/v1/session`, list the declared tenant, and keep card/reference
  bytes out of logs (`docs/DUPLEX-ENGINE.md:17-26`, `:91-96`).
- `core/server/audio_gen.cpp` — only if the shared audio prompt/tokenisation helpers are reused;
  otherwise unchanged.

Documentation and tests:

- `docs/PLUGIN.md` — the version and the architecture-declaration section (today §1 and §11).
- `docs/IMPLEMENTATION.md` — the `core/gen/` and `core/server/` layout lines if the new files are
  added.
- `docs/ARCHITECTURES.md` — a table row for the new plugin, if it ships in-tree.
- `tests/audio_gen_test.cpp` — unchanged by this proposal; it exercises the existing
  `RadAudioGenDecl` driver (`tests/audio_gen_test.cpp:1-5`, `:217-279`).
- new `tests/duplex_driver_test.cpp` (auto-globbed by `tests/CMakeLists.txt:163-167`) — fake plugin,
  prompt/step/flush/close, one-in/one-out, lane release, hybrid state staging.
- new `tests/duplex_tenant_test.cpp` — wire-level `echo`/`echo1` and the WebSocket seam, no model
  (`docs/DUPLEX-ENGINE.md:116-122`).
- `tests/arch_test.cpp` — if a shipped plugin declares a duplex tenant, its graph and pinned steps
  need a fixture entry; `tests/arch_test.cpp` declares every plugin's graph and is the existing
  place for that (`docs/PLUGIN.md:395-398`).
- `tests/shipped_plugins_test.cpp` — if the plugin is installed with the engine.

---

## 4. How a server-side `DuplexTenant` adapts

The `DuplexTenant` described in `docs/DUPLEX-ENGINE.md` owns the WebSocket, the frame clock, queues,
deadlines and cancellation; the engine is passive (`docs/DUPLEX-ENGINE.md:8-15`). The adapter is
therefore thin:

```
client message          DuplexTenant                     plugin-declared engine side
------------------------------------------------------------------------------------------
start + reference   ->  open(lane, card, ref_bytes)   ->  DuplexDriver::open_lane
                                                           allocate one sequence + state slots
                                                           stage reference samples
                                                           RAD_GEN_PASS_DUPLEX_PROMPT
                                                           -> ready
binary seq + 3200   ->  step(lane, samples)           ->  DuplexDriver::step
                                                           stage audio_in, frame, flags
                                                           RAD_GEN_PASS_DUPLEX_STEP
                                                           read audio_out/text_out/status
                                                           -> one binary reply
{"type":"flush"}    ->  flush(lane)                   ->  DuplexDriver::flush
                                                           return buffered frame or flags = 0
{"type":"close"}    ->  close(lane)                   ->  DuplexDriver::close
                                                           release sequence, slots, lane buffers
socket drop         ->  close(lane)                   ->  same, no resume
```

Contract points the adapter must enforce, all already in `docs/DUPLEX-ENGINE.md`:

- one lane is one conversation and one model state (`:15`);
- one input binary frame produces exactly one reply with the echoed `seq`, even when `flags = 0`
  (`:60-80`);
- `flush` is a step-shaped reply with the last `seq`, or `flags = 0` (`:84-89`);
- a wrong frame length is `bad_frame` and a wrong `seq` is `bad_sequence`, both fatal; after
  `ready`, every error is fatal (`:91-105`);
- `max_lanes` is the tenant's, and an unknown tenant name is `tenant_unavailable` (`:100-107`);
- the info route reports the declaration's `name`, `text`, `max_lanes` and the frame geometry
  (`:28-42`).

The engine-side object behind the adapter is the new `DuplexDriver`. It should mirror the audio
driver's threading and readback pattern: the engine thread calls `next()`/`done()`, the HTTP/WS
thread queues and waits through a ticket, and `done()` reads `audio_out`, `text_out` and `status`
back through the readback mechanism (`core/gen/audio_driver.cpp:984-1069`,
`core/gen/audio_driver.h:73-121`). The difference is that `next()` only ever returns a
`DUPLEX_PROMPT` or `DUPLEX_STEP` and that a step is owed for every queued wire frame; there is no
window scheduling.

For a plugin-declared tenant the engine registers exactly one backend per declared `name`. The two
test tenants (`echo`, `echo1`) are separate registrations with no model
(`docs/DUPLEX-ENGINE.md:116-122`), which lets the WebSocket seam be tested before any duplex plugin
exists.

---

## 5. Mapping the Qwen3.5-4B model onto it

The facts below are the brief's; the existing `qwen35` plugin shows how each is already read and
declared.

### 5.1 Trunk

- 32 layers, 24 gated-delta-net and 8 full attention. The family's rule is every
  `full_attention_interval`-th layer is full attention
  (`arch/qwen35_fp8/qwen35_fp8.cpp:550-562`; `arch/common/rad_arch.h:946-950`), and the plugin
  refuses a container whose `layer_types` disagree (`arch/common/rad_arch.h:952-971`). For 32
  layers at interval 4 that is layers 3, 7, ..., 31: 8 full and 24 GDN.
- hidden 2560, 16 query heads, 4 KV heads, head_dim 256. These are `RadModelMeta` fields
  (`abi/rad_builder.h:26-33`), read by `geom_from` (`arch/qwen35_fp8/qwen35_fp8.cpp:531`); the
  full-attention block is the existing gated-attention block.
- partial rotary 0.25, MROPE. `n_rot = head_dim * partial_rotary_factor` is read from the
  checkpoint (`arch/qwen35_fp8/qwen35_fp8.cpp:579-587`); MROPE sections are read by
  `rad_arch.h:1044-1053`, and the multi-component positions ride `RadBatch::rope_pos`
  (`abi/rad_runtime.h:211-227`). The duplex driver must stage positions per row. For a pure
  text/audio session every row can use the one-component form (`rope_posmc` falls back to
  `positions`, `arch/common/rad_arch.h:1328-1335`); if any row needs distinct MROPE components,
  the driver must stage a `[3, n_tok]` `rope_pos`, which the current audio driver does not (its
  regions are `core/gen/audio_driver.h:125-128`).
- tied head. `qwen35_fp8` maps `output.weight` with a fallback to the embedding table
  (`arch/qwen35_fp8/qwen35_fp8.cpp:918-933`), which is the existing tied-head pattern. The tied
  head is also the natural place to read the optional per-step text token.
- conv width 4. Read from `linear_conv_kernel_dim` (`arch/qwen35_fp8/qwen35_fp8.cpp:596-603`) and
  passed to `RadKVGroupDecl::conv_width` (`:694-700`).

### 5.2 State groups

The duplex plugin declares the same three groups as `qwen35_fp8`:

| group | layers | declaration |
|---|---|---|
| `RAD_KV_FULL` | 8 full-attention layers | paged attention cache, head_dim 256, 4 KV heads (`arch/qwen35_fp8/qwen35_fp8.cpp:646-670`) |
| `RAD_KV_LINEAR` | 24 GDN layers | recurrent state, one state a sequence (`:685-692`) |
| `RAD_KV_CONV` | 24 GDN layers | convolution history, width 4 (`:694-700`) |

The driver binds each layer to its group (`:1113-1130`) and stages one sequence per duplex lane.
The state geometry comes from the checkpoint, not from this note; the existing plugin reads it with
`meta_int2` (`arch/qwen35_fp8/qwen35_fp8.cpp:596-623`) and never hard-codes it.

### 5.3 Audio input projectors

The plugin declares two `RadDuplexAudioIn` entries with `source = USER` and `source = MODEL`,
`in_dim = 64`, `hidden = 2560`, `out_dim = 2560`, `rms_norm = 1`. Each is declared as ordinary
weights and ops (two linears and a norm) and used in the row embedding described in section 2.4.
The existing speech precedent is `fishomni`'s `op_fb`/`op_pfb`
(`arch/fishomni/fishomni.cpp:1113-1134`, `:1986-1997`). There is no generic per-frame audio
projector module, so this is the plugin's own block. I checked with
`grep -rniE "audio.*project|audio_in|projector" arch abi core`; the existing audio-input
projectors belong to encoder towers or a video generator (`arch/qwen3asr_bf16/qwen3asr_bf16.cpp:179-182`,
`arch/common/rad_block_gemma4_audio.h:174`, `arch/ltx2/ltx2.cpp:2569-2570`), not to a per-frame LM
input. The closest existing LM hook is the speech generator's `rvq_embed` into `b_x`; the generic
`mm_embd` row replacement is also available (`abi/rad_runtime.h:229-242`).

### 5.4 MeanFlow head and codec

- The MeanFlow head is 4 layers, hidden 2560, ffn 7680, one sampling step, producing one 64-dim
  latent per frame. It is a plugin head like the existing audio frame heads; it is not a
  `RAD_GEN_PASS_WINDOW` denoiser because there is no multi-step window and no guidance
  (`abi/rad_builder.h:1209-1228`, `core/gen/audio_driver.cpp:822-914`). In `DUPLEX_STEP` the
  plugin runs the trunk row, feeds its hidden through the 4-layer head and takes the single step to
  the 64-dim latent.
- The causal convolutional codec is 24 kHz, 3200 samples per 64-dim latent, 7.5 frames/s. That is
  `sample_rate = 24000`, `channels = 1`, `samples_per_frame = 3200`, `latent_dim = 64`,
  `frame_rate_num = 15`, `frame_rate_den = 2` in `RadDuplexDecl`. The existing generator has the
  same geometry fields (`abi/rad_builder.h:1204-1223`) but uses them for windows and a separate
  `VOCODE` pass (`:1251-1256`, `core/gen/audio_driver.cpp:916-946`). In the duplex pass the plugin
  encodes the input samples to a 64-dim user latent at the start of the step and decodes the
  model latent to samples at the end, carrying the codec's causal state in its own per-lane buffer.

### 5.5 Prompt

- Card tokens: tokenised from `tmpl`'s `{card}` region with special tokens, the same way the speech
  request is tokenised (`core/server/audio_gen.cpp:267-320`).
- Reference frames: the reference message is decoded to mono 24 kHz, padded to 3200-sample frames,
  and each frame becomes one prompt row with `ref_frame[t] >= 0`; the plugin runs its codec encoder
  and `P_model`, adds the result to the placeholder token's embedding, and appends the row. This is
  the `ref_frame`/`op_pfb` pattern (`abi/rad_runtime.h:615-627`,
  `arch/fishomni/fishomni.cpp:1809-1884`, `:1995-1997`).
- Live frames: after `ready`, each `DUPLEX_STEP` appends one row after the reference rows. Its
  embedding is the live placeholder/text token plus `P_user` of the current user frame and
  `P_model` of the previous model frame (section 2.4). The trunk advances one row, the MeanFlow head
  makes the model latent, and the codec decodes it. The order is card, references, live frames.

---

## 6. Unknowns and risks

1. **`core/server/duplex.h` does not exist in this worktree.** I checked with `ls
   core/server/duplex.h` (no such file) and `grep -rn "DuplexTenant\|duplex" core` (only an
   unrelated comment in `core/place/heat.h:58`). The adapter above assumes only the four operations
   the brief names and does not depend on the interface's actual shape.
2. **Card rendering.** `docs/DUPLEX-ENGINE.md:44-58` makes `card` opaque JSON. There is no ABI hook
   to parse it. The proposal serialises it to text and lets the plugin rewrite it through
   `rad_arch_text_prepare`; if the card needs structural interpretation, a new optional export is
   needed and its lifetime/ownership must be defined.
3. **MROPE rows.** The audio driver stages no `rope_pos` (`core/gen/audio_driver.h:125-128`). A
   one-component position is valid for every row whose components agree
   (`arch/common/rad_arch.h:1328-1335`); if the checkpoint's MROPE uses distinct components for
   audio rows, the new driver must stage `[3, n_tok]` and set `rope_mixed` (`abi/rad_runtime.h:211-227`).
   This must be checked against the checkpoint before Stage 3.
4. **Codec state.** A causal codec needs per-lane history. It is not a transformer layer and has no
   natural `rad_bind_layer_kv` layer index. The v0 proposal keeps it in a plugin-private persistent
   buffer; the engine cannot checkpoint or offload it. If the codec is large enough to need the
   state pool, a new state kind is a follow-up.
5. **Which projector is which.** The brief says two projectors but not that one is `USER` and one
   is `MODEL`; the checkpoint may use them for two codec streams or for two directions. The
   `source` enum is a declaration knob; confirm against the weights before Stage 3.
6. **MeanFlow conditioning.** "One sampling step" does not say whether the head needs a noise draw,
   a seed, or the previous frame. `RadDuplexBatch::seed` is provided so a stochastic head can be
   reproduced (`abi/rad_runtime.h:555-557` precedent); if the head is deterministic the field is
   ignored. The first live step's previous model latent (zero or last reference frame) must be
   decided from the training recipe.
7. **Text output.** The tied head can emit a token every step, but the wire only has a 1-bit
   "text present" flag plus a length (`docs/DUPLEX-ENGINE.md:69-77`). The tenant must decide when a
   token is worth sending; the plugin writes `-1` when there is none. This is a product decision,
   not an ABI one.
8. **Latency.** The step target is p95 under 100 ms against a 133.33 ms frame
   (`docs/DUPLEX-ENGINE.md:98-102`). A 32-layer 4B hybrid, a 4-layer MeanFlow head, a codec and
   possibly two projectors must fit that on the deployment hardware. The per-frame pass must not
   allocate, synchronise or batch a prompt on the step path
   (`docs/IMPLEMENTATION.md:88-90`). Measure before claiming the target.
9. **KV growth and lane count.** Each lane grows one row per live frame plus its prompt. The engine
   must size or evict for `max_lanes * (max_prompt + max live frames)`; the current audio driver
   reserves a whole piece's KV at admission (`core/gen/audio_driver.h:7-11`,
   `core/gen/audio_driver.cpp:312-315`). A duplex tenant cannot reserve an unbounded conversation;
   it needs a context bound or streaming. The engine's existing KV warning at
   `core/engine_bringup.cpp:3740-3778` is written for pieces and must be reworded for lanes.
10. **One engine thread, many lanes.** The pass driver runs one pass at a time
    (`core/gen/audio_driver.cpp:566-627`). A multi-lane tenant must batch steps fairly and must not
    let one lane's prompt or flush block another lane's deadline. The wire keeps at most one step
    in flight per lane (`docs/DUPLEX-ENGINE.md:78-80`), but the engine still needs a queue per lane
    and a cancellation path.
11. **ABI coordination.** 37 is the current version (`abi/rad_abi.h:22`); another in-flight change
    may also claim 38. The release that ships this must assign the number once and regenerate the
    layout table on a machine that can build the tree.
12. **This host cannot build the tree.** On this macOS worktree `cmake -S . -B build
    -DRAD_WITH_HIP=OFF` configures, but the build fails before tests: the shared modules do not
    resolve engine symbols on macOS (e.g. `_rad_bind_layer_kv` in `minimaxmusic3.so`), and `rad_core`
    includes Linux-only headers (`sys/vfs.h`, `sys/sysmacros.h`, `SIGRTMIN`, `SYS_tgkill`). The
    details are in `DUPLEX-REPORT.md`. That is an environment limitation, not a result of this
    note; verify every stage on a supported Linux host.

---

## 7. Staged implementation

Sizes are rough engineer-effort estimates, not calendar promises.

| stage | size | work | must be verified |
|---|---|---|---|
| 0. ABI surface | S (1-2 days) | `RAD_ABI_VERSION 38`; pass/source/flag enums; `RadDuplexDecl`, `RadDuplexBatch`, `RadArchProbe::gen_duplex`, `RadBatch::dup_b`; layout rows; regenerate the table | `abi_layout_test` (C and C++ agree, rows tile every byte, table matches 38), `abi_c_test` as part of it, every shipped plugin still loads and their layout fingerprint matches |
| 1. Builder and `Program` | M (3-5 days) | `Builder::declare_duplex_tenant` validation, string copies, `Program::is_duplex_tenant()`, finish diagnostics, startup probe sizing | builder unit tests; startup refuses a malformed declaration by name; `--debug-graph` shows the new pass; an audio generator still declares exactly as before |
| 2. Duplex driver, host only | L (1-2 weeks) | new `gen::DuplexDriver`: `open_lane`/`step`/`flush`/`close`; prompt and reference staging; one full-attention group plus state groups; `audio_in`/`audio_out`/`text_out`/`status` readback; ticket/cancel | new `duplex_driver_test`: `echo` and `echo1` style deterministic fake plugin, one-in/one-out, flush, lane release, wrong length/seq, lane limit; `audio_gen_test` unchanged |
| 3. Qwen3.5 trunk and reference prefill | XL (2-3 weeks) | reuse `qwen35_fp8`'s geometry, three state groups, bindings and blocks; declare card/reference prompt; two projectors; tied head; positions/MROPE | graph parity against qwen35_fp8 on a text-only prompt; 8 full / 24 GDN state groups staged; reference rows embed through `P_model`; no step-path allocation; one live step runs on the host oracle |
| 4. MeanFlow and codec | XL (2-3 weeks) | 4-layer MeanFlow head, one sampling step; causal codec encode/decode and per-lane state; 64-dim latents, 3200-sample frames | golden latent and sample tests against the reference implementation; 7.5 fps geometry; a run of N frames does not drift; p95 step time measured against 100 ms |
| 5. Text output and tenant seam | M (1 week) | optional text token from the tied head; `/duplex/v1/info` and the WebSocket adapter; `flags` bits; errors and close | wire tests for `ready`, binary step, `flush`, `close`, `bad_frame`, `bad_sequence`, `session_limit`, `tenant_unavailable`; no card/reference/audio in logs |
| 6. Multi-lane and hardening | L (1-2 weeks) | fair batching, per-lane deadline/cancel, KV bound/eviction or streaming, state reset on reconnect, metrics | concurrent lanes at `max_lanes`; one lane's flush does not stall another; close releases KV/state; memory budget and p95 at the target concurrency |
| 7. Quantised/TP deployment | L (1-2 weeks) | the trunk's quantised blocks and tensor-parallel all-reduces (projectors, embedding, tied head); per-rank readback of the output frame | TP2 and the quantised container match the single-rank output within the reference tolerance; audio and text are identical across ranks; ABI fingerprint matches |

Stages 0-2 can proceed without a model and are the right first merge: they make the seam and the
ABI reviewable while the model work is still exploratory.

---

## 8. ABI 39 addendum: host-side text sampling and host noise

The v0 note above is the ABI-38 design. The assembled Haru tenant needs the host, not the plugin,
to sample text, because DRY, the PAD/EOP replacement, `eop_bias` and the temperature/top-k/top-p
sampler belong to the session policy. ABI 39 appends (never moves) the following fields:

`RadDuplexDecl`, after `status`:

```c
int64_t n_vocab;        /* > 0 with host_sampling */
int32_t host_sampling;  /* 1: the host samples logits_out and writes text_in */
int32_t pad_id, eop_id, eos_id;   /* -1 when unused */
rad_buf logits_out;     /* f32 [max_lanes, n_vocab] device */
rad_buf pad_eop_out;    /* f32 [max_lanes, 2] device */
rad_buf text_in;        /* i32 [max_lanes] device */
double codec_noise_scale; /* latent_std / 0.8; > 0 and finite with host_sampling */
```

`RadDuplexBatch`, after `flags_h`:

```c
const float* codec_noise;
const float* codec_noise_h;
const float* head_noise;
const float* head_noise_h;
```

The plugin writes the current model row's raw f32 logits and the f32 PAD/EOP pair. The host applies
DRY, replaces the PAD/EOP entries, adds `eop_bias`, samples, and writes the chosen token to
`text_in[lane]`; the next STEP's embedding reads it. `text_out` is unused when
`host_sampling = 1`. `codec_noise` is the combined term added to the codec's latent means
(`scale * codec_noise_scale * raw`), one row per reference row on PROMPT and one row a lane on
STEP; `head_noise` is the raw vector the MeanFlow head starts from. The plugin declares
`codec_noise_scale` because it reads the checkpoint's `latent_std`; the host does not. Both noise
buffers are host inputs in production (seed-derived) and in teacher-forced tests (recorded values);
no RNG lives in a block.

A host-sampling tenant's production source (`core/server/duplex_host.{h,cpp}`) owns one
`LaneSampler`, one noise PRNG and one `StreamDetokenizer` per lane. The driver tells it when a lane
opens and closes (`DuplexHostSource::lane_opened`/`lane_closed`), and the source returns the text
delta through `reply_text`; the server maps it to the wire and sets `flags` bit 1 when it is
non-empty. The server clamps every engine reply's audio to `[-1, 1]` (NaN to 0, infinities to the
endpoints) because the plugin has no constant-bound clamp op; the echo test tenants do not go
through that path and stay bit-exact. A lane's seed is `duplex_seed(session_id)`, a pinned FNV-1a
64-bit hash (`core/server/duplex.h`), not `std::hash`, so a replay does not depend on the standard
library.

`RadDuplexBatch::seed` is now documented as two 32-bit words a lane: `seed[2*i]` low,
`seed[2*i+1]` high. ABI 39 regenerates the layout table and moves `RAD_ABI_VERSION` to 39; see
`abi/rad_abi_layout_rows.h` and `tests/abi_layout_table.h`.
