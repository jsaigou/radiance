# Duplex engine contract (v0)

**Status: v0, proposed.** This is the interface a model engine provides so that a *client* (a
service that owns the audio clock, the user connection and session policy) can run a continuous
speech-to-speech conversation on it. It is engine-neutral: nothing in the wire contract names a
model. Radiance implements the engine side of it.

## 1. Roles

The **client** owns the frame clock, queues, deadlines, cancellation and everything user-facing. The
**engine** owns the model, its per-lane state and its weights. The engine is **passive**: it never
decides when a frame is due. It answers one input frame with one reply, in order. Any buffering
inside the engine is private and must not change the one-in/one-out rule.

One lane is one conversation: one WebSocket, one model state.

## 2. Transport

- WebSocket on loopback, path `/duplex/v1/session`. Control messages are JSON **text** frames;
  audio is **binary** frames. The two kinds must be distinguishable by frame type, not by content.
- Discovery: `GET /duplex/v1/info` (JSON), see section 3.
- Authentication: none required on loopback by default. If the engine requires a key, the client sends
  `Authorization: Bearer <key>`; use the engine's existing API-key mechanism. A key is never put in a URL
  or a log.
- The WebSocket **is** the lane's lifetime. If it closes or drops for any reason, the engine must
  release the lane's model state. There is no resume.

## 3. `GET /duplex/v1/info`

```json
{
  "contract": "0",
  "engine": "radiance",
  "engine_version": "...",
  "tenants": [{"name": "echo", "text": false, "max_lanes": 2}],
  "frame": {"sample_rate": 24000, "samples": 3200, "format": "f32le"},
  "ready": true
}
```

`ready: false` means the model is still loading. The client treats that as retryable. Unknown fields
are ignored by both sides.

## 4. Session messages

### 4.1 `start` (client to engine, text, first message)

```json
{"type":"start","contract":"0","session_id":"...","tenant":"echo","text":false,
 "max_minutes":30,"card":{},"reference":{"id":"","bytes":0,"transcript":""}}
```

`card` is opaque to the engine except for what the tenant understands; unknown keys are ignored. If
`reference.bytes > 0`, exactly that many bytes follow as **one binary message** before the engine
replies. The engine replies with exactly one of:

- `{"type":"ready","text":false,"input_rates":[24000]}`
- `{"type":"error", ...}` (section 5), after which it closes the socket.

### 4.2 Step (lockstep, binary both ways)

Client to engine, one binary message per input frame:

| Offset | Size | Field |
|---|---|---|
| 0 | 4 | `seq`, uint32 little-endian, starts at 0, +1 per frame |
| 4 | 12800 | 3200 samples, float32 little-endian, mono, 24 kHz, range [-1, 1] |

Engine to client, exactly **one** binary message per input `seq`, in order:

| Offset | Size | Field |
|---|---|---|
| 0 | 4 | `seq` echoed |
| 4 | 1 | `flags`: bit0 = audio present, bit1 = text present |
| 5 | 4 | `text_len`, uint32 little-endian (0 when bit1 is clear) |
| 9 | `text_len` | UTF-8 text delta |
| ... | 12800 if bit0 | 3200 float32 little-endian samples |

A model with nothing to say replies with `flags = 0`: it still answers, so every frame is accounted
for. The client keeps at most one step in flight per lane and enforces its own deadline.

### 4.3 Other messages

| Direction | Message | Meaning |
|---|---|---|
| client to engine | `{"type":"flush"}` | The engine replies with one step-shaped binary message holding any frame it buffered (`seq` = the last step's), or `flags = 0` if it has none. |
| client to engine | `{"type":"close","reason":"..."}` | Release the lane, then close the socket. |
| engine to client | `{"type":"meta", ...}` | Optional per-frame diagnostics. Never required. |
| either | WebSocket ping/pong | Liveness only. |

## 5. Errors

An engine error is JSON text: `{"type":"error","code":"...","message":"...","fatal":true}`. Codes:
`model_not_ready`, `tenant_unavailable`, `session_limit`, `bad_frame`, `bad_sequence`, `timeout`,
`internal`, `bad_request`. After `ready`, every error is fatal for the lane: send it, then close.
A `message` must not contain reference audio, card text, keys or tokens.

## 6. Rules

- Step time target: p95 under 100 ms (the frame period is 133.33 ms). Never block the accept path
  while loading: answer `ready: false` until the weights are resident.
- Lane limits are the engine's: `max_lanes` in `/info`; beyond it, `session_limit`.
- An engine MAY close a lane that has received no message for its idle limit, or that has run past
  `start.max_minutes`, with a fatal `timeout` error.
- A lane holds no state that affects a later lane. Closing a lane releases all of its state.
- A step message of the wrong length is `bad_frame`; a `seq` other than the next expected is
  `bad_sequence`. Both are fatal.
- One engine may serve several tenants; the client selects by name in `start`; an unknown name is
  `tenant_unavailable`.
- Engines may cache per-voice work keyed on a hash of the card and reference audio. The contract does
  not require it.

## 7. Versioning

`contract: "0"` is the only version. Changes are additive: new optional fields and new `flags` bits.
A breaking change bumps `contract`. A client refuses an engine whose `contract` it does not know.

## 8. Test tenants

An engine should offer two **test tenants**, off by default and enabled only by an explicit flag,
so a client can verify the seam without a model:

- `echo`: replies in the same step with the audio it was given, unchanged (bit-exact).
- `echo1`: replies one step later (the first reply has `flags = 0`); `flush` returns the held frame.
