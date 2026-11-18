/* duplex_sampler.h -- host-side text sampling and stream detokenization for the Haru duplex
 * model (rad::gen). This is the host half of decision D1: the plugin returns one text-logits row
 * per frame and the host chooses the token, with no engine, tokenizer or device dependency.
 *
 * Contract
 * --------
 * LaneSampler::sample() applies, in this exact order (the reference is
 * .haru-ref/session.py DuplexSession._emit, lines 148-157):
 *
 *   1. DRY: subtract the lane's current penalties from the logits in place
 *      (.haru-ref/session.py:150-151, .haru-ref/sampling.py:71-77).
 *   2. Replace the PAD and EOP entries with the f32-recomputed pair the caller passes as
 *      pad_eop[0] = PAD, pad_eop[1] = EOP (.haru-ref/session.py:152; the ids are built as
 *      [pad, eop] at .haru-ref/session.py:80).
 *   3. Add eop_bias to the EOP entry (.haru-ref/session.py:153-154).
 *   4. If forced-text frames remain, set the banned set {PAD, EOP, EOS} to -inf and consume one
 *      frame (.haru-ref/session.py:155-156). EOS is present only when eos_id >= 0
 *      (.haru-ref/session.py:82).
 *   5. Sample (.haru-ref/session.py:157): temperature <= 0 is greedy, otherwise temperature
 *      scaling, top-k (partial sort) or a full descending sort, float softmax with max
 *      subtraction, the reference's top-p keep rule
 *      (cumsum - prob < top_p, .haru-ref/sampling.py:96-98), then inverse-CDF sampling over the
 *      kept probabilities in descending-value order.
 *   6. Append the chosen token to the DRY history unless it is exempt (PAD/EOP/EOS)
 *      (.haru-ref/session.py:164-165, .haru-ref/sampling.py:60-69).
 *
 * The logits array is modified in place by steps 1-4, exactly what the reference leaves in its
 * tensor before sample_token(); the temperature scaling and probability work happen on a private
 * copy. Ties: descending logit, then ascending token id, everywhere (top-k cut and inverse CDF).
 * Greedy returns the first index of the maximum, like torch.argmax.
 *
 * The DRY history is per lane, trimmed to lookback + max_match_length + 1 entries, and exempt
 * tokens are never recorded nor penalised (.haru-ref/sampling.py:20-23, 60-69). Penalties are
 * computed in double as the reference's Python integers/floats are, then cast to float32 when
 * subtracted, like DryState.apply's dtype cast (.haru-ref/sampling.py:71-77).
 *
 * What this is not
 * ----------------
 * It is not a tokenizer: StreamDetokenizer takes the server's decode callback. It is not an
 * engine, does not read config files, does not touch a device, and knows nothing of audio. It is
 * not thread-safe: one LaneSampler or StreamDetokenizer per lane, never shared between threads or
 * lanes. It does not reproduce torch's PRNG stream; the internal splitmix64 is only a source of
 * one uniform in [0,1) per sampled token, and the overload taking a uniform lets tests replay
 * recorded draws. Greedy consumes no draw, as the reference's greedy path never asks the
 * generator.
 */
#pragma once
#include <cstdint>
#include <functional>
#include <map>
#include <set>
#include <string>
#include <vector>

namespace rad {
namespace gen {

/* The reference's DryConfig (.haru-ref/sampling.py:10-23). */
struct DryConfig {
    float multiplier       = 0.8f;   /* .haru-ref/sampling.py:14 */
    float base             = 1.75f;  /* .haru-ref/sampling.py:15 */
    int   allowed_length   = 1;      /* .haru-ref/sampling.py:16 */
    int   lookback         = 1024;   /* .haru-ref/sampling.py:17 */
    int   max_match_length = 32;     /* .haru-ref/sampling.py:18 */
    float max_penalty      = 32.0f;  /* .haru-ref/sampling.py:19 */

    int history_size() const { return lookback + max_match_length + 1; }  /* sampling.py:22-23 */
};

/* DuplexConfig's sampling fields (.haru-ref/session.py:35-40) plus the DRY block. speak_first
 * defaults to false (.haru-ref/session.py:40), which the session turns into one forced-text
 * frame when it is true and zero otherwise (.haru-ref/session.py:112). */
struct SamplerConfig {
    float     temperature       = 0.7f;  /* .haru-ref/session.py:35 */
    int       top_k             = 25;    /* .haru-ref/session.py:36 */
    float     top_p             = 1.0f;  /* .haru-ref/session.py:37 */
    DryConfig dry;                       /* .haru-ref/session.py:38 */
    float     eop_bias          = 0.0f;  /* .haru-ref/session.py:39 */
    int       force_text_frames = 0;     /* false by default, .haru-ref/session.py:40,112 */
};

/* One lane's sampler: DRY history/penalties, the forced-text count and a private RNG. */
class LaneSampler {
public:
    /* eos_id may be -1: the reference only bans EOS when the checkpoint has one
     * (.haru-ref/session.py:82). The seed initialises splitmix64; equal seeds give equal streams. */
    LaneSampler(const SamplerConfig& config, int pad_id, int eop_id, int eos_id, uint64_t seed);

    /* The full step: DRY, PAD/EOP replacement, eop_bias, optional forced-text ban, sample, then
     * update the DRY history. Returns the token id. `logits` is n_vocab floats, modified in place
     * by the first four steps. `pad_eop` is [PAD, EOP] as the f32 path recomputed them. */
    int sample(float* logits, int n_vocab, const float pad_eop[2]);

    /* The same step with an externally supplied uniform in [0,1) for the inverse CDF. The internal
     * RNG is not advanced, so a test can replay recorded draws. Greedy ignores the value. */
    int sample(float* logits, int n_vocab, const float pad_eop[2], double uniform);

    /* Append a token to the DRY history (the reference's DryState.update, sampling.py:60-69):
     * exempt tokens are ignored, the history is trimmed, and penalties are recomputed. The full
     * sample() calls this with the token it chose; tests call it to replay a history. */
    void observe(int token);

    const std::map<int, float>& penalties() const { return penalties_; }
    const std::vector<int>&     history() const { return history_; }
    int                         force_text_left() const { return force_text_left_; }
    uint64_t                    rng_state() const { return rng_state_; }

private:
    double next_uniform();
    int    draw(float* logits, int n_vocab, double uniform);
    void   recompute_penalties();

    SamplerConfig       config_;
    int                 pad_id_;
    int                 eop_id_;
    int                 eos_id_;
    std::set<int>       exempt_;
    std::vector<int>    history_;
    std::map<int, float> penalties_;
    int                 force_text_left_;
    uint64_t            rng_state_;
};

/* The reference's StreamDetokenizer (.haru-ref/sampling.py:102-126). A run of silent ids renders
 * as one space; a token run whose decode still ends in U+FFFD (EF BF BD) is held until the next
 * token completes the character. `decode` receives the pending ids and returns UTF-8. One
 * instance per lane; it is not thread-safe. */
class StreamDetokenizer {
public:
    StreamDetokenizer(std::function<std::string(const std::vector<int>&)> decode,
                      std::set<int> silent_ids);

    std::string push(int token);

private:
    std::function<std::string(const std::vector<int>&)> decode_;
    std::set<int>    silent_ids_;
    std::vector<int> pending_;
    bool             in_gap_ = false;
};

}  // namespace gen
}  // namespace rad
