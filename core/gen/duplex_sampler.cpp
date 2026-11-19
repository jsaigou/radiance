/* duplex_sampler.cpp -- the host text sampler and stream detokenizer. See duplex_sampler.h for
 * the contract; the comments here name the reference lines each block ports. */
#include "gen/duplex_sampler.h"

#include <algorithm>
#include <cmath>
#include <limits>
#include <utility>

namespace rad {
namespace gen {

namespace {

struct Candidate {
    float value;
    int   id;
    float prob = 0.0f;
};

/* Descending value, then ascending id: the stable tie rule used for the top-k cut and for the
 * inverse CDF's order. */
bool better(const Candidate& a, const Candidate& b) {
    if (a.value != b.value) return a.value > b.value;
    return a.id < b.id;
}

/* Does `s` end with U+FFFD's UTF-8 bytes EF BF BD? (.haru-ref/sampling.py:123) */
bool ends_with_replacement(const std::string& s) {
    return s.size() >= 3 && static_cast<unsigned char>(s[s.size() - 1]) == 0xBD &&
           static_cast<unsigned char>(s[s.size() - 2]) == 0xBF &&
           static_cast<unsigned char>(s[s.size() - 3]) == 0xEF;
}

/* Length of the well-formed UTF-8 sequence starting at s[i], or 0 if there is none. Overlong forms,
 * surrogates and code points above U+10FFFF are not well-formed. `incomplete` is set when the bytes
 * are a valid prefix of a sequence that the end of the string cut short. */
size_t utf8_seq(const std::string& s, size_t i, bool* incomplete) {
    const unsigned char c = static_cast<unsigned char>(s[i]);
    *incomplete = false;
    if (c < 0x80) return 1;
    size_t n;
    unsigned char lo = 0x80, hi = 0xBF;
    if (c >= 0xC2 && c <= 0xDF) n = 2;
    else if (c == 0xE0) { n = 3; lo = 0xA0; }
    else if (c == 0xED) { n = 3; hi = 0x9F; }
    else if (c >= 0xE1 && c <= 0xEF) n = 3;
    else if (c == 0xF0) { n = 4; lo = 0x90; }
    else if (c == 0xF4) { n = 4; hi = 0x8F; }
    else if (c >= 0xF1 && c <= 0xF3) n = 4;
    else return 0;
    for (size_t k = 1; k < n; ++k) {
        if (i + k >= s.size()) { *incomplete = true; return 0; }
        const unsigned char d = static_cast<unsigned char>(s[i + k]);
        const unsigned char l = (k == 1) ? lo : 0x80, h = (k == 1) ? hi : 0xBF;
        if (d < l || d > h) return 0;
    }
    return n;
}

/* Does the string end inside a multi-byte character that later bytes could still complete? */
bool ends_incomplete(const std::string& s) {
    size_t i = 0;
    while (i < s.size()) {
        bool inc = false;
        const size_t n = utf8_seq(s, i, &inc);
        if (inc) return true;
        i += n ? n : 1;
    }
    return false;
}

/* The same text with every ill-formed byte replaced by U+FFFD, so the wire never carries bytes
 * that are not UTF-8. */
std::string sanitize_utf8(const std::string& s) {
    std::string out;
    out.reserve(s.size());
    size_t i = 0;
    while (i < s.size()) {
        bool inc = false;
        const size_t n = utf8_seq(s, i, &inc);
        if (n) { out.append(s, i, n); i += n; }
        else { out += "\xEF\xBF\xBD"; ++i; }
    }
    return out;
}

/* A character is at most 4 bytes; a held run longer than this is not going to complete. */
constexpr size_t kMaxPendingTokens = 8;

}  // namespace

LaneSampler::LaneSampler(const SamplerConfig& config, int pad_id, int eop_id, int eos_id,
                         uint64_t seed)
    : config_(config),
      pad_id_(pad_id),
      eop_id_(eop_id),
      eos_id_(eos_id),
      force_text_left_(config.force_text_frames),
      rng_state_(seed) {
    exempt_.insert(pad_id_);
    exempt_.insert(eop_id_);
    if (eos_id_ >= 0) exempt_.insert(eos_id_);  /* .haru-ref/session.py:82 */
}

double LaneSampler::next_uniform() {
    /* splitmix64: one draw per sampled token, in [0,1). Not torch's generator; the stream is
     * documented here so the statistical test can reproduce it with a plain script. */
    uint64_t z = (rng_state_ += 0x9E3779B97F4A7C15ULL);
    z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ULL;
    z = (z ^ (z >> 27)) * 0x94D049BB133111EBULL;
    z ^= z >> 31;
    return static_cast<double>(z >> 11) * (1.0 / 9007199254740992.0);
}

void LaneSampler::recompute_penalties() {
    penalties_.clear();
    const DryConfig& c = config_.dry;
    const int n = static_cast<int>(history_.size());
    if (n < std::max(1, c.allowed_length)) return;  /* .haru-ref/sampling.py:29 */

    std::map<int, int> longest;
    const int last = history_[static_cast<size_t>(n - 1)];
    const int low = std::max(0, n - 1 - c.lookback);
    for (int j = n - 2; j >= low; --j) {  /* .haru-ref/sampling.py:33 */
        if (history_[static_cast<size_t>(j)] != last) continue;
        int length = 1;
        while (length < c.max_match_length && j - length >= 0 &&
               history_[static_cast<size_t>(j - length)] ==
                   history_[static_cast<size_t>(n - 1 - length)]) {
            ++length;  /* .haru-ref/sampling.py:36-41 */
        }
        if (length >= c.allowed_length) {
            const int candidate = history_[static_cast<size_t>(j + 1)];
            auto it = longest.find(candidate);
            if (it == longest.end() || length > it->second) longest[candidate] = length;
        }
    }

    for (const auto& [token, length] : longest) {
        if (exempt_.count(token)) continue;  /* .haru-ref/sampling.py:68 */
        const double p =
            std::min(static_cast<double>(c.max_penalty),
                     static_cast<double>(c.multiplier) *
                         std::pow(static_cast<double>(c.base),
                                  static_cast<double>(length - c.allowed_length)));
        penalties_[token] = static_cast<float>(p);  /* apply casts to float32, sampling.py:74 */
    }
}

void LaneSampler::observe(int token) {
    if (exempt_.count(token)) return;  /* .haru-ref/sampling.py:61-62 */
    history_.push_back(token);
    int extra = static_cast<int>(history_.size()) - config_.dry.history_size();
    if (extra > 0) {  /* .haru-ref/sampling.py:66-68 */
        extra = std::min(extra, static_cast<int>(history_.size()));
        history_.erase(history_.begin(), history_.begin() + extra);
    }
    recompute_penalties();
}

int LaneSampler::sample(float* logits, int n_vocab, const float pad_eop[2]) {
    if (!logits || n_vocab <= 0) return -1;
    /* Greedy consumes no draw, exactly as the reference never asks the generator on that path. */
    if (config_.temperature <= 0) return sample(logits, n_vocab, pad_eop, 0.0);
    return sample(logits, n_vocab, pad_eop, next_uniform());
}

int LaneSampler::sample(float* logits, int n_vocab, const float pad_eop[2], double uniform) {
    if (!logits || n_vocab <= 0) return -1;
    for (const auto& [token, penalty] : penalties_) {
        if (token >= 0 && token < n_vocab) logits[token] -= penalty;  /* session.py:150-151 */
    }
    if (pad_id_ >= 0 && pad_id_ < n_vocab) logits[pad_id_] = pad_eop[0];  /* session.py:152 */
    if (eop_id_ >= 0 && eop_id_ < n_vocab) logits[eop_id_] = pad_eop[1];
    if (config_.eop_bias != 0.0f && eop_id_ >= 0 && eop_id_ < n_vocab)
        logits[eop_id_] += config_.eop_bias;  /* session.py:153-154 */
    if (force_text_left_ > 0) {  /* session.py:155-156 */
        for (int id : exempt_) {
            if (id >= 0 && id < n_vocab) logits[id] = -std::numeric_limits<float>::infinity();
        }
        --force_text_left_;
    }
    return draw(logits, n_vocab, uniform);
}

int LaneSampler::draw(float* logits, int n_vocab, double uniform) {
    int token = -1;
    if (n_vocab <= 0) return token;

    if (config_.temperature <= 0) {
        /* Greedy: first index on ties, like torch.argmax (.haru-ref/sampling.py:88-89). */
        token = 0;
        for (int i = 1; i < n_vocab; ++i)
            if (logits[i] > logits[token]) token = i;
    } else {
        std::vector<Candidate> cand;
        cand.reserve(static_cast<size_t>(n_vocab));
        const float temp = config_.temperature;
        for (int i = 0; i < n_vocab; ++i)
            cand.push_back(Candidate{ logits[i] / temp, i });  /* sampling.py:90 */

        if (config_.top_k > 0 && config_.top_k < n_vocab) {  /* sampling.py:91-93 */
            const auto k = static_cast<size_t>(config_.top_k);
            std::partial_sort(cand.begin(), cand.begin() + k, cand.end(), better);
            cand.resize(k);
        } else {
            std::sort(cand.begin(), cand.end(), better);  /* sampling.py:94-95 */
        }

        /* Float softmax with max subtraction over the kept values (sampling.py:96). */
        const float maxv = cand.front().value;
        if (!(maxv > -std::numeric_limits<float>::infinity())) {
            /* Every kept logit is -inf (only possible if the banned set filled the top-k).
             * The reference's torch.softmax yields NaN here; choose the first candidate rather
             * than propagate NaN, and keep the history update real. */
            token = cand.front().id;
        } else {
            float sum = 0.0f;
            for (auto& c : cand) {
                c.prob = std::exp(c.value - maxv);
                sum += c.prob;
            }
            for (auto& c : cand) c.prob /= sum;

            if (config_.top_p > 0.0f && config_.top_p < 1.0f) {
                /* keep = cumsum - prob < top_p: the sum strictly before this candidate. */
                float prev = 0.0f;
                for (auto& c : cand) {
                    const float p = c.prob;
                    if (!(prev < config_.top_p)) c.prob = 0.0f;
                    prev += p;
                }
            }

            double total = 0.0;
            for (const auto& c : cand)
                if (c.prob > 0.0f) total += c.prob;

            if (!(total > 0.0)) {
                token = cand.front().id;
            } else {
                if (uniform < 0.0) uniform = 0.0;
                double target = uniform * total;
                double acc = 0.0;
                int last_positive = cand.front().id;
                for (const auto& c : cand) {
                    if (!(c.prob > 0.0f)) continue;
                    last_positive = c.id;
                    acc += c.prob;
                    if (target < acc) {
                        token = c.id;
                        break;
                    }
                }
                if (token < 0) token = last_positive;
            }
        }
    }

    observe(token);
    return token;
}

StreamDetokenizer::StreamDetokenizer(std::function<std::string(const std::vector<int>&)> decode,
                                     std::set<int> silent_ids)
    : decode_(std::move(decode)), silent_ids_(std::move(silent_ids)) {}

std::string StreamDetokenizer::push(int token) {
    if (silent_ids_.count(token)) {  /* .haru-ref/sampling.py:112-116 */
        if (in_gap_) return std::string();
        in_gap_ = true;
        return std::string(" ");
    }
    in_gap_ = false;
    pending_.push_back(token);
    const std::string text = decode_ ? decode_(pending_) : std::string();
    /* The reference holds while the decoded text ends in U+FFFD (sampling.py:123-124). A tokenizer
     * that returns the raw bytes of a split character does not produce U+FFFD, so an incomplete
     * UTF-8 tail is held too, and what is finally emitted is always well-formed UTF-8. A run that
     * cannot complete (kMaxPendingTokens) is flushed with U+FFFD instead of being held forever. */
    if ((ends_with_replacement(text) || ends_incomplete(text)) &&
        pending_.size() < kMaxPendingTokens)
        return std::string();
    pending_.clear();
    return sanitize_utf8(text);
}

}  // namespace gen
}  // namespace rad
