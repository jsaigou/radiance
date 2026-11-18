/* duplex_host.cpp -- the production host source for a host-sampling duplex tenant. See
 * duplex_host.h for the contract and docs/HARU-H6-DESIGN.md sections 2-5 for the reasoning. */
#include "server/duplex_host.h"

#include <algorithm>
#include <cmath>
#include <limits>
#include <set>
#include <utility>

namespace rad {
namespace server {

namespace {

constexpr double kPi = 3.14159265358979323846;

std::set<int> silent_ids(const HostSampleSource::Options& o) {
    std::set<int> s;
    if (o.pad_id >= 0) s.insert(o.pad_id);
    if (o.eop_id >= 0) s.insert(o.eop_id);
    return s;
}

void fail(std::string* why, const char* what) {
    if (why) *why = what;
}

}  /* namespace */

HostSampleSource::NoisePrng::NoisePrng(uint64_t seed)
    : state_(seed ^ 0xD1B54A32D192ED03ULL) {}

double HostSampleSource::NoisePrng::uniform() {
    uint64_t z = (state_ += 0x9E3779B97F4A7C15ULL);
    z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ULL;
    z = (z ^ (z >> 27)) * 0x94D049BB133111EBULL;
    z ^= z >> 31;
    return static_cast<double>(z >> 11) * (1.0 / 9007199254740992.0);
}

void HostSampleSource::NoisePrng::normals(int n, float* out) {
    if (!out || n <= 0) return;
    for (int i = 0; i < n; i += 2) {
        double u1 = uniform();
        if (!(u1 > 0.0)) u1 = 1.0 / 9007199254740992.0;  /* 2^-53 */
        const double u2 = uniform();
        const double r = std::sqrt(-2.0 * std::log(u1));
        const double a = 2.0 * kPi * u2;
        out[i] = static_cast<float>(r * std::cos(a));
        if (i + 1 < n) out[i + 1] = static_cast<float>(r * std::sin(a));
    }
}

HostSampleSource::Lane::Lane(const HostSampleSource::Options& o, uint64_t seed)
    : sampler(o.sampler, o.pad_id, o.eop_id, o.eos_id, seed),
      noise(seed),
      detok(o.decode ? o.decode
                     : std::function<std::string(const std::vector<int>&)>(
                           [](const std::vector<int>&) { return std::string(); }),
            silent_ids(o)) {}

HostSampleSource::HostSampleSource(HostSampleSource::Options o) : o_(std::move(o)) {
    if (o_.n_vocab > 0) logits_.assign(static_cast<size_t>(o_.n_vocab), 0.0f);
    lanes_.resize(static_cast<size_t>(std::max<int64_t>(o_.max_lanes, 1)));
}

HostSampleSource::Lane* HostSampleSource::lane_locked(int32_t lane) {
    if (lane < 0 || static_cast<size_t>(lane) >= lanes_.size()) return nullptr;
    return lanes_[static_cast<size_t>(lane)].get();
}

int HostSampleSource::check_pass(const gen::DuplexHostPass& p, std::string* why) const {
    if (o_.latent_dim > 0 && p.latent_dim != o_.latent_dim) {
        fail(why, "the host source's latent width does not match the pass");
        return RAD_E_INVAL;
    }
    if (o_.n_vocab > 0 && p.n_vocab != 0 && p.n_vocab != o_.n_vocab) {
        fail(why, "the host source's vocabulary does not match the pass");
        return RAD_E_INVAL;
    }
    if (!p.lane) {
        fail(why, "the pass carries no lane array");
        return RAD_E_INVAL;
    }
    return RAD_OK;
}

void HostSampleSource::lane_opened(int32_t lane, uint64_t seed) {
    if (lane < 0) return;
    std::lock_guard<std::mutex> lk(mu_);
    if (static_cast<size_t>(lane) >= lanes_.size())
        lanes_.resize(static_cast<size_t>(lane) + 1);
    lanes_[static_cast<size_t>(lane)].reset(new Lane(o_, seed));
}

void HostSampleSource::lane_closed(int32_t lane) {
    if (lane < 0) return;
    std::lock_guard<std::mutex> lk(mu_);
    if (static_cast<size_t>(lane) < lanes_.size())
        lanes_[static_cast<size_t>(lane)].reset();
}

int HostSampleSource::noise(const gen::DuplexHostPass& p, float* codec_noise, float* head_noise,
                            std::string* why) {
    std::lock_guard<std::mutex> lk(mu_);
    if (check_pass(p, why) < 0) return RAD_E_INVAL;
    if (!codec_noise || !head_noise) {
        fail(why, "the pass carries no noise buffers");
        return RAD_E_INVAL;
    }
    if (!(o_.codec_noise_scale > 0.0) || !std::isfinite(o_.codec_noise_scale)) {
        fail(why, "the tenant's codec_noise_scale must be > 0 and finite");
        return RAD_E_INVAL;
    }
    const int64_t D = p.latent_dim;
    if (D <= 0 || D > (int64_t)std::numeric_limits<int>::max()) {
        fail(why, "the pass's latent width is not usable");
        return RAD_E_INVAL;
    }
    const int dim = static_cast<int>(D);
    const float scale_term = static_cast<float>(o_.codec_noise_scale);

    /* A test may supply the normal stream; production uses the lane's own PRNG. */
    auto draw = [&](Lane& lane, int n, float* out) {
        if (o_.normal) {
            for (int i = 0; i < n; ++i) out[i] = static_cast<float>(o_.normal());
        } else {
            lane.noise.normals(n, out);
        }
    };

    if (p.kind == RAD_GEN_PASS_DUPLEX_PROMPT) {
        if (p.n_lane != 1) {
            fail(why, "a PROMPT pass carries one lane");
            return RAD_E_INVAL;
        }
        Lane* lane = lane_locked(p.lane[0]);
        if (!lane) {
            fail(why, "the PROMPT lane has no host state");
            return RAD_E_STATE;
        }
        const int64_t T = p.n_tok;
        if (T < 0) {
            fail(why, "the PROMPT row count is negative");
            return RAD_E_INVAL;
        }
        std::fill(codec_noise, codec_noise + T * D, 0.0f);

        int64_t frames = 0;
        if (p.ref_frame) {
            for (int64_t t = 0; t < T; ++t)
                if (p.ref_frame[t] >= 0) frames = std::max<int64_t>(frames, p.ref_frame[t] + 1);
            /* The server appends one row a reference frame; two rows for one frame would give
             * that frame two different noises. */
            std::vector<int> seen(static_cast<size_t>(frames), 0);
            for (int64_t t = 0; t < T; ++t) {
                const int32_t f = p.ref_frame[t];
                if (f < 0) continue;
                if (f >= frames || seen[static_cast<size_t>(f)]++) {
                    fail(why, "the PROMPT names a reference frame twice");
                    return RAD_E_INVAL;
                }
            }
        }

        if (frames > 0) {
            float scale = 0.0f;
            draw(*lane, 1, &scale);
            std::vector<float> raw(static_cast<size_t>(D));
            for (int64_t f = 0; f < frames; ++f) {
                draw(*lane, dim, raw.data());
                bool used = false;
                for (int64_t t = 0; t < T; ++t) {
                    if (p.ref_frame[t] != f) continue;
                    used = true;
                    for (int64_t d = 0; d < D; ++d)
                        codec_noise[t * D + d] = scale * scale_term * raw[static_cast<size_t>(d)];
                }
                if (!used) {
                    fail(why, "the PROMPT reference frames are not contiguous");
                    return RAD_E_INVAL;
                }
            }
        }
        /* Frame 0's head noise, one row for the one prompt lane. */
        draw(*lane, dim, head_noise);
        return RAD_OK;
    }

    /* STEP: the noise buffers are entry-indexed, exactly as the plugin reads them. */
    for (int64_t i = 0; i < p.n_lane; ++i) {
        Lane* lane = lane_locked(p.lane[i]);
        if (!lane) {
            fail(why, "a STEP lane has no host state");
            return RAD_E_STATE;
        }
        if (p.frame) {
            if (p.frame[i] != lane->next_frame) {
                fail(why, "a STEP frame is not the lane's next frame");
                return RAD_E_STATE;
            }
        }
        float scale = 0.0f;
        draw(*lane, 1, &scale);
        std::vector<float> raw(static_cast<size_t>(D));
        draw(*lane, dim, raw.data());
        for (int64_t d = 0; d < D; ++d)
            codec_noise[i * D + d] = scale * scale_term * raw[static_cast<size_t>(d)];
        draw(*lane, dim, head_noise + i * D);
        if (p.frame) ++lane->next_frame;
    }
    return RAD_OK;
}

int HostSampleSource::tokens(const gen::DuplexHostPass& p, int32_t* out, std::string* why) {
    std::lock_guard<std::mutex> lk(mu_);
    if (check_pass(p, why) < 0) return RAD_E_INVAL;
    if (!out || !p.logits || !p.pad_eop) {
        fail(why, "the pass carries no token readbacks");
        return RAD_E_INVAL;
    }
    if (p.n_vocab <= 0 || p.n_vocab > (int64_t)std::numeric_limits<int>::max()) {
        fail(why, "the pass's vocabulary is not usable");
        return RAD_E_INVAL;
    }
    if (logits_.size() != static_cast<size_t>(p.n_vocab))
        logits_.resize(static_cast<size_t>(p.n_vocab));
    const int n_vocab = static_cast<int>(p.n_vocab);

    for (int64_t i = 0; i < p.n_lane; ++i) {
        Lane* lane = lane_locked(p.lane[i]);
        if (!lane) {
            fail(why, "a token lane has no host state");
            return RAD_E_STATE;
        }
        /* The readbacks are lane-indexed, not entry-indexed (core/gen/duplex_driver.h). */
        const float* row = p.logits + static_cast<size_t>(p.lane[i]) * p.n_vocab;
        std::copy(row, row + n_vocab, logits_.begin());
        const float* pad_eop = p.pad_eop + static_cast<size_t>(p.lane[i]) * 2;
        const int token = lane->sampler.sample(logits_.data(), n_vocab, pad_eop);
        if (token < 0) {
            fail(why, "the lane's sampler returned no token");
            return RAD_E_STATE;
        }
        out[i] = token;
        std::string delta = lane->detok.push(token);
        if (p.kind == RAD_GEN_PASS_DUPLEX_STEP)
            lane->last_text = std::move(delta);
        /* On PROMPT token 0's delta is discarded with frame 0; the push still advanced the gap
         * and partial-character state. */
    }
    return RAD_OK;
}

int HostSampleSource::reply_text(int32_t lane, std::string* out, std::string* why) {
    std::lock_guard<std::mutex> lk(mu_);
    Lane* state = lane_locked(lane);
    if (!state || !out) {
        fail(why, "the reply's lane has no host state");
        return RAD_E_NOTFOUND;
    }
    *out = std::move(state->last_text);
    state->last_text.clear();
    return RAD_OK;
}

}  /* namespace server */
}  /* namespace rad */
