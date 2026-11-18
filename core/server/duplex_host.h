/* duplex_host.h -- the production host source for a host-sampling duplex tenant (Haru).
 *
 * The driver's ABI 39 seam (gen::DuplexHostSource, core/gen/duplex_driver.h) asks for the noise
 * before a pass and for the sampled token after it. This class supplies both, plus the per-lane
 * streamed text the server puts on the wire:
 *
 *   - noise:  the reference's codec noise, scale * (latent_std / 0.8) * N(0,1), and the head's
 *             raw N(0,1) vector (.haru-ref/vibevoice_codec.py:196-200,
 *             .haru-ref/session.py:159-161);
 *   - token:  gen::LaneSampler, the S2 DRY / PAD-EOP / eop_bias / temperature / top-k / top-p
 *             sampler (core/gen/duplex_sampler.h);
 *   - text:   gen::StreamDetokenizer, one per lane, so PAD/EOP runs and partial UTF-8 match the
 *             reference (.haru-ref/sampling.py:103-128).
 *
 * One lane is one conversation: one LaneSampler, one noise PRNG and one StreamDetokenizer,
 * created by lane_opened() and destroyed by lane_closed(). The noise PRNG's seed is the
 * DuplexRequest seed, domain-separated from the sampler's splitmix64 (see the .cpp).
 *
 * The class does not own a tokenizer or a device buffer. `Options::decode` is the model's
 * detokenizer callback; production passes the engine Tokenizer's const decode.
 */
#pragma once

#include "gen/duplex_driver.h"
#include "gen/duplex_sampler.h"

#include <cstdint>
#include <functional>
#include <memory>
#include <mutex>
#include <string>
#include <vector>

namespace rad {
namespace server {

class HostSampleSource final : public gen::DuplexHostSource {
public:
    struct Options {
        gen::SamplerConfig sampler;            /* reference defaults, overridden by server flags */
        double  codec_noise_scale = 0.0;       /* plugin-declared latent_std / 0.8 */
        int32_t pad_id = -1, eop_id = -1, eos_id = -1;
        int64_t n_vocab = 0, latent_dim = 0, max_lanes = 0;
        /* HF-equivalent decode: .haru-ref/sampling.py:124 uses skip_special_tokens=False. Empty
         * returns no text, which is only useful to a test that does not inspect text. */
        std::function<std::string(const std::vector<int>&)> decode;
        /* Test seam only: null uses the internal splitmix64/Box-Muller stream. A test supplies
         * the golden normal draws in the caller's order; production never sets it. */
        std::function<double()> normal;
    };

    explicit HostSampleSource(Options o);

    /* One conversation's lifetime. Called by DuplexDriver while its mutex is held. */
    void lane_opened(int32_t lane, uint64_t seed) override;
    void lane_closed(int32_t lane) override;

    int noise(const gen::DuplexHostPass& pass, float* codec_noise, float* head_noise,
              std::string* why) override;
    int tokens(const gen::DuplexHostPass& pass, int32_t* out, std::string* why) override;
    int reply_text(int32_t lane, std::string* out, std::string* why) override;

private:
    /* splitmix64 + Box-Muller, one stream per lane. The first two uniforms of a pair produce
     * cos/sin; a group of one normal keeps the cosine and discards the sine, as documented in
     * docs/HARU-H6-DESIGN.md section 3.3. */
    class NoisePrng {
    public:
        explicit NoisePrng(uint64_t seed);
        void normals(int n, float* out);

    private:
        double uniform();
        uint64_t state_;
    };

    struct Lane {
        Lane(const Options& o, uint64_t seed);
        gen::LaneSampler       sampler;
        NoisePrng              noise;
        gen::StreamDetokenizer detok;
        std::string            last_text;
        int64_t                next_frame = 0;
    };

    int  check_pass(const gen::DuplexHostPass& p, std::string* why) const;
    Lane* lane_locked(int32_t lane);

    Options                            o_;
    std::vector<std::unique_ptr<Lane>> lanes_;
    std::vector<float>                 logits_;   /* n_vocab scratch, reused under mu_ */
    mutable std::mutex                 mu_;
};

}  /* namespace server */
}  /* namespace rad */
