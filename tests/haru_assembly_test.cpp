/* haru_assembly_test.cpp -- the assembled Haru duplex tenant against the committed tiny goldens.
 *
 * The engine is started in-process (Engine::start()) and its rank/generator loops are run without
 * the HTTP server (Engine::start_workers()). The test then drives gen::DuplexDriver directly, as
 * tests/duplex_driver_test.cpp does, with one difference: a TeacherSource implements the ABI 39
 * host seam from the recorded golden tokens and noise. The plugin itself cannot tell teacher mode
 * from production; only who fills text_in/codec_noise/head_noise changes.
 *
 * The request is built from the goldens, not through the server's card renderer: prompt.tokens is
 * the reference's exact 37-row prompt (23 card ids, 13 PAD, one EOP), prompt.ref_audio is the
 * 44,000-sample clip, and each reference row names its frame. This is the driver/block path; the
 * server's rendering has its own tests in tests/duplex_bridge_test.cpp.
 *
 * Frame 0 is produced by PROMPT and dropped: its token/latent/logits are compared, its audio is
 * not sent (the plugin decodes it only to advance the decoder state). Each wire step s is compared
 * to frame s+1: token, latent (prev_latent), audio and logits/pad_eop. A second lane and a
 * close/reopen with the same schedule follow.
 */
#include "haru_harness.h"

#include "engine.h"
#include "util/stallwatch.h"
#include "gen/duplex_driver.h"
#include "server/duplex_host.h"

#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <filesystem>
#include <memory>
#include <string>
#include <vector>

namespace {

using haru_test::Diff;
using haru_test::Golden;
using haru_test::Tol;

constexpr int64_t kSamples = 3200;
constexpr int64_t kLatent = 64;
constexpr int64_t kSteps = 24;

/* A per-lane recorded schedule. The reference's tokens are frame{t}.token_out; the STEP s input
 * token is the previous frame's output, so tokens[0] is frame 0 and tokens[t] is frame t. */
struct Schedule {
    std::vector<float> prompt_codec;            /* [T, latent] */
    std::vector<float> prompt_head;             /* [latent] */
    std::vector<std::vector<float>> step_codec; /* [step][latent] */
    std::vector<std::vector<float>> step_head;  /* [step][latent] */
    std::vector<int32_t> tokens;                /* [25] */
    std::vector<float> prompt_logits, prompt_pad;
    int32_t prompt_token = -1;
    std::vector<float> last_logits, last_pad;
};

int32_t i64_one(const Golden& g, const std::string& name) {
    const std::vector<int64_t> v = g.i64(name);
    return v.empty() ? -1 : (int32_t)v[0];
}

void make_schedule(const Golden& g, Schedule* s) {
    const int64_t T = g.shape("prompt.tokens")[0];
    const std::vector<float> ref_noise = g.f32("prompt.ref_noise");
    s->prompt_codec.assign((size_t)(T * kLatent), 0.0f);
    for (int64_t t = 23, f = 0; t < T; ++t, ++f)
        std::memcpy(&s->prompt_codec[(size_t)(t * kLatent)], &ref_noise[(size_t)(f * kLatent)],
                    (size_t)(kLatent * 4));
    s->prompt_head = g.f32("frame0.head_noise");
    s->tokens.assign(1, i64_one(g, "frame0.token_out"));
    s->step_codec.resize((size_t)kSteps);
    s->step_head.resize((size_t)kSteps);
    for (int64_t t = 1; t <= kSteps; ++t) {
        const std::string pre = "frame" + std::to_string(t);
        s->step_codec[(size_t)(t - 1)] = g.f32(pre + ".user_noise");
        s->step_head[(size_t)(t - 1)] = g.f32(pre + ".head_noise");
        s->tokens.push_back(i64_one(g, pre + ".token_out"));
    }
}

/* ABI 39's host seam, replaying one lane's schedule. */
struct TeacherSource final : rad::gen::DuplexHostSource {
    Schedule s;

    int noise(const rad::gen::DuplexHostPass& p, float* codec_noise, float* head_noise,
              std::string*) override {
        if (p.kind == RAD_GEN_PASS_DUPLEX_PROMPT) {
            const size_t n = (size_t)(p.n_tok * p.latent_dim);
            for (size_t i = 0; i < n; ++i)
                codec_noise[i] = i < s.prompt_codec.size() ? s.prompt_codec[i] : 0.0f;
            for (int64_t i = 0; i < p.n_lane * p.latent_dim; ++i)
                head_noise[i] = i < (int64_t)s.prompt_head.size() ? s.prompt_head[(size_t)i] : 0.0f;
        } else {
            for (int64_t l = 0; l < p.n_lane; ++l) {
                const int64_t f = p.frame ? p.frame[l] : 0;
                const std::vector<float>& cn =
                    (f >= 0 && f < (int64_t)s.step_codec.size()) ? s.step_codec[(size_t)f]
                                                                 : std::vector<float>{};
                const std::vector<float>& hn =
                    (f >= 0 && f < (int64_t)s.step_head.size()) ? s.step_head[(size_t)f]
                                                                : std::vector<float>{};
                for (int64_t d = 0; d < p.latent_dim; ++d) {
                    codec_noise[l * p.latent_dim + d] = d < (int64_t)cn.size() ? cn[(size_t)d] : 0.0f;
                    head_noise[l * p.latent_dim + d] = d < (int64_t)hn.size() ? hn[(size_t)d] : 0.0f;
                }
            }
        }
        return RAD_OK;
    }

    int tokens(const rad::gen::DuplexHostPass& p, int32_t* out, std::string*) override {
        if (p.kind == RAD_GEN_PASS_DUPLEX_PROMPT) {
            const int32_t lane = p.lane ? p.lane[0] : 0;
            /* Readbacks are lane-indexed, not entry-indexed (core/gen/duplex_driver.h). */
            if (p.logits)
                s.prompt_logits.assign(p.logits + (size_t)lane * p.n_vocab,
                                       p.logits + (size_t)(lane + 1) * p.n_vocab);
            if (p.pad_eop) s.prompt_pad.assign(p.pad_eop + (size_t)lane * 2,
                                               p.pad_eop + (size_t)(lane + 1) * 2);
            s.prompt_token = s.tokens.empty() ? -1 : s.tokens[0];
            out[0] = s.prompt_token;
        } else {
            for (int64_t l = 0; l < p.n_lane; ++l) {
                const int64_t f = p.frame ? p.frame[l] : 0;
                const int32_t lane = p.lane ? p.lane[l] : (int32_t)l;
                /* Record one lane's row; the test drives one lane at a time. */
                if (p.logits)
                    s.last_logits.assign(p.logits + (size_t)lane * p.n_vocab,
                                         p.logits + (size_t)(lane + 1) * p.n_vocab);
                if (p.pad_eop)
                    s.last_pad.assign(p.pad_eop + (size_t)lane * 2,
                                      p.pad_eop + (size_t)(lane + 1) * 2);
                out[l] = (f + 1 < (int64_t)s.tokens.size()) ? s.tokens[(size_t)(f + 1)] : -1;
            }
        }
        return RAD_OK;
    }
};

void expect_close(const char* what, const std::vector<float>& got,
                  const std::vector<float>& want, Tol tol) {
    const Diff d = haru_test::compare(got, want, tol);
    std::fprintf(stderr, "    %-38s %s\n", what, d.text().c_str());
    ++radtest::checks();
    if (!d.ok) radtest::fail(__FILE__, __LINE__, std::string(what) + ": " + d.text());
}

const float* buffer_ptr(const rad::Program& p, const char* name, int64_t* n) {
    for (const rad::BufferInfo& b : p.buffers) {
        if (b.name == name) {
            if (n) *n = b.bytes / 4;
            return static_cast<const float*>(b.ptr);
        }
    }
    return nullptr;
}

struct Workers {
    rad::Engine* e = nullptr;
    ~Workers() { if (e) e->stop_workers(); }
};

bool make_home(const std::string& home, std::string* why) {
    namespace fs = std::filesystem;
    std::error_code ec;
    fs::create_directories(home + "/kernels", ec);
    fs::create_directories(home + "/architectures", ec);
    fs::create_directories(home + "/quantizers", ec);
    if (ec) { *why = "home: " + ec.message(); return false; }
    fs::copy_file(std::string(RAD_E2E_HOME) + "/kernels/libref.so", home + "/kernels/libref.so",
                  fs::copy_options::overwrite_existing, ec);
    if (ec) { *why = "libref: " + ec.message(); return false; }
    fs::copy_file(RAD_HARU_PLUGIN, home + "/architectures/haru.so",
                  fs::copy_options::overwrite_existing, ec);
    if (ec) { *why = "haru plugin: " + ec.message(); return false; }
    return true;
}

std::unique_ptr<rad::gen::DuplexRequest> request_from_goldens(const Golden& g) {
    auto r = std::make_unique<rad::gen::DuplexRequest>();
    const std::vector<int64_t> tok = g.i64("prompt.tokens");
    r->tokens.reserve(tok.size());
    for (int64_t v : tok) r->tokens.push_back((int32_t)v);
    r->ref_audio = g.f32("prompt.ref_audio");
    r->ref_cu = { 0, (int32_t)r->ref_audio.size() };
    r->ref_frame.assign(tok.size(), -1);
    const int32_t n_ref = (int32_t)((r->ref_audio.size() + kSamples - 1) / kSamples);
    for (int32_t f = 0; f < n_ref; ++f)
        r->ref_frame[(size_t)(23 + f)] = f;
    r->seed = 1234;
    return r;
}

}  /* namespace */

/* The same arithmetic HaruProj::unscale runs: r0 = 0.75 and six Newton steps. The documented
 * range is latent_scale in [0.5, 2.0]; the tiny checkpoint is 0.8 and the real scalar is not
 * host-readable here, so the real value is covered only by this range statement. */
TEST(unscale_newton_covers_the_documented_latent_scale_range) {
    for (float s : { 0.5f, 0.8f, 1.0f, 1.5f, 2.0f }) {
        float r = 0.75f;
        for (int i = 0; i < 6; ++i) r = r * (2.0f - s * r);
        CHECK(std::fabs(r * s - 1.0f) < 1e-6f);
    }
}

TEST(the_assembled_tenant_matches_the_tiny_goldens_for_the_prompt_and_24_frames) {
    tiny::TempDir dir("haru_assembly", std::getenv("RAD_DUPLEX_KEEP") != nullptr);
    REQUIRE(dir.ok());
    const std::string home = dir / "home";
    std::string why;
    REQUIRE(make_home(home, &why));

    setenv("RADIANCE_HOME", home.c_str(), 1);
    setenv("ROCR_VISIBLE_DEVICES", "99", 1);
    setenv("RADIANCE_HOST_VRAM_MIB", "512", 1);
    setenv("OMP_WAIT_POLICY", "passive", 1);

    rad::Config cfg;
    cfg.model = std::string(RAD_HARU_DATA) + "/haru_tiny";
    cfg.radiance_home = home;
    cfg.kernel_hierarchy = { "libref" };
    cfg.accept_reference_kernels = true;
    cfg.max_tok = 4096;
    cfg.max_seqs = 2;
    cfg.duplex_max_ctx = 128;
    cfg.port = 0;
    cfg.http_threads = 4;

    rad::Engine engine(std::move(cfg));
    if (engine.start() < 0) {
        REQUIRE(false);
        return;
    }
    if (engine.serve_prepare(nullptr) < 0) {
        REQUIRE(false);
        return;
    }
    Workers guard{ &engine };
    if (engine.start_workers() < 0) {
        REQUIRE(false);
        return;
    }
    rad::gen::DuplexDriver* drv = engine.duplex_driver();
    REQUIRE(drv != nullptr);
    if (!drv) return;

    const Golden& g = Golden::get();
    const int64_t T = g.shape("prompt.tokens")[0];
    int64_t nbuf = 0;
    const float* latent_buf = buffer_ptr(engine.program(), "duplex.prev_latent", &nbuf);
    REQUIRE(latent_buf != nullptr);
    if (!latent_buf) return;

    const Tol tol{ 1e-4, 1e-4 };
    auto run_lane = [&](const char* label, int64_t steps) -> bool {
        TeacherSource src;
        make_schedule(g, &src.s);
        drv->set_host_source(&src);
        std::shared_ptr<rad::gen::DuplexTicket> open;
        std::string w;
        if (drv->open(request_from_goldens(g), &open, &w) < 0) {
            std::fprintf(stderr, "    %s open: %s\n", label, w.c_str());
            return false;
        }
        bool to = false;
        if (!open->wait(300000, &to) || open->status() != RAD_OK) {
            std::fprintf(stderr, "    %s open did not finish\n", label);
            return false;
        }
        const int32_t lane = open->lane();

        /* Frame 0, from PROMPT. The audio is decoded but intentionally not compared here. */
        CHECK_EQ(src.s.prompt_token, i64_one(g, "frame0.token_out"));
        expect_close((std::string(label) + " frame0 latent").c_str(),
                     std::vector<float>(latent_buf + lane * kLatent,
                                        latent_buf + lane * kLatent + kLatent),
                     g.f32("frame0.latent_out"), tol);
        expect_close((std::string(label) + " frame0 logits").c_str(), src.s.prompt_logits,
                     g.f32("frame0.logits"), tol);
        expect_close((std::string(label) + " frame0 pad_eop").c_str(), src.s.prompt_pad,
                     g.f32("frame0.logits_fp32_pad_eop"), tol);

        for (int64_t s = 0; s < steps; ++s) {
            const std::string pre = "frame" + std::to_string(s + 1);
            const std::vector<float> in = g.f32(pre + ".user_audio");
            std::shared_ptr<rad::gen::DuplexTicket> t;
            if (drv->step(lane, in.data(), (int64_t)in.size(), (uint64_t)s, &t, &w) < 0) {
                std::fprintf(stderr, "    %s step %lld: %s\n", label, (long long)s, w.c_str());
                return false;
            }
            rad::gen::DuplexReply r;
            if (!t->next(&r, 300000, &to)) {
                std::fprintf(stderr, "    %s step %lld did not finish\n", label, (long long)s);
                return false;
            }
            CHECK_EQ(r.token, i64_one(g, pre + ".token_out"));
            expect_close((std::string(label) + " " + pre + " audio").c_str(), r.audio,
                         g.f32(pre + ".audio_out"), tol);
            expect_close((std::string(label) + " " + pre + " latent").c_str(),
                         std::vector<float>(latent_buf + lane * kLatent,
                                            latent_buf + lane * kLatent + kLatent),
                         g.f32(pre + ".latent_out"), tol);
            expect_close((std::string(label) + " " + pre + " logits").c_str(), src.s.last_logits,
                         g.f32(pre + ".logits"), tol);
            expect_close((std::string(label) + " " + pre + " pad_eop").c_str(), src.s.last_pad,
                         g.f32(pre + ".logits_fp32_pad_eop"), tol);
        }
        drv->close(lane);
        return true;
    };

    REQUIRE(run_lane("lane0", kSteps));
    REQUIRE(run_lane("lane1", 2));        /* a second lane, same schedule */
    REQUIRE(run_lane("reopen", 1));       /* close/reopen: fresh codec and latent state */
    engine.stop_workers();
    guard.e = nullptr;
}

/* The production host source on the tiny model: the same seed replays the same tokens and audio,
 * a different seed does not. This exercises the driver, the plugin, the codec, the sampler and the
 * noise stream together, with no teacher feed. */
TEST(the_production_host_source_is_deterministic_on_the_tiny_model) {
    tiny::TempDir dir("haru_host_det", std::getenv("RAD_DUPLEX_KEEP") != nullptr);
    REQUIRE(dir.ok());
    const std::string home = dir / "home";
    std::string why;
    REQUIRE(make_home(home, &why));

    setenv("RADIANCE_HOME", home.c_str(), 1);
    setenv("ROCR_VISIBLE_DEVICES", "99", 1);
    setenv("RADIANCE_HOST_VRAM_MIB", "512", 1);
    setenv("OMP_WAIT_POLICY", "passive", 1);

    rad::Config cfg;
    cfg.model = std::string(RAD_HARU_DATA) + "/haru_tiny";
    cfg.radiance_home = home;
    cfg.kernel_hierarchy = { "libref" };
    cfg.accept_reference_kernels = true;
    cfg.max_tok = 4096;
    cfg.max_seqs = 2;
    cfg.duplex_max_ctx = 128;
    cfg.port = 0;
    cfg.http_threads = 4;

    rad::Engine engine(std::move(cfg));
    if (engine.start() < 0) {
        REQUIRE(false);
        return;
    }
    if (engine.serve_prepare(nullptr) < 0) {
        REQUIRE(false);
        return;
    }
    Workers guard{ &engine };
    if (engine.start_workers() < 0) {
        REQUIRE(false);
        return;
    }
    rad::gen::DuplexDriver* drv = engine.duplex_driver();
    REQUIRE(drv != nullptr);
    if (!drv) return;

    const Golden& g = Golden::get();
    const rad::RadDuplexDecl& dd = engine.program().duplex;
    constexpr int64_t kRunSteps = 3;

    auto run = [&](uint64_t seed, std::vector<int32_t>* tokens,
                   std::vector<float>* audio) -> bool {
        rad::server::HostSampleSource::Options o;
        o.sampler.temperature = 0.7f;
        o.sampler.top_k = 25;
        o.sampler.top_p = 1.0f;
        o.sampler.eop_bias = 0.0f;
        o.sampler.force_text_frames = 0;
        o.codec_noise_scale = dd.codec_noise_scale;
        o.pad_id = dd.pad_id;
        o.eop_id = dd.eop_id;
        o.eos_id = dd.eos_id;
        o.n_vocab = dd.n_vocab;
        o.latent_dim = dd.latent_dim;
        o.max_lanes = dd.max_lanes;
        o.decode = [](const std::vector<int>&) { return std::string(); };
        rad::server::HostSampleSource src(std::move(o));
        struct ClearHost {
            rad::gen::DuplexDriver* d;
            ~ClearHost() { if (d) d->set_host_source(nullptr); }
        } clear_host{ drv };
        drv->set_host_source(&src);

        auto req = request_from_goldens(g);
        req->seed = seed;
        std::shared_ptr<rad::gen::DuplexTicket> open;
        if (drv->open(std::move(req), &open, &why) < 0) return false;
        bool to = false;
        if (!open->wait(300000, &to) || open->status() != RAD_OK) return false;
        const int32_t lane = open->lane();
        for (int64_t s = 0; s < kRunSteps; ++s) {
            const std::vector<float> in =
                g.f32("frame" + std::to_string(s + 1) + ".user_audio");
            std::shared_ptr<rad::gen::DuplexTicket> t;
            if (drv->step(lane, in.data(), (int64_t)in.size(), (uint64_t)s, &t, &why) < 0)
                return false;
            rad::gen::DuplexReply r;
            if (!t->next(&r, 300000, &to)) return false;
            tokens->push_back(r.token);
            audio->insert(audio->end(), r.audio.begin(), r.audio.end());
        }
        drv->close(lane);
        return true;
    };

    std::vector<int32_t> ta, tb, tc;
    std::vector<float> aa, ab, ac;
    REQUIRE(run(1234, &ta, &aa));
    REQUIRE(run(1234, &tb, &ab));
    CHECK(ta == tb);
    CHECK(aa == ab);
    REQUIRE(run(5678, &tc, &ac));
    CHECK(aa != ac);

    engine.stop_workers();
    guard.e = nullptr;
}

RAD_TEST_MAIN()
