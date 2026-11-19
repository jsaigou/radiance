/* haru_real_parity_test.cpp -- the REAL Haru-4B on the GPU, through the pass driver, teacher-forced.
 *
 * The tiny-model test (haru_assembly_test.cpp) proves the plumbing at f32 on the host. This one
 * drives the converted real checkpoint (a .rad) on the card with the Phase 0 recording of the
 * Python reference: the reference's sampled tokens and noise are replayed through the same ABI 39
 * seam, so every frame is comparable with the reference's, and each step is timed.
 *
 * It is a device test and needs artifacts that are not in the tree, so it is skipped unless all of:
 *   RAD_HARU_REAL_RAD     the converted container (rad-convert of higurashi-ai/Haru-4B)
 *   RAD_HARU_REAL_GOLDEN  a golden_real_<lang>_<dtype>.safetensors from the Python reference
 *   RAD_HARU_REAL_HOME    a radiance home (kernels/ with libhalo and libref, architectures/haru.so)
 * are set. Optional: RAD_HARU_REAL_OUT (a JSON file receiving per-frame metrics and step times),
 * RAD_HARU_REAL_FRAMES (steps to run, default every recorded frame).
 *
 * The comparison is by cosine similarity, top-1 token agreement and spectral distance, not by
 * element: a bf16 engine on a GPU cannot match an f32 CPU reference bit for bit, and the
 * reference's own bf16-versus-f32 floor (docs/duplex/tohil-plan.md section 1a) is what the
 * thresholds are measured against. The plugin feeds its own previous latent back, as production
 * does, so these numbers include that compounding; they are not a per-frame replay. */
#include "rad_test.h"
#include "tiny_model.h"

#include "engine.h"
#include "gen/duplex_driver.h"
#include "util/stallwatch.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <memory>
#include <string>
#include <vector>

namespace {

constexpr int64_t kSamples = 3200;
constexpr int64_t kLatent = 64;

const char* env(const char* k) {
    const char* v = std::getenv(k);
    return (v && *v) ? v : nullptr;
}

struct Rec {
    std::vector<tiny::Tensor> t;
    const tiny::Tensor* find(const std::string& n) const {
        for (const tiny::Tensor& x : t) if (x.name == n) return &x;
        return nullptr;
    }
    std::vector<float> f32(const std::string& n) const {
        const tiny::Tensor* x = find(n);
        std::vector<float> o;
        if (!x || x->dtype != "F32") return o;
        o.resize(x->bytes.size() / 4);
        std::memcpy(o.data(), x->bytes.data(), o.size() * 4);
        return o;
    }
    std::vector<int64_t> i64(const std::string& n) const {
        const tiny::Tensor* x = find(n);
        std::vector<int64_t> o;
        if (!x || x->dtype != "I64") return o;
        o.resize(x->bytes.size() / 8);
        std::memcpy(o.data(), x->bytes.data(), o.size() * 8);
        return o;
    }
};

int32_t one(const Rec& g, const std::string& n) {
    const std::vector<int64_t> v = g.i64(n);
    return v.empty() ? -1 : (int32_t)v[0];
}

/* Replays the recording's tokens and noise; keeps the last logits row the driver read back. */
struct Teacher final : rad::gen::DuplexHostSource {
    std::vector<float> prompt_codec, prompt_head;
    std::vector<std::vector<float>> step_codec, step_head;
    std::vector<int32_t> script;     /* the recorded token for each frame */
    std::vector<float> logits, pad;   /* the most recent readback (prompt or step) */

    int noise(const rad::gen::DuplexHostPass& p, float* codec_noise, float* head_noise,
              std::string*) override {
        if (p.kind == RAD_GEN_PASS_DUPLEX_PROMPT) {
            const size_t n = (size_t)(p.n_tok * p.latent_dim);
            for (size_t i = 0; i < n; ++i) codec_noise[i] = i < prompt_codec.size() ? prompt_codec[i] : 0.0f;
            for (int64_t i = 0; i < p.n_lane * p.latent_dim; ++i)
                head_noise[i] = i < (int64_t)prompt_head.size() ? prompt_head[(size_t)i] : 0.0f;
        } else {
            for (int64_t l = 0; l < p.n_lane; ++l) {
                const int64_t f = p.frame ? p.frame[l] : 0;
                static const std::vector<float> none;
                const std::vector<float>& cn = (f >= 0 && f < (int64_t)step_codec.size()) ? step_codec[(size_t)f] : none;
                const std::vector<float>& hn = (f >= 0 && f < (int64_t)step_head.size()) ? step_head[(size_t)f] : none;
                for (int64_t d = 0; d < p.latent_dim; ++d) {
                    codec_noise[l * p.latent_dim + d] = d < (int64_t)cn.size() ? cn[(size_t)d] : 0.0f;
                    head_noise[l * p.latent_dim + d] = d < (int64_t)hn.size() ? hn[(size_t)d] : 0.0f;
                }
            }
        }
        return RAD_OK;
    }

    int tokens_for(const rad::gen::DuplexHostPass& p, int32_t* out) {
        const bool prompt = p.kind == RAD_GEN_PASS_DUPLEX_PROMPT;
        for (int64_t l = 0; l < p.n_lane; ++l) {
            const int32_t lane = p.lane ? p.lane[l] : (int32_t)l;
            if (p.logits) logits.assign(p.logits + (size_t)lane * p.n_vocab, p.logits + (size_t)(lane + 1) * p.n_vocab);
            if (p.pad_eop) pad.assign(p.pad_eop + (size_t)lane * 2, p.pad_eop + (size_t)(lane + 1) * 2);
            const int64_t f = prompt ? -1 : (p.frame ? p.frame[l] : 0);
            out[l] = (f + 1 < (int64_t)script.size()) ? script[(size_t)(f + 1)] : -1;
        }
        return RAD_OK;
    }
    int tokens(const rad::gen::DuplexHostPass& p, int32_t* out, std::string*) override { return tokens_for(p, out); }
};

double cosine(const std::vector<float>& a, const std::vector<float>& b) {
    if (a.size() != b.size() || a.empty()) return -2.0;
    double ab = 0, aa = 0, bb = 0;
    for (size_t i = 0; i < a.size(); ++i) { ab += (double)a[i] * b[i]; aa += (double)a[i] * a[i]; bb += (double)b[i] * b[i]; }
    return (aa > 0 && bb > 0) ? ab / std::sqrt(aa * bb) : -2.0;
}

int argmax(const std::vector<float>& v) {
    int best = 0;
    for (size_t i = 1; i < v.size(); ++i) if (v[i] > v[(size_t)best]) best = (int)i;
    return best;
}

/* A crude log-mel distance: a Hann-windowed DFT power spectrum on 64 quadratically spaced bands
 * over 512-sample frames, mean absolute difference of the logs. Matches measure_real.py's idea,
 * not its exact numbers; the same function is applied to both signals, which is what matters. */
std::vector<double> logmel(const std::vector<float>& x) {
    const int N = 512, H = 256, B = 64;
    std::vector<double> out;
    if ((int)x.size() < N) return out;
    std::vector<double> win(N);
    for (int i = 0; i < N; ++i) win[(size_t)i] = 0.5 - 0.5 * std::cos(2.0 * M_PI * i / N);
    std::vector<double> edges(B + 1);
    for (int b = 0; b <= B; ++b) edges[(size_t)b] = std::min<double>(N / 2, (double)(N / 2) * b * b / (double)(B * B));
    for (int s = 0; s + N <= (int)x.size(); s += H) {
        std::vector<double> pw((size_t)(N / 2 + 1), 0.0);
        for (int k = 0; k <= N / 2; ++k) {
            double re = 0, im = 0;
            for (int n = 0; n < N; ++n) {
                const double v = x[(size_t)(s + n)] * win[(size_t)n];
                const double ang = 2.0 * M_PI * k * n / N;
                re += v * std::cos(ang); im -= v * std::sin(ang);
            }
            pw[(size_t)k] = re * re + im * im;
        }
        for (int b = 0; b < B; ++b) {
            const int lo = (int)edges[(size_t)b], hi = std::max(lo + 1, (int)edges[(size_t)b + 1]);
            double m = 0;
            for (int k = lo; k < hi && k <= N / 2; ++k) m += pw[(size_t)k];
            m /= (double)(hi - lo);
            out.push_back(std::log(m + 1e-8));
        }
    }
    return out;
}

struct Workers {
    rad::Engine* e = nullptr;
    ~Workers() { if (e) e->stop_workers(); }
};

}  /* namespace */

TEST(the_real_haru_on_the_gpu_matches_the_reference_teacher_forced) {
    const char* rad_path = env("RAD_HARU_REAL_RAD");
    const char* gold_path = env("RAD_HARU_REAL_GOLDEN");
    const char* home = env("RAD_HARU_REAL_HOME");
    if (!rad_path || !gold_path || !home)
        ARTIFACT_MISSING("RAD_HARU_REAL_RAD, RAD_HARU_REAL_GOLDEN and RAD_HARU_REAL_HOME");
    NEED_ARTIFACT(rad_path);
    NEED_ARTIFACT(gold_path);

    Rec g;
    g.t = tiny::read_safetensors(gold_path);
    REQUIRE(!g.t.empty());
    const std::vector<int64_t> prompt_tokens = g.i64("prompt.tokens");
    const std::vector<float> ref_audio = g.f32("prompt.ref_audio");
    const std::vector<float> ref_noise = g.f32("prompt.ref_noise");
    REQUIRE(!prompt_tokens.empty() && !ref_audio.empty() && !ref_noise.empty());
    const int64_t T = (int64_t)prompt_tokens.size();
    const int64_t n_ref = (int64_t)((ref_audio.size() + kSamples - 1) / kSamples);
    const int64_t n_card = T - n_ref;
    REQUIRE(n_card > 0 && (int64_t)ref_noise.size() == n_ref * kLatent);

    int64_t frames = 0;
    while (g.find("frame" + std::to_string(frames + 1) + ".user_audio")) ++frames;
    if (const char* f = env("RAD_HARU_REAL_FRAMES")) frames = std::min<int64_t>(frames, std::atoll(f));
    REQUIRE(frames > 0);

    Teacher src;
    src.prompt_codec.assign((size_t)(T * kLatent), 0.0f);
    for (int64_t f = 0; f < n_ref; ++f)
        std::memcpy(&src.prompt_codec[(size_t)((n_card + f) * kLatent)], &ref_noise[(size_t)(f * kLatent)], (size_t)(kLatent * 4));
    src.prompt_head = g.f32("frame0.head_noise");
    src.script.push_back(one(g, "frame0.token_out"));
    src.step_codec.resize((size_t)frames);
    src.step_head.resize((size_t)frames);
    for (int64_t t = 1; t <= frames; ++t) {
        const std::string pre = "frame" + std::to_string(t);
        src.step_codec[(size_t)(t - 1)] = g.f32(pre + ".user_noise");
        src.step_head[(size_t)(t - 1)] = g.f32(pre + ".head_noise");
        src.script.push_back(one(g, pre + ".token_out"));
    }

    setenv("RADIANCE_HOME", home, 1);
    setenv("OMP_WAIT_POLICY", "passive", 1);
    rad::Config cfg;
    cfg.model = rad_path;
    cfg.radiance_home = home;
    cfg.kernel_hierarchy = { "libhalo" };   // every op resolves on the device: libref is not needed
    if (const char* k = env("RAD_HARU_REAL_KERNELS")) {   // e.g. "libhalo,libref" to include the host oracle
        cfg.kernel_hierarchy.clear();
        std::string v = k;
        for (size_t at = 0; at <= v.size();) {
            size_t e = v.find(',', at);
            if (e == std::string::npos) e = v.size();
            if (e > at) cfg.kernel_hierarchy.push_back(v.substr(at, e - at));
            at = e + 1;
        }
    }
    cfg.accept_reference_kernels = false;
    cfg.max_tok = 4096;
    cfg.max_seqs = 2;
    cfg.duplex_max_ctx = std::max<int64_t>(1024, T + frames + 64);
    cfg.port = 0;
    cfg.http_threads = 4;
    if (env("RAD_HARU_REAL_PROFILE")) cfg.profile_ops = true;   // per-op device timing at stop

    rad::Engine engine(std::move(cfg));
    REQUIRE(engine.start() >= 0);
    REQUIRE(engine.serve_prepare(nullptr) >= 0);
    Workers guard{ &engine };
    REQUIRE(engine.start_workers() >= 0);
    rad::gen::DuplexDriver* drv = engine.duplex_driver();
    REQUIRE(drv != nullptr);
    drv->set_host_source(&src);

    auto req = std::make_unique<rad::gen::DuplexRequest>();
    for (int64_t v : prompt_tokens) req->tokens.push_back((int32_t)v);
    req->ref_audio = ref_audio;
    req->ref_cu = { 0, (int32_t)ref_audio.size() };
    req->ref_frame.assign((size_t)T, -1);
    for (int64_t f = 0; f < n_ref; ++f) req->ref_frame[(size_t)(n_card + f)] = (int32_t)f;
    req->seed = 1234;

    using clock = std::chrono::steady_clock;
    std::shared_ptr<rad::gen::DuplexTicket> open;
    std::string why;
    const auto t_open = clock::now();
    REQUIRE(drv->open(std::move(req), &open, &why) >= 0);
    bool to = false;
    REQUIRE(open->wait(600000, &to) && open->status() == RAD_OK);
    const double open_ms = std::chrono::duration<double, std::milli>(clock::now() - t_open).count();
    const int32_t lane = open->lane();

    /* RAD_HARU_REAL_LANES=2: a second lane runs the same recording alongside, stepped together, so the
     * step time is that of a pass carrying two lanes. Its outputs are not compared. */
    const bool two = env("RAD_HARU_REAL_LANES") && std::atoi(env("RAD_HARU_REAL_LANES")) >= 2;
    int32_t lane_b = -1;
    if (two) {
        auto reqb = std::make_unique<rad::gen::DuplexRequest>();
        for (int64_t v : prompt_tokens) reqb->tokens.push_back((int32_t)v);
        reqb->ref_audio = ref_audio;
        reqb->ref_cu = { 0, (int32_t)ref_audio.size() };
        reqb->ref_frame.assign((size_t)T, -1);
        for (int64_t f = 0; f < n_ref; ++f) reqb->ref_frame[(size_t)(n_card + f)] = (int32_t)f;
        reqb->seed = 4321;
        std::shared_ptr<rad::gen::DuplexTicket> ob;
        REQUIRE(drv->open(std::move(reqb), &ob, &why) >= 0);
        REQUIRE(ob->wait(600000, &to) && ob->status() == RAD_OK);
        lane_b = ob->lane();
    }

    struct Row { double cos_logits, cos_pad, mel; int top1; double ms; };
    std::vector<Row> rows;
    std::vector<float> all_got, all_want;
    {   /* frame 0 comes from the prompt; its audio is decoded and dropped, so only logits compare */
        Row r{};
        r.cos_logits = cosine(src.logits, g.f32("frame0.logits"));
        r.cos_pad = cosine(src.pad, g.f32("frame0.logits_fp32_pad_eop"));
        r.top1 = argmax(src.logits) == argmax(g.f32("frame0.logits"));
        r.mel = 0; r.ms = open_ms;
        rows.push_back(r);
    }
    std::vector<double> step_ms;
    for (int64_t s = 0; s < frames; ++s) {
        const std::string pre = "frame" + std::to_string(s + 1);
        const std::vector<float> in = g.f32(pre + ".user_audio");
        std::shared_ptr<rad::gen::DuplexTicket> t;
        const auto t0 = clock::now();
        REQUIRE(drv->step(lane, in.data(), (int64_t)in.size(), (uint64_t)s, &t, &why) >= 0);
        std::shared_ptr<rad::gen::DuplexTicket> tb;
        if (two) REQUIRE(drv->step(lane_b, in.data(), (int64_t)in.size(), (uint64_t)s, &tb, &why) >= 0);
        rad::gen::DuplexReply rep, repb;
        REQUIRE(t->next(&rep, 600000, &to));
        if (two) REQUIRE(tb->next(&repb, 600000, &to));
        const double ms = std::chrono::duration<double, std::milli>(clock::now() - t0).count();
        step_ms.push_back(ms);
        const std::vector<float> want_audio = g.f32(pre + ".audio_out");
        Row r{};
        r.cos_logits = cosine(src.logits, g.f32(pre + ".logits"));
        r.cos_pad = cosine(src.pad, g.f32(pre + ".logits_fp32_pad_eop"));
        r.top1 = argmax(src.logits) == argmax(g.f32(pre + ".logits"));
        r.ms = ms;
        const std::vector<double> a = logmel(rep.audio), b = logmel(want_audio);
        double d = 0;
        if (a.size() == b.size() && !a.empty()) { for (size_t i = 0; i < a.size(); ++i) d += std::fabs(a[i] - b[i]); d /= (double)a.size(); }
        else d = -1;
        r.mel = d;
        rows.push_back(r);
        all_got.insert(all_got.end(), rep.audio.begin(), rep.audio.end());
        all_want.insert(all_want.end(), want_audio.begin(), want_audio.end());
    }
    drv->close(lane);
    if (two) drv->close(lane_b);
    engine.stop_workers();
    guard.e = nullptr;

    /* Summaries over the steps (frame 0 excluded from the audio and timing numbers). */
    double cos_min = 2, cos_sum = 0, mel_sum = 0, pad_min = 2;
    int64_t top1 = 0, mel_n = 0;
    for (size_t i = 0; i < rows.size(); ++i) {
        cos_min = std::min(cos_min, rows[i].cos_logits);
        cos_sum += rows[i].cos_logits;
        pad_min = std::min(pad_min, rows[i].cos_pad);
        top1 += rows[i].top1;
        if (i > 0 && rows[i].mel >= 0) { mel_sum += rows[i].mel; ++mel_n; }
    }
    std::vector<double> sorted = step_ms;
    std::sort(sorted.begin(), sorted.end());
    auto pct = [&](double q) { return sorted.empty() ? 0.0 : sorted[std::min(sorted.size() - 1, (size_t)(q * (double)sorted.size()))]; };
    std::fprintf(stderr,
                 "    frames %zu | logits cosine min %.5f mean %.5f | pad/eop cosine min %.5f | top-1 %.1f%% | "
                 "logmel %.4f\n    open %.0f ms | step ms: p50 %.2f p95 %.2f p99 %.2f max %.2f\n",
                 rows.size(), cos_min, cos_sum / (double)rows.size(), pad_min,
                 100.0 * (double)top1 / (double)rows.size(), mel_n ? mel_sum / (double)mel_n : -1.0, open_ms,
                 pct(0.50), pct(0.95), pct(0.99), sorted.empty() ? 0.0 : sorted.back());

    if (const char* out = env("RAD_HARU_REAL_OUT")) {
        std::ofstream f(out);
        f << "{\"frames\":" << rows.size() << ",\"open_ms\":" << open_ms << ",\"rows\":[";
        for (size_t i = 0; i < rows.size(); ++i)
            f << (i ? "," : "") << "{\"cos_logits\":" << rows[i].cos_logits << ",\"cos_pad\":" << rows[i].cos_pad
              << ",\"top1\":" << rows[i].top1 << ",\"mel\":" << rows[i].mel << ",\"ms\":" << rows[i].ms << "}";
        f << "]}\n";
    }

    /* Sanity only: the thresholds are set from the measured noise floor in the plan, not here. */
    CHECK(std::isfinite(cos_sum));
    CHECK(cos_min > 0.5);
    CHECK(top1 * 2 > (int64_t)rows.size());
}

RAD_TEST_MAIN()
