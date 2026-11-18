/* tests/haru_head_test.cpp -- the MeanFlow head and the two audio projectors against the recorded
 * exchange in tests/data/haru/golden.safetensors (schema: tests/data/haru/README.md).
 *
 * Two duplex tenants carry the blocks over the shared tensor tunnel (tests/HARU_HARNESS.md):
 * "head" takes [hidden | noise | guidance] and returns the normalized latent, "embed" takes
 * [token | model_latent | user_latent | model_mask | user_mask] and returns the LM embedding. The
 * engine's duplex ABI holds one tenant a program, so this test starts one engine per tenant; the
 * temporary checkpoint copy names the shape in its config.json ("haru_test_tenant").
 *
 * The checkpoint's LM dimensions live under "lm" in config.json, which the engine's metadata
 * reader does not alias to top level, so the test starts the engine on a copy of haru_tiny with
 * those dims mirrored at the top level. The golden directory itself is never written.
 */
#include "rad_test.h"
#include "haru_harness.h"

#include <nlohmann/json.hpp>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstddef>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <memory>
#include <string>
#include <vector>

namespace {

/* Started at the harness's float32 default. The authoring machine cannot build the C++ tree, so
 * the measured maximum error is to be recorded here by the reviewer's Linux run. */
constexpr haru_test::Tol kTol{ 1e-4, 1e-4 };

std::string frame_name(int t, const char* leaf) {
    return "frame" + std::to_string(t) + "." + leaf;
}

/* A private copy of the tiny checkpoint whose config.json also states the LM dims at top level.
 * The engine's typed metadata reader knows top-level, text_config and llm_config keys; Haru's
 * config nests them under "lm", so a bare haru_tiny is refused before any plugin loads. */
std::string prepare_model(const char* tenant) {
    namespace fs = std::filesystem;
    const std::string src = std::string(RAD_HARU_DATA) + "/haru_tiny";
    const std::string dst = radtest::tmp_path(std::string("haru_head_model_") + tenant);
    std::error_code ec;
    if (!fs::exists(src + "/config.json", ec) || !fs::exists(src + "/model.safetensors", ec))
        return std::string();
    fs::remove_all(dst, ec);
    fs::create_directories(dst, ec);
    if (ec) return std::string();
    for (const char* f : { "config.json", "model.safetensors", "tokenizer.json" }) {
        fs::copy_file(src + "/" + f, dst + "/" + f, fs::copy_options::overwrite_existing, ec);
        if (ec) return std::string();
    }
    nlohmann::json c;
    try {
        std::ifstream in(dst + "/config.json");
        if (!in) return std::string();
        in >> c;
    } catch (const std::exception&) {
        return std::string();
    }
    if (!c.is_object() || !c.contains("lm") || !c["lm"].is_object()) return std::string();
    const nlohmann::json lm = c["lm"];   /* a copy: inserting top-level keys can move it */
    for (const char* k : { "num_hidden_layers", "hidden_size", "num_attention_heads",
                           "num_key_value_heads", "head_dim", "intermediate_size", "vocab_size",
                           "max_position_embeddings", "rms_norm_eps" })
        if (lm.contains(k)) c[k] = lm[k];
    c["haru_test_tenant"] = tenant;
    try {
        std::ofstream out(dst + "/config.json");
        if (!out) return std::string();
        out << c.dump(2) << "\n";
        if (!out) return std::string();
    } catch (const std::exception&) {
        return std::string();
    }
    return dst;
}

bool check_close(const char* what, const std::vector<float>& got,
                 const std::vector<float>& want) {
    const haru_test::Diff d = haru_test::compare(got, want, kTol);
    if (!d.ok) {
        fprintf(stderr, "    %s: %s\n", what, d.text().c_str());
        return false;
    }
    return true;
}

/* The engine on the test's temporary checkpoint copy. The copy names the tenant the plugin should
 * declare, so the lane's tenant and the block under test always agree. */
std::unique_ptr<haru_test::Engine> start_tenant(const std::string& model, std::string* why) {
    /* The engine's duplex context default is 13500 rows a lane; these blocks never read the KV,
     * so the smaller bound only keeps the reservation honest. */
    return haru_test::Engine::start({ "--model", model, "--duplex-max-ctx", "512" }, why);
}

}  /* namespace */

TEST(the_meanflow_head_matches_the_goldens) {
    const std::string model = prepare_model("head");
    if (model.empty()) {
        fprintf(stderr, "    could not build a loadable copy of %s/haru_tiny\n", RAD_HARU_DATA);
        REQUIRE(false);
    }
    std::string why;
    std::unique_ptr<haru_test::Engine> engine = start_tenant(model, &why);
    if (!engine) {
        fprintf(stderr, "    %s\n", why.c_str());
        REQUIRE(false);
    }
    std::unique_ptr<haru_test::Lane> lane = haru_test::Lane::open(*engine, "head", &why);
    if (!lane) {
        fprintf(stderr, "    %s\n", why.c_str());
        REQUIRE(false);
    }

    const haru_test::Golden& g = haru_test::Golden::get();
    for (int t = 0; t <= 24; ++t) {
        const std::string hn = frame_name(t, "hidden");
        const std::string nn = frame_name(t, "head_noise");
        const std::string on = frame_name(t, "head_latent_normalized");
        if (!g.has(hn) || !g.has(nn) || !g.has(on)) {
            fprintf(stderr, "    missing golden tensor for frame %d\n", t);
            REQUIRE(false);
        }
        const std::vector<float> hidden = g.f32(hn);
        const std::vector<float> noise = g.f32(nn);
        const std::vector<float> want = g.f32(on);
        std::vector<float> in;
        in.reserve(hidden.size() + noise.size() + 1);
        in.insert(in.end(), hidden.begin(), hidden.end());
        in.insert(in.end(), noise.begin(), noise.end());
        in.push_back(1.5f);   /* the recording's guidance_scale (golden_meta.json) */

        std::vector<float> out;
        if (!lane->run(in, want.size(), &out, &why)) {
            fprintf(stderr, "    frame %d: %s\n", t, why.c_str());
            REQUIRE(false);
        }
        if (!check_close(("frame " + std::to_string(t) + " head").c_str(), out, want))
            REQUIRE(false);
    }

    /* No golden exists for other guidance values; check the runtime scalar actually reaches the
     * network: 1.0 and 2.0 must give different, finite latents. */
    auto run_guidance = [&](float guidance, std::vector<float>* out) {
        const std::vector<float> hidden = g.f32(frame_name(0, "hidden"));
        const std::vector<float> noise = g.f32(frame_name(0, "head_noise"));
        std::vector<float> in;
        in.reserve(hidden.size() + noise.size() + 1);
        in.insert(in.end(), hidden.begin(), hidden.end());
        in.insert(in.end(), noise.begin(), noise.end());
        in.push_back(guidance);
        return lane->run(in, noise.size(), out, &why);
    };
    std::vector<float> at1, at2;
    if (!run_guidance(1.0f, &at1) || !run_guidance(2.0f, &at2)) {
        fprintf(stderr, "    %s\n", why.c_str());
        REQUIRE(false);
    }
    REQUIRE_EQ(at1.size(), at2.size());
    bool finite = true;
    double max_diff = 0.0;
    for (size_t i = 0; i < at1.size(); ++i) {
        finite = finite && std::isfinite(at1[i]) && std::isfinite(at2[i]);
        max_diff = std::max(max_diff, (double)std::fabs(at1[i] - at2[i]));
    }
    CHECK(finite);
    CHECK(max_diff > 1e-6);
}

TEST(the_audio_embedding_matches_the_goldens) {
    const std::string model = prepare_model("embed");
    if (model.empty()) {
        fprintf(stderr, "    could not build a loadable copy of %s/haru_tiny\n", RAD_HARU_DATA);
        REQUIRE(false);
    }
    std::string why;
    std::unique_ptr<haru_test::Engine> engine = start_tenant(model, &why);
    if (!engine) {
        fprintf(stderr, "    %s\n", why.c_str());
        REQUIRE(false);
    }
    std::unique_ptr<haru_test::Lane> lane = haru_test::Lane::open(*engine, "embed", &why);
    if (!lane) {
        fprintf(stderr, "    %s\n", why.c_str());
        REQUIRE(false);
    }

    const haru_test::Golden& g = haru_test::Golden::get();

    /* Frame t's position. frame 0 is the prompt's last position (model mask 1, user mask 0 and
     * no user audio); t >= 1 is a live step, where the reference passes no masks (both 1). */
    for (int t = 0; t <= 24; ++t) {
        const std::string tn = frame_name(t, "token_in");
        const std::string mn = frame_name(t, "model_latent_in");
        const std::string en = frame_name(t, "embed");
        if (!g.has(tn) || !g.has(mn) || !g.has(en)) {
            fprintf(stderr, "    missing golden tensor for frame %d\n", t);
            REQUIRE(false);
        }
        const std::vector<int64_t> tok = g.i64(tn);
        const std::vector<float> model_latent = g.f32(mn);
        std::vector<float> user_latent;
        float user_mask = 0.0f;
        if (t == 0) {
            user_latent.assign(model_latent.size(), 0.0f);
        } else {
            user_latent = g.f32(frame_name(t, "user_latent"));
            user_mask = 1.0f;
        }
        if (tok.size() != 1 || user_latent.size() != model_latent.size()) {
            fprintf(stderr, "    frame %d has an unexpected golden shape\n", t);
            REQUIRE(false);
        }
        std::vector<float> in;
        in.reserve(1 + model_latent.size() + user_latent.size() + 2);
        in.push_back((float)tok[0]);
        in.insert(in.end(), model_latent.begin(), model_latent.end());
        in.insert(in.end(), user_latent.begin(), user_latent.end());
        in.push_back(1.0f);   /* model mask */
        in.push_back(user_mask);

        const std::vector<float> want = g.f32(en);
        std::vector<float> out;
        if (!lane->run(in, want.size(), &out, &why)) {
            fprintf(stderr, "    frame %d: %s\n", t, why.c_str());
            REQUIRE(false);
        }
        if (!check_close(("frame " + std::to_string(t) + " embed").c_str(), out, want))
            REQUIRE(false);
    }

    /* Every prompt position, with the masks the reference used. */
    const std::vector<int64_t> toks = g.i64("prompt.tokens");
    const std::vector<float> model_latents = g.f32("prompt.model_latents");
    const std::vector<float> user_latents = g.f32("prompt.user_latents");
    const std::vector<int64_t> model_mask = g.i64("prompt.model_mask");
    const std::vector<int64_t> user_mask = g.i64("prompt.user_mask");
    const std::vector<float> want_all = g.f32("prompt.embed");
    const size_t n = toks.size();
    REQUIRE(n > 0);
    REQUIRE(model_latents.size() % n == 0);
    const size_t latent = model_latents.size() / n;
    REQUIRE(latent > 0);
    REQUIRE(user_latents.size() == n * latent);
    REQUIRE(model_mask.size() == n);
    REQUIRE(user_mask.size() == n);
    REQUIRE(want_all.size() == n * latent);

    for (size_t i = 0; i < n; ++i) {
        std::vector<float> in;
        in.reserve(1 + 2 * latent + 2);
        in.push_back((float)toks[i]);
        in.insert(in.end(), model_latents.begin() + (std::ptrdiff_t)(i * latent),
                  model_latents.begin() + (std::ptrdiff_t)((i + 1) * latent));
        in.insert(in.end(), user_latents.begin() + (std::ptrdiff_t)(i * latent),
                  user_latents.begin() + (std::ptrdiff_t)((i + 1) * latent));
        in.push_back((float)model_mask[i]);
        in.push_back((float)user_mask[i]);

        const std::vector<float> want(want_all.begin() + (std::ptrdiff_t)(i * latent),
                                      want_all.begin() + (std::ptrdiff_t)((i + 1) * latent));
        std::vector<float> out;
        if (!lane->run(in, want.size(), &out, &why)) {
            fprintf(stderr, "    prompt row %zu: %s\n", i, why.c_str());
            REQUIRE(false);
        }
        if (!check_close(("prompt row " + std::to_string(i)).c_str(), out, want)) REQUIRE(false);
    }
}

RAD_TEST_MAIN()
