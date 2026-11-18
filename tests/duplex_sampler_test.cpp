/* tests/duplex_sampler_test.cpp -- the host text sampler (core/gen/duplex_sampler.h) against
 * independently generated vectors and against its own statistical and mutation checks.
 *
 * tests/data/haru/gen_sampler_vectors.py re-implements the reference's DRY, top-k/top-p, float
 * softmax and inverse CDF in plain Python from .haru-ref/sampling.py and writes the cases to
 * tests/data/haru/sampler_vectors.json. The C++ side only parses that file; it does not recompute
 * the expectations. Penalties are compared within 1e-6 and tokens exactly.
 *
 * The fixture is checked in, so a missing or malformed file is a failed case, not a skip.
 */
#include "rad_test.h"
#include "gen/duplex_sampler.h"

#include <nlohmann/json.hpp>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <map>
#include <string>
#include <vector>

using nlohmann::json;
using rad::gen::DryConfig;
using rad::gen::LaneSampler;
using rad::gen::SamplerConfig;
using rad::gen::StreamDetokenizer;

namespace {

const json& golden() {
    static const json g = [] {
        const std::string path = std::string(RAD_GEN_SAMPLER_VECTORS) + "/sampler_vectors.json";
        std::ifstream f(path);
        if (!f.is_open()) {
            std::fprintf(stderr, "  FAIL %s missing -- regenerate with "
                                 "tests/data/haru/gen_sampler_vectors.py\n",
                         path.c_str());
            std::exit(1);
        }
        return json::parse(f);
    }();
    return g;
}

DryConfig dry_from(const json& j) {
    DryConfig d;
    d.multiplier = j.value("multiplier", d.multiplier);
    d.base = j.value("base", d.base);
    d.allowed_length = j.value("allowed_length", d.allowed_length);
    d.lookback = j.value("lookback", d.lookback);
    d.max_match_length = j.value("max_match_length", d.max_match_length);
    d.max_penalty = j.value("max_penalty", d.max_penalty);
    return d;
}

SamplerConfig config_from(const json& c) {
    SamplerConfig cfg;
    cfg.temperature = c.value("temperature", cfg.temperature);
    cfg.top_k = c.value("top_k", cfg.top_k);
    cfg.top_p = c.value("top_p", cfg.top_p);
    cfg.dry = dry_from(c["dry"]);
    cfg.eop_bias = c.value("eop_bias", cfg.eop_bias);
    cfg.force_text_frames = c.value("force_text_frames", cfg.force_text_frames);
    return cfg;
}

void check_penalties(const std::map<int, float>& got, const json& want) {
    CHECK_EQ(got.size(), want.size());
    for (auto it = want.begin(); it != want.end(); ++it) {
        const int id = std::stoi(it.key());
        auto found = got.find(id);
        if (found == got.end()) {
            CHECK(false);
            continue;
        }
        CHECK_NEAR(found->second, it.value().get<double>(), 1e-6);
    }
}

void check_history(const LaneSampler& sampler, const json& want) {
    if (sampler.history().size() != want.size()) {
        CHECK_EQ(sampler.history().size(), want.size());
        return;
    }
    for (size_t i = 0; i < want.size(); ++i)
        CHECK_EQ(sampler.history()[i], want[i].get<int>());
}

float number_or_inf(const json& v) {
    if (!v.is_string()) return v.get<float>();
    const std::string s = v.get<std::string>();
    if (s == "inf") return INFINITY;
    if (s == "-inf") return -INFINITY;
    std::fprintf(stderr, "  FAIL unknown non-finite encoding \"%s\"\n", s.c_str());
    std::exit(1);
}

void check_logits(const std::vector<float>& got, const json& want) {
    if (got.size() != want.size()) {
        CHECK_EQ(got.size(), want.size());
        return;
    }
    for (size_t i = 0; i < got.size(); ++i) {
        const float e = number_or_inf(want[i]);
        if (std::isinf(e)) {
            CHECK(std::isinf(got[i]) && ((got[i] > 0) == (e > 0)));
        } else {
            CHECK_NEAR(got[i], e, 1e-6);
        }
    }
}

void replay_observe(LaneSampler& sampler, const json& observed) {
    for (const auto& t : observed) sampler.observe(t.get<int>());
}

/* The documented splitmix64 step LaneSampler::next_uniform uses, for the RNG contract checks. */
uint64_t splitmix_next(uint64_t& state) {
    uint64_t z = (state += 0x9E3779B97F4A7C15ULL);
    z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ULL;
    z = (z ^ (z >> 27)) * 0x94D049BB133111EBULL;
    return z ^ (z >> 31);
}

}  // namespace

TEST(sampler_dry_vectors) {
    for (const auto& c : golden()["penalty_cases"]) {
        const SamplerConfig cfg = config_from(c);
        LaneSampler sampler(cfg, c["pad_id"].get<int>(), c["eop_id"].get<int>(),
                            c["eos_id"].get<int>(), 99);
        replay_observe(sampler, c["observe"]);
        check_history(sampler, c["expected_history"]);
        check_penalties(sampler.penalties(), c["expected_penalties"]);
    }
}

TEST(sampler_sample_vectors) {
    for (const auto& c : golden()["sample_cases"]) {
        const SamplerConfig cfg = config_from(c);
        const int pad = c["pad_id"].get<int>();
        const int eop = c["eop_id"].get<int>();
        const int eos = c["eos_id"].get<int>();
        const uint64_t seed = c["seed"].get<uint64_t>();
        LaneSampler sampler(cfg, pad, eop, eos, seed);
        replay_observe(sampler, c["observe"]);
        check_penalties(sampler.penalties(), c["expected_penalties_before"]);

        std::vector<float> logits;
        for (const auto& v : c["logits"]) logits.push_back(v.get<float>());
        REQUIRE(!logits.empty());
        const float pad_eop[2] = { c["pad_eop"][0].get<float>(),
                                   c["pad_eop"][1].get<float>() };
        const int token = sampler.sample(logits.data(), static_cast<int>(logits.size()), pad_eop,
                                         c["uniform"].get<double>());

        CHECK_EQ(token, c["expected_token"].get<int>());
        check_logits(logits, c["expected_logits"]);
        check_penalties(sampler.penalties(), c["expected_penalties_after"]);
        check_history(sampler, c["expected_history_after"]);
        CHECK_EQ(sampler.force_text_left(), c["expected_force_text_after"].get<int>());
        /* The injected overload is documented not to touch the internal PRNG. */
        CHECK_EQ(sampler.rng_state(), c["expected_rng_state"].get<uint64_t>());
    }
}

TEST(sampler_rng_contract) {
    /* Greedy consumes no draw; a sampled token consumes exactly one. */
    SamplerConfig greedy;
    greedy.temperature = 0.0f;
    std::vector<float> row = { 0.0f, 2.0f, 1.0f };
    const float pad_eop[2] = { 0.0f, 2.0f };
    LaneSampler g(greedy, 3, 4, -1, 0xABCDEFULL);
    CHECK_EQ(g.sample(row.data(), 3, pad_eop), 1);
    CHECK_EQ(g.rng_state(), 0xABCDEFULL);

    SamplerConfig sampled;
    sampled.temperature = 1.0f;
    sampled.top_k = 0;
    sampled.dry = DryConfig{ 0.0f, 1.0f, 1, 0, 0, 0.0f };
    LaneSampler s(sampled, 3, 4, -1, 0xABCDEFULL);
    row = { 0.0f, 2.0f, 1.0f };
    s.sample(row.data(), 3, pad_eop);
    uint64_t expected = 0xABCDEFULL;
    splitmix_next(expected);
    CHECK_EQ(s.rng_state(), expected);
}

TEST(sampler_internal_rng_statistics) {
    /* A fixed row, 20000 samples from the internal splitmix64, every token's count within three
     * binomial sigma of its softmax probability. DRY is inert: lookback 0 and max_match_length 0
     * leave history_size 1 and dry_penalties with no j to visit. */
    SamplerConfig cfg;
    cfg.temperature = 1.0f;
    cfg.top_k = 0;  /* sort every token, as top_k <= 0 does in the reference */
    cfg.top_p = 1.0f;
    cfg.dry = DryConfig{ 0.0f, 1.0f, 1, 0, 0, 0.0f };
    std::vector<float> base = { 2.0f, 1.5f, 1.0f, 0.5f, 0.0f, -0.5f, -1.0f };
    const int n = static_cast<int>(base.size());
    const float pad_eop[2] = { base[0], base[1] };  /* replacement leaves the row unchanged */
    LaneSampler sampler(cfg, 0, 1, -1, 0x5EED5EEDULL);

    double maxv = *std::max_element(base.begin(), base.end());
    std::vector<double> p(base.size());
    double sum = 0.0;
    for (size_t i = 0; i < base.size(); ++i) {
        p[i] = std::exp(static_cast<double>(base[i]) - maxv);
        sum += p[i];
    }
    for (double& v : p) v /= sum;

    const int draws = 20000;
    std::vector<int> counts(base.size(), 0);
    for (int i = 0; i < draws; ++i) {
        std::vector<float> row = base;
        const int token = sampler.sample(row.data(), n, pad_eop);
        REQUIRE(token >= 0 && token < n);
        ++counts[static_cast<size_t>(token)];
    }

    /* One uniform per sampled token: the state is exactly `draws` splitmix64 increments on. */
    uint64_t expect_state = 0x5EED5EEDULL;
    for (int i = 0; i < draws; ++i) splitmix_next(expect_state);
    CHECK_EQ(sampler.rng_state(), expect_state);

    for (int i = 0; i < n; ++i) {
        const double expected = draws * p[static_cast<size_t>(i)];
        const double sigma = std::sqrt(draws * p[static_cast<size_t>(i)] *
                                       (1.0 - p[static_cast<size_t>(i)]));
        CHECK(std::fabs(counts[static_cast<size_t>(i)] - expected) <= 3.0 * sigma + 1.0);
    }
}

TEST(stream_detokenizer_rules) {
    /* A fake byte-level tokenizer: 10/11 are "a"/"b", 100 alone is an incomplete multi-byte
     * character (the reference's decode then ends in U+FFFD), 100+101 completes "é". */
    auto decode = [](const std::vector<int>& ids) -> std::string {
        if (ids == std::vector<int>{ 100 }) return "\xEF\xBF\xBD";
        if (ids == std::vector<int>{ 100, 101 }) return "\xC3\xA9";
        if (ids.size() == 1) {
            if (ids[0] == 10) return "a";
            if (ids[0] == 11) return "b";
        }
        return std::string();
    };
    StreamDetokenizer detok(decode, { 0, 1 });

    CHECK(detok.push(0) == " ");
    CHECK(detok.push(1) == "");   /* still in the same gap */
    CHECK(detok.push(0) == "");
    CHECK(detok.push(10) == "a");
    CHECK(detok.push(0) == " ");
    CHECK(detok.push(11) == "b");
    CHECK(detok.push(0) == " ");
    CHECK(detok.push(100) == ""); /* held: the decoded text ends with U+FFFD */
    CHECK(detok.push(101) == "\xC3\xA9");
    CHECK(detok.push(10) == "a"); /* the pending ids were cleared */
}

TEST(sampler_mutation_checks) {
    /* These prove the vectors can fail: each mutant follows the same inputs but deliberately
     * breaks one rule, and the expected token or logits must differ. Note the DRY/replacement
     * swap the task suggests is *not* observable: PAD and EOP are in the exempt set, so DRY never
     * penalises them and replacing those two entries commutes with it. The bias/replacement order
     * is the observable one, tested below. */
    const json& g = golden();

    /* Skip DRY: dry_changes_token's token 1 loses its 0.8 penalty and wins again. */
    {
        const auto& c = *std::find_if(g["sample_cases"].begin(), g["sample_cases"].end(),
                                      [](const json& x) {
                                          return x["name"] == "dry_changes_token";
                                      });
        const SamplerConfig cfg = config_from(c);
        LaneSampler sampler(cfg, c["pad_id"].get<int>(), c["eop_id"].get<int>(),
                            c["eos_id"].get<int>(), c["seed"].get<uint64_t>());
        replay_observe(sampler, c["observe"]);
        std::vector<float> raw;
        for (const auto& v : c["logits"]) raw.push_back(v.get<float>());
        std::vector<float> with_dry = raw;
        const float pad_eop[2] = { c["pad_eop"][0].get<float>(),
                                   c["pad_eop"][1].get<float>() };
        const int expect_token = c["expected_token"].get<int>();
        const int got = sampler.sample(with_dry.data(), static_cast<int>(with_dry.size()), pad_eop,
                                       c["uniform"].get<double>());
        CHECK_EQ(got, expect_token);

        /* The same row with no DRY subtraction and no history update changes the call. */
        std::vector<float> no_dry = raw;
        no_dry[c["pad_id"].get<int>()] = pad_eop[0];
        no_dry[c["eop_id"].get<int>()] = pad_eop[1];
        int mutant_token = 0;
        for (size_t i = 1; i < no_dry.size(); ++i)
            if (no_dry[i] > no_dry[static_cast<size_t>(mutant_token)]) mutant_token = (int)i;
        CHECK(mutant_token != expect_token);
    }

    /* Apply eop_bias before the replacement: the replacement overwrites it, changing the token. */
    {
        const auto& c = *std::find_if(
            g["sample_cases"].begin(), g["sample_cases"].end(),
            [](const json& x) { return x["name"] == "pad_eop_bias_after_replacement"; });
        std::vector<float> correct;
        for (const auto& v : c["expected_logits"]) correct.push_back(number_or_inf(v));
        std::vector<float> mutant;
        for (const auto& v : c["logits"]) mutant.push_back(v.get<float>());
        const int pad = c["pad_id"].get<int>();
        const int eop = c["eop_id"].get<int>();
        const float bias = c["eop_bias"].get<float>();
        const float pe0 = c["pad_eop"][0].get<float>(), pe1 = c["pad_eop"][1].get<float>();
        mutant[pad] = pe0;
        mutant[eop] = pe1 + bias; /* bias first ... */
        mutant[eop] = pe1;        /* ... then the replacement erases it */
        auto argmax = [](const std::vector<float>& r) {
            int best = 0;
            for (size_t i = 1; i < r.size(); ++i)
                if (r[i] > r[static_cast<size_t>(best)]) best = (int)i;
            return best;
        };
        CHECK(argmax(mutant) != argmax(correct));
        CHECK(argmax(correct) == c["expected_token"].get<int>());
    }

    /* top_p with a non-strict keep rule keeps one candidate too many. */
    {
        const auto& c = *std::find_if(
            g["sample_cases"].begin(), g["sample_cases"].end(),
            [](const json& x) { return x["name"] == "top_p_strict_and_ties"; });
        std::vector<float> row;
        for (const auto& v : c["expected_logits"]) row.push_back(number_or_inf(v));
        const double uniform = c["uniform"].get<double>();
        auto pick = [&](bool strict) {
            std::vector<int> order(row.size());
            for (size_t i = 0; i < order.size(); ++i) order[i] = (int)i;
            std::sort(order.begin(), order.end(), [&](int a, int b) {
                if (row[(size_t)a] != row[(size_t)b]) return row[(size_t)a] > row[(size_t)b];
                return a < b;
            });
            const double maxv = row[(size_t)order[0]];
            std::vector<double> p(row.size());
            double sum = 0.0;
            for (size_t k = 0; k < order.size(); ++k) {
                p[k] = std::exp((double)row[(size_t)order[k]] - maxv);
                sum += p[k];
            }
            for (double& v : p) v /= sum;

            const double top_p = c["top_p"].get<double>();
            double prev = 0.0;
            for (size_t k = 0; k < p.size(); ++k) {
                const double original = p[k];
                if (!(strict ? prev < top_p : prev <= top_p)) p[k] = 0.0;
                prev += original;
            }

            double total = 0.0;
            for (double v : p) total += v;
            const double target = uniform * total;
            double acc = 0.0;
            for (size_t k = 0; k < p.size(); ++k) {
                if (p[k] <= 0.0) continue;
                acc += p[k];
                if (target < acc) return order[k];
            }
            return order.back();
        };
        const int strict_token = pick(true);
        const int loose_token = pick(false);
        CHECK(strict_token == c["expected_token"].get<int>());
        CHECK(loose_token != strict_token);
    }
}
