/* w3_emulate_test.cpp -- stage 1 of the 3-bit routed-expert format (w3nl64a8h): quantise real
 * expert tensors of a container to the proposed 3-bit layout and measure the reconstruction error
 * against the 4- and 5-bit forms already served, and against llama.cpp's 3- to 5-bit formats from
 * libquant's `ggml` quantiser, on the very same tensors.
 *
 * Needs $W3_RAD, a container whose experts are fp8 (the Qwen3.6 35B-A3B .rad); skips without it.
 * Optional: $W3_LAYERS "0,10,20,30,39", $W3_EXPERTS "0,64,128,255", $W3_FIT_LAYERS (the layers
 * the codebook is fitted on, default W3_LAYERS), $W3_TABLE (a ;-list of the four magnitudes,
 * skipping the fit), $W3_TOP (the fitted table's top level, default 24), $W3_DUMP (a directory:
 * the w3nl codes and scale planes of every tensor measured are written there as <name>.codes and
 * <name>.scale, which libhalo/halo_w3_bench.hip reads), $W3_ONLY (comma-separated labels to keep).
 *
 * It measures; it asserts only that the 3-bit form is worse than the 4-bit and better than
 * nothing, which a broken quantiser or table order would violate.
 */
#include "quant_fixture.h"
#include "format/radfile.h"

#include <algorithm>
#include <map>
#include <numeric>
#include <sstream>

using namespace rad;
using namespace qfix;

namespace {

std::vector<int> ints_env(const char* k, const char* dflt) {
    const char* v = std::getenv(k);
    std::string s = v ? v : dflt;
    std::vector<int> out;
    std::stringstream ss(s);
    std::string t;
    while (std::getline(ss, t, ',')) out.push_back(std::atoi(t.c_str()));
    return out;
}

bool load(const RadFile& f, const std::string& name, int64_t* rows, int64_t* cols,
          std::vector<float>* w) {
    const RadFileEntry* e = f.find(name);
    if (!e) return false;
    *rows = e->shape[0];
    *cols = e->shape[1];
    std::vector<const void*> d;
    for (uint32_t k = 0; k < e->n_planes; ++k) d.push_back(f.data(f.plane(*e, (int)k)));
    EncodedView v;
    enc_view(f.encoding(*e), *rows, *cols, d.data(), &v);
    w->resize((size_t)(*rows * *cols));
    std::string why;
    if (enc_decode_rows(v, 0, *rows, w->data(), &why) != RAD_OK) {
        std::fprintf(stderr, "decode %s: %s\n", name.c_str(), why.c_str());
        return false;
    }
    return true;
}

/* "a=b,c=d" into Opts (the values may hold ';' and '*'). */
void parse_opts(Opts& o, const std::string& s) {
    std::stringstream ss(s);
    std::string t;
    while (std::getline(ss, t, ' ')) {
        if (t.empty()) continue;
        const size_t eq = t.find('=');
        const std::string k = t.substr(0, eq), v = t.substr(eq + 1);
        if (k == "group" || k == "group2" || k == "cd") o.i(k.c_str(), std::atoll(v.c_str()));
        else if (k == "scale_value" || k == "scale2_value") o.f(k.c_str(), std::atof(v.c_str()));
        else o.s(k.c_str(), v.c_str());
    }
}

/* ---- the codebook fit: alternating least squares over the levels and the per-64 block scales of
 * the rotated weights, magnitudes only (the codebook is symmetric: sign plane + magnitude plane) */
std::vector<double> fit_levels(const std::vector<std::vector<float>>& rot, int nlev, double top,
                               int iters) {
    /* start from the Gaussian Lloyd-Max levels for the count, scaled to `top` */
    std::vector<double> L(nlev);
    for (int k = 0; k < nlev; ++k) L[k] = top * (k + 0.5) / nlev;
    for (int it = 0; it < iters; ++it) {
        std::vector<double> num(nlev, 0.0), den(nlev, 0.0);
        for (const auto& w : rot) {
            for (size_t b = 0; b + 64 <= w.size(); b += 64) {
                /* the block's scale: least squares over the assignment, starting from amax/top */
                double amax = 0;
                for (int i = 0; i < 64; ++i) amax = std::max(amax, (double)std::fabs(w[b + i]));
                if (amax == 0) continue;
                double s = amax / L[nlev - 1];
                int code[64];
                for (int r = 0; r < 6; ++r) {
                    double sxl = 0, sll = 0;
                    for (int i = 0; i < 64; ++i) {
                        const double a = std::fabs(w[b + i]) / s;
                        int best = 0;
                        double bd = 1e300;
                        for (int k = 0; k < nlev; ++k) {
                            const double d = std::fabs(a - L[k]);
                            if (d < bd) { bd = d; best = k; }
                        }
                        code[i] = best;
                        sxl += std::fabs(w[b + i]) * L[best];
                        sll += L[best] * L[best];
                    }
                    if (sll > 0) s = sxl / sll;
                }
                for (int i = 0; i < 64; ++i) {
                    num[code[i]] += std::fabs(w[b + i]) * s;
                    den[code[i]] += s * s;
                }
            }
        }
        for (int k = 0; k < nlev; ++k)
            if (den[k] > 0) L[k] = num[k] / den[k];
        const double sc = top / L[nlev - 1];   /* the top level is the normalisation */
        for (double& l : L) l *= sc;
    }
    return L;
}

double e4m3_round(double v) {   /* nearest E4M3 value, v >= 0 */
    if (v <= 0) return 0;
    int e;
    std::frexp(v, &e);            /* v = m 2^e, m in [0.5,1) */
    const int ex = std::max(e - 1, -6);
    const double step = std::ldexp(1.0, ex - 3);
    return std::floor(v / step + 0.5) * step;
}

std::string levels_str(const std::vector<double>& m) {
    std::string t;
    char b[64];
    for (double v : m) { std::snprintf(b, sizeof b, "%.9g;", v); t += b; }
    for (double v : m) { std::snprintf(b, sizeof b, "%.9g;", -v); t += b; }
    t.pop_back();
    return t;
}

struct Acc { double num = 0, den = 0; int n = 0; double worst = 0; };

}  // namespace

TEST(w3_emulation_error_against_4_and_5_bit_on_real_experts) {
    NEED_LIBQUANT();
    const char* rad = std::getenv("W3_RAD");
    if (!rad) SKIP("W3_RAD is not set");
    RadFile f;
    REQUIRE_EQ(f.open(rad), RAD_OK);
    const std::vector<int> layers = ints_env("W3_LAYERS", "0,10,20,30,39");
    const std::vector<int> experts = ints_env("W3_EXPERTS", "0,64,128,255");
    const std::vector<int> fitl = ints_env("W3_FIT_LAYERS", std::getenv("W3_LAYERS")
                                                               ? std::getenv("W3_LAYERS")
                                                               : "0,10,20,30,39");
    std::vector<std::string> names;
    for (int l : layers)
        for (int e : experts)
            for (const char* p : { "ffn_gate_up", "ffn_down" })
                names.push_back("blk." + std::to_string(l) + "." + p + "_exps." +
                                std::to_string(e) + ".weight");

    /* ---- the fit, on rotated weights of the fit layers */
    std::vector<double> mag;
    if (const char* t = std::getenv("W3_TABLE")) {
        std::stringstream ss(t);
        std::string x;
        while (std::getline(ss, x, ';')) mag.push_back(std::atof(x.c_str()));
    } else {
        std::vector<std::vector<float>> rot;
        for (int l : fitl)
            for (int e : experts)
                for (const char* p : { "ffn_gate_up", "ffn_down" }) {
                    std::vector<float> w;
                    int64_t R, C;
                    if (!load(f, "blk." + std::to_string(l) + "." + p + "_exps." +
                                     std::to_string(e) + ".weight", &R, &C, &w))
                        continue;
                    for (int64_t r = 0; r < R; ++r)
                        for (int64_t c = 0; c + 128 <= C; c += 128)
                            fwht_inplace(w.data() + r * C + c, 128);
                    rot.push_back(std::move(w));
                }
        const double top = std::getenv("W3_TOP") ? std::atof(std::getenv("W3_TOP")) : 24.0;
        std::vector<double> m = fit_levels(rot, 4, top, 12);
        std::printf("W3 fit (4 magnitudes, top %.4g, %zu tensors): ", top, rot.size());
        for (double v : m) std::printf("%.5f ", v);
        std::printf("\n");
        for (double v : m) mag.push_back(e4m3_round(v));
        std::printf("W3 fit rounded to E4M3: ");
        for (double v : mag) std::printf("%g ", v);
        std::printf("\n");
    }
    REQUIRE_EQ(mag.size(), (size_t)4);
    std::vector<double> uni = { 0.5, 1.5, 2.5, 3.5 };
    for (double& u : uni) u *= mag[3] / 3.5;   /* the uniform grid at the same top level */

    const std::string base = " group=64 scale=fp8_e4m3 scale2=f32 block2=*x* "
                             "scale2_value=0.0001220703125 transform=fwht128 rule=search";
    struct Var { std::string label; std::string quantizer; std::string opts; double bpw; };
    std::vector<Var> vars = {
        { "w5nl64a8h",  "rtn", "table=w5nl" + base, 5.125 },
        { "w4nl64a8h",  "rtn", "table=w4nl" + base, 4.125 },
        { "w3nl-fit",   "rtn", "table=" + levels_str(mag) + base, 3.125 },
        { "w3nl-named", "rtn", "table=w3nl" + base, 3.125 },
        { "w3-uniform", "rtn", "table=" + levels_str(uni) + base, 3.125 },
        { "q6_K",       "ggml", "type=q6_K", 6.5625 },
        { "q5_K",       "ggml", "type=q5_K", 5.5 },
        { "q4_K",       "ggml", "type=q4_K", 4.5 },
        { "iq4_xs",     "ggml", "type=iq4_xs", 4.25 },
        { "q3_K",       "ggml", "type=q3_K", 3.4375 },
        { "iq3_xxs",    "ggml", "type=iq3_xxs", 3.0625 },
    };
    if (const char* extra = std::getenv("W3_EXTRA")) {   /* "label|quantizer|opts|bpw;..." */
        std::stringstream ss(extra);
        std::string it;
        while (std::getline(ss, it, '#')) {
            std::vector<std::string> p;
            std::stringstream s2(it);
            std::string x;
            while (std::getline(s2, x, '|')) p.push_back(x);
            if (p.size() == 4) vars.push_back({ p[0], p[1], p[2], std::atof(p[3].c_str()) });
        }
    }

    if (const char* only = std::getenv("W3_ONLY")) {   /* "a,b": keep these labels */
        std::vector<Var> keep;
        for (const Var& v : vars)
            if (std::string(",") + only + "," != "" &&
                (std::string(",") + only + ",").find("," + v.label + ",") != std::string::npos)
                keep.push_back(v);
        vars = keep;
    }
    std::map<std::string, Acc> by;   /* label|kind */
    std::map<std::string, std::vector<int>> scale_hist;
    for (const std::string& name : names) {
        std::vector<float> w;
        int64_t R, C;
        if (!load(f, name, &R, &C, &w)) continue;
        const bool gu = name.find("gate_up") != std::string::npos;
        for (const Var& v : vars) {
            if (v.quantizer == "ggml" && C % 256) continue;
            Opts o;
            parse_opts(o, v.opts);
            Quantised q = quantise(quantizer(v.quantizer.c_str()), o, name.c_str(), R, C, w);
            if (q.status != RAD_OK) { std::printf("  %s %s status %d\n", v.label.c_str(), name.c_str(), q.status); continue; }
            const std::vector<float> d = decode(q, R, C);
            if (d.empty()) continue;
            double num = 0, den = 0;
            for (size_t i = 0; i < d.size(); ++i) {
                num += (double)(w[i] - d[i]) * (w[i] - d[i]);
                den += (double)w[i] * w[i];
            }
            for (const char* kind : { gu ? "gate_up" : "down", "all" }) {
                Acc& a = by[v.label + "|" + kind];
                a.num += num; a.den += den; a.n++;
                a.worst = std::max(a.worst, std::sqrt(num / den));
            }
            if (const char* dump = std::getenv("W3_DUMP"); dump && v.label == "w3nl-named") {
                /* the quantised planes of this tensor, for libhalo's halo_w3_bench `check real` */
                for (int k = 0; k < 2; ++k) {
                    const std::string fn = std::string(dump) + "/" + name + (k ? ".scale" : ".codes");
                    FILE* fp = std::fopen(fn.c_str(), "wb");
                    if (fp) {
                        std::fwrite(q.planes[(size_t)k].data(), 1, q.planes[(size_t)k].size(), fp);
                        std::fclose(fp);
                    }
                }
            }
            if (std::getenv("W3_DEBUG") && v.quantizer == "rtn") {
                std::printf("  enc %s:", enc_name(q.enc).c_str());
                for (auto& pl : q.planes) std::printf(" %zu", pl.size());
                std::printf("\n");
            }
            if (v.quantizer == "rtn" && q.planes.size() > 1) {   /* the E4M3 scale bytes */
                std::vector<int>& h = scale_hist[v.label];
                h.resize(256, 0);
                for (uint8_t b : q.planes[1]) h[b]++;
            }
        }
    }
    std::printf("\nW3 error on %zu tensors (%zu layers x %zu experts x gate_up,down), relative to the fp8 source\n",
                names.size(), layers.size(), experts.size());
    std::printf("%-12s %6s | %-9s %-9s %-9s | %-9s %-9s\n", "format", "bpw", "rel_rmse", "snr_dB",
                "worst", "gate_up", "down");
    for (const Var& v : vars) {
        const Acc &a = by[v.label + "|all"], &g = by[v.label + "|gate_up"], &d = by[v.label + "|down"];
        if (!a.n) continue;
        auto rr = [](const Acc& x) { return x.den > 0 ? std::sqrt(x.num / x.den) : 0.0; };
        std::printf("%-12s %6.3f | %-9.5f %-9.2f %-9.5f | %-9.5f %-9.5f\n", v.label.c_str(), v.bpw,
                    rr(a), -20 * std::log10(rr(a)), a.worst, rr(g), rr(d));
    }
    /* the E4M3 scale bytes: where the scales sit under the fixed 2^-13 */
    for (auto& kv : scale_hist) {
        const std::vector<int>& h = kv.second;
        double tot = 0, sub = 0, zer = 0, sat = 0, mn = 1e30, mx = 0;
        std::map<int, double> ex;
        for (int b = 0; b < 256; ++b) {
            if (!h[b]) continue;
            tot += h[b];
            const int e = (b >> 3) & 15, m = b & 7;
            if ((b & 0x7F) == 0x7E) sat += h[b];
            double val = e == 0 ? m / 8.0 * std::ldexp(1.0, -6) : (1 + m / 8.0) * std::ldexp(1.0, e - 7);
            if (e == 0) sub += h[b];
            if (b == 0) zer += h[b];
            mn = std::min(mn, val); mx = std::max(mx, val);
            ex[e] += h[b];
        }
        std::printf("W3 scale bytes %-12s n=%.0f e4m3 value range [%.4g, %.4g] (x2^-13), subnormal-or-zero %.4f%% (of which exactly zero %.4f%%), at the top code %.4f%%; exponent field histogram:", kv.first.c_str(), tot, mn, mx, 100 * sub / tot, 100 * zer / tot, 100 * sat / tot);
        for (auto& e : ex) std::printf(" %d:%.2f%%", e.first, 100 * e.second / tot);
        std::printf("\n");
    }
    const Acc &a3 = by["w3nl-fit|all"], &a4 = by["w4nl64a8h|all"];
    REQUIRE(a3.n > 0 && a4.n > 0);
    CHECK(a3.num / a3.den > a4.num / a4.den);
    CHECK(a3.num / a3.den < 1.0);
}

RAD_TEST_MAIN()
