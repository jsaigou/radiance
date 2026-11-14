/* avx_iq4_check.cpp -- embed_lookup_q over a real IQ4_NL table on the host, against a reference
 * decoded by tools/ud_repack, and a throughput measurement of random-row gathers.
 *
 * The n-gram table in the Unsloth UD release is IQ4_NL: u4 codes and an f16 scale per 32 values,
 * 26.8 GiB. This tool loads libavx and libref by path (nothing is linked), maps the codes and scale
 * planes of a real slice, and launches embed_lookup_q with `wform=iq4_nl` on a list of ids. With
 * `--expected` it compares the output to the bytes a Python decode of the same rows wrote, bit for
 * bit, and it compares libavx against libref. With `--bench N` it times N random-row gathers
 * through libavx's row cache and prints rows and bytes a second.
 *
 * The planes are the ones the container stores (rad_encoding.h): codes [rows, n_embd] packed two
 * nibbles a byte, and scale [rows, n_embd / 32] f16. tools/ple_iq4/extract.py writes both from a
 * GGUF, and expected.bf16 from tools/ud_repack's decode.
 *
 *   rad-avx-iq4 --avx libavx.so --ref libref.so --codes codes.bin --scales scales.bin \
 *               --rows 320001536 --n_embd 160 --ids ids.bin --expected expected.bf16
 *   rad-avx-iq4 ... --bench 200000
 */
#include "avx_harness.h"

#include <cerrno>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <time.h>
#include <unistd.h>

namespace {

struct Mapped {
    void*  p = nullptr;
    size_t n = 0;
    void* map(const char* path, size_t bytes) {
        const int fd = ::open(path, O_RDONLY | O_CLOEXEC);
        if (fd < 0) { std::fprintf(stderr, "open %s: %s\n", path, std::strerror(errno)); return nullptr; }
        struct stat st {};
        if (::fstat(fd, &st) != 0) { std::fprintf(stderr, "fstat %s: %s\n", path, std::strerror(errno)); ::close(fd); return nullptr; }
        const size_t want = bytes ? (bytes < (size_t)st.st_size ? bytes : (size_t)st.st_size)
                                  : (size_t)st.st_size;
        p = ::mmap(nullptr, want, PROT_READ, MAP_SHARED, fd, 0);
        ::close(fd);
        if (p == MAP_FAILED) { p = nullptr; std::fprintf(stderr, "mmap %s: %s\n", path, std::strerror(errno)); return nullptr; }
        n = want;
        return p;
    }
};

bool read_file(const char* path, void* dst, size_t bytes) {
    std::FILE* f = std::fopen(path, "rb");
    if (!f) { std::fprintf(stderr, "open %s: %s\n", path, std::strerror(errno)); return false; }
    const size_t got = std::fread(dst, 1, bytes, f);
    std::fclose(f);
    if (got != bytes) { std::fprintf(stderr, "%s: %zu bytes wanted, %zu read\n", path, bytes, got); return false; }
    return true;
}

double now_s() {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (double)ts.tv_sec + (double)ts.tv_nsec * 1e-9;
}

RadTensor tensor(void* data, uint32_t dt, std::initializer_list<int64_t> shape) {
    RadTensor t{};
    t.data = data;
    t.dtype = dt;
    t.rank = (uint32_t)shape.size();
    int i = 0;
    for (int64_t s : shape) t.shape[i++] = s;
    int64_t st = 1;
    for (int k = (int)t.rank - 1; k >= 0; --k) { t.stride[k] = st; st *= t.shape[k]; }
    return t;
}

struct Opts {
    const char* avx = nullptr;
    const char* ref = nullptr;
    const char* codes = nullptr;
    const char* scales = nullptr;
    const char* ids = nullptr;
    const char* expected = nullptr;
    int64_t rows = 0;
    int64_t n_embd = 160;
    int64_t M = 0;          /* ids in the file, or the bench row count */
    int64_t bench = 0;
    bool quiet = false;
};

}  /* namespace */

int main(int argc, char** argv) {
    Opts o;
    for (int i = 1; i < argc; ++i) {
        auto val = [&]() -> const char* {
            if (i + 1 >= argc) { std::fprintf(stderr, "%s needs a value\n", argv[i]); std::exit(2); }
            return argv[++i];
        };
        if (!std::strcmp(argv[i], "--avx")) o.avx = val();
        else if (!std::strcmp(argv[i], "--ref")) o.ref = val();
        else if (!std::strcmp(argv[i], "--codes")) o.codes = val();
        else if (!std::strcmp(argv[i], "--scales")) o.scales = val();
        else if (!std::strcmp(argv[i], "--ids")) o.ids = val();
        else if (!std::strcmp(argv[i], "--expected")) o.expected = val();
        else if (!std::strcmp(argv[i], "--rows")) o.rows = std::atoll(val());
        else if (!std::strcmp(argv[i], "--n_embd")) o.n_embd = std::atoll(val());
        else if (!std::strcmp(argv[i], "--M")) o.M = std::atoll(val());
        else if (!std::strcmp(argv[i], "--bench")) o.bench = std::atoll(val());
        else if (!std::strcmp(argv[i], "--quiet")) o.quiet = true;
        else { std::fprintf(stderr, "unknown option %s\n", argv[i]); return 2; }
    }
    if (!o.avx || !o.codes || !o.scales || o.rows <= 0 || o.n_embd <= 0 || o.n_embd % 32) {
        std::fprintf(stderr, "--avx, --codes, --scales, --rows and an n_embd of whole 32s are "
                             "required\n");
        return 2;
    }

    Lib avx, ref;
    if (!avx.open(o.avx)) return 2;
    if (o.ref && !ref.open(o.ref)) return 2;

    const size_t code_bytes = (size_t)(o.rows * (o.n_embd / 2));
    const size_t scale_bytes = (size_t)(o.rows * (o.n_embd / 32) * 2);
    Mapped codes, scales;
    if (!codes.map(o.codes, code_bytes)) return 2;
    if (!scales.map(o.scales, scale_bytes)) return 2;
    if (codes.n < code_bytes || scales.n < scale_bytes) {
        std::fprintf(stderr, "planes are shorter than %lld rows of %lld: codes %zu/%zu, scales "
                             "%zu/%zu\n", (long long)o.rows, (long long)o.n_embd, codes.n,
                   code_bytes, scales.n, scale_bytes);
        return 2;
    }

    std::vector<int32_t> ids;
    if (o.ids) {
        struct stat st {};
        if (::stat(o.ids, &st) != 0) { std::fprintf(stderr, "stat %s: %s\n", o.ids, std::strerror(errno)); return 2; }
        ids.resize((size_t)(st.st_size / 4));
        if (!read_file(o.ids, ids.data(), ids.size() * 4)) return 2;
    }
    const int64_t M = o.M ? o.M : (o.bench ? o.bench : (int64_t)ids.size());
    if (M <= 0) { std::fprintf(stderr, "no ids: pass --ids or --M with --bench\n"); return 2; }

    /* The bench case draws its own ids from a seeded generator so a run is reproducible. */
    if ((int64_t)ids.size() < M && o.bench) {
        ids.resize((size_t)M);
        uint64_t x = 0x9e3779b97f4a7c15ull;
        for (int64_t m = 0; m < M; ++m) {
            x ^= x << 13; x ^= x >> 7; x ^= x << 17;
            ids[(size_t)m] = (int32_t)(x % (uint64_t)o.rows);
        }
    }
    if ((int64_t)ids.size() < M) { std::fprintf(stderr, "ids file has %zu, %lld wanted\n", ids.size(), (long long)M); return 2; }

    std::vector<uint16_t> out((size_t)(M * o.n_embd), 0);
    std::vector<RadTensor> ts = {
        tensor(ids.data(), RAD_I32, { M }),
        tensor(codes.p, RAD_U4, { o.rows, o.n_embd }),
        tensor(scales.p, RAD_F16, { o.rows, o.n_embd / 32 }),
        tensor(out.data(), RAD_BF16, { M, o.n_embd }),
        tensor(nullptr, RAD_I32, { 0 }),
    };
    RadParam ps[6] = {};
    ps[0].key = "M";       ps[0].kind = RAD_P_INT; ps[0].ival = M;
    ps[1].key = "n_embd";  ps[1].kind = RAD_P_INT; ps[1].ival = o.n_embd;
    ps[2].key = "n_vocab"; ps[2].kind = RAD_P_INT; ps[2].ival = o.rows;
    ps[3].key = "dtype";   ps[3].kind = RAD_P_STR; ps[3].sval = "bf16";
    ps[4].key = "wform";   ps[4].kind = RAD_P_STR; ps[4].sval = "iq4_nl";
    RadArgs a{};
    a.t = ts.data(); a.n_t = (int)ts.size();
    a.p = ps; a.n_p = 5;               /* wform is optional; libavx dispatches on wte's dtype */
    a.rank = 0; a.world_size = 1;

    const RadKernelInfo* ka = avx.select("embed_lookup_q", ps, 5);
    if (!ka || !ka->launch) { std::fprintf(stderr, "libavx has no host embed_lookup_q row\n"); return 2; }

    std::vector<uint16_t> want;
    if (o.expected) {
        want.resize((size_t)(M * o.n_embd));
        if (!read_file(o.expected, want.data(), want.size() * 2)) return 2;
    }

    int rc = ka->launch(&a, nullptr);
    if (rc != RAD_OK) { std::fprintf(stderr, "libavx embed_lookup_q returned %d\n", rc); return 1; }

    bool ok = true;
    if (!want.empty()) {
        int64_t bad = 0;
        for (size_t i = 0; i < want.size(); ++i) bad += out[i] != want[i];
        const int64_t n = (int64_t)want.size();
        std::printf("host vs python: %lld of %lld values differ%s\n", (long long)bad, (long long)n,
                    bad ? " -- FAIL" : " (bit for bit)");
        ok = ok && bad == 0;
    }

    if (o.ref) {
        const RadKernelInfo* kr = ref.select("embed_lookup_q", ps, 5);
        std::vector<uint16_t> r((size_t)(M * o.n_embd), 0);
        std::vector<RadTensor> rt = ts;
        rt[3] = tensor(r.data(), RAD_BF16, { M, o.n_embd });
        RadArgs ra = a;
        ra.t = rt.data();
        if (!kr || !kr->launch) {
            std::fprintf(stderr, "libref has no host embed_lookup_q row\n");
            ok = false;
        } else {
            const int rr = kr->launch(&ra, nullptr);
            int64_t bad = 0;
            for (size_t i = 0; i < r.size(); ++i) bad += r[i] != out[i];
            std::printf("host vs libref: %lld of %lld values differ%s\n", (long long)bad,
                        (long long)r.size(), (rr == RAD_OK && bad == 0) ? " (bit for bit)"
                                                                        : " -- FAIL");
            ok = ok && rr == RAD_OK && bad == 0;
        }
    }

    if (o.bench) {
        std::fill(out.begin(), out.end(), 0);
        const double t0 = now_s();
        rc = ka->launch(&a, nullptr);
        const double t1 = now_s();
        if (rc != RAD_OK) { std::fprintf(stderr, "bench launch returned %d\n", rc); return 1; }
        const double dt = t1 - t0;
        const double row_bytes = (double)o.n_embd / 2.0 + (double)o.n_embd / 32.0 * 2.0;
        std::printf("bench: %lld random rows in %.3f s = %.0f rows/s, %.2f GiB/s of table "
                    "(%.0f bytes a row)\n", (long long)M, dt, (double)M / dt,
                    (double)M * row_bytes / dt / (1024.0 * 1024.0 * 1024.0), row_bytes);
    }

    return ok ? 0 : 1;
}
