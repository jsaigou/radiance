// Dumps what the fork's own C++ reader (core/format/ggml_planes.h) makes of raw GGUF blocks, one
// file per plane, so crosscheck_cpp.py can compare it byte for byte with the python layout.
// Build (from the repo root): c++ -std=c++17 -I. -Iabi -Icore -Icore/format -Icore/text \
//     tools/ud_repack/ggml_planes_dump.cpp -o /tmp/ggml_planes_dump
// Usage: ggml_planes_dump TYPE(8|12|13|14|42) blocks.bin ROWS K outprefix
#include "core/format/ggml_planes.h"
// gguf_dtype lives in core/text/gguf.cpp (which drags in the whole text stack); the three block
// types this tool splits are all it needs, with the sizes ggml defines.
namespace rad {
const GgufDtype* gguf_dtype(uint32_t t) {
    static const GgufDtype q8_0{32, 34, "q8_0"}, q4_K{256, 144, "q4_K"}, q5_K{256, 176, "q5_K"}, q6_K{256, 210, "q6_K"}, q2_0{64, 18, "q2_0"};
    return t == 8 ? &q8_0 : t == 12 ? &q4_K : t == 13 ? &q5_K : t == 14 ? &q6_K : t == 42 ? &q2_0 : nullptr;
}
}  // namespace rad
#include <cstdio>
#include <cstdlib>
#include <vector>
#include <string>
int main(int argc, char** argv) {
    if (argc != 6) { fprintf(stderr, "usage\n"); return 2; }
    const uint32_t ty = (uint32_t)atoi(argv[1]);
    const int64_t rows = atoll(argv[3]), K = atoll(argv[4]);
    const rad::ggml_planes::Type* t = rad::ggml_planes::type_of(ty);
    if (!t) { fprintf(stderr, "type %u not split by ggml_planes.h\n", ty); return 3; }
    const rad::GgufDtype& gd = *rad::gguf_dtype(ty);
    const int64_t nb = K / gd.blck, rb = nb * (int64_t)gd.block_bytes;
    std::vector<uint8_t> in((size_t)(rows * rb));
    FILE* f = fopen(argv[2], "rb"); if (!f || fread(in.data(), 1, in.size(), f) != in.size()) { fprintf(stderr, "read\n"); return 4; } fclose(f);
    // plane row bytes: bits per element of the plane's dtype times elements per row
    auto bits_of = [](uint32_t dt) { switch (dt) { case RAD_I8: return 8; case RAD_U2: return 2; case RAD_U4: return 4; case RAD_U6: return 6; case RAD_U5: return 5; case RAD_I6: return 6; case RAD_F16: return 16; default: return -1; } };
    std::vector<std::vector<uint8_t>> out; std::vector<int64_t> rowbytes; int np = 0;
    for (const auto& p : t->planes) { if (!p.role) break; ++np; }
    for (int i = 0; i < np; ++i) {
        const auto& p = t->planes[i];
        int64_t elems = K / p.block;
        int64_t by = (elems * bits_of(p.dtype) + 7) / 8;
        rowbytes.push_back(by); out.emplace_back((size_t)(rows * by));
    }
    for (int64_t r = 0; r < rows; ++r) {
        std::vector<uint8_t*> ptr(np);
        for (int i = 0; i < np; ++i) ptr[i] = out[i].data() + r * rowbytes[i];
        rad::ggml_plane_row(ty, in.data() + r * rb, K, ptr.data());
    }
    for (int i = 0; i < np; ++i) {
        std::string fn = std::string(argv[5]) + "." + t->planes[i].role;
        FILE* o = fopen(fn.c_str(), "wb"); fwrite(out[i].data(), 1, out[i].size(), o); fclose(o);
        printf("%s %lld\n", t->planes[i].role, (long long)rowbytes[i]);
    }
    return 0;
}
