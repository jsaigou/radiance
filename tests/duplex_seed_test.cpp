/* duplex_seed_test.cpp -- the pinned FNV-1a 64-bit session seed.
 *
 * std::hash is implementation-defined, so a replay that must survive a different standard library
 * cannot use it. The two vectors are FNV's published ones; a change here is a wire-visible change
 * to every host-sampled lane's tokens and noise. */
#include "rad_test.h"

#include "server/duplex.h"

#include <cstdint>

using rad::server::duplex_seed;

TEST(the_duplex_seed_is_pinned_fnv1a_64) {
    CHECK_EQ(duplex_seed(""), (uint64_t)0xcbf29ce484222325ULL);
    CHECK_EQ(duplex_seed("a"), (uint64_t)0xaf63dc4c8601ec8cULL);
    CHECK_EQ(duplex_seed("foobar"), (uint64_t)0x85944171f73967e8ULL);
    CHECK(duplex_seed("session-1") != duplex_seed("session-2"));
}

RAD_TEST_MAIN()
