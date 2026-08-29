#pragma once
#include <cstdint>
#include <string>
#include <vector>

// sieve/rung_scan3.cpp is #included by ref_host.cpp and NOWHERE ELSE. A second
// #include of the same .cpp in another translation unit is an ODR violation -- the
// same trap census/cuda/ref_host.cpp documents on the census reference. Everything
// the device translation units need from the CPU scanner comes through this header.

// No scan3_main: rung_scan3.cpp's own #undef main defeats the census port's rename
// trick, so its main is suppressed with RUNG_SCAN3_NO_MAIN instead. --verify delegates
// straight to the self-test, which is all it needed from main anyway.
void ref3_build_tables(uint64_t hi);      // init_phi + init_rfac + build_base_primes
int  ref3_verify();                       // scan3_self_test()

// The host tail: stage D and stage E, verbatim from the CPU scanner. They run on a
// handful of survivors per window, so reusing them costs nothing -- and it keeps the
// emitted certificates identical to the CPU scanner's BY CONSTRUCTION rather than by
// agreement, which is a much stronger property for the differential gate to rest on.
bool ref3_solve_rung(uint64_t p, std::string &out);
bool ref3_solve_direct(uint64_t p, std::string &out);
bool ref3_is_prime(uint64_t n);           // the reference's 12-base MR, for --check-primality

// The stage-E ceiling, derived in rung_scan3.cpp from the frozen reference's isqrt_u64.
// The driver refuses HI above it for the same reason rung_scan3 does: silently skipping
// stage E turns a numeric limit into what reads as a mathematical result.
uint64_t ref3_direct_pmax();

// The wheel and the filter set, so the device driver never re-derives them.
uint32_t ref3_wheel_size();
uint64_t ref3_wheel(uint32_t i);
uint64_t ref3_wheel_pmin();
uint64_t ref3_cover_pmax(uint64_t mmax);
uint64_t ref3_class_fnv64_of(uint64_t M);      // digest of the derived class table
bool     ref3_class_pinned(uint64_t M, uint64_t *fnv64, const char **sha256);

// Builds the wheel for M and caches it; must be called before the accessors above.
void ref3_build_wheel(uint64_t M, uint64_t mmax);

// The flattened filter set, exactly as rung_scan3's main builds it: covers with
// m | M excluded (already spent on the wheel), then small primes appended if sieve_p.
// ncov is the boundary -- entries below it are certified covers, above are sieve
// primes, and the two must stay separable because `covered` and `sieved` mean
// different things in SUMMARY.
void ref3_filters(uint64_t M, uint64_t mmax, const char *file, uint64_t sieve_p,
                  std::vector<uint64_t> &mods, std::vector<uint64_t> &ress,
                  size_t &ncov);

// Where each filter first hits lane `first`, and how far apart its hits are, indexed by
// the GLOBAL position index J = (p - first)/M. This is sweep_lane's own setup loop, in
// the same TU as the scanner's gcd3/inv3, so the extended Euclid the offsets rest on is
// the reference's and not a re-implementation. It stays on the HOST on purpose: one
// extended Euclid per (lane, filter) per shard is invisible at production range, and
// keeping it off the device removes a whole class of divergence.
void ref3_lane_offsets(uint64_t M, uint64_t first,
                       const std::vector<uint64_t> &mods,
                       const std::vector<uint64_t> &ress,
                       std::vector<uint32_t> &off, std::vector<uint32_t> &step);

// The same thing with the extended Euclid hoisted out of the lane loop, where it does
// not belong: g, s and inv3((M/g) % s, s) depend only on M and the filter. The naive
// form above stays as the reference --check-mark compares against.
// prep is built once per (M, filter set, lo); the per-lane call then takes only
// delta = first - lo, which is what makes the remainder 32-bit.
void ref3_prep_offsets(uint64_t M, const std::vector<uint64_t> &mods,
                       const std::vector<uint64_t> &ress, uint64_t lo,
                       std::vector<uint32_t> &prep);
void ref3_lane_offsets_fast(const std::vector<uint32_t> &prep, uint64_t delta,
                            uint32_t *off, uint32_t *step, size_t nall);
