// Zero-modification reuse of the CPU scanner's `static` internals (gen_wheel_covers,
// gen_ucerts, build_wheel, solve_rung3, solve_direct3, factorA, is_prime, ...).
// sieve/rung_scan3.cpp is never edited, copied, or forked.
//
// NOTE THE DIFFERENCE FROM census/cuda/ref_host.cpp, which renames `main` by macro:
// that CANNOT work here. rung_scan3.cpp itself does `#define main rung_scan_main` /
// `#include "rung_scan.cpp"` / `#undef main`, and that inner #undef cancels any outer
// rename before its own main is reached -- so `main` would survive and collide at link
// time. RUNG_SCAN3_NO_MAIN exists for exactly this; it suppresses the scanner's main
// instead. (Discovered the same way twice: tests/bench_staged.cpp hit it first.)
#define RUNG_SCAN3_NO_MAIN
#include "../rung_scan3.cpp"

#include "ref_host.h"

void ref3_build_tables(uint64_t hi) {
  init_phi();
  init_rfac();
  build_base_primes(hi);
}

int  ref3_verify()                                    { return scan3_self_test(); }
bool ref3_solve_rung(uint64_t p, std::string &out)    { return solve_rung3(p, out); }
bool ref3_solve_direct(uint64_t p, std::string &out)  { return solve_direct3(p, out); }
bool ref3_is_prime(uint64_t n)                        { return is_prime(n); }
uint64_t ref3_direct_pmax()                           { return DIRECT_PMAX; }

// ---------------------------------------------------------------- wheel cache
//
// Built once per (M, mmax) and reused. The device driver asks for lanes thousands of
// times; re-deriving the wheel each time would dominate startup at M = 2,042,040.

static u64 g_cached_M = 0, g_cached_mmax = 0;
static WheelResult g_cached_wheel;
static u64 g_cached_cover_pmax = 0;

void ref3_build_wheel(uint64_t M, uint64_t mmax) {
  if (g_cached_M == M && g_cached_mmax == mmax) return;
  const vector<Cover> covers = gen_covers(mmax);
  const vector<Cover> wcov = gen_wheel_covers(M);
  const vector<Ucert> uc = gen_ucerts(UCERT_RMAX, UCERT_UMAX);
  g_cached_wheel = build_wheel(M, wcov, uc);
  g_cached_cover_pmax = 0;
  for (const Cover &f : covers) g_cached_cover_pmax = max(g_cached_cover_pmax, f.pmin);
  g_cached_M = M;
  g_cached_mmax = mmax;
}

uint32_t ref3_wheel_size()          { return (uint32_t)g_cached_wheel.classes.size(); }
uint64_t ref3_wheel(uint32_t i)     { return g_cached_wheel.classes[i]; }
uint64_t ref3_wheel_pmin()          { return g_cached_wheel.max_pmin; }
uint64_t ref3_cover_pmax(uint64_t)  { return g_cached_cover_pmax; }

uint64_t ref3_class_fnv64_of(uint64_t M) {
  ref3_build_wheel(M, g_cached_mmax ? g_cached_mmax : 10000);
  return fnv64_classes(g_cached_wheel.classes);
}

bool ref3_class_pinned(uint64_t M, uint64_t *fnv64, const char **sha256) {
  const ClassPin *p = class_pin(M);
  if (!p) return false;
  if (fnv64) *fnv64 = p->fnv64;
  if (sha256) *sha256 = p->sha256;
  return true;
}

// ---------------------------------------------------------------- filter set
//
// Mirrors rung_scan3's main exactly: covers with m | M are excluded because those
// classes are already gone from the wheel, and small primes are appended AFTER, with
// ncov marking the boundary. Getting either wrong would change `covered` or `sieved`
// and break the SUMMARY contract with the CPU scanner.

void ref3_filters(uint64_t M, uint64_t mmax, const char *file, uint64_t sieve_p,
                  std::vector<uint64_t> &mods, std::vector<uint64_t> &ress,
                  size_t &ncov) {
  mods.clear();
  ress.clear();
  if (file) {
    FILE *fp = fopen(file, "r");
    if (!fp) { fprintf(stderr, "FATAL: cannot open %s\n", file); abort(); }
    char line[256];
    while (fgets(line, sizeof line, fp)) {
      unsigned long long mod, res;
      if (sscanf(line, "FILTER %llu %llu", &mod, &res) == 2) {
        mods.push_back((u64)mod); ress.push_back((u64)res);
      }
    }
    fclose(fp);
    if (mods.empty()) { fprintf(stderr, "FATAL: no FILTER lines in %s\n", file); abort(); }
  } else {
    for (const Cover &f : gen_covers(mmax))
      if (M % f.mod) { mods.push_back(f.mod); ress.push_back(f.res); }
  }
  ncov = mods.size();
  if (sieve_p)
    for (u64 q : small_primes(sieve_p, M)) { mods.push_back(q); ress.push_back(0); }
}

// ---------------------------------------------------------------- lane offsets
//
// sweep_lane's setup loop, verbatim. `rhs % g` is the ONLY solvability condition: per
// prime the minimum exponent goes wholly into g, so M/g and mod/g are automatically
// coprime. step[i] == 0 means the filter misses this lane entirely.
void ref3_lane_offsets(uint64_t M, uint64_t first,
                       const std::vector<uint64_t> &mods,
                       const std::vector<uint64_t> &ress,
                       std::vector<uint32_t> &off, std::vector<uint32_t> &step) {
  const size_t nall = mods.size();
  off.assign(nall, 0);
  step.assign(nall, 0);
  for (size_t i = 0; i < nall; ++i) {
    const u64 mod = mods[i];
    const u64 g = gcd3(M, mod);
    const u64 rhs = (ress[i] + mod - first % mod) % mod;
    if (rhs % g) { step[i] = 0; continue; }
    const u64 s = mod / g;
    step[i] = (uint32_t)s;
    off[i] = s == 1 ? 0 : (uint32_t)((u128)(rhs / g) * inv3((M / g) % s, s) % s);
  }
}

// ------------------------------------------------------- lane offsets, hoisted
//
// The loop above runs once per (lane, filter): 2,308 x 64,708 = 149 MILLION extended
// Euclids at production settings, which --profile showed to be 79% of the whole run --
// sixteen times the device work behind it.
//
// But g, s and inv3((M/g) % s, s) depend only on M and the FILTER. Nothing in them
// varies with the lane; only `rhs` does, through `first % mod`. So the extended Euclid
// is loop-invariant and lifts clean out: 64,708 of them instead of 149 million, leaving
// one 64-bit remainder and one 128-bit multiply per (lane, filter).
//
// This is a hoist, not a different formula -- every operation below appears above in the
// same order -- and --check-mark proves it by computing both ways and comparing.
// (The CPU scanner has the same invariant and does NOT exploit it; see the write-up.)
// The second hoist is `first % mod`. Every lane's `first` lies in [lo, lo + M), so
// first % mod = (lo % mod + (first - lo)) % mod, and BOTH operands of that are small:
// lo % mod < mod and first - lo < M. A 64-bit remainder of a 10^12-scale numerator
// becomes a 32-bit one, which is where the rest of the time was going.
//
// Everything here still fits: mod and M are checked below, rhs/g < mod and inv < s <=
// mod, so the product is under 2^64 and the u128 above was never load-bearing.
void ref3_prep_offsets(uint64_t M, const std::vector<uint64_t> &mods,
                       const std::vector<uint64_t> &ress, uint64_t lo,
                       std::vector<uint32_t> &prep) {
  const size_t nall = mods.size();
  prep.resize(nall * 6);
  if (M > 0x7fffffffull) { fprintf(stderr, "FATAL: wheel M too large for the 32-bit "
                                   "offset setup\n"); abort(); }
  for (size_t i = 0; i < nall; ++i) {
    const u64 mod = mods[i];
    if (mod > 0x7fffffffull) { fprintf(stderr, "FATAL: filter modulus %" PRIu64
                                       " too large\n", mod); abort(); }
    const u64 g = gcd3(M, mod), s = mod / g;
    prep[i * 6 + 0] = (uint32_t)mod;
    prep[i * 6 + 1] = (uint32_t)ress[i];
    prep[i * 6 + 2] = (uint32_t)g;
    prep[i * 6 + 3] = (uint32_t)s;
    // inv3 returns 0 for m == 1, which is exactly what the s == 1 special case in the
    // naive form produces, so that branch disappears rather than being reproduced.
    prep[i * 6 + 4] = (uint32_t)inv3((M / g) % s, s);
    prep[i * 6 + 5] = (uint32_t)(lo % mod);
  }
}

void ref3_lane_offsets_fast(const std::vector<uint32_t> &prep, uint64_t delta,
                            uint32_t *off, uint32_t *step, size_t nall) {
  const uint32_t d = (uint32_t)delta;               // first - lo, always < M
  for (size_t i = 0; i < nall; ++i) {
    const uint32_t mod = prep[i * 6 + 0], res = prep[i * 6 + 1];
    const uint32_t g = prep[i * 6 + 2], s = prep[i * 6 + 3];
    const uint32_t inv = prep[i * 6 + 4], lomod = prep[i * 6 + 5];
    const uint32_t fm = (lomod + d) % mod;
    const uint32_t rhs = (res + mod - fm) % mod;
    if (rhs % g) { step[i] = 0; off[i] = 0; continue; }
    step[i] = s;
    off[i] = (uint32_t)((uint64_t)(rhs / g) * inv % s);
  }
}
