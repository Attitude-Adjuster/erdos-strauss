// sieve_cuda -- the verification sieve on an NVIDIA L4.
//
// Trust model, identical to the census port's: byte-identical SUMMARY and an identical
// SORTED certificate set against the CPU scanner (tests/diff_cuda.sh), plus the CPU
// scanner's own self-test running inside this binary via ref_host.
#include "ref_host.h"
#include "mark.cuh"
#include "scan.cuh"
#include "survivors.cuh"
#include <cinttypes>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>
#ifdef _OPENMP
#include <omp.h>
#endif

// ---------------------------------------------------------------- --check-mark
//
// Device marking must agree with the CPU scanner BIT FOR BIT on the same segment. That
// is the kernel's whole correctness claim and it is cheap to check exactly, so it is
// checked exactly rather than sampled statistically.
//
// Three things are compared, not one. The bitset is the obvious one. The CARRIED
// offsets (off[i] = j - n) matter just as much: a carry bug is invisible in segment 0
// and corrupts every segment after it, so it would survive a bits-only check and then
// surface as a certificate-set difference thousands of positions later. And the
// between-stages popcount is what separates `covered` from `sieved` in SUMMARY.
static int check_mark(uint64_t M, uint64_t mmax, uint64_t lo, uint64_t sieve_p,
                      uint32_t seg, uint32_t nthreads) {
  ref3_build_wheel(M, mmax);
  std::vector<uint64_t> mods, ress;
  size_t ncov = 0;
  ref3_filters(M, mmax, nullptr, sieve_p, mods, ress, ncov);

  const uint32_t nlanes = ref3_wheel_size();
  const uint32_t stride = nlanes > 64 ? nlanes / 64 : 1;
  // Both a full segment and a partial one: the tail mask (n & 63) and the carry out of
  // a short segment are separate code paths and only the partial run exercises them.
  const uint32_t ns[2] = {seg, seg > 12345 ? seg - 12345 : seg / 2};

  // The driver does not use ref3_lane_offsets -- it uses the hoisted form, which lifts
  // the extended Euclid out of the lane loop. That hoist is only sound if the two agree
  // exactly, so they are computed side by side here and compared.
  std::vector<uint32_t> prep;
  ref3_prep_offsets(M, mods, ress, lo, prep);

  uint64_t bad_bits = 0, bad_off = 0, bad_alive = 0, bad_prep = 0, lanes = 0;
  for (uint32_t li = 0; li < nlanes; li += stride) {
    const uint64_t rho = ref3_wheel(li);
    const uint64_t first = lo + ((rho + M - lo % M) % M);
    std::vector<uint32_t> off0, step;
    ref3_lane_offsets(M, first, mods, ress, off0, step);
    {
      std::vector<uint32_t> foff(mods.size()), fstep(mods.size());
      ref3_lane_offsets_fast(prep, first - lo, foff.data(), fstep.data(), mods.size());
      for (size_t i = 0; i < mods.size(); ++i)
        if (foff[i] != off0[i] || fstep[i] != step[i]) ++bad_prep;
    }
    for (uint32_t k = 0; k < 2; ++k) {
      std::vector<uint32_t> hoff, doff;
      uint64_t halive = 0, dalive = 0, hsurv = 0, dsurv = 0;
      const std::vector<uint32_t> h =
          mark_reference(off0, step, ncov, ns[k], hoff, halive, hsurv);
      const std::vector<uint32_t> d =
          mark_device(off0, step, ncov, ns[k], seg, doff, dalive, dsurv, nthreads);
      for (size_t w = 0; w < h.size(); ++w) if (h[w] != d[w]) ++bad_bits;
      for (size_t i = 0; i < hoff.size(); ++i) if (hoff[i] != doff[i]) ++bad_off;
      if (halive != dalive || hsurv != dsurv) ++bad_alive;
      ++lanes;
    }
  }
  const uint64_t bad = bad_bits + bad_off + bad_alive + bad_prep;
  printf("%s check-mark M=%" PRIu64 " filters=%zu (ncov=%zu) lanes=%" PRIu64
         " seg=%u -- mismatched words: %" PRIu64 ", offsets: %" PRIu64
         ", counts: %" PRIu64 ", hoisted-setup: %" PRIu64 "\n",
         bad ? "FAIL" : "ok  ", M, mods.size(), ncov, lanes, seg,
         bad_bits, bad_off, bad_alive, bad_prep);
  return bad ? 1 : 0;
}

// ---------------------------------------------------------------- --check-primality
//
// Every odd integer in a dense range classified by BOTH paths. Primality is where an
// emulated 64-bit modmul could silently differ from the host's -- the L4 has no
// 64x64->128 multiply-high instruction -- so this is checked exhaustively over a range
// rather than sampled.
static int check_primality(uint64_t lo, uint32_t N) {
  std::vector<uint64_t> cand(N);
  for (uint32_t i = 0; i < N; ++i) cand[i] = lo + 2 * (uint64_t)i + 1;
  const std::vector<uint8_t> dev = primality_device(cand);
  uint64_t mism = 0, primes = 0;
  for (uint32_t i = 0; i < N; ++i) {
    const uint8_t want = ref3_is_prime(cand[i]) ? 1 : 0;
    primes += want;
    if (dev[i] != want) ++mism;
  }
  printf("%s check-primality mismatches: %" PRIu64 " over %u (%" PRIu64 " primes)\n",
         mism ? "FAIL" : "ok  ", mism, N, primes);
  return mism ? 1 : 0;
}

// ---------------------------------------------------------------- --check-compact
//
// The compacted survivor list, compared ELEMENT FOR ELEMENT against the host's own
// bit-extraction. Order is the property under test, not membership: an atomicAdd would
// produce the right set in a schedule-dependent order and pass a set comparison, and the
// certificate stream would stop being reproducible.
static int check_compact(uint64_t M, uint64_t mmax, uint64_t lo, uint64_t sieve_p,
                         uint32_t seg, uint32_t nthreads) {
  ref3_build_wheel(M, mmax);
  std::vector<uint64_t> mods, ress;
  size_t ncov = 0;
  ref3_filters(M, mmax, nullptr, sieve_p, mods, ress, ncov);

  const uint32_t nlanes = ref3_wheel_size();
  const uint32_t stride = nlanes > 32 ? nlanes / 32 : 1;
  const uint32_t ns[2] = {seg, seg > 12345 ? seg - 12345 : seg / 2};
  const uint64_t segbase = 7;                 // a non-zero base: p must fold it in

  uint64_t bad_n = 0, bad_p = 0, bad_prime = 0, lanes = 0, total = 0;
  for (uint32_t li = 0; li < nlanes; li += stride) {
    const uint64_t rho = ref3_wheel(li);
    const uint64_t first = lo + ((rho + M - lo % M) % M);
    std::vector<uint32_t> off0, step;
    ref3_lane_offsets(M, first, mods, ress, off0, step);
    for (uint32_t k = 0; k < 2; ++k) {
      std::vector<uint32_t> hoff;
      uint64_t halive = 0, hsurv = 0;
      const std::vector<uint32_t> bits =
          mark_reference(off0, step, ncov, ns[k], hoff, halive, hsurv);
      std::vector<uint64_t> want;
      for (size_t w = 0; w < bits.size(); ++w) {
        uint32_t x = bits[w];
        while (x) {
          const uint32_t j = (uint32_t)(w * 32 + __builtin_ctz(x));
          want.push_back(first + M * (segbase + j));
          x &= x - 1;
        }
      }
      std::vector<uint8_t> prime;
      const std::vector<uint64_t> got =
          survivors_device(off0, step, ncov, ns[k], seg, first, M, segbase, nthreads,
                           prime);
      if (got.size() != want.size()) { ++bad_n; ++lanes; continue; }
      for (size_t i = 0; i < want.size(); ++i) {
        if (got[i] != want[i]) ++bad_p;
        else if (prime[i] != (ref3_is_prime(got[i]) ? 1 : 0)) ++bad_prime;
      }
      total += want.size();
      ++lanes;
    }
  }
  const uint64_t bad = bad_n + bad_p + bad_prime;
  printf("%s check-compact M=%" PRIu64 " lanes=%" PRIu64 " candidates=%" PRIu64
         " -- wrong counts: %" PRIu64 ", wrong/out-of-order p: %" PRIu64
         ", wrong primality: %" PRIu64 "\n",
         bad ? "FAIL" : "ok  ", M, lanes, total, bad_n, bad_p, bad_prime);
  return bad ? 1 : 0;
}

// ---------------------------------------------------------------- --bench-mark
static int bench_mark(uint64_t M, uint64_t mmax, uint64_t lo, uint64_t sieve_p,
                      uint32_t seg, uint32_t nthreads, int reps) {
  uint64_t positions = 0;
  const double s = mark_bench(M, mmax, lo, sieve_p, seg, nthreads, reps, positions);
  printf("bench-mark M=%" PRIu64 " seg=%u threads=%u positions=%" PRIu64
         " best=%.4f s  %.2f Gpos/s\n",
         M, seg, nthreads, positions, s, positions / s / 1e9);
  return 0;
}

// ---------------------------------------------------------------- main
// The scan itself. Option names, defaults and shard arithmetic match rung_scan3's main
// so the two can be driven by the same command line and diffed field for field.
static int run_scan(int argc, char **argv) {
  ScanOpts o;
  o.lo = strtoull(argv[1], 0, 10);
  o.hi = strtoull(argv[2], 0, 10);
  for (int i = 3; i < argc; ++i) {
    if (!strcmp(argv[i], "--mmax") && i + 1 < argc) o.mmax = strtoull(argv[++i], 0, 10);
    else if (!strcmp(argv[i], "--wheel") && i + 1 < argc) o.M = strtoull(argv[++i], 0, 10);
    else if (!strcmp(argv[i], "--seg") && i + 1 < argc) o.seg = strtoull(argv[++i], 0, 10);
    else if (!strcmp(argv[i], "--block") && i + 1 < argc) o.block = (uint32_t)strtoul(argv[++i], 0, 10);
    else if (!strcmp(argv[i], "--progress") && i + 1 < argc) o.progress = strtoull(argv[++i], 0, 10);
    else if (!strcmp(argv[i], "--threads") && i + 1 < argc) o.threads = atoi(argv[++i]);
    else if (!strcmp(argv[i], "--filters") && i + 1 < argc) o.filters = argv[++i];
    else if (!strcmp(argv[i], "--sieve") && i + 1 < argc) o.sieve_p = strtoull(argv[++i], 0, 10);
    else if (!strcmp(argv[i], "--shard") && i + 1 < argc) {
      unsigned long long a = 0, b = 1;
      if (sscanf(argv[++i], "%llu/%llu", &a, &b) == 2) { o.shard = a; o.nshard = b; }
    }
    else if (!strcmp(argv[i], "--no-emit-survivors")) o.emit_surv = false;
    else if (!strcmp(argv[i], "--profile")) o.profile = true;
    else if (!strcmp(argv[i], "--quiet-time")) o.quiet_time = true;
    else {
      // Refuse unknown options rather than ignoring them, exactly as rung_scan3 does: a
      // typo'd --mmax would quietly scan at the default and look entirely plausible.
      fprintf(stderr, "FATAL: unknown option '%s'\n", argv[i]);
      return 1;
    }
  }
  if (!o.seg) o.seg = 262144;
#ifdef _OPENMP
  if (o.threads > 0) omp_set_num_threads(o.threads);
#endif
  return scan_run(o);
}

static void usage(const char *me) {
  fprintf(stderr,
    "usage: %s LO HI [--mmax N] [--wheel M] [--seg N] [--block N] [--threads N]\n"
    "                 [--shard i/n] [--progress N] [--filters FILE] [--sieve P]\n"
    "                 [--no-emit-survivors] [--profile] [--quiet-time]\n"
    "       %s --verify\n"
    "       %s --check-mark    [--wheel M] [--mmax N] [--lo P] [--seg N] [--sieve P]\n"
    "                          [--block N]\n"
    "       %s --check-compact [same options]\n"
    "       %s --check-primality LO N\n"
    "       (stage E lives in rung_scan3: ./rung_scan3 --check-direct LO N)\n"
    "       %s --bench-mark    [same options] [--reps N]\n",
    me, me, me, me, me, me);
}

int main(int argc, char **argv) {
  if (argc >= 2 && !strcmp(argv[1], "--verify")) return ref3_verify();
  if (argc < 2) { usage(argv[0]); return 1; }

  if (!strcmp(argv[1], "--check-primality")) {
    if (argc < 4) { usage(argv[0]); return 1; }
    return check_primality(strtoull(argv[2], 0, 10), (uint32_t)strtoul(argv[3], 0, 10));
  }

  const bool do_check = !strcmp(argv[1], "--check-mark");
  const bool do_compact = !strcmp(argv[1], "--check-compact");
  const bool do_bench = !strcmp(argv[1], "--bench-mark");
  if (!do_check && !do_compact && !do_bench) {
    if (argc < 3 || argv[1][0] == '-') { usage(argv[0]); return 1; }
    return run_scan(argc, argv);
  }

  // Measured on the L4 (cloud-gpu, 2026-08-17), M = 2042040, mmax = 10000, one segment
  // per lane, best of 5, in Gpos/s -- against each other, in one session, which is the
  // only comparison this box supports:
  //
  //          seg   32768  65536  131072  262144  393216  524288      (block 256)
  //               0.94   1.58    2.30    2.26    --      0.94
  //        block    128    256     384     512     768    1024      (seg 262144)
  //               1.54   2.26    2.45    2.51    2.17    1.40
  //
  // seg 262144 lands on the CPU's optimum too, but for an unrelated reason: the CPU's is
  // L1 residency, the device's is amortizing the per-segment fixed cost against shared
  // memory occupancy. 524288 needs 64 KB of shared memory and collapses to one block per
  // SM, which is the cliff on the right. block 512 over 256 is worth 11%.
  uint64_t M = 2042040, mmax = 10000, lo = 1000000000000ull, sieve_p = 0;
  uint32_t seg = 262144, nthreads = 512;
  int reps = 3;
  for (int i = 2; i < argc; ++i) {
    // Unknown options are a hard error, exactly as in rung_scan3: a silently dropped
    // flag would scan at the default and produce entirely plausible-looking output.
    if (!strcmp(argv[i], "--wheel") && i + 1 < argc) M = strtoull(argv[++i], 0, 10);
    else if (!strcmp(argv[i], "--mmax") && i + 1 < argc) mmax = strtoull(argv[++i], 0, 10);
    else if (!strcmp(argv[i], "--lo") && i + 1 < argc) lo = strtoull(argv[++i], 0, 10);
    else if (!strcmp(argv[i], "--seg") && i + 1 < argc) seg = (uint32_t)strtoul(argv[++i], 0, 10);
    else if (!strcmp(argv[i], "--sieve") && i + 1 < argc) sieve_p = strtoull(argv[++i], 0, 10);
    else if (!strcmp(argv[i], "--block") && i + 1 < argc) nthreads = (uint32_t)strtoul(argv[++i], 0, 10);
    else if (!strcmp(argv[i], "--reps") && i + 1 < argc) reps = atoi(argv[++i]);
    else { fprintf(stderr, "FATAL: unknown option '%s'\n", argv[i]); return 1; }
  }
  if (do_check) return check_mark(M, mmax, lo, sieve_p, seg, nthreads);
  if (do_compact) return check_compact(M, mmax, lo, sieve_p, seg, nthreads);
  return bench_mark(M, mmax, lo, sieve_p, seg, nthreads, reps);
}
