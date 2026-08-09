// Task 1: skeleton only -- no kernels. Delegates 100% of the work to the CPU
// reference so the diff harness can be proven before any GPU code exists.
// Task 2: adds device tables (cuda/tables.cu). The census path is still 100%
// delegated to the CPU reference; --check-tables is the new table self-check.
// Task 3: adds --gate (cuda/gate.cu), the factorization-pipeline GO/NO-GO gate.
// Task 4: adds the device classifier (cuda/classify.cu) with --check-classify and
// --check-s.
// Task 5: the census path now runs on the GPU (cuda/census.cu). --verify compares the
// GPU census against the reference in process. Only --occupancy still delegates.
#include "census.cuh"
#include "classify.cuh"
#include "factorize.cuh"
#include "sieve.cuh"
#include "ref_host.h"
#include "tables.cuh"
#include <cstdio>
#include <cstdlib>
#include <cstring>

int gate(uint64_t lo, uint64_t hi, bool hard840, int spanlog, int reps);

int main(int argc, char **argv) {
  if (argc >= 2 && !strcmp(argv[1], "--check-tables")) return check_tables();

  // --check-sieve LO HI [--hard840] [--spanlog L]
  if (argc >= 4 && !strcmp(argv[1], "--check-sieve")) {
    const uint64_t lo = strtoull(argv[2], nullptr, 10);
    const uint64_t hi = strtoull(argv[3], nullptr, 10);
    bool hard840 = false; int spanlog = 18;
    for (int i = 4; i < argc; ++i) {
      if (!strcmp(argv[i], "--hard840")) hard840 = true;
      else if (!strcmp(argv[i], "--spanlog") && i + 1 < argc) spanlog = atoi(argv[++i]);
      else { fprintf(stderr, "--check-sieve: unknown argument %s\n", argv[i]); return 2; }
    }
    return check_sieve(lo, hi, hard840, spanlog);
  }

  // --check-factor LO HI [--hard840] [--spanlog L]
  if (argc >= 4 && !strcmp(argv[1], "--check-factor")) {
    const uint64_t lo = strtoull(argv[2], nullptr, 10);
    const uint64_t hi = strtoull(argv[3], nullptr, 10);
    bool hard840 = false; int spanlog = 16;
    for (int i = 4; i < argc; ++i) {
      if (!strcmp(argv[i], "--hard840")) hard840 = true;
      else if (!strcmp(argv[i], "--spanlog") && i + 1 < argc) spanlog = atoi(argv[++i]);
      else { fprintf(stderr, "--check-factor: unknown argument %s\n", argv[i]); return 2; }
    }
    return check_factor(lo, hi, hard840, spanlog);
  }

  // --check-s FILE: replay known level-S certificates through the classifier.
  if (argc >= 3 && !strcmp(argv[1], "--check-s")) return check_s(argv[2]);

  // --check-classify LO HI [--hard840] [--spanlog L]
  if (argc >= 4 && !strcmp(argv[1], "--check-classify")) {
    const uint64_t lo = strtoull(argv[2], nullptr, 10);
    const uint64_t hi = strtoull(argv[3], nullptr, 10);
    bool hard840 = false;
    // 18, not the census-optimal 22: this harness calls the reference's test_rung
    // once per position per rung on the host, so the window is sized by the CPU side.
    int spanlog = 18;
    for (int i = 4; i < argc; ++i) {
      if (!strcmp(argv[i], "--hard840")) hard840 = true;
      else if (!strcmp(argv[i], "--spanlog") && i + 1 < argc) spanlog = atoi(argv[++i]);
      else { fprintf(stderr, "--check-classify: unknown argument %s\n", argv[i]); return 2; }
    }
    if (spanlog < 8) spanlog = 8;
    if (spanlog > 27) spanlog = 27;
    return check_classify(lo, hi, hard840, spanlog);
  }

  // --gate LO HI [--hard840] [--spanlog L] [--reps N]
  if (argc >= 4 && !strcmp(argv[1], "--gate")) {
    const uint64_t lo = strtoull(argv[2], nullptr, 10);
    const uint64_t hi = strtoull(argv[3], nullptr, 10);
    bool hard840 = false;
    int spanlog = 22, reps = 5;          // 22 is the measured-optimal window (RUNBOOK.md)
    for (int i = 4; i < argc; ++i) {
      if (!strcmp(argv[i], "--hard840")) hard840 = true;
      else if (!strcmp(argv[i], "--spanlog") && i + 1 < argc) spanlog = atoi(argv[++i]);
      else if (!strcmp(argv[i], "--reps") && i + 1 < argc) reps = atoi(argv[++i]);
      else { fprintf(stderr, "--gate: unknown argument %s\n", argv[i]); return 2; }
    }
    if (spanlog < 8) spanlog = 8;
    if (spanlog > 27) spanlog = 27;
    if (reps < 1) reps = 1;
    return gate(lo, hi, hard840, spanlog, reps);
  }

  if (argc >= 2 && !strcmp(argv[1], "--verify")) return census_verify();

  // --check-threshold LO HI [--hard840] [--spanlog L]
  if (argc >= 4 && !strcmp(argv[1], "--check-threshold")) {
    const uint64_t lo = strtoull(argv[2], nullptr, 10);
    const uint64_t hi = strtoull(argv[3], nullptr, 10);
    bool hard840 = false;
    int spanlog = 14;
    for (int i = 4; i < argc; ++i) {
      if (!strcmp(argv[i], "--hard840")) hard840 = true;
      else if (!strcmp(argv[i], "--spanlog") && i + 1 < argc) spanlog = atoi(argv[++i]);
      else { fprintf(stderr, "--check-threshold: unknown argument %s\n", argv[i]); return 2; }
    }
    return census_check_threshold(lo, hi, hard840, spanlog);
  }

  if (argc < 3) return rung_scan_main(argc, argv);       // usage text, verbatim

  // ---- the census. Argument parsing mirrors rung_scan.cpp:700-724 exactly; anything
  // it would accept and this does not must fall through to the reference rather than
  // be silently ignored, or the two binaries would be running different jobs.
  uint64_t lo = strtoull(argv[1], nullptr, 10), hi = strtoull(argv[2], nullptr, 10);
  CensusOpts opt;
  uint64_t shard = 0, nshard = 1;
  for (int i = 3; i < argc; ++i) {
    if (!strcmp(argv[i], "--shard") && i + 1 < argc) {
      unsigned long long a = 0, b = 1;
      if (sscanf(argv[++i], "%llu/%llu", &a, &b) == 2) { shard = a; nshard = b; }
    }
    else if (!strcmp(argv[i], "--rmax") && i + 1 < argc) opt.rmax = strtoull(argv[++i], nullptr, 10);
    else if (!strcmp(argv[i], "--emit-residual")) opt.emitRes = true;
    else if (!strcmp(argv[i], "--no-emit-support")) opt.emitSup = false;
    else if (!strcmp(argv[i], "--emit-deep") && i + 1 < argc) opt.deepThreshold = strtoull(argv[++i], nullptr, 10);
    else if (!strcmp(argv[i], "--sample") && i + 1 < argc) opt.sampleEvery = strtoull(argv[++i], nullptr, 10);
    else if (!strcmp(argv[i], "--spanlog") && i + 1 < argc) opt.spanlog = atoi(argv[++i]);
    else if (!strcmp(argv[i], "--hard840")) opt.hard840 = true;
    else if (!strcmp(argv[i], "--progress") && i + 1 < argc) opt.progress = strtoull(argv[++i], nullptr, 10);
    else if (!strcmp(argv[i], "--gpu-min") && i + 1 < argc) opt.gpuMin = (uint32_t)strtoul(argv[++i], nullptr, 10);
    else if (!strcmp(argv[i], "--profile")) opt.profile = true;
    else if (!strcmp(argv[i], "--direct-min") && i + 1 < argc)
      factorize_set_direct_min((uint32_t)strtoul(argv[++i], nullptr, 10));
    else if (!strcmp(argv[i], "--occupancy")) {
      // Needs support_size() per position and emits OCC lines. Rather than support it
      // halfway, hand the whole run to the reference -- correct, just not accelerated.
      fprintf(stderr, "[rung_scan_cuda] --occupancy is CPU-only; delegating to the reference\n");
      return rung_scan_main(argc, argv);
    }
  }
  if (opt.rmax > 255) opt.rmax = 255;
  if (opt.spanlog < 12) opt.spanlog = 12;
  if (opt.spanlog > 27) opt.spanlog = 27;
  if (nshard == 0) nshard = 1;
  if (nshard > 1) {
    const uint64_t span = (hi - lo + nshard - 1) / nshard;
    const uint64_t a = lo + shard * span;
    lo = a; hi = (hi < a + span) ? hi : a + span;
  }
  if (lo >= hi) {   // an empty shard still reports, so merge/resume see full coverage
    printf("SUMMARY lo=%llu hi=%llu primes=0 jac=0 rc=0 sup=0 res=0 escalated=0 "
           "pairpath=0 maxr=0 maxr_p=0 hist= sieve_s=0.00 scan_s=0.00 ns_per_prime=0\n",
           (unsigned long long)lo, (unsigned long long)hi);
    return 0;
  }
  return census_run(lo, hi, opt);
}
