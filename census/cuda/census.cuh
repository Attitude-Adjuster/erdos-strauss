// census.cuh -- the GPU census: wavefront over rungs, CPU handoff, statistics (Task 5).
//
// This is the first task whose output is a real census rather than a self-check, so
// from here on tests/diff_range.sh is testing GPU results.
#pragma once
#include <cstdint>
#include <vector>

// Mirrors the reference's Opts (rung_scan.cpp:400) for the fields the GPU path honours.
// --occupancy is deliberately absent: it needs support_size() per position and emits
// OCC lines, so a run that asks for it is delegated wholesale to the CPU reference
// rather than half-supported here.
struct CensusOpts {
  uint64_t rmax = 127;
  bool     emitRes = false, emitSup = true;
  uint64_t deepThreshold = 0, sampleEvery = 0;
  // 24, where the reference defaults to 20 and RUNBOOK.md recommends 22 for the CPU.
  // The optima differ because the constraints do: the CPU's is cache residency, the
  // GPU's is amortizing per-(class,rung) fixed cost -- kernel launches and the
  // latency-bound readback -- over more positions. Measured at 10^13, --hard840:
  //   spanlog   20    22    24    25    26
  //   ns/prime  38    24    17    17    18
  // Buffers scale with 2^spanlog: ~1 GB of device memory at 24. Lower it on a small
  // GPU; output is invariant either way (bit-identical statistics across --spanlog is
  // itself one of the project's free checks).
  int      spanlog = 24;
  uint64_t progress = 0;
  bool     hard840 = false;
  // Active-count below which the rung ladder hands the survivors to the CPU
  // reference. A PURE PERFORMANCE KNOB: both paths derive the same Level from the
  // same factorization, so moving it cannot change output -- which is exactly what
  // --check-threshold asserts by running the same range at two different values.
  //
  // 16 is measured, not guessed. Swept on an L4 at 10^13, --hard840 --spanlog 22:
  //   gpuMin  1024   256    64    16     4     1
  //   ns/prime 186   122   109   103   105   106
  // The cost of a large threshold is the CPU tail, where each prime pays a full
  // factorA (trial division to 65536, then Pollard) at ~40 us per rung test; at 1024
  // that was 48% of all runtime for 0.1% of the primes. Below ~16 the per-rung kernel
  // launch overhead on a nearly empty active list takes it back. Re-sweep on other
  // hardware -- this optimum is not portable.
  uint32_t gpuMin = 16;
  // Phase timing. Costs extra cudaDeviceSynchronize calls, so it is off by default and
  // the timed run is NOT the run to quote ns/prime from.
  bool     profile = false;
};

// Accumulated census statistics. Field-for-field the reference's Stats
// (rung_scan.cpp:375), including its merge() tie-break, because maxr_p depends on it.
struct CensusStats {
  uint64_t nprimes = 0, jac = 0, rc = 0, sup = 0, res = 0, escalated = 0;
  uint64_t maxr = 0, maxr_p = 0;
  uint64_t hist[256] = {0};
  std::vector<uint64_t> rc_list, sup_list, res_list, deep_list, sample_list;  // (p,A,r)
  void merge(const CensusStats &o);
  void clear_lists();
};

// Runs the whole census over [lo, hi) on the GPU and prints the same lines the
// reference does -- certificates streamed per window, then SUMMARY. Returns 0 on
// success. Aborts loudly if the classifier ever reports gcd(q, r) != 1.
int census_run(uint64_t lo, uint64_t hi, const CensusOpts &opt);

// --verify for the GPU binary: runs the GPU census and the reference over [0, 100000)
// in BOTH population modes, in process, and compares every statistic and every
// emitted certificate. Prints "SELF-TEST PASSED"/"FAILED"; returns nonzero on failure.
int census_verify();

// --check-threshold LO HI: same range at two different gpuMin values, asserting
// identical output. Proves the CPU handoff point is a performance knob and not a
// correctness one -- the claim Task 5 rests on.
int census_check_threshold(uint64_t lo, uint64_t hi, bool hard840, int spanlog);
