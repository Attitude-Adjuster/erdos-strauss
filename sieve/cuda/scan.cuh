#pragma once
#include <cstdint>

// The window/lane driver. Everything the CPU scanner's main decides is decided here in
// the same way and in the same order, so the two binaries can be diffed field for field.
struct ScanOpts {
  uint64_t lo = 0, hi = 0;
  uint64_t M = 2042040, mmax = 10000, sieve_p = 0;
  uint64_t seg = 262144, progress = 0;
  uint64_t shard = 0, nshard = 1;
  uint32_t block = 512;                  // device threads per lane block
  int threads = 0;                       // host tail pool; 0 leaves OpenMP's default
  const char *filters = nullptr;
  bool emit_surv = true;
  bool profile = false;                  // device / host-tail / wait split
  bool quiet_time = false;               // wall seconds only, for sweeps
};

// Returns the CPU scanner's own exit convention: 2 if any survivor remains, else 0.
int scan_run(const ScanOpts &o);
