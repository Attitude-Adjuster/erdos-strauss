// tables.cuh -- device-resident prime and per-rung lookup tables (Task 2).
//
// The per-rung tables turn the certificate path's Jacobi/Legendre computation into
// pure lookups. This is the "known, deliberately unexploited optimization" RUNBOOK.md
// calls out: `rung_scan.cpp` (the CPU reference) computes these symbols on the fly on
// purpose, so it stays the untouched thing everything else is benchmarked and diffed
// against. The CUDA port is where the lookup-table trick actually lives.
#pragma once
#include <cstdint>

// Tables are populated only for odd r == 3 (mod 4) with r <= TABLE_RMAX; all other
// rows are zero-filled and unused. TABLE_STRIDE is the row width (>= TABLE_RMAX+1,
// so every valid x < r fits) shared by both flat layouts below.
enum { TABLE_RMAX = 63, TABLE_STRIDE = 64 };

// facp/invastep mirror the reference's g_facp/g_invastep (rung_scan.cpp): primes
// <= sqrt(Amax) for factoring A, and g_astep^{-1} mod each such prime. jac/leg are
// the new per-rung symbol tables, flat-indexed:
//   jac[r*TABLE_STRIDE + x]            = Jacobi symbol (x/r),                x < r
//   leg[(r*4 + fi)*TABLE_STRIDE + x]    = Legendre symbol of (x mod p_fi), where p_fi
//                                         is the fi-th distinct prime factor of r
//                                         (fi < omega(r) <= 4, per g_rfac/g_rnf)
//   rnf[r]                              = omega(r), the count of distinct prime
//                                         factors -- how many leg rows row r has.
//                                         The classifier enumerates 2^omega(r) real
//                                         characters, so it needs this device-side.
// All five pointers are device pointers, valid only after upload_tables() returns.
// base/invmod are the PRIMALITY sieve's tables and are a different, longer list than
// facp/invastep: base is every prime <= sqrt(hi)+2 (the reference's g_base), whereas
// facp stops at sqrt(Amax) ~ sqrt(hi/4). invmod[i] is g_mod^{-1} mod base[i], and is 0
// for the few primes dividing g_mod -- those never divide p and the sieve skips them.
struct DevTables {
  const uint32_t *facp;
  const uint64_t *invastep;
  uint32_t        nfacp;
  const int8_t   *jac;
  const int8_t   *leg;
  const uint8_t  *rnf;
  const uint32_t *base;
  const uint64_t *invmod;
  uint32_t        nbase;
};

// Rebuilds the reference's tables for the given range/population mode (via
// ref_build_tables), stages the r<=TABLE_RMAX Jacobi/Legendre tables and the
// factor-prime tables on the host, uploads everything to freshly cudaMalloc'd
// device memory, and returns pointers/count. Task 2 scope: called once per
// process; repeated calls replace (leak) the previous allocation.
DevTables upload_tables(uint64_t hi, bool hard840);

// Host-side self-check for `--check-tables`: builds and uploads the tables, reads
// the device jac/leg tables back, and compares every populated (r, x) entry against
// a fresh, independent call to the reference's own jacobi()/legendre_sym() (via the
// ref_host accessors) -- not a self-comparison against the same build pass. Prints
// "TABLES OK" or "TABLES FAIL (n)" plus the pair count checked; returns n.
int check_tables();
