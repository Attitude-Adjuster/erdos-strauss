// sieve.cuh -- segmented primality sieve on the device (Task 6).
//
// WHY THIS IS NOW THE HOT SPOT. The plan scheduled this task last, on the grounds that
// the sieve is "only 6% of runtime". That 6% was measured on the CPU reference. Once
// factorization and classification moved to the GPU, everything ELSE got ~10x faster
// and the sieve inverted to 69.5% of wall clock (measured, Task 5 --profile at 10^13).
// Amdahl, exactly as advertised. It also gets worse with height: the base prime list
// is pi(sqrt(X)) -- 3.2e6 at 10^13, 3.2e7 at 10^15 -- so this is SCALING.md's Wall 1
// arriving on the host side of the port.
#pragma once
#include <cstdint>
#include <cuda_runtime.h>
#include "tables.cuh"

// Sieves one residue class of one window and returns its prime positions.
//
// Mirrors rung_scan.cpp:487-507: for every base prime l not dividing mod, clear the
// positions j = (l - first) * mod^{-1} (mod l), starting no earlier than l^2, then
// clear every position whose p is below 25. Composite marking writes a constant 0, so
// it is idempotent -- concurrent marks of the same position cannot disagree and the
// kernel needs no atomics, which is also what keeps it deterministic.
//
//   first, npos    the progression p = first + mod*j, j < npos
//   whi            window end; primes with l*l >= whi cannot mark anything
//   d_primebits    OUT, bitmap over positions, bit j set iff p is prime (nwords words)
//   d_actpos       OUT, ASCENDING list of prime positions, capacity >= npos
//   returns        the number of prime positions written to d_actpos
//
// Both outputs are what factorize_rung already consumes, so this is a drop-in
// replacement for the host ref_sieve_class + upload it retires.
// `slot` selects an independent workspace (two exist) -- see factorize.cuh on slots.
uint32_t sieve_class(const DevTables &T, uint64_t first, uint32_t npos, uint64_t mod,
                     uint64_t whi, uint64_t *d_primebits, uint32_t *d_actpos,
                     cudaStream_t s, int slot = 0);

// Frees the persistent workspace (the composite byte map and CUB scratch, both grown
// on demand and reused, so a steady-state window allocates nothing).
void sieve_release();

// --check-sieve LO HI: for one window, compare the device sieve against the reference
// two ways -- position list identical to ref_sieve_class, and every position's
// primality agreeing with the reference's own is_prime(). The second is the one that
// matters: is_prime is Miller-Rabin, a completely different algorithm from the sieve,
// so agreement is evidence rather than a tautology. Returns the mismatch count.
int check_sieve(uint64_t lo, uint64_t hi, bool hard840, int spanlog);
