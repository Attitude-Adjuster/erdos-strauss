// factorize.cuh -- device factorization of the rung-r denominator A over a whole
// window (Task 3, the GO/NO-GO gate).
//
// The CPU reference factors an entire window with one strided walk per factor prime
// (rung_scan.cpp:531-542): for l not dividing g_astep, l | A_j exactly when
// j = -A0 * astep^{-1} (mod l), so the hits of l form an arithmetic progression in j.
// The GPU version keeps that algorithm but replaces the reference's in-place
// scatter-update of a per-position Fac (which would need atomics, or worse, a race)
// with an allocate-then-fill pipeline:
//
//   stage 1  count       closed-form hit count per work item          (no atomics)
//   stage 2  scan        cub::DeviceScan::ExclusiveSum -> slice base  (no atomics)
//   stage 3  generate    walk + filter, write into own slice          (no atomics)
//            compact     scan of the ACTUAL counts + gather -> dense  (no atomics)
//   stage 4  sort        cub::DeviceRadixSort::SortPairs by position
//            segment     cub::DeviceRunLengthEncode + scan -> segments
//            reduce      one thread per position: divide out its factors
//
// Every write lands at an index that is a pure function of the inputs, so the output
// is bit-identical run to run. That determinism is the whole reason for the
// count->scan->slice shape; see RUNBOOK.md's verification discipline.
#pragma once
#include <cstdint>
#include <cuda_runtime.h>
#include "tables.cuh"

// A < 2.5e17 has at most 14 distinct prime factors, so 16 slots is safe to 10^18 --
// the same bound and the same reasoning as the reference's Fac (rung_scan.cpp:144).
// Overflow aborts loudly on the device rather than silently dropping a factor.
enum { FACR_MAX = 16 };

// The factorization of A reduced mod r. The classifier only ever needs the residues
// (Jacobi/Legendre arguments) and the exponents (tau(q^2) = prod(2e_i+1)), never the
// primes themselves, so storing (l mod r) instead of l keeps this at 33 bytes.
//
// Entries are NOT merged by residue: two distinct primes can share a residue mod r,
// and merging them would change tau(q^2). Compare two FacR as sorted MULTISETS of
// (res, ex) pairs.
struct FacR {
  uint8_t res[FACR_MAX];   // (l mod r)
  uint8_t ex [FACR_MAX];   // exponent of l in A
  uint8_t k;               // number of entries
};

// Fills d_out[i] with the factorization of A = A0 + astep * d_actpos[i], for each of
// the nact active positions, all mod r.
//
//   T           device tables; T.facp / T.invastep / T.nfacp are used
//   A0, astep   A = A0 + astep * j, matching rung_scan.cpp:520
//   npos        number of positions in the window (j < npos)
//   Amax        A0 + astep * (npos - 1); factor primes stop at l*l > Amax
//   r           the rung (only used to reduce residues)
//   d_actpos    ASCENDING list of active positions j, length nact
//   d_primebits bitmask over positions, bit j set iff position j is prime; may be null
//   d_out       output, length nact
//   s           stream; the call synchronizes internally twice (two device->host
//               counts are needed to size the sort), so it is not fully async
// `finish` controls the final cofactor append (k_finish). The census passes false and
// fuses that append into the classify-tally kernel (classify.cuh), reading the
// leftover cofactors via factorize_last_remA(); every harness leaves it true. Both
// paths append through the ONE inline in facr_append.cuh, so they cannot drift.
void factorize_rung(const DevTables &T, uint64_t A0, uint64_t astep, uint32_t npos,
                    uint64_t Amax, uint32_t r,
                    const uint32_t *d_actpos, uint32_t nact,
                    const uint64_t *d_primebits, FacR *d_out, cudaStream_t s,
                    bool finish = true, int slot = 0);

// Device pointer to the per-position cofactors of slot's LAST factorize_rung call
// (remA[i] for d_actpos[i]; 1 when the walk consumed A entirely). Valid until the
// slot's next call. Only meaningful after finish = false.
//
// `slot` selects an independent workspace (two exist). The double-buffered census
// drives one window per host thread, each on its own slot and stream; a slot is only
// ever touched by its one thread, which is what makes the whole file lock-free. All
// harnesses use the default slot 0.
const uint64_t *factorize_last_remA(int slot = 0);

// Factors each active position's A directly -- trial division then Miller-Rabin and
// Pollard-Brent, one thread per position -- with no sieve walk, sort or compaction.
// Correct at any nact; it is only FASTER than factorize_rung below the crossover.
void factorize_direct(const DevTables &T, uint64_t A0, uint64_t astep, uint32_t r,
                      const uint32_t *d_actpos, uint32_t nact, FacR *d_out,
                      cudaStream_t s, int slot = 0);

// The crossover: factorize_rung dispatches to factorize_direct when nact is below it.
// A pure performance knob -- both paths return the same factorization, which
// --check-factor asserts against each other AND against the reference's factorA.
void     factorize_set_direct_min(uint32_t n);
uint32_t factorize_get_direct_min();

// Frees the persistent workspace (buffers are grown on demand and reused across
// calls, so a steady-state call performs no allocation).
void factorize_release();

// --check-factor LO HI: assert the sieve pipeline, the direct path, and the
// reference's factorA all produce the same factorization. Returns the mismatch count.
int check_factor(uint64_t lo, uint64_t hi, bool hard840, int spanlog);

// Last call's stage sizes, for reporting only. Not part of the pipeline.
struct FacStats { uint32_t nitems, cap, nhits, nseg, nfacp_r, bA, bB; };
FacStats factorize_last_stats();
