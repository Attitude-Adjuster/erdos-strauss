// classify.cuh -- the device certificate classifier (Task 4).
//
// Mirrors rung_scan.cpp's test_rung() (rung_scan.cpp:314-371) exactly, with two
// representation changes and one theorem-driven shortcut:
//
//   * m = r <= TABLE_RMAX = 63, so every subset of Z/r is ONE uint64_t. The
//     reference's `bool S[RCAP+1]` divisor-DP and its subgroup-closure worklist both
//     become register-resident bitmask loops.
//   * Jacobi and Legendre symbols are table lookups (tables.cuh), not computed. This
//     is the optimization RUNBOOK.md reserves for the GPU; the CPU reference keeps
//     computing them on the fly on purpose.
//   * Level S is tested at r = 51 ONLY. See S_RUNG below.
//
// Rungs above 63 are the CPU's by construction, so this file never sees them.
#pragma once
#include <cstdint>
#include <cuda_runtime.h>
#include "factorize.cuh"
#include "tables.cuh"

// Values match the reference's `enum Level` (rung_scan.cpp:214) so a device level and
// a reference level compare directly, with no translation table to get wrong.
enum DevLevel {
  DL_HIT = 0, DL_J = 1, DL_RC = 2, DL_S = 3, DL_R = 4,
  // Not a level. gcd(q, r) != 1, where the reference switches to a 2-D pair DP
  // (rung_scan.cpp:339-360). Lemma "coprime" makes this unreachable for r < p, and
  // this classifier only ever runs at r <= 63 << p, so it cannot occur here. It is
  // reported rather than assumed: a silent wrong answer is the one outcome the
  // verification discipline cannot tolerate.
  DL_PAIRPATH = 255
};

// ---------------------------------------------------------------- S-confinement
//
// Level S is the only level no real character detects, and the only place the
// classifier would need an unbounded subgroup closure. It is confined to r = 51.
//
// WHY (es_paper.tex, Lemma "the four-constraint" + Proposition "when level S is
// possible"): 4A = p + r forces 4 in H, because H contains p and every prime factor
// of A, hence A itself, hence 4 = p*A^{-1}. A rung can therefore carry a level-S
// failure only if some subgroup H <= (Z/r)* has
//
//     4 in H,      -1 not in H,      (-1)H a square in (Z/r)*/H.
//
// Every real character is blind to the first condition (chi(4) = chi(2)^2 = 1), which
// is exactly why level S -- and only level S -- is confined by it. Brute-force
// enumeration of every subgroup of (Z/r)* for all r = 3 (mod 4), r <= 255 gives
//
//     admissible:              51, 119, 123, 187, 195, 219, 255
//     admissible without "4":  15, 35, 39, 51, 55, 75, 87, 91, 95, 111, 115, 119, ...
//
// so the four-constraint is what eliminates 15/35/39/55 -- the sub-63 rungs a naive
// reading would leave in. Below TABLE_RMAX only 51 survives, and there the admissible
// subgroup is unique: H = <2> = {1,2,4,8,13,16,26,32}, index 4 in (Z/51)*.
//
// The shortcut is checked, not trusted: `strict` re-enables the closure at every rung
// (see classify_rung), and --check-classify diffs BOTH modes against the reference
// over every rung <= 63. A disagreement would falsify the proposition, which is the
// point -- the paper states it as falsifiable at fixed cost.
enum { S_RUNG = 51 };

// The unique admissible subgroup at r = 51, as a bitmask over residues. Used only by
// the strict-mode audit, never to decide a level.
static const uint64_t S_RUNG_H_MASK = 0x0000000104012116ull;   // {1,2,4,8,13,16,26,32}

// Classifies one rung over a window of active positions. p and A are derived from the
// position index rather than passed as arrays, so this consumes exactly what Task 3
// produces: for i < nact and j = d_actpos[i],
//
//     p = first + mod * j,      A = A0 + astep * j.
//
// Writes d_level[i] in {DL_HIT, DL_J, DL_RC, DL_S, DL_R, DL_PAIRPATH}.
// `strict` runs the subgroup closure at every rung instead of only at S_RUNG.
void classify_rung(const DevTables &T, uint32_t r,
                   uint64_t first, uint64_t mod, uint64_t A0, uint64_t astep,
                   const uint32_t *d_actpos, const FacR *d_fac,
                   uint32_t nact, uint8_t *d_level, bool strict, cudaStream_t s);

// ---------------------------------------------------------------- fused census path
//
// Per-rung tallies, reduced on the device with atomics. Integer sums and a min, so
// the VALUE is independent of block order -- atomics without losing determinism.
struct RungTally {
  unsigned long long lv[6];    // indexed by DevLevel; slot 5 is DL_PAIRPATH
  unsigned int min_hit_j;      // lowest position that hit, 0xffffffff if none
};

// The census's per-position tail, FUSED: cofactor append (facr_append.cuh, shared
// with k_finish) + classify_one + the tally/flag/encode bookkeeping, one thread per
// active position. Unfused this was three kernels (k_finish, k_classify_window,
// k_bookkeep) passing a 33-byte FacR and a level byte through global memory between
// launches; fused, the FacR is completed and classified in registers and the level
// never exists in memory at all. The harness path (classify_rung, --check-classify)
// stays unfused on purpose -- it tests the same classify_one this calls.
//
//   d_fac    factorizations WITHOUT the prime cofactor (factorize_rung finish=false)
//   d_remA   the cofactors (factorize_last_remA())
//   emits    encoded (level << 32 | position) for the selected positions
void classify_tally_rung(const DevTables &T, uint32_t r,
                         uint64_t first, uint64_t mod, uint64_t A0, uint64_t astep,
                         const uint32_t *d_actpos, const FacR *d_fac,
                         const uint64_t *d_remA, uint32_t nact,
                         uint64_t deepThr, uint64_t sampleEvery,
                         bool emitRes, bool emitSup,
                         uint8_t *d_survflag, uint8_t *d_emitflag, uint64_t *d_enc,
                         RungTally *d_tally, cudaStream_t s);

// Classifies an explicit list of (p, A, r) triples. Same device code as
// classify_rung -- only the addressing differs. This exists because level S occurs
// roughly 0.13 times per 10^9 of height (1,309 events in the whole 10^13 census), so
// no window a test can afford contains one; DL_S would otherwise ship unexecuted.
// --check-s replays the census's known level-S certificates through it.
void classify_explicit(const DevTables &T,
                       const uint32_t *d_r, const uint64_t *d_p, const uint64_t *d_A,
                       const FacR *d_fac, uint32_t n,
                       uint8_t *d_level, bool strict, cudaStream_t s);

// --check-classify LO HI: diff the device classifier against the reference's
// test_rung() at every position of one window, for every rung r = 3 (mod 4), r <= 63
// -- the classifier's entire domain. Runs both fast and strict mode. Returns the
// mismatch count; prints "CLASSIFY OK" or "CLASSIFY FAIL (n)".
int check_classify(uint64_t lo, uint64_t hi, bool hard840, int spanlog);

// --check-s FILE: replay known level-S certificates ("SUPPORT p A r" lines) through
// the device classifier and require DL_S on every one. Returns the failure count.
int check_s(const char *path);
