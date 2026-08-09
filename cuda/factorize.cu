// factorize.cu -- the generate/sort/segment factorization pipeline (Task 3).
// See factorize.cuh for the stage overview and the no-atomics determinism argument.
#include "common.cuh"
#include "factorize.cuh"
#include "ref_host.h"
#include "sieve.cuh"
#include "tables.cuh"
#include "facr_append.cuh"
#include <cub/cub.cuh>
#include <algorithm>
#include <cstdio>
#include <vector>

// ------------------------------------------------------ load balancing by l
//
// The progressions are wildly uneven: at npos = 2^22 the prime l = 11 lands ~380k
// hits and a prime near sqrt(Amax) ~ 1.6e6 lands two or three. One thread per prime
// would put a million-iteration loop next to a two-iteration loop in the same warp,
// so the prime list -- which is sorted, hence contiguous in every band -- is split
// into three regimes and each prime is handed a different amount of hardware:
//
//   regime A   l < npos/1024              NWA warps per prime (>= 1024 hits)
//   regime B   npos/1024 <= l < npos/WARP one warp per prime  (32..1024 hits)
//   regime C   l >= npos/WARP             one thread per prime (< 32 hits)
//
// At spanlog 22 that splits 119853 primes into 564 / 11687 / 107602, holding roughly
// 69% / 20% / 11% of the hits.
//
// The A/B split is the brief's. Two deviations, both measured on the L4:
//
//  * The B/C split is at npos/32, not the brief's npos. A prime with l in
//    [npos/32, npos) yields at most 32 hits, so a whole warp there leaves at most one
//    lane busy.
//  * Regime A is a group of warps, not a block. The block form (cub::BlockScan to
//    compact the block's survivors each iteration) is the obvious way to keep stores
//    coalesced, but its per-iteration __syncthreads costs more than the coalescing
//    saves: it made generation 3.65 ms/window against 3.46 for the naive form. The
//    warp group gets the same coalescing from __ballot_sync/__popc with no barrier
//    at all.
//
// A WORK ITEM is one warp's share of one prime -- (prime i, warp slot w) with nw
// warps on that prime. It owns a slice, and its 32 lanes ballot-compact their
// survivors into it, so each store is a contiguous 32-lane run. Items are numbered
// so that every item of prime i precedes every item of prime i' > i, which is what
// makes the sorted output's within-position factor order a pure function of the
// inputs.
#ifndef NWA_VAL           // warps per regime-A prime; see the sweep in task-3-report.md
#define NWA_VAL 32
#endif
enum { WARP = 32, TPB = 256, NWA = NWA_VAL };

struct Part {                       // host-side regime layout for one rung
  uint32_t nfacp_r;                 // primes with l*l <= Amax (replaces the ref's break)
  uint32_t bA, bB;                  // regime boundaries as indices into facp
  uint32_t offB, offC, total;       // work-item index bases and count
};

__device__ __forceinline__ uint64_t mulmod_dev(uint64_t a, uint64_t b, uint64_t m) {
  // Every factor prime is < 2^32 (SCALING.md S3.2), and a, b < m <= l, so the plain
  // 64-bit product is exact -- no 128-bit path needed.
  return (a * b) % m;
}

// First position j in [0, npos) that l divides: j = -A0 * astep^{-1} (mod l).
// Same expression as rung_scan.cpp:535.
__device__ __forceinline__ uint64_t first_hit(uint64_t l, uint64_t inv, uint64_t A0) {
  return mulmod_dev((l - A0 % l) % l, inv, l);
}

__device__ __forceinline__ bool keep(const uint64_t *filt, uint64_t j) {
  return (filt[j >> 6] >> (j & 63)) & 1ull;
}

// Decodes work item x into (prime index, warp slot, warps on this prime). nw == 0
// marks regime C, where the item is a single thread covering the whole progression.
__device__ __forceinline__ void decode(uint32_t x, uint32_t bA, uint32_t bB,
                                       uint32_t offB, uint32_t offC,
                                       uint32_t &i, uint32_t &w, uint32_t &nw) {
  if (x < offB)      { i = x / NWA;          w = x % NWA; nw = NWA; }
  else if (x < offC) { i = bA + (x - offB);  w = 0;       nw = 1;   }
  else               { i = bB + (x - offC);  w = 0;       nw = 0;   }
}

// ------------------------------------------------------------ stage 1: count
// Closed form, no walk. The hits of l are the progression j0, j0+l, j0+2l, ...; of
// those U hits, item (w, nw) takes the ones whose ordinal lies in [w*32, w*32+32)
// modulo nw*32. This is the PRE-filter upper bound, which is what sizes the slice.
__global__ void k_count(const uint32_t *facp, const uint64_t *invastep,
                        uint64_t A0, uint64_t astep, uint32_t npos,
                        uint32_t bA, uint32_t bB, uint32_t offB, uint32_t offC,
                        uint32_t total, uint32_t *counts) {
  uint32_t x = blockIdx.x * blockDim.x + threadIdx.x;
  if (x > total) return;
  if (x == total) { counts[x] = 0; return; }   // scan sentinel: slice[total] = grand total
  uint32_t i, w, nw; decode(x, bA, bB, offB, offC, i, w, nw);
  const uint64_t l = facp[i];
  if (astep % l == 0) { counts[x] = 0; return; }   // stripped per-position instead
  const uint64_t j0 = first_hit(l, invastep[i], A0);
  if (j0 >= npos) { counts[x] = 0; return; }
  const uint32_t U = (uint32_t)((npos - 1 - j0) / l + 1);
  if (nw == 0) { counts[x] = U; return; }
  const uint32_t G = nw * WARP, t0 = w * WARP;
  const uint32_t rem = U % G;
  counts[x] = (U / G) * WARP + (rem > t0 ? (rem - t0 < WARP ? rem - t0 : WARP) : 0u);
}

// ------------------------------------- stage 3: generate + filter, regimes A and B
// One warp per work item. Each iteration the warp tests 32 consecutive multiples of
// l, ballots the survivors, and writes them as one contiguous run at
// slice[x] + base + popc(mask below this lane) -- coalesced, no barrier, and the
// slice fills in ascending j, which is exactly the reference's walk order.
__global__ void k_gen_warp(const uint32_t *facp, const uint64_t *invastep,
                           uint64_t A0, uint64_t astep, uint32_t npos,
                           uint32_t lo, uint32_t nprimes, uint32_t nw,
                           uint32_t itembase, const uint64_t *filt,
                           const uint32_t *slice,
                           uint32_t *out_j, uint32_t *out_i, uint32_t *actual) {
  const uint32_t gid = blockIdx.x * blockDim.x + threadIdx.x;
  const uint32_t lane = gid & 31u, wid = gid >> 5;
  if (wid >= nprimes * nw) return;             // warp-uniform: __ballot_sync stays legal
  const uint32_t i = lo + wid / nw, w = wid % nw, x = itembase + wid;
  const uint64_t l = facp[i];
  if (astep % l == 0) { if (!lane) actual[x] = 0; return; }
  const uint64_t j0 = first_hit(l, invastep[i], A0);
  const uint64_t start = j0 + (uint64_t)(w * WARP) * l;   // this item's first position
  const uint64_t step  = (uint64_t)(nw * WARP) * l;
  const uint32_t niter = (start < npos) ? (uint32_t)((npos - 1 - start) / step + 1) : 0u;
  const uint32_t base0 = slice[x];
  uint64_t j = start + (uint64_t)lane * l;
  uint32_t base = 0;
  for (uint32_t it = 0; it < niter; ++it, j += step) {
    const bool pred = (j < npos) && keep(filt, j);
    const uint32_t m = __ballot_sync(0xffffffffu, pred);
    const uint32_t off = __popc(m & ((1u << lane) - 1u));
    if (pred) { out_j[base0 + base + off] = (uint32_t)j; out_i[base0 + base + off] = i; }
    base += __popc(m);
  }
  if (!lane) actual[x] = base;
}

// ------------------------------------------- stage 3: generate + filter, regime C
// One thread per prime -- fewer than WARP hits each, so there is nothing to share.
__global__ void k_gen_thread(const uint32_t *facp, const uint64_t *invastep,
                             uint64_t A0, uint64_t astep, uint32_t npos,
                             uint32_t bB, uint32_t nC, uint32_t offC,
                             const uint64_t *filt, const uint32_t *slice,
                             uint32_t *out_j, uint32_t *out_i, uint32_t *actual) {
  const uint32_t t = blockIdx.x * blockDim.x + threadIdx.x;
  if (t >= nC) return;
  const uint32_t i = bB + t, x = offC + t;
  const uint64_t l = facp[i];
  if (astep % l == 0) { actual[x] = 0; return; }
  const uint32_t base0 = slice[x];
  uint32_t n = 0;
  for (uint64_t j = first_hit(l, invastep[i], A0); j < npos; j += l) {
    if (!keep(filt, j)) continue;
    out_j[base0 + n] = (uint32_t)j; out_i[base0 + n] = i; ++n;
  }
  actual[x] = n;
}

// Sparse -> dense gather, ONE THREAD PER OUTPUT ELEMENT.
//
// The obvious shape -- one warp per work item -- was measured at 16.6% of all GPU time
// at 10^15, the second most expensive kernel in the program, for a pure gather. The
// reason is that it launches `total` warps whether or not there is anything to move:
// 90% of items are regime C with one or zero survivors, and at late rungs essentially
// all of them are empty. Its cost had a FLOOR of 150 us per call across every rung.
//
// Indexing by output instead makes the cost proportional to the data. `dense` is the
// exclusive scan of `actual`, so it is non-decreasing with dense[total] = nhits; the
// owning item of output k is the last x with dense[x] <= k, found by binary search.
// dense is a few MB and L2-resident, so the ~20 dependent loads are cache hits.
__global__ void k_compact(uint32_t total, uint32_t nhits,
                          const uint32_t *slice, const uint32_t *dense,
                          const uint32_t *in_j, const uint32_t *in_i,
                          uint32_t *out_j, uint32_t *out_i) {
  const uint32_t k = blockIdx.x * blockDim.x + threadIdx.x;
  if (k >= nhits) return;
  uint32_t a = 0, b = total;                  // find last x with dense[x] <= k
  while (a < b) {
    const uint32_t mid = a + ((b - a + 1) >> 1);
    if (dense[mid] <= k) a = mid; else b = mid - 1;
  }
  const uint32_t src = slice[a] + (k - dense[a]);
  out_j[k] = in_j[src];
  out_i[k] = in_i[src];
}

// -------------------------------------------------- filter mask (active AND prime)
//
// The walk's inner test has to be a single cheap load, so the active-position LIST is
// turned into a bitmask once per call. Building it without atomics: one thread per
// 64-position word binary-searches the (sorted) active list for the start of its own
// word and ORs the bits it owns into a register. npos/64 words is 512 KB at
// spanlog 22 -- L2-resident, so the strided walk reads it at cache speed.
__global__ void k_buildmask(const uint32_t *actpos, uint32_t nact, uint32_t nwords,
                            const uint64_t *primebits, uint64_t *filt) {
  uint32_t w = blockIdx.x * blockDim.x + threadIdx.x;
  if (w >= nwords) return;
  const uint32_t lo = w * 64u, hi = lo + 64u;
  uint32_t a = 0, b = nact;
  while (a < b) { uint32_t mid = a + ((b - a) >> 1); if (actpos[mid] < lo) a = mid + 1; else b = mid; }
  uint64_t mask = 0;
  for (uint32_t k = a; k < nact; ++k) {
    const uint32_t p = actpos[k];
    if (p >= hi) break;
    mask |= 1ull << (p - lo);
  }
  filt[w] = primebits ? (mask & primebits[w]) : mask;
}

// -------------------------------------------------- stage 4: per-position reduce
//
// Split in three so that every write target is owned by exactly one thread:
//   k_base    one thread per active position: strip the primes dividing astep
//   k_reduce  one thread per SEGMENT (a position with at least one factor-prime hit)
//   k_finish  one thread per active position: append the prime cofactor
// A position belongs to at most one segment, so k_reduce's read-modify-write of
// remA/out is race-free without atomics.

// stripmask: bit b set iff STRIPP[b] divides astep AND divides A0 -- i.e. iff it
// divides every A in the window. The reference tests this per position
// (rung_scan.cpp:525-528); because astep*j is a multiple of each such prime,
// A = A0 (mod sp), so the host can decide it once for the whole window.
__constant__ uint32_t STRIPP_DEV[4] = {2u, 3u, 5u, 7u};

__global__ void k_base(uint64_t A0, uint64_t astep, uint32_t r, uint32_t stripmask,
                       const uint32_t *actpos, uint32_t nact,
                       uint64_t *remA, FacR *out) {
  uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= nact) return;
  uint64_t A = A0 + astep * (uint64_t)actpos[i];
  uint8_t k = 0;
  #pragma unroll
  for (int b = 0; b < 4; ++b) {
    if (!((stripmask >> b) & 1u)) continue;
    const uint64_t sp = STRIPP_DEV[b];
    uint8_t e = 0;
    while (A % sp == 0) { A /= sp; ++e; }
    out[i].res[k] = (uint8_t)(sp % r); out[i].ex[k] = (uint8_t)e; ++k;
  }
  remA[i] = A;
  out[i].k = k;
}

// nseg arrives as a DEVICE pointer (RunLengthEncode's own output), not a host value:
// reading it back to launch this kernel would cost a pipeline drain per rung, so the
// grid is sized by the host-known upper bound nhits and the excess threads exit here.
__global__ void k_reduce(const uint32_t *uniq_j, const uint32_t *seg_off,
                         const uint32_t *seg_len, const uint32_t *d_nseg,
                         const uint32_t *sorted_i, const uint32_t *facp, uint32_t r,
                         const uint32_t *actpos, uint32_t nact,
                         uint64_t *remA, FacR *out) {
  uint32_t s = blockIdx.x * blockDim.x + threadIdx.x;
  if (s >= *d_nseg) return;
  const uint32_t j = uniq_j[s];
  // Every generated hit passed the active filter, so j is in actpos: exact match.
  uint32_t a = 0, b = nact;
  while (a < b) { uint32_t mid = a + ((b - a) >> 1); if (actpos[mid] < j) a = mid + 1; else b = mid; }
  const uint32_t idx = a;
  uint64_t A = remA[idx];
  uint8_t k = out[idx].k;
  const uint32_t off = seg_off[s], len = seg_len[s];
  for (uint32_t t = 0; t < len; ++t) {
    const uint64_t l = facp[sorted_i[off + t]];
    uint8_t e = 0;
    while (A % l == 0) { A /= l; ++e; }
    if (e) {
      if (k >= FACR_MAX) {
        printf("FATAL: more than %d distinct prime factors at j=%u\n", (int)FACR_MAX, j);
        __trap();
      }
      out[idx].res[k] = (uint8_t)(l % r); out[idx].ex[k] = e; ++k;
    }
  }
  remA[idx] = A;
  out[idx].k = k;
}

// Prime by construction: the walk used every l <= sqrt(Amax) (SCALING.md S1.2), so no
// primality test or Pollard appears on this path -- same argument as rung_scan.cpp:544.
__global__ void k_finish(uint32_t nact, uint32_t r, const uint64_t *remA, FacR *out) {
  uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= nact) return;
  facr_append_cofactor(remA[i], r, out[i]);
}


// ================================================================ direct path
//
// WHY THIS EXISTS. The sieve pipeline above costs O(pi(sqrt Amax)) per rung no matter
// how few positions are still active: it launches ~976k regime-C threads (one per
// factor prime) every single time. Measured at 10^15, the per-rung cost goes FLAT at
// ~20 ms from rung 23 onward while the active count falls 375x -- rungs 23..63 were
// ~37% of runtime for under 0.5% of the work. At rung 47 that is 18 ms to serve 483
// positions, 7000x the per-position cost at rung 3.
//
// The fix is the one the CPU reference already makes (rung_scan.cpp:477-478, BULK_MIN):
// when the ladder has thinned, stop walking primes across the window and factor each
// surviving A directly. The reference hands those to the CPU; doing it on the device
// instead is what lets the crossover move up by three orders of magnitude.
//
// The algorithm is the reference's factorA: trial division by primes <= TD_BOUND, then
// Miller-Rabin, then Pollard-Brent on what is left.

enum { TD_BOUND_DEV = 65536 };

__device__ __forceinline__ uint64_t mulmod64(uint64_t a, uint64_t b, uint64_t m) {
  return (uint64_t)((unsigned __int128)a * b % m);   // A < 2.5e17, so 64x64 overflows
}

__device__ __forceinline__ uint64_t powmod64(uint64_t a, uint64_t e, uint64_t m) {
  uint64_t r = 1; a %= m;
  while (e) { if (e & 1) r = mulmod64(r, a, m); a = mulmod64(a, a, m); e >>= 1; }
  return r;
}

__device__ __forceinline__ uint64_t gcd64(uint64_t a, uint64_t b) {
  while (b) { const uint64_t t = a % b; a = b; b = t; }
  return a;
}

// Deterministic Miller-Rabin: the first 12 primes are a proven-correct base set for
// every n < 3.3e24, far past the 2.5e17 this ever sees.
__device__ bool is_prime_dev(uint64_t n) {
  const uint32_t small[12] = {2,3,5,7,11,13,17,19,23,29,31,37};
  if (n < 2) return false;
  for (int i = 0; i < 12; ++i) { if (n % small[i] == 0) return n == small[i]; }
  uint64_t d = n - 1; int s = 0;
  while (!(d & 1)) { d >>= 1; ++s; }
  for (int i = 0; i < 12; ++i) {
    uint64_t x = powmod64(small[i], d, n);
    if (x == 1 || x == n - 1) continue;
    bool composite = true;
    for (int k = 1; k < s; ++k) {
      x = mulmod64(x, x, n);
      if (x == n - 1) { composite = false; break; }
    }
    if (composite) return false;
  }
  return true;
}

// Pollard-Brent. Note the `!diff` branch: it ABORTS this choice of c rather than
// substituting 1. Substituting 1 is the latent bug SCALING.md S1.1 documents -- it
// destroys cycle detection and turns a 0.1 ms factorization into 11 seconds. Porting
// the algorithm means porting that fix, not just the shape.
__device__ uint64_t pollard_dev(uint64_t n) {
  if (!(n & 1)) return 2;
  for (uint64_t c = 1; c < 64; ++c) {
    uint64_t x = 2, y = 2, d = 1;
    do {
      x = (mulmod64(x, x, n) + c) % n;
      y = (mulmod64(y, y, n) + c) % n;
      y = (mulmod64(y, y, n) + c) % n;
      const uint64_t diff = (x > y) ? x - y : y - x;
      if (!diff) { d = n; break; }              // cycle closed: try the next c
      d = gcd64(diff, n);
    } while (d == 1);
    if (d != n) return d;
  }
  return n;                                      // caller treats a failure as prime
}

// Full factorization of A into (prime, exponent), reduced mod r into `out`.
__device__ void factor_A_dev(uint64_t A, const uint32_t *facp, uint32_t ntd,
                             uint32_t r, FacR &out) {
  out.k = 0;
  auto add = [&](uint64_t l, uint32_t e) {
    if (out.k < FACR_MAX) {
      out.res[out.k] = (uint8_t)(l % r);
      out.ex[out.k]  = (uint8_t)e;
      ++out.k;
    }
  };

  for (uint32_t i = 0; i < ntd && A > 1; ++i) {
    const uint64_t l = facp[i];
    if (l * l > A) break;
    if (A % l == 0) {
      uint32_t e = 0;
      do { A /= l; ++e; } while (A % l == 0);
      add(l, e);
    }
  }
  if (A == 1) return;

  // What is left has no factor <= TD_BOUND, so it is prime, or a product of at most
  // three primes above it. Explicit stack; recursion on device buys nothing here.
  uint64_t stack[8]; int sp = 0;
  stack[sp++] = A;
  while (sp) {
    uint64_t n = stack[--sp];
    if (n == 1) continue;
    if (is_prime_dev(n)) {
      // Multiplicity: a repeated large prime is rare but must still be counted.
      uint32_t e = 1;
      for (int i = 0; i < sp; ++i)
        if (stack[i] == n) { stack[i] = 1; ++e; }
      add(n, e);
      continue;
    }
    const uint64_t d = pollard_dev(n);
    if (d == n || d <= 1) { add(n, 1); continue; }   // give up loudly-ish, never silently drop
    if (sp + 2 <= 8) { stack[sp++] = d; stack[sp++] = n / d; }
    else add(n, 1);
  }
}

// One thread per active position. No sieve, no sort, no compaction.
__global__ void k_factor_direct(uint64_t A0, uint64_t astep, uint32_t r,
                                const uint32_t *actpos, uint32_t nact,
                                const uint32_t *facp, uint32_t ntd, FacR *out) {
  const uint32_t t = blockIdx.x * blockDim.x + threadIdx.x;
  if (t >= nact) return;
  factor_A_dev(A0 + astep * (uint64_t)actpos[t], facp, ntd, r, out[t]);
}

// ---------------------------------------------------------------- workspace
//
// Buffers are grown on demand and reused, so a steady-state call (every call after
// the first, which is what the warm timing measures) allocates nothing.
namespace {
// cudaFree SYNCHRONIZES the device -- measured at ~620 us per call in the census's
// api_sum, each one a full pipeline drain. Growing a buffer therefore never frees the
// old allocation mid-run; it parks it in a per-slot graveyard drained at
// factorize_release. Growth is geometric, so the parked waste stays bounded.
struct Buf {
  void *p = nullptr; size_t n = 0;
  std::vector<void *> *graveyard = nullptr;
  void *get(size_t bytes) {
    if (n < bytes) {
      if (p && graveyard) graveyard->push_back(p);
      else if (p) CUDA_CHECK(cudaFree(p));
      n = bytes + bytes / 4 + 1024;
      CUDA_CHECK(cudaMalloc(&p, n));
    }
    return p;
  }
};
struct Work {
  Buf counts, slice, actual, dense;      // per factor prime
  Buf spj, spi;                          // sparse hits
  Buf dj, di, sj, si;                    // dense + sorted hits
  Buf uniq, seglen, segoff, nruns;       // segments
  Buf remA, filt, temp;
  std::vector<uint32_t> h_facp;          // host mirror of T.facp, for the partition
  const uint32_t *facp_key = nullptr;
  uint32_t facp_n = 0;
  std::vector<void *> graveyard;
  uint32_t *pinned_scalar = nullptr;     // pinned staging, the one surviving readback
  const uint64_t *last_remA = nullptr;
  // cap_bound cache -- see cap_bound() below
  uint64_t capb = 0;
  uint32_t capb_npos = 0, capb_nf = 0;
  FacStats stats = {};

  Work() {
    Buf *bufs[] = {&counts, &slice, &actual, &dense, &spj, &spi, &dj, &di, &sj, &si,
                   &uniq, &seglen, &segoff, &nruns, &remA, &filt, &temp};
    for (Buf *b : bufs) b->graveyard = &graveyard;
  }
};

// One workspace per window-in-flight (FAC_SLOTS = 2 for the double-buffered census).
// Every entry point takes a `slot`; the harnesses use the default 0, so a slot is
// only ever touched by the one host thread driving its window -- no locks anywhere.
enum { FAC_SLOTS = 2 };
Work g_w[FAC_SLOTS];
}  // namespace

static const std::vector<uint32_t> &host_facp(const DevTables &T, Work &w);

// Host-side bound on the generation capacity, replacing the per-rung `cap` readback.
// k_count's per-item counts are PRE-filter and partition U exactly, so summing
// npos/l + 1 over the walked primes bounds slice[ni] for every rung of every window.
static uint64_t cap_bound(const DevTables &T, uint64_t astep, uint32_t npos,
                          uint32_t nfacp_r, Work &w) {
  if (npos == w.capb_npos && nfacp_r <= w.capb_nf) return w.capb;
  const std::vector<uint32_t> &fp = host_facp(T, w);
  // Pad the prefix so Amax's slow growth across windows doesn't force a recompute.
  const uint32_t nf = std::min<uint32_t>(T.nfacp, nfacp_r + nfacp_r / 64 + 16);
  uint64_t b = 0;
  for (uint32_t i = 0; i < nf; ++i)
    if (astep % fp[i] != 0) b += (uint64_t)npos / fp[i] + 1;
  w.capb = b; w.capb_npos = npos; w.capb_nf = nf;
  return b;
}

static const std::vector<uint32_t> &host_facp(const DevTables &T, Work &w) {
  if (w.facp_key != T.facp || w.facp_n != T.nfacp) {
    w.h_facp.resize(T.nfacp);
    CUDA_CHECK(cudaMemcpy(w.h_facp.data(), T.facp, (size_t)T.nfacp * sizeof(uint32_t),
                          cudaMemcpyDeviceToHost));
    w.facp_key = T.facp; w.facp_n = T.nfacp;
  }
  return w.h_facp;
}

static Part partition_primes(const DevTables &T, uint64_t Amax, uint32_t npos, Work &w) {
  const std::vector<uint32_t> &fp = host_facp(T, w);
  Part P;
  // The reference stops the walk at l*l > Amax (rung_scan.cpp:534); the host resolves
  // that bound once here instead of testing it in every thread.
  P.nfacp_r = (uint32_t)(std::upper_bound(fp.begin(), fp.end(), Amax,
                   [](uint64_t v, uint32_t l) { return v < (uint64_t)l * l; }) - fp.begin());
  const uint32_t LA = std::max<uint32_t>(2u, npos / 1024u);
  const uint32_t LB = std::max<uint32_t>(LA, npos / WARP);
  P.bA = (uint32_t)(std::lower_bound(fp.begin(), fp.begin() + P.nfacp_r, LA) - fp.begin());
  P.bB = (uint32_t)(std::lower_bound(fp.begin() + P.bA, fp.begin() + P.nfacp_r, LB) - fp.begin());
  P.offB  = P.bA * NWA;
  P.offC  = P.offB + (P.bB - P.bA);
  P.total = P.offC + (P.nfacp_r - P.bB);
  return P;
}

static size_t cub_bytes(uint32_t nitems, uint32_t nhits, cudaStream_t s) {
  size_t a = 0, b = 0, c = 0;
  cub::DeviceScan::ExclusiveSum(nullptr, a, (uint32_t *)nullptr, (uint32_t *)nullptr,
                                (int)nitems + 1, s);
  if (nhits) {
    cub::DeviceRadixSort::SortPairs(nullptr, b, (uint32_t *)nullptr, (uint32_t *)nullptr,
                                    (uint32_t *)nullptr, (uint32_t *)nullptr, (int)nhits,
                                    0, 32, s);
    cub::DeviceRunLengthEncode::Encode(nullptr, c, (uint32_t *)nullptr, (uint32_t *)nullptr,
                                       (uint32_t *)nullptr, (uint32_t *)nullptr, (int)nhits, s);
  }
  return std::max(a, std::max(b, c));
}

// Active count below which factorize_rung uses the direct path. DEFAULT 0 = never.
//
// It was built to kill the flat per-rung tail (cost stops tracking nact from rung 23
// on) and it does NOT pay off: measured at 10^15, --direct-min 1024 took the census
// from 14 to 56 ns/prime, and forcing it everywhere gave 1023. At nact ~ 500 the
// direct path costs ~28 us per position -- trial division by 6542 primes, then
// Miller-Rabin and Pollard-Brent, run by 500 threads that occupy 7% of one wave.
//
// The premise was wrong, and nsys says why: the late-rung cost is not the prime-list
// scan, it is LATENCY. factorize_rung makes three blocking readbacks per call (cap,
// nhits, nseg -- CUB needs num_items on the host), and at 10^15 those cost 255 ms of
// wall clock against 359 ms of total kernel time. A rung serving 483 positions still
// pays four full pipeline drains per class. Batching classes, not changing the
// factorization algorithm, is what fixes that.
//
// The path is KEPT, correct and tested, because --check-factor uses it as a third
// independent implementation to diff the sieve pipeline against.
static uint32_t g_direct_min = 0;
void     factorize_set_direct_min(uint32_t n) { g_direct_min = n; }
uint32_t factorize_get_direct_min()           { return g_direct_min; }

void factorize_direct(const DevTables &T, uint64_t A0, uint64_t astep, uint32_t r,
                      const uint32_t *d_actpos, uint32_t nact, FacR *d_out,
                      cudaStream_t s, int slot) {
  if (!nact) return;
  // Trial-divide only by primes <= TD_BOUND_DEV; beyond that Miller-Rabin + Pollard is
  // cheaper than more division. Same split as the reference's factorA.
  const std::vector<uint32_t> &fp = host_facp(T, g_w[slot]);
  const uint32_t ntd = (uint32_t)(std::lower_bound(fp.begin(), fp.end(),
                                                   (uint32_t)TD_BOUND_DEV) - fp.begin());
  k_factor_direct<<<(nact + TPB - 1) / TPB, TPB, 0, s>>>(A0, astep, r, d_actpos, nact,
                                                         T.facp, ntd, d_out);
  CUDA_CHECK(cudaPeekAtLastError());
}

const uint64_t *factorize_last_remA(int slot) { return g_w[slot].last_remA; }

void factorize_rung(const DevTables &T, uint64_t A0, uint64_t astep, uint32_t npos,
                    uint64_t Amax, uint32_t r,
                    const uint32_t *d_actpos, uint32_t nact,
                    const uint64_t *d_primebits, FacR *d_out, cudaStream_t s,
                    bool finish, int slot) {
  Work &w = g_w[slot];
  if (!nact || !npos) { w.stats = {}; return; }
  // The sieve walk amortizes over the window; the direct path costs O(nact). Below the
  // crossover the walk is paying pi(sqrt Amax) of fixed cost to serve a handful of
  // positions. Both produce the same factorization -- --check-factor asserts it -- so
  // this is a performance dispatch, not a semantic one.
  if (nact < g_direct_min) {
    factorize_direct(T, A0, astep, r, d_actpos, nact, d_out, s, slot);
    w.stats = {0, 0, 0, 0, 0, 0, 0};
    return;
  }
#ifdef GATE_STUB   // TDD step 2: prove --gate actually discriminates
  CUDA_CHECK(cudaMemsetAsync(d_out, 0, (size_t)nact * sizeof(FacR), s));
  w.stats = {}; return;
#endif
  const Part P = partition_primes(T, Amax, npos, w);
  const uint32_t nwords = (npos + 63u) / 64u;

  uint64_t *d_filt = (uint64_t *)w.filt.get((size_t)nwords * sizeof(uint64_t));
  k_buildmask<<<(nwords + TPB - 1) / TPB, TPB, 0, s>>>(d_actpos, nact, nwords,
                                                       d_primebits, d_filt);

  // ---- stage 1: count, and stage 2: exclusive scan -> per-item slice base
  const uint32_t ni = P.total;
  uint32_t *d_counts = (uint32_t *)w.counts.get((size_t)(ni + 1) * 4);
  uint32_t *d_slice  = (uint32_t *)w.slice.get((size_t)(ni + 1) * 4);
  k_count<<<(ni + 1 + TPB - 1) / TPB, TPB, 0, s>>>(T.facp, T.invastep, A0, astep, npos,
                                                   P.bA, P.bB, P.offB, P.offC, ni, d_counts);
  size_t tb = cub_bytes(ni, 0, s);
  void *d_tmp = w.temp.get(tb);
  CUDA_CHECK(cub::DeviceScan::ExclusiveSum(d_tmp, tb, d_counts, d_slice, (int)ni + 1, s));

  // No readback of slice[ni]: the sparse buffers are sized by cap_bound(), a host-side
  // bound on it that is valid for every rung, so the sync this used to force is gone.
  const uint64_t cap = cap_bound(T, astep, npos, P.nfacp_r, w);

  // ---- stage 3: generate into the (sparse) slices, then compact
  uint32_t *d_spj = (uint32_t *)w.spj.get((size_t)std::max<uint64_t>(cap, 1) * 4);
  uint32_t *d_spi = (uint32_t *)w.spi.get((size_t)std::max<uint64_t>(cap, 1) * 4);
  uint32_t *d_act = (uint32_t *)w.actual.get((size_t)(ni + 1) * 4);
  uint32_t *d_den = (uint32_t *)w.dense.get((size_t)(ni + 1) * 4);
  const uint32_t nA = P.bA, nB = P.bB - P.bA, nC = P.nfacp_r - P.bB;
  if (nA) k_gen_warp<<<(nA * NWA * WARP + TPB - 1) / TPB, TPB, 0, s>>>(
              T.facp, T.invastep, A0, astep, npos, 0u, nA, NWA, 0u,
              d_filt, d_slice, d_spj, d_spi, d_act);
  if (nB) k_gen_warp<<<(nB * WARP + TPB - 1) / TPB, TPB, 0, s>>>(
              T.facp, T.invastep, A0, astep, npos, P.bA, nB, 1u, P.offB,
              d_filt, d_slice, d_spj, d_spi, d_act);
  if (nC) k_gen_thread<<<(nC + TPB - 1) / TPB, TPB, 0, s>>>(
              T.facp, T.invastep, A0, astep, npos, P.bB, nC, P.offC,
              d_filt, d_slice, d_spj, d_spi, d_act);
  CUDA_CHECK(cudaMemsetAsync(d_act + ni, 0, 4, s));
  CUDA_CHECK(cub::DeviceScan::ExclusiveSum(d_tmp, tb, d_act, d_den, (int)ni + 1, s));
  // The one surviving readback: the sort and compaction need num_items host-side.
  // Pinned staging -- an async D2H into pageable memory degrades to a synchronous
  // copy, which would put back the very stall being removed.
  if (!w.pinned_scalar) CUDA_CHECK(cudaMallocHost(&w.pinned_scalar, sizeof(uint32_t)));
  CUDA_CHECK(cudaMemcpyAsync(w.pinned_scalar, d_den + ni, 4, cudaMemcpyDeviceToHost, s));
  CUDA_CHECK(cudaStreamSynchronize(s));
  const uint32_t nhits = *w.pinned_scalar;

  uint64_t *d_remA = (uint64_t *)w.remA.get((size_t)nact * 8);
  uint32_t stripmask = 0;
  { const uint32_t sp[4] = {2u, 3u, 5u, 7u};
    for (int b = 0; b < 4; ++b)
      if (astep % sp[b] == 0 && A0 % sp[b] == 0) stripmask |= 1u << b; }
  k_base<<<(nact + TPB - 1) / TPB, TPB, 0, s>>>(A0, astep, r, stripmask,
                                                d_actpos, nact, d_remA, d_out);

  if (nhits) {
    uint32_t *d_dj = (uint32_t *)w.dj.get((size_t)nhits * 4);
    uint32_t *d_di = (uint32_t *)w.di.get((size_t)nhits * 4);
    k_compact<<<(nhits + TPB - 1) / TPB, TPB, 0, s>>>(ni, nhits, d_slice, d_den,
                                                      d_spj, d_spi, d_dj, d_di);

    // ---- stage 4: sort by position, run-length encode into segments, reduce
    uint32_t *d_sj = (uint32_t *)w.sj.get((size_t)nhits * 4);
    uint32_t *d_si = (uint32_t *)w.si.get((size_t)nhits * 4);
    tb = cub_bytes(ni, nhits, s);
    d_tmp = w.temp.get(tb);
    // CUB's radix sort is stable, so equal positions keep their input order, which is
    // ascending work-item index and hence ascending prime index: the contents AND the
    // order of every segment are a pure function of the inputs.
    // Keys are positions j < npos, so only ceil(log2 npos) bits can be nonzero --
    // sorting 25 bits instead of 32 drops a whole onesweep pass, and the order is
    // identical because the dropped bits are identically zero.
    int end_bit = 1;
    while ((1u << end_bit) < npos) ++end_bit;
    CUDA_CHECK(cub::DeviceRadixSort::SortPairs(d_tmp, tb, d_dj, d_sj, d_di, d_si,
                                               (int)nhits, 0, end_bit, s));
    uint32_t *d_uq = (uint32_t *)w.uniq.get((size_t)nhits * 4);
    uint32_t *d_sl = (uint32_t *)w.seglen.get((size_t)(nhits + 1) * 4);
    uint32_t *d_so = (uint32_t *)w.segoff.get((size_t)(nhits + 1) * 4);
    uint32_t *d_nr = (uint32_t *)w.nruns.get(4);
    // nseg never comes back to the host. seglen is zeroed first so the scan over the
    // full nhits bound is correct past the nseg RunLengthEncode actually writes, and
    // k_reduce reads nseg from d_nr on the device (its grid is the nhits bound).
    CUDA_CHECK(cudaMemsetAsync(d_sl, 0, (size_t)(nhits + 1) * 4, s));
    CUDA_CHECK(cub::DeviceRunLengthEncode::Encode(d_tmp, tb, d_sj, d_uq, d_sl, d_nr,
                                                  (int)nhits, s));
    CUDA_CHECK(cub::DeviceScan::ExclusiveSum(d_tmp, tb, d_sl, d_so, (int)nhits, s));
    k_reduce<<<(nhits + TPB - 1) / TPB, TPB, 0, s>>>(d_uq, d_so, d_sl, d_nr, d_si,
                                                     T.facp, r, d_actpos, nact,
                                                     d_remA, d_out);
  }

  if (finish)
    k_finish<<<(nact + TPB - 1) / TPB, TPB, 0, s>>>(nact, r, d_remA, d_out);
  w.last_remA = d_remA;
  CUDA_CHECK(cudaPeekAtLastError());
  // cap is now the host bound (an overestimate of slice[ni]) and nseg stays on the
  // device, so the stats line reports 0 for it. Correctness never depended on either.
  w.stats = {ni, (uint32_t)std::min<uint64_t>(cap, 0xffffffffu), nhits, 0,
             P.nfacp_r, P.bA, P.bB};
}

FacStats factorize_last_stats() { return g_w[0].stats; }

void factorize_release() {
  for (int sl = 0; sl < FAC_SLOTS; ++sl) {
    Work &w = g_w[sl];
    Buf *bufs[] = {&w.counts, &w.slice, &w.actual, &w.dense, &w.spj, &w.spi,
                   &w.dj, &w.di, &w.sj, &w.si, &w.uniq, &w.seglen,
                   &w.segoff, &w.nruns, &w.remA, &w.filt, &w.temp};
    for (Buf *b : bufs) { if (b->p) cudaFree(b->p); b->p = nullptr; b->n = 0; }
    for (void *p : w.graveyard) cudaFree(p);
    w.graveyard.clear();
    if (w.pinned_scalar) { cudaFreeHost(w.pinned_scalar); w.pinned_scalar = nullptr; }
    w.capb = 0; w.capb_npos = w.capb_nf = 0; w.facp_key = nullptr; w.facp_n = 0;
  }
}

// ---------------------------------------------------------------- --check-factor
//
// The dispatch in factorize_rung is only legitimate if the two paths agree. They share
// no code -- one is a strided sieve walk with a sort, the other is trial division plus
// Pollard-Brent -- so agreement is evidence rather than a tautology. Both are also
// compared against the reference's own factorA, a third independent implementation.
int check_factor(uint64_t lo, uint64_t hi, bool hard840, int spanlog) {
  const uint64_t hi_eff = std::max<uint64_t>(hi, lo + (840ull << spanlog));
  DevTables T = upload_tables(hi_eff, hard840);
  const uint64_t mod = ref_mod(), astep = ref_astep();
  const uint64_t SPAN = mod << spanlog, wlo = lo, whi = lo + SPAN;
  const uint32_t ncls = ref_nclasses();

  printf("FACTOR  window [%llu, %llu)  classes=%u\n",
         (unsigned long long)wlo, (unsigned long long)whi, ncls);

  int bad = 0;
  uint64_t tot = 0;
  const uint32_t RUNGS[4] = {3, 19, 43, 63};
  for (uint32_t c = 0; c < ncls; ++c) {
    const uint64_t first = wlo + ((ref_class(c) + mod - wlo % mod) % mod);
    if (first >= whi) continue;
    const uint64_t npos = (whi - first + mod - 1) / mod;
    const uint32_t nwords = (uint32_t)((npos + 63) / 64);

    uint64_t *d_bits; uint32_t *d_pos; FacR *d_a, *d_b;
    CUDA_CHECK(cudaMalloc(&d_bits, (size_t)nwords * 8));
    CUDA_CHECK(cudaMalloc(&d_pos, (size_t)npos * 4));
    const uint32_t nact = sieve_class(T, first, (uint32_t)npos, mod, whi, d_bits, d_pos, 0);
    if (!nact) { cudaFree(d_bits); cudaFree(d_pos); continue; }
    CUDA_CHECK(cudaMalloc(&d_a, (size_t)nact * sizeof(FacR)));
    CUDA_CHECK(cudaMalloc(&d_b, (size_t)nact * sizeof(FacR)));

    std::vector<uint32_t> pjs(nact);
    CUDA_CHECK(cudaMemcpy(pjs.data(), d_pos, (size_t)nact * 4, cudaMemcpyDeviceToHost));

    for (uint32_t r : RUNGS) {
      const uint64_t A0 = (first + r) / 4, Amax = A0 + astep * (npos - 1);
      const uint32_t saved = factorize_get_direct_min();
      factorize_set_direct_min(0);            // force the sieve pipeline
      factorize_rung(T, A0, astep, (uint32_t)npos, Amax, r, d_pos, nact, d_bits, d_a, 0);
      factorize_set_direct_min(saved);
      factorize_direct(T, A0, astep, r, d_pos, nact, d_b, 0);
      CUDA_CHECK(cudaDeviceSynchronize());

      std::vector<FacR> A(nact), B(nact);
      CUDA_CHECK(cudaMemcpy(A.data(), d_a, (size_t)nact * sizeof(FacR), cudaMemcpyDeviceToHost));
      CUDA_CHECK(cudaMemcpy(B.data(), d_b, (size_t)nact * sizeof(FacR), cudaMemcpyDeviceToHost));

      int rbad = 0;
      #pragma omp parallel for schedule(dynamic, 2048) reduction(+ : rbad)
      for (long long ii = 0; ii < (long long)nact; ++ii) {
        const uint32_t i = (uint32_t)ii;
        const uint64_t Av = A0 + astep * (uint64_t)pjs[i];
        uint64_t pr[16]; uint8_t ex[16];
        const int k = ref_factorA(Av, pr, ex);
        std::vector<std::pair<int,int>> want, va, vb;
        for (int t = 0; t < k; ++t) want.push_back({(int)(pr[t] % r), (int)ex[t]});
        for (int t = 0; t < A[i].k; ++t) va.push_back({(int)A[i].res[t], (int)A[i].ex[t]});
        for (int t = 0; t < B[i].k; ++t) vb.push_back({(int)B[i].res[t], (int)B[i].ex[t]});
        std::sort(want.begin(), want.end());
        std::sort(va.begin(), va.end());
        std::sort(vb.begin(), vb.end());
        if (va != want || vb != want) {
          if (rbad < 5)
            printf("FACTOR mismatch A=%llu r=%u: pipeline %s, direct %s\n",
                   (unsigned long long)Av, r,
                   va == want ? "ok" : "DIFFERS", vb == want ? "ok" : "DIFFERS");
          ++rbad;
        }
      }
      if (rbad) printf("FACTOR  r=%-2u  %d mismatches of %u\n", r, rbad, nact);
      bad += rbad;
      tot += nact;
    }
    cudaFree(d_bits); cudaFree(d_pos); cudaFree(d_a); cudaFree(d_b);
  }
  printf("FACTOR  %llu (position, rung) factorizations checked three ways: sieve "
         "pipeline vs direct vs the reference's factorA\n", (unsigned long long)tot);
  factorize_release();
  sieve_release();
  printf(bad ? "FACTOR FAIL (%d)\n" : "FACTOR OK\n", bad);
  return bad;
}
