// sieve.cu -- the device primality sieve (Task 6). See sieve.cuh for why it matters.
#include "common.cuh"
#include "ref_host.h"
#include "sieve.cuh"
#include "tables.cuh"
#include <algorithm>
#include <cstdio>
#include <cub/cub.cuh>
#include <thrust/iterator/counting_iterator.h>
#include <mutex>
#include <vector>

namespace {

enum { WARP = 32, TPB = 256, NWS = 8 };   // NWS: warps per regime-A prime

// Same three-regime split as factorize.cu, and for the same reason: hits per prime is
// npos/l, which spans six orders of magnitude across the base list, so one thread per
// prime would leave a handful of threads marking millions of positions while the rest
// mark none.
//
//   regime A   l < npos/(NWS*WARP)      NWS warps per prime
//   regime B   .. < l < npos/WARP       one warp per prime
//   regime C   l >= npos/WARP           one thread per prime (< 32 hits)
struct Part { uint32_t nb_eff, bA, bB, offB, offC, total; };

__device__ __forceinline__ void decode(uint32_t x, uint32_t bA, uint32_t bB,
                                       uint32_t offB, uint32_t offC,
                                       uint32_t &i, uint32_t &w, uint32_t &nw) {
  if      (x < offB) { i = x / NWS;          w = x % NWS; nw = NWS; }
  else if (x < offC) { i = bA + (x - offB);  w = 0;       nw = 1;   }
  else               { i = bB + (x - offC);  w = 0;       nw = 0;   }
}

// First position of prime l's progression in this window, or >= npos if it has none.
// rung_scan.cpp:492-496, verbatim in effect.
__device__ __forceinline__ uint64_t first_hit(uint64_t l, uint64_t invm, uint64_t first,
                                              uint64_t mod) {
  // j0 = (l - first mod l) * mod^{-1} (mod l). Both factors are < l < 2^32, so the
  // product fits in 64 bits and needs no mulmod helper.
  uint64_t j = ((l - first % l) % l) * invm % l;
  const uint64_t ll = l * l;
  if (first < ll) {                                // start no earlier than l^2
    const uint64_t jmin = (ll - first + mod - 1) / mod;
    if (j < jmin) j += ((jmin - j + l - 1) / l) * l;
  }
  return j;
}

// Regimes A and B: ONE WARP PER ITEM. Writing 0 is idempotent, so overlapping
// progressions need no coordination and the kernel needs no atomics.
__global__ void k_mark_warp(const uint32_t *base, const uint64_t *invmod,
                            uint64_t first, uint64_t mod, uint64_t whi, uint32_t npos,
                            uint32_t bA, uint32_t bB, uint32_t offB, uint32_t offC,
                            uint8_t *isp) {
  const uint32_t x = (blockIdx.x * blockDim.x + threadIdx.x) / WARP;   // item = warp
  if (x >= offC) return;
  uint32_t i, w, nw;
  decode(x, bA, bB, offB, offC, i, w, nw);

  const uint64_t l = base[i];
  if (mod % l == 0) return;                        // l | mod never divides p
  if (l * l >= whi) return;                        // rung_scan.cpp:491

  const uint64_t j = first_hit(l, invmod[i], first, mod);
  // This item covers hits w*WARP + lane, stepping by nw*WARP, so a warp's writes are
  // l apart -- as coalesced as a strided mark can be.
  const uint32_t lane = threadIdx.x & (WARP - 1);
  const uint64_t start  = j + (uint64_t)(w * WARP + lane) * l;
  const uint64_t stride = (uint64_t)(nw * WARP) * l;
  for (uint64_t jj = start; jj < npos; jj += stride) isp[jj] = 0;
}

// Regime C: fewer than WARP hits per prime, so one thread each and nothing to share.
__global__ void k_mark_thread(const uint32_t *base, const uint64_t *invmod,
                              uint64_t first, uint64_t mod, uint64_t whi, uint32_t npos,
                              uint32_t bB, uint32_t nC, uint8_t *isp) {
  const uint32_t t = blockIdx.x * blockDim.x + threadIdx.x;
  if (t >= nC) return;
  const uint32_t i = bB + t;
  const uint64_t l = base[i];
  if (mod % l == 0) return;
  if (l * l >= whi) return;
  for (uint64_t j = first_hit(l, invmod[i], first, mod); j < npos; j += l) isp[j] = 0;
}

// rung_scan.cpp:501 -- 1 and anything below 25 is not prime, and the sieve never marks
// them because their smallest factor is not below their square root.
__global__ void k_clear_small(uint64_t first, uint64_t mod, uint32_t npos, uint8_t *isp) {
  const uint32_t j = blockIdx.x * blockDim.x + threadIdx.x;
  if (j < npos && first + mod * (uint64_t)j < 25) isp[j] = 0;
}

// One thread per 64-position word of the output bitmap.
__global__ void k_pack(const uint8_t *isp, uint32_t npos, uint32_t nwords,
                       uint64_t *bits) {
  const uint32_t w = blockIdx.x * blockDim.x + threadIdx.x;
  if (w >= nwords) return;
  uint64_t v = 0;
  const uint32_t b0 = w * 64, n = min(64u, npos - b0);
  for (uint32_t k = 0; k < n; ++k) if (isp[b0 + k]) v |= 1ull << k;
  bits[w] = v;
}

// Persistent workspace, one per window-in-flight (see factorize.cuh on slots). A slot
// is only ever driven by one host thread, so no locks; the shared h_base mirror is
// initialized under a mutex because both threads read the same uploaded table.
struct SieveWork {
  uint8_t  *isp = nullptr;
  uint32_t *num = nullptr;
  uint32_t *pinned = nullptr;
  void     *tmp = nullptr;
  size_t    cap_isp = 0, cap_tmp = 0;
  Part      part;
};
enum { SIEVE_SLOTS = 2 };
SieveWork g_sw[SIEVE_SLOTS];

void ensure(SieveWork &w, uint32_t npos) {
  if (npos > w.cap_isp) {
    if (w.isp) cudaFree(w.isp);
    CUDA_CHECK(cudaMalloc(&w.isp, npos));
    w.cap_isp = npos;
  }
  if (!w.num) CUDA_CHECK(cudaMalloc(&w.num, sizeof(uint32_t)));
  if (!w.pinned) CUDA_CHECK(cudaMallocHost(&w.pinned, sizeof(uint32_t)));
}

std::mutex g_base_mu;

}  // namespace

uint32_t sieve_class(const DevTables &T, uint64_t first, uint32_t npos, uint64_t mod,
                     uint64_t whi, uint64_t *d_primebits, uint32_t *d_actpos,
                     cudaStream_t s, int slot) {
  if (!npos) return 0;
  SieveWork &W = g_sw[slot];
  ensure(W, npos);

  // Only primes with l*l < whi can mark anything (rung_scan.cpp:491). The base list is
  // ascending, so that is a prefix -- found once on the host rather than tested by
  // every thread. The mirror is shared between slots; the mutex covers its (rare)
  // initialization, both threads read it thereafter.
  static std::vector<uint32_t> h_base;
  static const uint32_t *h_base_src = nullptr;
  {
    std::lock_guard<std::mutex> lk(g_base_mu);
    if (h_base_src != T.base || h_base.size() != T.nbase) {   // uploaded once per run
      h_base.resize(T.nbase);
      CUDA_CHECK(cudaMemcpy(h_base.data(), T.base, (size_t)T.nbase * sizeof(uint32_t),
                            cudaMemcpyDeviceToHost));
      h_base_src = T.base;
    }
  }
  // l*l is monotonic in l, so the primes that can mark anything are a prefix; find it
  // by binary search in exact integer arithmetic (l <= 10^9 at 10^18, so l*l fits).
  const uint32_t nb_eff = (uint32_t)(
      std::partition_point(h_base.begin(), h_base.begin() + T.nbase,
                           [whi](uint32_t l) { return (uint64_t)l * l < whi; })
      - h_base.begin());

  Part &P = W.part;
  P.nb_eff = nb_eff;
  const uint32_t LA = std::max<uint32_t>(2, npos / (NWS * WARP));
  const uint32_t LB = std::max<uint32_t>(LA, npos / WARP);
  P.bA = (uint32_t)(std::lower_bound(h_base.begin(), h_base.begin() + nb_eff, LA)
                    - h_base.begin());
  P.bB = (uint32_t)(std::lower_bound(h_base.begin() + P.bA, h_base.begin() + nb_eff, LB)
                    - h_base.begin());
  P.offB  = P.bA * NWS;
  P.offC  = P.offB + (P.bB - P.bA);
  P.total = P.offC + (nb_eff - P.bB);

  CUDA_CHECK(cudaMemsetAsync(W.isp, 1, npos, s));
  if (P.offC) {                                    // regimes A and B, one warp per item
    const uint32_t nthr = P.offC * WARP;
    k_mark_warp<<<(nthr + TPB - 1) / TPB, TPB, 0, s>>>(
        T.base, T.invmod, first, mod, whi, npos, P.bA, P.bB, P.offB, P.offC, W.isp);
    CUDA_CHECK(cudaGetLastError());
  }
  if (nb_eff > P.bB) {                             // regime C, one thread per item
    const uint32_t nC = nb_eff - P.bB;
    k_mark_thread<<<(nC + TPB - 1) / TPB, TPB, 0, s>>>(
        T.base, T.invmod, first, mod, whi, npos, P.bB, nC, W.isp);
    CUDA_CHECK(cudaGetLastError());
  }
  k_clear_small<<<(std::min(npos, 64u) + TPB - 1) / TPB, TPB, 0, s>>>(
      first, mod, std::min(npos, 64u), W.isp);
  CUDA_CHECK(cudaGetLastError());

  const uint32_t nwords = (npos + 63) / 64;
  k_pack<<<(nwords + TPB - 1) / TPB, TPB, 0, s>>>(W.isp, npos, nwords, d_primebits);
  CUDA_CHECK(cudaGetLastError());

  // Stable, atomic-free compaction to the ascending position list.
  thrust::counting_iterator<uint32_t> idx(0);
  size_t bytes = 0;
  CUDA_CHECK(cub::DeviceSelect::Flagged(nullptr, bytes, idx, W.isp, d_actpos, W.num,
                                        npos, s));
  if (bytes > W.cap_tmp) {
    if (W.tmp) cudaFree(W.tmp);
    CUDA_CHECK(cudaMalloc(&W.tmp, bytes));
    W.cap_tmp = bytes;
  }
  CUDA_CHECK(cub::DeviceSelect::Flagged(W.tmp, bytes, idx, W.isp, d_actpos, W.num,
                                        npos, s));

  CUDA_CHECK(cudaMemcpyAsync(W.pinned, W.num, sizeof(uint32_t),
                             cudaMemcpyDeviceToHost, s));
  CUDA_CHECK(cudaStreamSynchronize(s));
  return *W.pinned;
}

void sieve_release() {
  for (int sl = 0; sl < SIEVE_SLOTS; ++sl) {
    SieveWork &W = g_sw[sl];
    if (W.isp) cudaFree(W.isp);
    if (W.num) cudaFree(W.num);
    if (W.pinned) cudaFreeHost(W.pinned);
    if (W.tmp) cudaFree(W.tmp);
    W.isp = nullptr; W.num = nullptr; W.pinned = nullptr; W.tmp = nullptr;
    W.cap_isp = W.cap_tmp = 0;
  }
}

int check_sieve(uint64_t lo, uint64_t hi, bool hard840, int spanlog) {
  const uint64_t hi_eff = std::max<uint64_t>(hi, lo + (840ull << spanlog));
  DevTables T = upload_tables(hi_eff, hard840);

  const uint64_t mod = ref_mod();
  const uint64_t SPAN = mod << spanlog;
  const uint64_t wlo = lo, whi = lo + SPAN;
  const uint32_t ncls = ref_nclasses();

  printf("SIEVE  window [%llu, %llu)  span=%llu  classes=%u  base primes=%u\n",
         (unsigned long long)wlo, (unsigned long long)whi,
         (unsigned long long)SPAN, ncls, T.nbase);

  int bad = 0;
  uint64_t tot = 0, totpos = 0;
  for (uint32_t c = 0; c < ncls; ++c) {
    std::vector<uint32_t> want;
    uint64_t first = 0, npos = 0;
    const uint32_t nwant = ref_sieve_class(wlo, whi, ref_class(c), &first, &npos, want);
    if (!npos) continue;
    totpos += npos;

    uint64_t *d_bits; uint32_t *d_pos;
    const uint32_t nwords = (uint32_t)((npos + 63) / 64);
    CUDA_CHECK(cudaMalloc(&d_bits, (size_t)nwords * sizeof(uint64_t)));
    CUDA_CHECK(cudaMalloc(&d_pos, (size_t)npos * sizeof(uint32_t)));

    const uint32_t ngot = sieve_class(T, first, (uint32_t)npos, mod, whi, d_bits, d_pos, 0);
    std::vector<uint32_t> got(ngot);
    CUDA_CHECK(cudaMemcpy(got.data(), d_pos, (size_t)ngot * sizeof(uint32_t),
                          cudaMemcpyDeviceToHost));
    std::vector<uint64_t> bits(nwords);
    CUDA_CHECK(cudaMemcpy(bits.data(), d_bits, (size_t)nwords * sizeof(uint64_t),
                          cudaMemcpyDeviceToHost));
    cudaFree(d_bits); cudaFree(d_pos);
    tot += ngot;

    // (1) identical to the reference's own sieve, position for position
    if (ngot != nwant || got != want) {
      printf("SIEVE class %llu: position list differs (gpu %u, ref %u)\n",
             (unsigned long long)ref_class(c), ngot, nwant);
      for (uint32_t i = 0, shown = 0; i < std::max(ngot, nwant) && shown < 5; ++i) {
        const uint32_t a = i < ngot ? got[i] : 0xffffffffu;
        const uint32_t b = i < nwant ? want[i] : 0xffffffffu;
        if (a != b) { printf("   first diff at %u: gpu j=%u ref j=%u\n", i, a, b); ++shown; }
      }
      ++bad;
    }

    // (2) the bitmap agrees with Miller-Rabin at EVERY position -- an independent
    // algorithm, so this cannot pass by sharing the sieve's bug
    int cbad = 0;
    #pragma omp parallel for schedule(dynamic, 8192) reduction(+ : cbad)
    for (long long j = 0; j < (long long)npos; ++j) {
      const uint64_t p = first + mod * (uint64_t)j;
      const bool sieved = (bits[j >> 6] >> (j & 63)) & 1ull;
      const bool truth  = (p >= 25) && ref_is_prime(p);
      if (sieved != truth) {
        if (cbad < 5)
          printf("SIEVE mismatch j=%lld p=%llu: sieve says %s, is_prime says %s\n",
                 j, (unsigned long long)p, sieved ? "prime" : "composite",
                 truth ? "prime" : "composite");
        ++cbad;
      }
    }
    bad += cbad;
  }
  printf("SIEVE  %llu positions checked against Miller-Rabin, %llu primes found\n",
         (unsigned long long)totpos, (unsigned long long)tot);
  sieve_release();
  printf(bad ? "SIEVE FAIL (%d)\n" : "SIEVE OK\n", bad);
  return bad;
}
