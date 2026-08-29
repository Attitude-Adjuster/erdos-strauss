// Survivor compaction and primality.
//
// Both stages are ported operation for operation from the CPU scanner, and both are
// deterministic BY CONSTRUCTION rather than by agreement: placement is a prefix sum,
// never an atomicAdd, so a survivor's index in the compacted list is a pure function of
// the bitset -- not of which block happened to reach the allocator first.
#include "survivors.cuh"
#include "common.cuh"
#include "mark.cuh"

#include <cuda_runtime.h>
#include <vector>

// ---------------------------------------------------------------- primality

// rung_scan3's sprp2, operation for operation. A FAIL is conclusive and nearly every
// survivor is composite, so this rejects almost all of them for 1/12 of the full test's
// cost; a pass falls through to is_prime, which is why the composite count is identical
// either way.
__device__ __forceinline__ bool sprp2_dev(uint64_t n) {
  if (n < 2) return false;
  if (!(n & 1)) return n == 2;
  uint64_t d = n - 1;
  int s = 0;
  while (!(d & 1)) { d >>= 1; ++s; }
  uint64_t x = powmod64(2, d, n);
  if (x == 1 || x == n - 1) return true;
  for (int i = 1; i < s; ++i) { x = mulmod64(x, x, n); if (x == n - 1) return true; }
  return false;
}

// The frozen census reference's is_prime: deterministic for n < 3.3e24 on the 12 bases
// {2,...,37}, with the same trial-division prologue. The base set is not a choice here --
// it has to match the reference exactly, or a composite the CPU rejects could reach
// stage D on the device and emit a certificate the CPU never emits.
__device__ __forceinline__ bool is_prime_dev(uint64_t n) {
  const uint64_t sp[12] = {2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37};
  if (n < 2) return false;
  for (int i = 0; i < 12; ++i) {
    if (n == sp[i]) return true;
    if (n % sp[i] == 0) return false;
  }
  uint64_t d = n - 1;
  int s = 0;
  while (!(d & 1)) { d >>= 1; ++s; }
  for (int i = 0; i < 12; ++i) {
    uint64_t x = powmod64(sp[i], d, n);
    if (x == 1 || x == n - 1) continue;
    bool composite = true;
    for (int j = 1; j < s; ++j) {
      x = mulmod64(x, x, n);
      if (x == n - 1) { composite = false; break; }
    }
    if (composite) return false;
  }
  return true;
}

// Grid-stride over the count the device itself computed. Reading `*ntot` on the device
// rather than on the host is what keeps compaction and primality free of a sync between
// them -- the census port's hard-won lesson is that per-rung readbacks are where the
// wall time goes.
__global__ void k_primality(const uint64_t *cand, const uint32_t *ntot, uint32_t cap,
                            uint8_t *prime) {
  // `*ntot` is the true survivor count; the BUFFERS are only cap long. When compaction
  // overflowed, the two differ and the caller is about to replay the whole round at a
  // larger cap -- but this kernel is already in flight behind it on the same stream, so
  // it has to clamp. Without the clamp an overflowing segment walks off the end of both
  // cand and prime, which is an illegal access, not a wrong answer.
  const uint32_t tot = *ntot;
  const uint32_t n = tot < cap ? tot : cap;
  for (uint32_t i = blockIdx.x * blockDim.x + threadIdx.x; i < n;
       i += gridDim.x * blockDim.x) {
    const uint64_t p = cand[i];
    prime[i] = (sprp2_dev(p) && is_prime_dev(p)) ? 1u : 0u;
  }
}

// ---------------------------------------------------------------- compaction

// One block, scanning the per-lane counts into per-lane bases. nlanes is at most a few
// thousand, so a single block is ample and a single block is also what makes the running
// offset trivially sequential -- no multi-block scan, no decoupled look-back, nothing
// whose result could depend on block scheduling.
__global__ void k_scan_counts(const uint32_t *cnt, uint32_t nlanes, uint32_t *base,
                              uint32_t *total) {
  uint32_t running = 0;
  for (uint32_t b = 0; b < nlanes; b += blockDim.x) {
    const uint32_t i = b + threadIdx.x;
    const uint32_t v = i < nlanes ? cnt[i] : 0u;
    uint32_t t = 0;
    const uint32_t pos = block_exclusive_scan(v, t);
    if (i < nlanes) base[i] = running + pos;
    running += t;
  }
  if (threadIdx.x == 0) *total = running;
}

// One block per lane, walking the bitset in chunks of blockDim words. Within a chunk the
// write offsets come from an exclusive scan over per-word popcounts, so the output is
// ascending by position inside a lane and lanes sit in wheel order -- exactly the order
// the CPU scanner emits, without any sorting.
//
// Candidates are stored as p, not as a position index: the host tail wants p, and
// recomputing first + M*(base+j) on the host would duplicate arithmetic that has to
// agree exactly.
__global__ void k_compact(const uint32_t *bits, uint32_t nwseg, const uint32_t *nlane,
                          const uint64_t *firsts, uint64_t M, uint64_t segbase,
                          const uint32_t *lbase, uint32_t cap,
                          uint64_t *cand, uint32_t *over) {
  const uint32_t lane = blockIdx.x;
  const uint32_t n = nlane[lane];
  if (!n) return;
  const uint32_t nw = (n + 31u) >> 5;
  const uint32_t *b = bits + (size_t)lane * nwseg;
  const uint64_t first = firsts[lane];
  uint32_t running = lbase[lane];

  for (uint32_t wb = 0; wb < nw; wb += blockDim.x) {   // uniform bound: nw is per-lane
    const uint32_t w = wb + threadIdx.x;
    uint32_t x = w < nw ? b[w] : 0u;
    uint32_t t = 0;
    uint32_t o = running + block_exclusive_scan(__popc(x), t);
    while (x) {
      const uint32_t j = (w << 5) + (uint32_t)__ffs((int)x) - 1u;
      if (o < cap) cand[o] = first + M * (segbase + j);
      else atomicOr(over, 1u);      // the caller grows cap and re-runs the segment
      ++o;
      x &= x - 1;
    }
    running += t;
  }
}

// ---------------------------------------------------------------- workspace

void survivors_alloc(SurvivorWork &w, uint32_t nlanes, uint32_t cap) {
  survivors_free(w);
  w.nlanes = nlanes;
  w.cap = cap;
  CUDA_OK(cudaMalloc(&w.d_base, (size_t)nlanes * 4));
  CUDA_OK(cudaMalloc(&w.d_total, 4));
  CUDA_OK(cudaMalloc(&w.d_over, 4));
  CUDA_OK(cudaMalloc(&w.d_cand, (size_t)cap * 8));
  CUDA_OK(cudaMalloc(&w.d_prime, (size_t)cap));
  CUDA_OK(cudaMemset(w.d_over, 0, 4));      // cudaMalloc does not zero
  CUDA_OK(cudaMemset(w.d_total, 0, 4));
}

void survivors_free(SurvivorWork &w) {
  if (w.d_base) CUDA_OK(cudaFree(w.d_base));
  if (w.d_total) CUDA_OK(cudaFree(w.d_total));
  if (w.d_over) CUDA_OK(cudaFree(w.d_over));
  if (w.d_cand) CUDA_OK(cudaFree(w.d_cand));
  if (w.d_prime) CUDA_OK(cudaFree(w.d_prime));
  w = SurvivorWork{};
}

// ---------------------------------------------------------------- test helpers

std::vector<uint8_t> primality_device(const std::vector<uint64_t> &cand) {
  const uint32_t n = (uint32_t)cand.size();
  uint64_t *d_cand = nullptr;
  uint8_t *d_prime = nullptr;
  uint32_t *d_n = nullptr;
  CUDA_OK(cudaMalloc(&d_cand, (size_t)n * 8));
  CUDA_OK(cudaMalloc(&d_prime, n));
  CUDA_OK(cudaMalloc(&d_n, 4));
  CUDA_OK(cudaMemcpy(d_cand, cand.data(), (size_t)n * 8, cudaMemcpyHostToDevice));
  CUDA_OK(cudaMemcpy(d_n, &n, 4, cudaMemcpyHostToDevice));

  k_primality<<<1024, 128>>>(d_cand, d_n, n, d_prime);
  CUDA_OK(cudaGetLastError());
  CUDA_OK(cudaDeviceSynchronize());

  std::vector<uint8_t> out(n);
  CUDA_OK(cudaMemcpy(out.data(), d_prime, n, cudaMemcpyDeviceToHost));
  CUDA_OK(cudaFree(d_cand));
  CUDA_OK(cudaFree(d_prime));
  CUDA_OK(cudaFree(d_n));
  return out;
}

std::vector<uint64_t> survivors_device(const std::vector<uint32_t> &off_in,
                                       const std::vector<uint32_t> &step,
                                       size_t ncov, uint32_t n, uint32_t seg,
                                       uint64_t first, uint64_t M, uint64_t base,
                                       uint32_t nthreads,
                                       std::vector<uint8_t> &prime_out) {
  const uint32_t nall = (uint32_t)step.size();
  const uint32_t nwseg = (seg + 31u) / 32;
  const size_t shmem = (size_t)nwseg * 4;
  mark_enable_shared(shmem);

  uint32_t *d_off = nullptr, *d_step = nullptr, *d_n = nullptr, *d_surv = nullptr;
  uint32_t *d_bits = nullptr;
  uint64_t *d_alive = nullptr, *d_first = nullptr;
  CUDA_OK(cudaMalloc(&d_off, (size_t)nall * 4));
  CUDA_OK(cudaMalloc(&d_step, (size_t)nall * 4));
  CUDA_OK(cudaMalloc(&d_n, 4));
  CUDA_OK(cudaMalloc(&d_surv, 4));
  CUDA_OK(cudaMalloc(&d_bits, (size_t)nwseg * 4));
  CUDA_OK(cudaMalloc(&d_alive, 8));
  CUDA_OK(cudaMalloc(&d_first, 8));
  CUDA_OK(cudaMemcpy(d_off, off_in.data(), (size_t)nall * 4, cudaMemcpyHostToDevice));
  CUDA_OK(cudaMemcpy(d_step, step.data(), (size_t)nall * 4, cudaMemcpyHostToDevice));
  CUDA_OK(cudaMemcpy(d_n, &n, 4, cudaMemcpyHostToDevice));
  CUDA_OK(cudaMemcpy(d_first, &first, 8, cudaMemcpyHostToDevice));

  SurvivorWork w;
  survivors_alloc(w, 1, n ? n : 1);            // every position could survive; n is safe
  CUDA_OK(cudaMemset(w.d_over, 0, 4));

  k_mark<<<1, nthreads, shmem>>>(d_off, d_step, nall, (uint32_t)ncov, nwseg, d_n,
                                 d_bits, d_alive, d_surv);
  k_scan_counts<<<1, 256>>>(d_surv, 1, w.d_base, w.d_total);
  k_compact<<<1, nthreads>>>(d_bits, nwseg, d_n, d_first, M, base, w.d_base, w.cap,
                             w.d_cand, w.d_over);
  k_primality<<<256, 128>>>(w.d_cand, w.d_total, w.cap, w.d_prime);
  CUDA_OK(cudaGetLastError());
  CUDA_OK(cudaDeviceSynchronize());

  uint32_t total = 0, over = 0;
  CUDA_OK(cudaMemcpy(&total, w.d_total, 4, cudaMemcpyDeviceToHost));
  CUDA_OK(cudaMemcpy(&over, w.d_over, 4, cudaMemcpyDeviceToHost));
  if (over) { fprintf(stderr, "FATAL: compaction overflowed cap=%u\n", w.cap); abort(); }
  std::vector<uint64_t> cand(total);
  prime_out.resize(total);
  if (total) {
    CUDA_OK(cudaMemcpy(cand.data(), w.d_cand, (size_t)total * 8, cudaMemcpyDeviceToHost));
    CUDA_OK(cudaMemcpy(prime_out.data(), w.d_prime, total, cudaMemcpyDeviceToHost));
  }

  survivors_free(w);
  CUDA_OK(cudaFree(d_off));
  CUDA_OK(cudaFree(d_step));
  CUDA_OK(cudaFree(d_n));
  CUDA_OK(cudaFree(d_surv));
  CUDA_OK(cudaFree(d_bits));
  CUDA_OK(cudaFree(d_alive));
  CUDA_OK(cudaFree(d_first));
  return cand;
}
