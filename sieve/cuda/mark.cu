// The marking kernel -- stage B (certified covers) and stage B' (small primes) -- plus
// the host-side replay the gate compares it against.
//
// One block per lane. The segment bitset lives in SHARED memory and every clear is an
// atomicAnd, which needs no coordination whatsoever: marking only ever CLEARS bits, so
// it is idempotent and order-independent. Two threads clearing the same bit cannot
// disagree, and no execution order is observable in the result. The census port makes
// the same argument for its composite marking writing a constant 0.
#include "mark.cuh"
#include "common.cuh"
#include "ref_host.h"

#include <cuda_runtime.h>
#include <algorithm>
#include <vector>

extern __shared__ uint32_t s_bits[];

// off is IN/OUT: on return it holds the carried offsets for the next segment, exactly as
// sweep_lane's `off[i] = j - n`. That carry is why the offsets are computed once per
// (lane, shard) and never again -- the pattern is fixed for the whole lane.
__global__ void k_mark(uint32_t *off, const uint32_t *step, uint32_t nall, uint32_t ncov,
                       uint32_t nwseg, const uint32_t *nlane,
                       uint32_t *bits_out, uint64_t *alive_out, uint32_t *surv_out) {
  const uint32_t lane = blockIdx.x;
  const uint32_t n = nlane[lane];
  if (!n) {
    if (threadIdx.x == 0) { alive_out[lane] = 0; surv_out[lane] = 0; }
    return;
  }
  const uint32_t nw = (n + 31u) >> 5;
  uint32_t *l_off = off + (size_t)lane * nall;
  const uint32_t *l_step = step + (size_t)lane * nall;

  for (uint32_t w = threadIdx.x; w < nw; w += blockDim.x) s_bits[w] = ~0u;
  // The tail mask must not race the fill above: without this barrier the thread that
  // owns word nw-1 in the strided fill can overwrite the mask with ~0 afterwards.
  __syncthreads();
  if (threadIdx.x == 0 && (n & 31u)) s_bits[nw - 1] = (1u << (n & 31u)) - 1;
  __syncthreads();

  // Stage B -- certified covers. Filters arrive sorted by ascending modulus, so a strided
  // assignment hands each warp 32 filters of similar stride and the hit counts inside a
  // warp stay comparable; the expensive small moduli all land in the first iteration.
  for (uint32_t i = threadIdx.x; i < ncov; i += blockDim.x) {
    const uint32_t s = l_step[i];
    if (!s) continue;                     // this filter misses the lane entirely
    uint32_t j = l_off[i];
    for (; j < n; j += s)
      atomicAnd(&s_bits[j >> 5], ~(1u << (j & 31u)));
    l_off[i] = j - n;                     // carry into the next segment
  }
  __syncthreads();

  // Count what the covers left BEFORE stage B' runs. `covered` and `sieved` mean
  // entirely different things -- a cover proves 4/p SOLVABLE and is part of the
  // certificate chain, a small prime only proves the position composite and certifies
  // nothing -- so the two counters must never be conflated. The caller applies the CPU
  // scanner's rule (this count is unused when there is no stage B'); computing it
  // unconditionally costs one popcount pass and keeps --check-mark able to compare it.
  uint32_t alive = 0;
  for (uint32_t w = threadIdx.x; w < nw; w += blockDim.x) alive += __popc(s_bits[w]);
  alive = block_reduce_sum(alive);
  if (threadIdx.x == 0) alive_out[lane] = alive;
  __syncthreads();                        // popcount reads before stage B' writes

  // Stage B' -- small primes. Same loop, same idempotent clear; a prime q is just the
  // filter (mod = q, res = 0), which has the same congruence shape as a cover.
  for (uint32_t i = ncov + threadIdx.x; i < nall; i += blockDim.x) {
    const uint32_t s = l_step[i];
    if (!s) continue;
    uint32_t j = l_off[i];
    for (; j < n; j += s)
      atomicAnd(&s_bits[j >> 5], ~(1u << (j & 31u)));
    l_off[i] = j - n;
  }
  __syncthreads();

  // Copy out, and count the survivors in the same pass -- compaction needs a per-lane
  // count before it can scan them into their slices, and re-reading the bitset from
  // global memory in a separate kernel just to popcount it would be pure waste.
  uint32_t *bits = bits_out + (size_t)lane * nwseg;
  uint32_t surv = 0;
  for (uint32_t w = threadIdx.x; w < nw; w += blockDim.x) {
    const uint32_t x = s_bits[w];
    bits[w] = x;
    surv += __popc(x);
  }
  surv = block_reduce_sum(surv);
  if (threadIdx.x == 0) surv_out[lane] = surv;
}

// sm_89 allows up to 100 KB of dynamic shared memory per block, but only 48 KB without
// opting in. Do it once; a --seg the hardware cannot host is a hard refusal, not a
// silent fallback to a smaller segment.
void mark_enable_shared(size_t bytes) {
  static size_t granted = 0;
  if (bytes <= granted) return;
  cudaFuncAttributes a{};
  CUDA_OK(cudaFuncGetAttributes(&a, k_mark));
  int maxopt = 0;
  CUDA_OK(cudaDeviceGetAttribute(&maxopt, cudaDevAttrMaxSharedMemoryPerBlockOptin, 0));
  if (bytes > (size_t)maxopt) {
    fprintf(stderr, "FATAL: --seg needs %zu B of shared memory, device allows %d\n",
            bytes, maxopt);
    abort();
  }
  CUDA_OK(cudaFuncSetAttribute(k_mark, cudaFuncAttributeMaxDynamicSharedMemorySize,
                               (int)bytes));
  granted = bytes;
}

// ---------------------------------------------------------------- host replay

std::vector<uint32_t> mark_reference(const std::vector<uint32_t> &off_in,
                                     const std::vector<uint32_t> &step,
                                     size_t ncov, uint32_t n,
                                     std::vector<uint32_t> &off_out,
                                     uint64_t &alive_after_covers, uint64_t &nsurv) {
  const size_t nall = step.size(), nw = (n + 31u) / 32;
  off_out = off_in;
  std::vector<uint32_t> bits(nw, ~0u);
  if (n & 31u) bits[nw - 1] = (1u << (n & 31u)) - 1;

  auto mark = [&](size_t from, size_t to) {
    for (size_t i = from; i < to; ++i) {
      const uint32_t s = step[i];
      if (!s) continue;
      uint32_t j = off_out[i];
      for (; j < n; j += s) bits[j >> 5] &= ~(1u << (j & 31u));
      off_out[i] = j - n;
    }
  };
  mark(0, ncov);
  alive_after_covers = 0;
  for (size_t w = 0; w < nw; ++w) alive_after_covers += (uint64_t)__builtin_popcount(bits[w]);
  mark(ncov, nall);
  nsurv = 0;
  for (size_t w = 0; w < nw; ++w) nsurv += (uint64_t)__builtin_popcount(bits[w]);
  return bits;
}

// ---------------------------------------------------------------- device, one lane

std::vector<uint32_t> mark_device(const std::vector<uint32_t> &off_in,
                                  const std::vector<uint32_t> &step,
                                  size_t ncov, uint32_t n, uint32_t seg,
                                  std::vector<uint32_t> &off_out,
                                  uint64_t &alive_after_covers, uint64_t &nsurv,
                                  uint32_t nthreads) {
  const uint32_t nall = (uint32_t)step.size();
  const uint32_t nwseg = (seg + 31u) / 32, nw = (n + 31u) / 32;
  const size_t shmem = (size_t)nwseg * 4;
  mark_enable_shared(shmem);

  uint32_t *d_off = nullptr, *d_step = nullptr, *d_n = nullptr, *d_surv = nullptr;
  uint32_t *d_bits = nullptr;
  uint64_t *d_alive = nullptr;
  CUDA_OK(cudaMalloc(&d_off, (size_t)nall * 4));
  CUDA_OK(cudaMalloc(&d_step, (size_t)nall * 4));
  CUDA_OK(cudaMalloc(&d_n, 4));
  CUDA_OK(cudaMalloc(&d_surv, 4));
  CUDA_OK(cudaMalloc(&d_bits, (size_t)nwseg * 4));
  CUDA_OK(cudaMalloc(&d_alive, 8));
  CUDA_OK(cudaMemcpy(d_off, off_in.data(), (size_t)nall * 4, cudaMemcpyHostToDevice));
  CUDA_OK(cudaMemcpy(d_step, step.data(), (size_t)nall * 4, cudaMemcpyHostToDevice));
  CUDA_OK(cudaMemcpy(d_n, &n, 4, cudaMemcpyHostToDevice));

  k_mark<<<1, nthreads, shmem>>>(d_off, d_step, nall, (uint32_t)ncov, nwseg, d_n,
                                 d_bits, d_alive, d_surv);
  CUDA_OK(cudaGetLastError());
  CUDA_OK(cudaDeviceSynchronize());

  std::vector<uint32_t> bits(nw);
  off_out.resize(nall);
  uint32_t surv32 = 0;
  CUDA_OK(cudaMemcpy(bits.data(), d_bits, (size_t)nw * 4, cudaMemcpyDeviceToHost));
  CUDA_OK(cudaMemcpy(off_out.data(), d_off, (size_t)nall * 4, cudaMemcpyDeviceToHost));
  CUDA_OK(cudaMemcpy(&alive_after_covers, d_alive, 8, cudaMemcpyDeviceToHost));
  CUDA_OK(cudaMemcpy(&surv32, d_surv, 4, cudaMemcpyDeviceToHost));
  nsurv = surv32;

  CUDA_OK(cudaFree(d_off));
  CUDA_OK(cudaFree(d_step));
  CUDA_OK(cudaFree(d_n));
  CUDA_OK(cudaFree(d_surv));
  CUDA_OK(cudaFree(d_bits));
  CUDA_OK(cudaFree(d_alive));
  return bits;
}

// ---------------------------------------------------------------- segment sweep

// One segment for every lane, timed. The CPU's --seg optimum is 32 KB because that is
// its L1; the device's constraint is different (shared memory per block against
// occupancy), so the number is re-measured rather than inherited.
double mark_bench(uint64_t M, uint64_t mmax, uint64_t lo, uint64_t sieve_p,
                  uint32_t seg, uint32_t nthreads, int reps, uint64_t &positions) {
  ref3_build_wheel(M, mmax);
  std::vector<uint64_t> mods, ress;
  size_t ncov = 0;
  ref3_filters(M, mmax, nullptr, sieve_p, mods, ress, ncov);
  const uint32_t nall = (uint32_t)mods.size();
  const uint32_t nlanes = ref3_wheel_size();
  const uint32_t nwseg = (seg + 31u) / 32;

  std::vector<uint32_t> off((size_t)nlanes * nall), step((size_t)nlanes * nall);
  std::vector<uint32_t> nseg(nlanes, seg);
#pragma omp parallel for schedule(dynamic, 1)
  for (int li = 0; li < (int)nlanes; ++li) {
    const uint64_t rho = ref3_wheel((uint32_t)li);
    const uint64_t first = lo + ((rho + M - lo % M) % M);
    std::vector<uint32_t> o, s;
    ref3_lane_offsets(M, first, mods, ress, o, s);
    std::copy(o.begin(), o.end(), off.begin() + (size_t)li * nall);
    std::copy(s.begin(), s.end(), step.begin() + (size_t)li * nall);
  }

  const size_t shmem = (size_t)nwseg * 4;
  mark_enable_shared(shmem);

  uint32_t *d_off = nullptr, *d_step = nullptr, *d_n = nullptr, *d_surv = nullptr;
  uint32_t *d_bits = nullptr;
  uint64_t *d_alive = nullptr;
  CUDA_OK(cudaMalloc(&d_off, off.size() * 4));
  CUDA_OK(cudaMalloc(&d_step, step.size() * 4));
  CUDA_OK(cudaMalloc(&d_n, (size_t)nlanes * 4));
  CUDA_OK(cudaMalloc(&d_surv, (size_t)nlanes * 4));
  CUDA_OK(cudaMalloc(&d_bits, (size_t)nlanes * nwseg * 4));
  CUDA_OK(cudaMalloc(&d_alive, (size_t)nlanes * 8));
  CUDA_OK(cudaMemcpy(d_step, step.data(), step.size() * 4, cudaMemcpyHostToDevice));
  CUDA_OK(cudaMemcpy(d_n, nseg.data(), (size_t)nlanes * 4, cudaMemcpyHostToDevice));

  cudaEvent_t t0, t1;
  CUDA_OK(cudaEventCreate(&t0));
  CUDA_OK(cudaEventCreate(&t1));
  double best = 1e30;
  for (int r = 0; r < reps; ++r) {
    // Fresh offsets each repetition: k_mark carries them, so a second run over the same
    // buffer would mark a different segment and the timings would not be comparable.
    CUDA_OK(cudaMemcpy(d_off, off.data(), off.size() * 4, cudaMemcpyHostToDevice));
    CUDA_OK(cudaEventRecord(t0));
    k_mark<<<nlanes, nthreads, shmem>>>(d_off, d_step, nall, (uint32_t)ncov, nwseg, d_n,
                                        d_bits, d_alive, d_surv);
    CUDA_OK(cudaEventRecord(t1));
    CUDA_OK(cudaGetLastError());
    CUDA_OK(cudaEventSynchronize(t1));
    float ms = 0;
    CUDA_OK(cudaEventElapsedTime(&ms, t0, t1));
    best = std::min(best, (double)ms / 1e3);
  }
  positions = (uint64_t)nlanes * seg;

  CUDA_OK(cudaEventDestroy(t0));
  CUDA_OK(cudaEventDestroy(t1));
  CUDA_OK(cudaFree(d_off));
  CUDA_OK(cudaFree(d_step));
  CUDA_OK(cudaFree(d_n));
  CUDA_OK(cudaFree(d_surv));
  CUDA_OK(cudaFree(d_bits));
  CUDA_OK(cudaFree(d_alive));
  return best;
}
