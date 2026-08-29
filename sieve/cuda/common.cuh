#pragma once
#include <cstdint>
#include <cstdio>
#include <cstdlib>

#define CUDA_OK(call) do {                                                        \
    cudaError_t _e = (call);                                                      \
    if (_e != cudaSuccess) {                                                      \
      fprintf(stderr, "FATAL: %s:%d %s\n", __FILE__, __LINE__,                    \
              cudaGetErrorString(_e));                                            \
      abort();                                                                    \
    }                                                                             \
  } while (0)

// 64-bit modmul via __uint128_t. The L4 has no 64x64->128 multiply-high instruction, so
// this is EMULATED and is the device's weak operation -- which is why survivors are
// expensive here while marking is cheap, and why the small-prime sieve's economics may
// invert relative to the CPU (spec S2, tested in Task 6).
__device__ __forceinline__ uint64_t mulmod64(uint64_t a, uint64_t b, uint64_t m) {
  return (uint64_t)((unsigned __int128)a * b % m);
}

__device__ __forceinline__ uint64_t powmod64(uint64_t b, uint64_t e, uint64_t m) {
  uint64_t r = 1;
  b %= m;
  while (e) {
    if (e & 1) r = mulmod64(r, b, m);
    b = mulmod64(b, b, m);
    e >>= 1;
  }
  return r;
}

// Block-wide sum, valid in thread 0. A sum is order-invariant over the integers, so this
// is one of the few places where the result cannot depend on the schedule -- unlike
// survivor PLACEMENT, which must be a prefix sum rather than an atomic.
//
// Every thread in the block must reach this call: it contains __syncthreads(). The
// shared scratch is function-static, so the leading barrier is what makes a second call
// safe while the first call's readers may still be in flight.
__device__ __forceinline__ uint32_t block_reduce_sum(uint32_t v) {
  __shared__ uint32_t warpsum[32];
  const uint32_t lane = threadIdx.x & 31u, wid = threadIdx.x >> 5;
  __syncthreads();
  for (int o = 16; o; o >>= 1) v += __shfl_down_sync(0xffffffffu, v, o);
  if (lane == 0) warpsum[wid] = v;
  __syncthreads();
  const uint32_t nwarp = (blockDim.x + 31u) >> 5;
  v = threadIdx.x < nwarp ? warpsum[threadIdx.x] : 0u;
  if (wid == 0)
    for (int o = 16; o; o >>= 1) v += __shfl_down_sync(0xffffffffu, v, o);
  return v;
}

// Block-wide EXCLUSIVE prefix sum; every thread gets its own base, and `total` (valid in
// every thread) is the block's sum.
//
// This is the primitive survivor compaction rests on, and the reason it is a scan and
// not an atomicAdd: an atomic hands out slots in arrival order, so the compacted list
// would be a function of the schedule. A scan makes every write index a pure function of
// the inputs, which is what "deterministic by construction" means here. Same barrier
// discipline as above: uniform entry, leading __syncthreads().
__device__ __forceinline__ uint32_t block_exclusive_scan(uint32_t v, uint32_t &total) {
  __shared__ uint32_t warpsum[32];
  __shared__ uint32_t blocktotal;
  const uint32_t lane = threadIdx.x & 31u, wid = threadIdx.x >> 5;
  const uint32_t nwarp = (blockDim.x + 31u) >> 5;    // <= 32, since blockDim <= 1024
  __syncthreads();

  uint32_t x = v;                                    // inclusive scan within the warp
  for (int o = 1; o < 32; o <<= 1) {
    const uint32_t y = __shfl_up_sync(0xffffffffu, x, o);
    if (lane >= (uint32_t)o) x += y;
  }
  if (lane == 31u || threadIdx.x == blockDim.x - 1) warpsum[wid] = x;
  __syncthreads();

  if (wid == 0) {                                    // scan the warp totals in warp 0
    const uint32_t w = threadIdx.x < nwarp ? warpsum[threadIdx.x] : 0u;
    uint32_t s = w;
    for (int o = 1; o < 32; o <<= 1) {
      const uint32_t y = __shfl_up_sync(0xffffffffu, s, o);
      if (lane >= (uint32_t)o) s += y;
    }
    if (threadIdx.x == nwarp - 1) blocktotal = s;
    __syncwarp();
    if (threadIdx.x < nwarp) warpsum[threadIdx.x] = s - w;
  }
  __syncthreads();

  total = blocktotal;
  return warpsum[wid] + (x - v);
}
