#pragma once
#include <cstdint>
#include <vector>

// The marking stage: 91% of the CPU scanner's work, and the reason this port exists.
//
// One block per lane, the segment bitset in SHARED memory, cleared with atomicAnd.
//
// The bitset is 32-BIT words, not 64. Measured on the L4 in exactly this access pattern,
// 32-bit shared atomicAnd is 2.27x faster than 64-bit (36.2 ms against 81.9 ms, each at
// its own best block size, identical results). Two reasons compound: a 64-bit shared
// access spans two banks so it conflicts twice as often, and the word halves the
// segment's shared footprint, which lifts occupancy. k_mark is 99.1% of all GPU time
// here, so this is the port's single largest lever.
// The atomics need no coordination: marking only ever CLEARS bits, so it is idempotent
// and order-independent -- two threads clearing the same bit cannot disagree and no
// execution order is observable in the result. That is what makes the kernel
// deterministic without any ordering machinery.

// One lane's sweep plan, in the CPU scanner's own terms: the first position of the lane
// at or above lo, and how many positions the lane holds below hi.
struct LanePlan { uint64_t first, npos; };

// The kernel itself, so the driver and the survivor stage can launch it directly. off is
// IN/OUT: on return it holds the carried offsets for the next segment.
__global__ void k_mark(uint32_t *off, const uint32_t *step, uint32_t nall, uint32_t ncov,
                       uint32_t nwseg, const uint32_t *nlane,
                       uint32_t *bits_out, uint64_t *alive_out, uint32_t *surv_out);

// Opts the kernel into more than 48 KB of dynamic shared memory when --seg needs it.
void mark_enable_shared(size_t bytes);

// The CPU scanner's marking, replayed on the host over ONE segment of n positions.
// Consumes the offsets ref3_lane_offsets produced; returns the bitset, the carried
// offsets (off[i] = j - n, exactly as sweep_lane does) and the popcount taken BETWEEN
// stage B and stage B' -- `covered` and `sieved` must stay separable.
std::vector<uint32_t> mark_reference(const std::vector<uint32_t> &off_in,
                                     const std::vector<uint32_t> &step,
                                     size_t ncov, uint32_t n,
                                     std::vector<uint32_t> &off_out,
                                     uint64_t &alive_after_covers, uint64_t &nsurv);

// The same segment marked by k_mark on the device. Same inputs, same three outputs, so
// --check-mark can compare all three word for word.
std::vector<uint32_t> mark_device(const std::vector<uint32_t> &off_in,
                                  const std::vector<uint32_t> &step,
                                  size_t ncov, uint32_t n, uint32_t seg,
                                  std::vector<uint32_t> &off_out,
                                  uint64_t &alive_after_covers, uint64_t &nsurv,
                                  uint32_t nthreads);

// Marks one segment for every lane, repeatedly; returns the best wall time in seconds
// and the number of positions marked per repetition. Used by --bench-mark to pick --seg
// on the device rather than inheriting the CPU's L1-driven 32 KB.
double mark_bench(uint64_t M, uint64_t mmax, uint64_t lo, uint64_t sieve_p,
                  uint32_t seg, uint32_t nthreads, int reps, uint64_t &positions);
