#pragma once
#include <cstdint>
#include <vector>

// Survivor compaction and primality.
//
// Compaction is a COUNT -> EXCLUSIVE SCAN -> write-into-your-own-slice, never an atomic.
// An atomicAdd would hand out slots in arrival order, making the compacted list a
// function of the schedule; a scan makes every write index a pure function of the
// inputs. That is the difference between "deterministic in practice" and "deterministic
// by construction", and the certificate stream depends on the second.
//
// Primality is the reference's own test, ported operation for operation: base-2 SPRP
// first (conclusive on failure, and nearly every survivor is composite), then the
// deterministic 12-base Miller-Rabin over {2,3,5,7,11,13,17,19,23,29,31,37}.

// Everything one segment needs on the device, for one launch shape (nlanes blocks).
// Owns nothing the caller has to size by guesswork except `cap`, which grows on demand.
struct SurvivorWork {
  uint32_t *d_base = nullptr;    // [nlanes]   exclusive scan of the per-lane counts
  uint32_t *d_total = nullptr;   // [1]        total survivors this segment
  uint32_t *d_over = nullptr;    // [1]        set if cap was too small
  uint64_t *d_cand = nullptr;    // [cap]      compacted candidates, p not index
  uint8_t  *d_prime = nullptr;   // [cap]      1 iff the reference would call it prime
  uint32_t cap = 0;
  uint32_t nlanes = 0;
};

void survivors_alloc(SurvivorWork &w, uint32_t nlanes, uint32_t cap);
void survivors_free(SurvivorWork &w);

// One block: the per-lane counts scanned into per-lane bases. A single block is what
// makes the running offset trivially sequential -- no multi-block scan, nothing whose
// result could depend on block scheduling.
__global__ void k_scan_counts(const uint32_t *cnt, uint32_t nlanes, uint32_t *base,
                              uint32_t *total);

// One block per lane; ascending by position inside a lane, lanes in wheel order. Writes
// past `cap` are dropped and flagged in `over` -- the caller grows and replays.
__global__ void k_compact(const uint32_t *bits, uint32_t nwseg, const uint32_t *nlane,
                          const uint64_t *firsts, uint64_t M, uint64_t segbase,
                          const uint32_t *lbase, uint32_t cap,
                          uint64_t *cand, uint32_t *over);

// Grid-stride over a count read from DEVICE memory, so no sync is needed between
// compaction and primality.
__global__ void k_primality(const uint64_t *cand, const uint32_t *ntot, uint32_t cap,
                            uint8_t *prime);

// --check-primality: classify a candidate list on the device, one thread each.
std::vector<uint8_t> primality_device(const std::vector<uint64_t> &cand);

// --check-compact: mark ONE segment of one lane, compact it on the device, and return
// the candidates in the order the device produced them. The caller compares against the
// host's own bit-extraction -- element for element, not as a set, because the ORDER is
// the property under test.
std::vector<uint64_t> survivors_device(const std::vector<uint32_t> &off_in,
                                       const std::vector<uint32_t> &step,
                                       size_t ncov, uint32_t n, uint32_t seg,
                                       uint64_t first, uint64_t M, uint64_t base,
                                       uint32_t nthreads,
                                       std::vector<uint8_t> &prime_out);
