// The driver: one segment of every lane per device round, with the host tail for round s
// overlapped against the device work for round s+1.
//
// The overlap is the task the whole port turns on. Stage D is 7.8% of the CPU scanner's
// work; once marking is twenty times faster it is 63% of the total, so a good kernel
// behind an unoverlapped tail reads as a bad port. Everything else here is bookkeeping.
//
// DETERMINISM. The device visits segments in ascending order and lanes in wheel order,
// and compaction places every candidate by prefix sum, so the candidate stream is a pure
// function of the inputs. Certificates are buffered PER LANE and printed in lane order at
// the end, which reproduces the CPU scanner's emission order exactly -- the same thing
// rung_scan3 gets by merging its per-lane buffers after the parallel region. The gate
// checks the sorted sets (which is all correctness requires) and reports whether the raw
// order matched too.
#include "scan.cuh"
#include "common.cuh"
#include "mark.cuh"
#include "survivors.cuh"
#include "ref_host.h"

#include <cuda_runtime.h>
#include <algorithm>
#include <chrono>
#include <cinttypes>
#include <cstdio>
#include <cstring>
#include <string>
#include <vector>

namespace {

struct Stats3 {
  uint64_t positions = 0, covered = 0, sieved = 0, mr_calls = 0, composite = 0,
           rung_solved = 0, direct_solved = 0, survivors = 0;
};

// n for this segment, per lane, computed on the device so the host never uploads a
// per-segment array. npos differs between lanes by at most one position, so the segment
// boundaries line up with the CPU's per-lane segmentation exactly.
__global__ void k_seg_n(const uint64_t *npos, uint64_t base, uint32_t seg,
                        uint32_t nlanes, uint32_t *nlane) {
  const uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= nlanes) return;
  const uint64_t np = npos[i];
  nlane[i] = base >= np ? 0u : (uint32_t)(np - base < seg ? np - base : seg);
}

double now_s() {
  using namespace std::chrono;
  return duration<double>(steady_clock::now().time_since_epoch()).count();
}

}  // namespace

int scan_run(const ScanOpts &o) {
  const double t_start = now_s();
  uint64_t lo = o.lo, hi = o.hi;

  // Shard arithmetic mirrors rung_scan3, which mirrors rung_scan.cpp, so all three tile
  // a range the same way. An empty shard still prints its SUMMARY.
  uint64_t nshard = o.nshard ? o.nshard : 1;
  if (nshard > 1) {
    const uint64_t span = (hi - lo + nshard - 1) / nshard;
    const uint64_t a = lo + o.shard * span, b = std::min(hi, a + span);
    lo = a;
    hi = std::max(a, b);
  }

  ref3_build_wheel(o.M, o.mmax);
  const uint32_t nlanes = ref3_wheel_size();

  // The class table is the whole of layer L1. A binary whose derivation drifted from the
  // published table must REFUSE to run, not warn.
  uint64_t pin_fnv = 0;
  const char *pin_sha = nullptr;
  const bool pinned = ref3_class_pinned(o.M, &pin_fnv, &pin_sha);
  if (pinned) {
    const uint64_t got = ref3_class_fnv64_of(o.M);
    if (got != pin_fnv) {
      fprintf(stderr, "FATAL: class table digest mismatch at M=%" PRIu64 " -- derived "
              "fnv64 %016" PRIx64 ", expected %016" PRIx64 "\n", o.M, got, pin_fnv);
      return 1;
    }
  } else {
    fprintf(stderr, "[sieve_cuda] WARNING: no pinned digest for M=%" PRIu64
            " -- running UNPINNED\n", o.M);
  }
  // Same ceiling rung_scan3 enforces, for the same reason: above it stage E silently
  // stops firing and a prime stage D misses is reported as a SURVIVOR, which reads as a
  // mathematical result rather than the numeric limit it is.
  if (hi > ref3_direct_pmax()) {
    fprintf(stderr, "FATAL: HI must not exceed %" PRIu64 ", the stage-E ceiling\n",
            ref3_direct_pmax());
    return 1;
  }
  const uint64_t wheel_pmin = ref3_wheel_pmin(), cover_pmax = ref3_cover_pmax(o.mmax);
  if (lo <= std::max(wheel_pmin, cover_pmax)) {
    fprintf(stderr, "FATAL: LO must exceed max pmin of the filter table (%" PRIu64 "); "
            "the region below is settled by the existing censuses\n",
            std::max(wheel_pmin, cover_pmax));
    return 1;
  }

  std::vector<uint64_t> mods, ress;
  size_t ncov = 0;
  ref3_filters(o.M, o.mmax, o.filters, o.sieve_p, mods, ress, ncov);
  const uint32_t nall = (uint32_t)mods.size();

  // FNV-1a over the CERTIFIED filters only, reported on stderr exactly as rung_scan3
  // reports it -- it ties the binary to the table it actually read.
  uint64_t fhash = 14695981039346656037ull;
  for (size_t i = 0; i < ncov; ++i) {
    char b[48];
    const int n = snprintf(b, sizeof b, "%" PRIu64 ",%" PRIu64 ";", mods[i], ress[i]);
    for (int j = 0; j < n; ++j) { fhash ^= (unsigned char)b[j]; fhash *= 1099511628211ull; }
  }

  ref3_build_tables(hi);       // init_phi + init_rfac + build_base_primes, ONCE

  // ------------------------------------------------------------ lane plans
  std::vector<uint64_t> first(nlanes), npos(nlanes);
  uint64_t maxnpos = 0;
  for (uint32_t i = 0; i < nlanes; ++i) {
    const uint64_t rho = ref3_wheel(i);
    const uint64_t f = lo + ((rho + o.M - lo % o.M) % o.M);
    first[i] = f;
    npos[i] = f >= hi ? 0 : (hi - f + o.M - 1) / o.M;
    maxnpos = std::max(maxnpos, npos[i]);
  }

  // Offsets stay on the HOST: one setup per (lane, filter) per shard, and keeping it off
  // the device removes a whole class of divergence. It is not free, though -- --profile
  // measured it at 79% of the run before the extended Euclid was hoisted out of the lane
  // loop, which is why ref3_prep_offsets exists.
  const double t_off0 = now_s();
  std::vector<uint32_t> prep;
  ref3_prep_offsets(o.M, mods, ress, lo, prep);  // the extended Euclid, once per filter
  std::vector<uint32_t> off((size_t)nlanes * nall), step((size_t)nlanes * nall);
#pragma omp parallel for schedule(static)
  for (int li = 0; li < (int)nlanes; ++li)
    ref3_lane_offsets_fast(prep, first[li] - lo, off.data() + (size_t)li * nall,
                           step.data() + (size_t)li * nall, nall);
  const double t_off = now_s() - t_off0;

  // Marks, i.e. atomicAnd executions, for the device cost model t = marks*c_m +
  // survivors*c_s. Counted exactly rather than estimated as npos/step: a filter whose
  // first hit lies past the end of the lane contributes nothing, and at large moduli
  // that is a large fraction of them. Only under --profile -- it is a second pass over
  // 149M step entries.
  uint64_t marks = 0;
  if (o.profile) {
#pragma omp parallel for schedule(static) reduction(+ : marks)
    for (int li = 0; li < (int)nlanes; ++li) {
      const uint32_t *st_l = step.data() + (size_t)li * nall;
      const uint32_t *of_l = off.data() + (size_t)li * nall;
      const uint64_t np = npos[li];
      uint64_t m = 0;
      for (uint32_t i = 0; i < nall; ++i)
        if (st_l[i] && of_l[i] < np) m += (np - 1 - of_l[i]) / st_l[i] + 1;
      marks += m;
    }
  }

  fprintf(stderr, "[sieve_cuda] wheel M=%" PRIu64 " lanes=%u (spacing %.0f)  covers=%zu "
          "(mmax=%" PRIu64 ")  seg=%" PRIu64 "  wheel_pmin=%" PRIu64 "\n"
          "[sieve_cuda] filters=%zu fnv64=%016" PRIx64 " source=%s sieve=%" PRIu64
          " (+%u primes)  offsets=%.2f s\n",
          o.M, nlanes, (double)o.M / nlanes, ncov, o.mmax, o.seg, wheel_pmin,
          ncov, fhash, o.filters ? o.filters : "generated", o.sieve_p,
          nall - (uint32_t)ncov, t_off);

  // ------------------------------------------------------------ device state
  const uint32_t seg = (uint32_t)(o.seg ? o.seg : 262144);
  const uint32_t nwseg = (seg + 31u) / 32;
  const size_t shmem = (size_t)nwseg * 4;
  mark_enable_shared(shmem);

  uint32_t *d_off = nullptr, *d_step = nullptr, *d_n = nullptr, *d_surv = nullptr;
  uint32_t *d_bits = nullptr;
  uint64_t *d_alive = nullptr, *d_first = nullptr, *d_npos = nullptr;
  CUDA_OK(cudaMalloc(&d_off, off.size() * 4));
  CUDA_OK(cudaMalloc(&d_step, step.size() * 4));
  CUDA_OK(cudaMalloc(&d_n, (size_t)nlanes * 4));
  CUDA_OK(cudaMalloc(&d_surv, (size_t)nlanes * 4));
  CUDA_OK(cudaMalloc(&d_bits, (size_t)nlanes * nwseg * 4));
  CUDA_OK(cudaMalloc(&d_alive, (size_t)nlanes * 8));
  CUDA_OK(cudaMalloc(&d_first, (size_t)nlanes * 8));
  CUDA_OK(cudaMalloc(&d_npos, (size_t)nlanes * 8));
  CUDA_OK(cudaMemcpy(d_off, off.data(), off.size() * 4, cudaMemcpyHostToDevice));
  CUDA_OK(cudaMemcpy(d_step, step.data(), step.size() * 4, cudaMemcpyHostToDevice));
  CUDA_OK(cudaMemcpy(d_first, first.data(), (size_t)nlanes * 8, cudaMemcpyHostToDevice));
  CUDA_OK(cudaMemcpy(d_npos, npos.data(), (size_t)nlanes * 8, cudaMemcpyHostToDevice));

  SurvivorWork w;
  survivors_alloc(w, nlanes, 1u << 20);

  // One round's readback, double-buffered: round s is on the host tail while round s+1
  // runs on the device, so each needs its own landing area.
  struct Round {
    std::vector<uint64_t> cand;
    std::vector<uint8_t> prime;
    std::vector<uint32_t> base, surv;
    std::vector<uint64_t> alive, nl;
    uint32_t total = 0;
    bool live = false;
  } rounds[2];

  Stats3 st;
  double t_wait = 0, t_tail = 0, t_launch = 0;
  const bool have_sieve = ncov != nall;

  auto launch = [&](uint64_t base) {
    k_seg_n<<<(nlanes + 255) / 256, 256>>>(d_npos, base, seg, nlanes, d_n);
    k_mark<<<nlanes, o.block, shmem>>>(d_off, d_step, nall, (uint32_t)ncov, nwseg, d_n,
                                       d_bits, d_alive, d_surv);
    k_scan_counts<<<1, 256>>>(d_surv, nlanes, w.d_base, w.d_total);
    k_compact<<<nlanes, o.block>>>(d_bits, nwseg, d_n, d_first, o.M, base, w.d_base,
                                   w.cap, w.d_cand, w.d_over);
    k_primality<<<2048, 128>>>(w.d_cand, w.d_total, w.cap, w.d_prime);
    CUDA_OK(cudaGetLastError());
  };

  // Compaction is the one stage whose output size is not known in advance. Rather than
  // guess a capacity, overflow is DETECTED and the round is replayed at twice the size:
  // k_mark has already carried its offsets, so only compaction and primality re-run --
  // the bitset for this segment is still sitting in d_bits.
  auto collect = [&](uint64_t base, Round &r) {
    uint32_t over = 1;
    while (over) {
      CUDA_OK(cudaMemcpy(&r.total, w.d_total, 4, cudaMemcpyDeviceToHost));
      CUDA_OK(cudaMemcpy(&over, w.d_over, 4, cudaMemcpyDeviceToHost));
      if (!over) break;
      const uint32_t want = std::max(r.total, w.cap * 2);
      fprintf(stderr, "[sieve_cuda] compaction cap %u -> %u\n", w.cap, want);
      survivors_alloc(w, nlanes, want);
      CUDA_OK(cudaMemset(w.d_over, 0, 4));
      k_scan_counts<<<1, 256>>>(d_surv, nlanes, w.d_base, w.d_total);
      k_compact<<<nlanes, o.block>>>(d_bits, nwseg, d_n, d_first, o.M, base, w.d_base,
                                     w.cap, w.d_cand, w.d_over);
      k_primality<<<2048, 128>>>(w.d_cand, w.d_total, w.cap, w.d_prime);
      CUDA_OK(cudaGetLastError());
    }
    r.cand.resize(r.total);
    r.prime.resize(r.total);
    r.base.resize(nlanes);
    r.surv.resize(nlanes);
    r.alive.resize(nlanes);
    r.nl.resize(nlanes);
    std::vector<uint32_t> nl32(nlanes);
    if (r.total) {
      CUDA_OK(cudaMemcpy(r.cand.data(), w.d_cand, (size_t)r.total * 8, cudaMemcpyDeviceToHost));
      CUDA_OK(cudaMemcpy(r.prime.data(), w.d_prime, r.total, cudaMemcpyDeviceToHost));
    }
    CUDA_OK(cudaMemcpy(r.base.data(), w.d_base, (size_t)nlanes * 4, cudaMemcpyDeviceToHost));
    CUDA_OK(cudaMemcpy(r.surv.data(), d_surv, (size_t)nlanes * 4, cudaMemcpyDeviceToHost));
    CUDA_OK(cudaMemcpy(r.alive.data(), d_alive, (size_t)nlanes * 8, cudaMemcpyDeviceToHost));
    CUDA_OK(cudaMemcpy(nl32.data(), d_n, (size_t)nlanes * 4, cudaMemcpyDeviceToHost));
    for (uint32_t i = 0; i < nlanes; ++i) r.nl[i] = nl32[i];
    r.live = true;
  };

  // Stage D and stage E on the host, in the CPU scanner's own code. The candidates are
  // independent, so the pool is a plain parallel-for; results land in per-candidate slots
  // and are folded into the per-lane buffers afterwards, in index order, which is what
  // keeps the emission stream independent of the schedule.
  auto tail = [&](Round &r) {
    if (!r.live) return;
    for (uint32_t i = 0; i < nlanes; ++i) {
      const uint64_t n = r.nl[i];
      st.positions += n;
      if (!have_sieve) {
        st.covered += n - r.surv[i];
      } else {
        st.covered += n - r.alive[i];
        st.sieved += r.alive[i] - r.surv[i];
      }
    }
    st.mr_calls += r.total;
    if (r.total) {
      std::vector<std::string> lines(r.total);
      uint64_t comp = 0, rung = 0, direct = 0, surv = 0;
#pragma omp parallel for schedule(dynamic, 16) reduction(+ : comp, rung, direct, surv)
      for (int64_t i = 0; i < (int64_t)r.total; ++i) {
        if (!r.prime[i]) { ++comp; continue; }
        std::string s;
        if (ref3_solve_rung(r.cand[i], s)) ++rung;
        else if (ref3_solve_direct(r.cand[i], s)) ++direct;
        else { ++surv; s = "SURVIVOR " + std::to_string(r.cand[i]) + "\n"; }
        lines[i] = std::move(s);
      }
      st.composite += comp;
      st.rung_solved += rung;
      st.direct_solved += direct;
      st.survivors += surv;
      // STREAMED, not buffered. The old fold pushed every certificate into per-lane
      // vectors and printed them after the last round -- fine at 10^15 (390k lines per
      // shard), fatal at 10^19, where a shard holds ~10^8 lines and the buffers alone
      // exceed the box's 48 GB. Emitting here, per round in lane order, is equally
      // deterministic (rounds advance in order, lanes in wheel order, candidates by
      // prefix-sum position), so repeated runs stay byte-identical; what changes is the
      // ORDER relative to the CPU scanner (round-major here, lane-major there), which
      // the gates already treat as cosmetic -- diff_cuda.sh sorts, and reports the raw
      // comparison as information.
      for (uint32_t l = 0; l < nlanes; ++l)
        for (uint32_t k = 0; k < r.surv[l]; ++k) {
          const std::string &s = lines[r.base[l] + k];
          if (!s.empty() && (o.emit_surv || s.compare(0, 9, "SURVIVOR ") != 0))
            fputs(s.c_str(), stdout);
        }
    }
    r.live = false;
  };

  // ------------------------------------------------------------ the pipeline
  const uint64_t nrounds = (maxnpos + seg - 1) / seg;
  if (nrounds) { const double l = now_s(); launch(0); t_launch += now_s() - l; }
  for (uint64_t s = 0; s < nrounds; ++s) {
    const double a = now_s();
    collect(s * seg, rounds[s & 1]);        // blocks until round s is on the host
    t_wait += now_s() - a;
    if (s + 1 < nrounds) {                  // round s+1 runs DURING the tail below
      const double l = now_s();
      launch((s + 1) * seg);
      t_launch += now_s() - l;
    }
    const double b = now_s();
    tail(rounds[s & 1]);
    t_tail += now_s() - b;
    if (o.progress && (s + 1) % o.progress == 0)
      fprintf(stderr, "[sieve_cuda] segment %" PRIu64 "/%" PRIu64 " done\n", s + 1, nrounds);
  }

  survivors_free(w);
  CUDA_OK(cudaFree(d_off));
  CUDA_OK(cudaFree(d_step));
  CUDA_OK(cudaFree(d_n));
  CUDA_OK(cudaFree(d_surv));
  CUDA_OK(cudaFree(d_bits));
  CUDA_OK(cudaFree(d_alive));
  CUDA_OK(cudaFree(d_first));
  CUDA_OK(cudaFree(d_npos));

  printf("SUMMARY lo=%" PRIu64 " hi=%" PRIu64 " wheelM=%" PRIu64 " lanes=%u mmax=%" PRIu64
         " ucert_rmax=63 ucert_umax=64 wheel_pmin=%" PRIu64 " classes_sha256=%s "
         "positions=%" PRIu64 " covered=%" PRIu64 " sieved=%" PRIu64 " mr=%" PRIu64
         " composite=%" PRIu64 " rung=%" PRIu64 " direct=%" PRIu64 " survivors=%" PRIu64 "\n",
         lo, hi, o.M, nlanes, o.mmax, wheel_pmin, pinned ? pin_sha : "unpinned",
         st.positions, st.covered, st.sieved, st.mr_calls, st.composite,
         st.rung_solved, st.direct_solved, st.survivors);

  const double wall = now_s() - t_start;
  if (o.quiet_time) fprintf(stderr, "%.3f\n", wall);
  if (o.profile)
    fprintf(stderr, "[sieve_cuda] PROFILE wall=%.3f s  offsets=%.3f  launch=%.3f  "
            "device-wait=%.3f  host-tail=%.3f  rounds=%" PRIu64
            "  marks=%" PRIu64 "  survivors_in=%" PRIu64 "  device=%.3f\n",
            wall, t_off, t_launch, t_wait, t_tail, nrounds, marks, st.mr_calls,
            t_launch + t_wait);
  return st.survivors ? 2 : 0;
}
