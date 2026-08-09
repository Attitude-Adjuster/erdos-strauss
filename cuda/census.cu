// census.cu -- GPU census: rung wavefront, CPU handoff, statistics (Task 5).
//
// SHAPE, AND WHY. The active list is created by the device sieve (Task 6) and stays on
// the device through the whole rung ladder: factorize (Task 3), classify (Task 4),
// tally + compact (the fused classify-tally kernel + two cub::DeviceSelect::Flagged,
// see classify.cuh). Per rung the host
// receives ONE Readback struct and, only when something is actually emitted, the
// handful of encoded certificate positions. Nothing per-position crosses PCIe.
//
// This arrived in two steps on purpose. Task 5 did the bookkeeping on the host, which
// made the statistics literally the reference's own record_hit/record_fail logic and
// so could not drift from it; that bought correctness first. Profiling then showed the
// host loop at 47.8% of runtime, which is what justified moving it -- the plan's Step
// 3/5, done when there was a number behind it rather than on faith.
//
// Determinism survives the move: the tallies are integer sums and a min, whose VALUE
// does not depend on block order, and both compactions are stable. Allocation still
// never uses atomics -- that remains count -> prefix-sum -> slice.
//
// The CPU handoff threshold is a pure performance knob: both paths derive the same
// Level from the same factorization of the same A, so it cannot change output.
// --check-threshold asserts that rather than asserting it in a comment.
#include "census.cuh"
#include "classify.cuh"
#include "common.cuh"
#include "factorize.cuh"
#include "ref_host.h"
#include "sieve.cuh"
#include "tables.cuh"
#include <algorithm>
#include <chrono>
#include <condition_variable>
#include <mutex>
#include <thread>
#include <cstdio>
#include <cstring>
#include <cub/cub.cuh>
#include <vector>

namespace {

// Per-rung tallies, reduced on the device. Integer sums and a min, so the VALUE is
// independent of the order the blocks happen to run in -- which is what lets these use
// atomics without giving up determinism. (Allocation still never uses them: that is
// prefix-sum + slice, per the plan's constraint.)
// RungTally (classify.cuh) plays the role the local RungCounts used to.
using RungCounts = RungTally;

// Everything the host needs back from a rung, in ONE struct so it is ONE transfer.
// Each cudaMemcpy here is a latency-bound sync, and there are ~4 rungs' worth per
// class; reading the two DeviceSelect counts separately from the tallies cost 15.8%
// of runtime as four round trips where two do.
struct Readback {
  RungCounts c;
  uint32_t   nemit, nsurv;     // written directly by cub::DeviceSelect::Flagged
};

// Device buffers, grown on demand and reused for the life of the process, so a
// steady-state window performs no allocation.
struct Work {
  uint32_t   *d_actpos = nullptr, *d_actpos2 = nullptr;   // ping-pong across rungs
  uint64_t   *d_primebits = nullptr;
  FacR       *d_fac = nullptr;
  uint8_t    *d_survflag = nullptr, *d_emitflag = nullptr;
  uint64_t   *d_enc = nullptr, *d_encout = nullptr;       // (level, position) pairs
  Readback   *d_rb = nullptr;                             // one struct, one transfer
  Readback   *h_rb = nullptr;                             // pinned host mirror of d_rb
  void       *d_tmp = nullptr;
  size_t      cap_pos = 0, cap_bits = 0, cap_tmp = 0;

  void ensure(size_t npos_cap, size_t nbits) {
    if (npos_cap > cap_pos) {
      free_pos();
      CUDA_CHECK(cudaMalloc(&d_actpos,  npos_cap * sizeof(uint32_t)));
      CUDA_CHECK(cudaMalloc(&d_actpos2, npos_cap * sizeof(uint32_t)));
      CUDA_CHECK(cudaMalloc(&d_fac,     npos_cap * sizeof(FacR)));
      CUDA_CHECK(cudaMalloc(&d_survflag, npos_cap));
      CUDA_CHECK(cudaMalloc(&d_emitflag, npos_cap));
      CUDA_CHECK(cudaMalloc(&d_enc,     npos_cap * sizeof(uint64_t)));
      CUDA_CHECK(cudaMalloc(&d_encout,  npos_cap * sizeof(uint64_t)));
      cap_pos = npos_cap;
    }
    if (nbits > cap_bits) {
      if (d_primebits) cudaFree(d_primebits);
      CUDA_CHECK(cudaMalloc(&d_primebits, nbits * sizeof(uint64_t)));
      cap_bits = nbits;
    }
    if (!d_rb) CUDA_CHECK(cudaMalloc(&d_rb, sizeof(Readback)));
    if (!h_rb) CUDA_CHECK(cudaMallocHost(&h_rb, sizeof(Readback)));
  }
  void ensure_tmp(size_t bytes) {
    if (bytes > cap_tmp) {
      if (d_tmp) cudaFree(d_tmp);
      CUDA_CHECK(cudaMalloc(&d_tmp, bytes));
      cap_tmp = bytes;
    }
  }
  void free_pos() {
    if (d_actpos) {
      cudaFree(d_actpos); cudaFree(d_actpos2); cudaFree(d_fac);
      cudaFree(d_survflag); cudaFree(d_emitflag); cudaFree(d_enc); cudaFree(d_encout);
    }
    d_actpos = nullptr; d_actpos2 = nullptr; d_fac = nullptr;
    d_survflag = nullptr; d_emitflag = nullptr; d_enc = nullptr; d_encout = nullptr;
    cap_pos = 0;
  }
  void release() {
    free_pos();
    if (d_primebits) cudaFree(d_primebits);
    if (d_rb) cudaFree(d_rb);
    if (h_rb) cudaFreeHost(h_rb);
    if (d_tmp) cudaFree(d_tmp);
    d_primebits = nullptr; d_rb = nullptr; h_rb = nullptr; d_tmp = nullptr;
    cap_bits = cap_tmp = 0;
  }
};

// One workspace per window-in-flight, matching the factorize/sieve slots. A slot is
// only ever driven by the one host thread pipelining its windows, so nothing here
// needs a lock.
Work g_work[2];

// One prime the GPU ladder left unresolved: the CPU tail resumes it at rung r0.
struct TailItem { uint64_t p, r0; };

// ---------------------------------------------------------------- profiling
// Where the wall clock goes, phase by phase. Only populated under --profile, which
// inserts extra syncs -- so a profiled run's ns/prime is not the number to quote.
struct Prof {
  double sieve = 0, upload = 0, factorize = 0, classify = 0, download = 0,
         bookkeep = 0, tail = 0;
  uint64_t nrung = 0, nclass = 0, nclassified = 0, ntail = 0, ntail_rungs = 0;
  double   per_rung[256] = {0};
  uint64_t act_rung[256] = {0};
};
Prof g_prof;

inline double now_s() {
  return std::chrono::duration<double>(
      std::chrono::steady_clock::now().time_since_epoch()).count();
}

// rung_scan.cpp:447-452 and 454-459, verbatim in effect. Kept as free functions so the
// bulk path and the CPU-tail path cannot record differently.
inline void record_hit(uint64_t p, uint64_t A, uint64_t r,
                       const CensusOpts &opt, CensusStats &st) {
  st.hist[r]++;
  if (r > st.maxr) { st.maxr = r; st.maxr_p = p; }
  if (opt.deepThreshold && r >= opt.deepThreshold) {
    st.deep_list.push_back(p); st.deep_list.push_back(A); st.deep_list.push_back(r);
  }
  if (opt.sampleEvery && (p % opt.sampleEvery) < 24) {
    st.sample_list.push_back(p); st.sample_list.push_back(A); st.sample_list.push_back(r);
  }
}

inline void record_fail(int lv, uint64_t p, uint64_t A, uint64_t r,
                        const CensusOpts &opt, CensusStats &st) {
  if (lv == DL_J) st.jac++;
  else if (lv == DL_RC) {
    st.rc++;
    if (opt.emitSup) { st.rc_list.push_back(p); st.rc_list.push_back(A); st.rc_list.push_back(r); }
  } else if (lv == DL_S) {
    st.sup++;
    if (opt.emitSup) { st.sup_list.push_back(p); st.sup_list.push_back(A); st.sup_list.push_back(r); }
  } else {
    st.res++;
    if (opt.emitRes) { st.res_list.push_back(p); st.res_list.push_back(A); st.res_list.push_back(r); }
  }
}

// rung_scan.cpp:416-438. Order within a window matches the reference's; across windows
// nothing is promised and merge.py sorts (see RUNBOOK.md).
void flush_emits(CensusStats &st) {
  auto dump = [](const char *tag, std::vector<uint64_t> &v) {
    for (size_t i = 0; i + 2 < v.size(); i += 3)
      printf("%s %llu %llu %llu\n", tag, (unsigned long long)v[i],
             (unsigned long long)v[i + 1], (unsigned long long)v[i + 2]);
    v.clear();
  };
  dump("REALCHAR", st.rc_list);
  dump("SUPPORT",  st.sup_list);
  dump("RESIDUAL", st.res_list);
  for (size_t i = 0; i + 2 < st.deep_list.size(); i += 3)
    printf("%s %llu %llu %llu\n", st.deep_list[i + 1] ? "DEEP" : "ESCALATE",
           (unsigned long long)st.deep_list[i], (unsigned long long)st.deep_list[i + 1],
           (unsigned long long)st.deep_list[i + 2]);
  st.deep_list.clear();
  dump("SAMPLE", st.sample_list);
}

// One residue class of one window: the rung wavefront, then the CPU tail.
void process_class_gpu(const DevTables &T, uint64_t wlo, uint64_t whi, uint64_t cls,
                       const CensusOpts &opt, CensusStats &st,
                       std::vector<TailItem> &tail, int slot, cudaStream_t sm) {
  Work &W = g_work[slot];
  if (wlo >= whi) return;
  const uint64_t mod = ref_mod(), astep = ref_astep();

  const bool prof = opt.profile;
  double t0 = prof ? now_s() : 0;

  // Position of the class in this window -- rung_scan.cpp:482-484.
  const uint64_t first = wlo + ((cls + mod - wlo % mod) % mod);
  if (first >= whi) return;
  const uint64_t npos = (whi - first + mod - 1) / mod;
  const uint32_t nwords = (uint32_t)((npos + 63) / 64);
  W.ensure((size_t)npos, nwords);

  // Task 6: the sieve runs on the device and leaves both the bitmap and the ascending
  // position list there, so nothing is uploaded here any more.
  const uint32_t npr = sieve_class(T, first, (uint32_t)npos, mod, whi,
                                   W.d_primebits, W.d_actpos, sm, slot);
  if (!npr) return;
  st.nprimes += npr;
  if (prof) { g_prof.sieve += now_s() - t0; ++g_prof.nclass; t0 = now_s(); }

  // The active list now lives on the device from the sieve through the whole ladder --
  // nothing per-position crosses PCIe except the handful of emitted certificates.
  uint32_t nact = npr;
  std::vector<uint64_t> emits;

  // r <= TABLE_RMAX is NOT optional: the device classifier's Jacobi/Legendre tables
  // only cover r <= 63 and are zero-filled above it, so a rung past that would be
  // silently misclassified rather than fail. This was latent until --spanlog rose to
  // 24 -- a 16x larger window keeps 16x more primes active, so the ladder reached r=67
  // still above gpuMin for the first time, and diff_range caught it as 18 level-J
  // failures turning into level-R plus a shifted depth histogram at r >= 67.
  uint64_t r = 3;
  for (; r <= opt.rmax && r <= (uint64_t)TABLE_RMAX && nact >= opt.gpuMin; r += 4) {
    const uint64_t A0 = (first + r) / 4;                    // rung_scan.cpp:520
    const uint64_t Amax = A0 + astep * (npos - 1);          // rung_scan.cpp:521

    const double tr0 = prof ? now_s() : 0;
    double t = tr0;

    // finish=false: the cofactor append is fused into the classify-tally kernel,
    // which reads the leftover cofactors via factorize_last_remA().
    factorize_rung(T, A0, astep, (uint32_t)npos, Amax, (uint32_t)r,
                   W.d_actpos, nact, W.d_primebits, W.d_fac, sm,
                   /*finish=*/false, slot);
    if (prof) { CUDA_CHECK(cudaDeviceSynchronize()); g_prof.factorize += now_s() - t; t = now_s(); }

    if (prof) { ++g_prof.nrung; g_prof.nclassified += nact; g_prof.act_rung[r] += nact; }

    // ---- classify + cofactor append + tally + flag, ONE fused kernel: all on device
    CUDA_CHECK(cudaMemsetAsync(&W.d_rb->c, 0, sizeof(RungCounts), sm));
    CUDA_CHECK(cudaMemsetAsync(&W.d_rb->c.min_hit_j, 0xff, sizeof(unsigned int), sm));
    classify_tally_rung(T, (uint32_t)r, first, mod, A0, astep, W.d_actpos,
                        W.d_fac, factorize_last_remA(slot), nact,
                        opt.deepThreshold, opt.sampleEvery, opt.emitRes, opt.emitSup,
                        W.d_survflag, W.d_emitflag, W.d_enc,
                        &W.d_rb->c, sm);
    if (prof) { CUDA_CHECK(cudaDeviceSynchronize()); g_prof.classify += now_s() - t; t = now_s(); }

    // Survivors compact into the ping-pong buffer and STAY on the device -- the old
    // code round-tripped this list through the host every single rung. Both counts are
    // written straight into the readback struct, so one copy collects them with the
    // tallies.
    size_t bytes = 0, b2 = 0;
    CUDA_CHECK(cub::DeviceSelect::Flagged(nullptr, bytes, W.d_actpos,
                                          W.d_survflag, W.d_actpos2,
                                          &W.d_rb->nsurv, nact, sm));
    CUDA_CHECK(cub::DeviceSelect::Flagged(nullptr, b2, W.d_enc,
                                          W.d_emitflag, W.d_encout,
                                          &W.d_rb->nemit, nact, sm));
    W.ensure_tmp(std::max(bytes, b2));
    CUDA_CHECK(cub::DeviceSelect::Flagged(W.d_tmp, b2, W.d_enc,
                                          W.d_emitflag, W.d_encout,
                                          &W.d_rb->nemit, nact, sm));
    CUDA_CHECK(cub::DeviceSelect::Flagged(W.d_tmp, bytes, W.d_actpos,
                                          W.d_survflag, W.d_actpos2,
                                          &W.d_rb->nsurv, nact, sm));

    // Plain cudaMemcpy runs on the LEGACY default stream; with the census's streams
    // created non-blocking that is a data race, not a slowdown. Stream-ordered async
    // into the pinned mirror + one stream sync is both correct and cheaper.
    CUDA_CHECK(cudaMemcpyAsync(W.h_rb, W.d_rb, sizeof(Readback),
                               cudaMemcpyDeviceToHost, sm));
    CUDA_CHECK(cudaStreamSynchronize(sm));
    const Readback rb = *W.h_rb;
    const RungCounts &hc = rb.c;
    emits.resize(rb.nemit);
    if (rb.nemit) {
      CUDA_CHECK(cudaMemcpyAsync(emits.data(), W.d_encout,
                                 (size_t)rb.nemit * sizeof(uint64_t),
                                 cudaMemcpyDeviceToHost, sm));
      CUDA_CHECK(cudaStreamSynchronize(sm));
    }
    std::swap(W.d_actpos, W.d_actpos2);
    if (prof) { g_prof.download += now_s() - t; t = now_s(); }

    if (hc.lv[5]) {
      fprintf(stderr, "census: gcd(q,r) != 1 at rung %llu (%llu positions) -- the "
                      "classifier's unreachable path was reached; refusing to report "
                      "a census\n", (unsigned long long)r, hc.lv[5]);
      abort();
    }

    // ---- fold the device tallies into Stats, exactly as record_hit/record_fail did
    st.hist[r] += hc.lv[DL_HIT];
    if (hc.lv[DL_HIT] && r > st.maxr) {
      st.maxr = r;
      st.maxr_p = first + mod * (uint64_t)hc.min_hit_j;   // lowest position => lowest p
    }
    st.jac += hc.lv[DL_J];  st.rc  += hc.lv[DL_RC];
    st.sup += hc.lv[DL_S];  st.res += hc.lv[DL_R];

    for (uint64_t e : emits) {
      const uint32_t lv = (uint32_t)(e >> 32), j = (uint32_t)e;
      const uint64_t p = first + mod * (uint64_t)j, A = A0 + astep * (uint64_t)j;
      if (lv == DL_HIT) {
        if (opt.deepThreshold && r >= opt.deepThreshold) {
          st.deep_list.push_back(p); st.deep_list.push_back(A); st.deep_list.push_back(r);
        }
        if (opt.sampleEvery && (p % opt.sampleEvery) < 24) {
          st.sample_list.push_back(p); st.sample_list.push_back(A);
          st.sample_list.push_back(r);
        }
      } else if (lv == DL_RC) {
        st.rc_list.push_back(p); st.rc_list.push_back(A); st.rc_list.push_back(r);
      } else if (lv == DL_S) {
        st.sup_list.push_back(p); st.sup_list.push_back(A); st.sup_list.push_back(r);
      } else if (lv == DL_R) {
        st.res_list.push_back(p); st.res_list.push_back(A); st.res_list.push_back(r);
      }
    }
    nact = rb.nsurv;
    if (prof) { g_prof.bookkeep += now_s() - t; g_prof.per_rung[r] += now_s() - tr0; }
  }

  // The tail is the only place the surviving positions come back to the host, and by
  // construction there are fewer than gpuMin of them.
  std::vector<uint32_t> act(nact);
  if (nact) {
    CUDA_CHECK(cudaMemcpyAsync(act.data(), W.d_actpos, (size_t)nact * sizeof(uint32_t),
                               cudaMemcpyDeviceToHost, sm));
    CUDA_CHECK(cudaStreamSynchronize(sm));
  }

  // ---- the deep tail is COLLECTED here and executed by run_tail on a worker thread,
  // overlapped with the next window's GPU ladder (see census_core). This is the one
  // interval where the GPU used to go completely dark: the tail runs the reference's
  // factorA + test_rung on the host, at ~40 us per rung test.
  for (uint32_t i = 0; i < act.size(); ++i)
    tail.push_back({first + mod * (uint64_t)act[i], r});
}

// The reference's per-prime deep-tail loop (rung_scan.cpp:557-572), unchanged logic,
// runnable off the main thread. Everything it calls in the reference (factorA,
// test_rung, is_prime) reads only tables that are constant after upload_tables; the
// only global it writes is g_pairpath, and only one tail runs at a time. It records
// exclusively into its own window's CensusStats, which the main thread does not touch
// until after join. Under --profile it owns the tail/ntail/ntail_rungs fields of
// g_prof and the main thread owns the rest -- distinct members, no shared location.
void run_tail(const std::vector<TailItem> &items, const CensusOpts &opt,
              CensusStats &st, bool prof) {
  const double ttail = prof ? now_s() : 0;
  if (prof) g_prof.ntail += items.size();
  for (const TailItem &it : items) {
    bool done = false;
    for (uint64_t rr = it.r0; rr <= opt.rmax; rr += 4) {
      const uint64_t A = (it.p + rr) / 4;
      const int lv = ref_test_rung(it.p, A, rr);
      if (prof) ++g_prof.ntail_rungs;
      if (lv == DL_HIT) { record_hit(it.p, A, rr, opt, st); done = true; break; }
      record_fail(lv, it.p, A, rr, opt, st);
    }
    if (!done) {
      st.escalated++;
      st.deep_list.push_back(it.p); st.deep_list.push_back(0); st.deep_list.push_back(0);
    }
  }
  if (prof) g_prof.tail += now_s() - ttail;
}

void process_window_gpu(const DevTables &T, uint64_t wlo, uint64_t whi,
                        const CensusOpts &opt, CensusStats &st,
                        std::vector<TailItem> &tail, int slot, cudaStream_t sm) {
  const uint32_t ncls = ref_nclasses();
  for (uint32_t c = 0; c < ncls; ++c)
    process_class_gpu(T, wlo, whi, ref_class(c), opt, st, tail, slot, sm);
}

// Runs the census into `total` without printing SUMMARY. Emission is streamed per
// window when `stream` is set, and retained in `total` otherwise (what --verify wants).
void census_core(const DevTables &T, uint64_t lo, uint64_t hi, const CensusOpts &opt,
                 CensusStats &total, bool stream) {
  ref_reset_pairpath();

  const uint64_t SPAN = ref_mod() << opt.spanlog;
  const uint64_t nwin = (hi - lo + SPAN - 1) / SPAN;
  auto tstart = std::chrono::steady_clock::now();

  // TWO WINDOWS IN FLIGHT. Windows are perfectly independent -- disjoint ranges,
  // nothing shared but the read-only tables -- so each of two host threads drives its
  // own windows (even/odd) on its own stream and its own workspace slot. When thread
  // A blocks on its stream's sync, the GPU keeps running thread B's kernels; while a
  // thread runs its window's CPU tail, the other window's ladder owns the GPU. This
  // subsumes the earlier tail-only overlap.
  //
  // The single-chain conservation law does not apply across chains: the host still
  // waits the same total on EACH chain, but it waits on one while the GPU runs the
  // other. Determinism is untouched because concurrency never crosses a window:
  // every window computes exactly what it computed single-threaded, and the main
  // thread folds, flushes, and merges strictly in window order below -- preserving
  // emission order and merge()'s order-sensitive maxr_p tie-break.
  //
  // --profile keeps one thread: its counters are unsynchronized by design, and a
  // profiled run's timing is not the number to quote anyway.
  const uint64_t nf = (opt.profile || nwin == 1) ? 1 : 2;
  enum { RING = 8 };                       // > 2*nf outstanding, enforced below
  cudaStream_t streams[2] = {nullptr, nullptr};
  for (uint64_t t = 0; t < nf; ++t)
    CUDA_CHECK(cudaStreamCreateWithFlags(&streams[t], cudaStreamNonBlocking));

  std::mutex mu;
  std::condition_variable cv;
  uint64_t retired = 0;
  CensusStats ring[RING];
  bool done[RING] = {false};

  auto worker = [&](uint64_t tid) {
    for (uint64_t w = tid; w < nwin; w += nf) {
      {
        // Bounded lead: never run more than 2*nf windows ahead of retirement, so the
        // ring can't wrap onto an unretired window and emit memory stays bounded.
        std::unique_lock<std::mutex> lk(mu);
        cv.wait(lk, [&] { return w < retired + 2 * nf; });
      }
      CensusStats loc;
      std::vector<TailItem> tail;
      process_window_gpu(T, lo + w * SPAN, std::min(hi, lo + (w + 1) * SPAN), opt,
                         loc, tail, (int)tid, streams[tid]);
      run_tail(tail, opt, loc, opt.profile);
      {
        std::lock_guard<std::mutex> lk(mu);
        ring[w % RING] = std::move(loc);
        done[w % RING] = true;
      }
      cv.notify_all();
    }
  };

  std::vector<std::thread> threads;
  for (uint64_t t = 0; t < nf; ++t) threads.emplace_back(worker, t);

  for (uint64_t w = 0; w < nwin; ++w) {    // retire strictly in window order
    CensusStats loc;
    {
      std::unique_lock<std::mutex> lk(mu);
      cv.wait(lk, [&] { return done[w % RING]; });
      loc = std::move(ring[w % RING]);
      ring[w % RING] = CensusStats();
      done[w % RING] = false;
      retired = w + 1;
    }
    cv.notify_all();
    if (stream) flush_emits(loc);
    total.merge(loc);
    if (opt.progress && ((w + 1) % opt.progress == 0)) {
      const double el = std::chrono::duration<double>(
          std::chrono::steady_clock::now() - tstart).count();
      const double f = (double)(w + 1) / (double)nwin;
      fprintf(stderr, "[rung_scan_cuda] %llu/%llu windows (%.1f%%)  %.0fs elapsed  ~%.0fs left\n",
              (unsigned long long)(w + 1), (unsigned long long)nwin, 100 * f, el, el * (1 / f - 1));
    }
  }
  for (std::thread &t : threads) t.join();
  for (uint64_t t = 0; t < nf; ++t) CUDA_CHECK(cudaStreamDestroy(streams[t]));
}

}  // namespace

void CensusStats::clear_lists() {
  rc_list.clear(); sup_list.clear(); res_list.clear();
  deep_list.clear(); sample_list.clear();
}

// rung_scan.cpp:385-397, including the tie-break that makes maxr_p well defined.
void CensusStats::merge(const CensusStats &o) {
  nprimes += o.nprimes; jac += o.jac; rc += o.rc; sup += o.sup; res += o.res;
  escalated += o.escalated;
  for (int i = 0; i < 256; ++i) hist[i] += o.hist[i];
  if (o.maxr > maxr || (o.maxr == maxr && o.maxr_p && (!maxr_p || o.maxr_p < maxr_p)))
    { maxr = o.maxr; maxr_p = o.maxr_p; }
  auto app = [](std::vector<uint64_t> &d, const std::vector<uint64_t> &s) {
    d.insert(d.end(), s.begin(), s.end());
  };
  app(rc_list, o.rc_list); app(sup_list, o.sup_list); app(res_list, o.res_list);
  app(deep_list, o.deep_list); app(sample_list, o.sample_list);
}

int census_run(uint64_t lo, uint64_t hi, const CensusOpts &opt) {
  auto t0 = std::chrono::steady_clock::now();
  // ONCE. This also runs the reference's build_base_primes, which sieves to sqrt(hi)
  // -- 1.9 s at 10^15 -- so calling it a second time inside the scan (which is what
  // census_core used to do) both double-counted that cost into scan_s and leaked the
  // first upload. It cost 40 of the 55 ns/prime measured at 10^15 and was invisible at
  // 10^13, because the wasted work scales with sqrt(hi) while the primes do not.
  DevTables T = upload_tables(hi, opt.hard840);
  auto t1 = std::chrono::steady_clock::now();

  CensusStats total;
  census_core(T, lo, hi, opt, total, /*stream=*/true);

  auto t2 = std::chrono::steady_clock::now();
  const double ts = std::chrono::duration<double>(t1 - t0).count();
  const double tc = std::chrono::duration<double>(t2 - t1).count();

  printf("SUMMARY lo=%llu hi=%llu primes=%llu jac=%llu rc=%llu sup=%llu res=%llu "
         "escalated=%llu pairpath=%llu maxr=%llu maxr_p=%llu hist=",
         (unsigned long long)lo, (unsigned long long)hi,
         (unsigned long long)total.nprimes, (unsigned long long)total.jac,
         (unsigned long long)total.rc, (unsigned long long)total.sup,
         (unsigned long long)total.res, (unsigned long long)total.escalated,
         (unsigned long long)ref_pairpath(),
         (unsigned long long)total.maxr, (unsigned long long)total.maxr_p);
  for (uint64_t r = 3; r <= 255; r += 4)
    if (total.hist[r]) printf("%llu:%llu,", (unsigned long long)r,
                              (unsigned long long)total.hist[r]);
  printf(" sieve_s=%.2f scan_s=%.2f ns_per_prime=%.0f\n", ts, tc,
         total.nprimes ? 1e9 * tc / (double)total.nprimes : 0.0);

  if (opt.profile) {
    const Prof &P = g_prof;
    const double np = (double)total.nprimes;
    const double sum = P.sieve + P.upload + P.factorize + P.classify + P.download
                     + P.bookkeep + P.tail;
    auto row = [&](const char *what, double s, const char *note) {
      printf("PROFILE  %-22s %8.3f s  %5.1f%%  %7.1f ns/prime   %s\n",
             what, s, sum > 0 ? 100 * s / sum : 0.0, np ? 1e9 * s / np : 0.0, note);
    };
    printf("PROFILE  --- phase breakdown, %llu primes, %llu (class,rung) steps, "
           "%llu classifications ---\n",
           (unsigned long long)total.nprimes, (unsigned long long)P.nrung,
           (unsigned long long)P.nclassified);
    row("host sieve",   P.sieve,     "CPU; Task 6 moves this to the device");
    row("H2D upload",   P.upload,    "primebits + actpos, per class and per rung");
    row("factorize",    P.factorize, "GPU, Task 3 pipeline");
    row("classify",     P.classify,  "GPU, Task 4 kernel");
    row("D2H download", P.download,  "one level byte per active position");
    row("host bookkeep", P.bookkeep, "record_hit/record_fail + compaction");
    row("CPU tail",     P.tail,      "handoff below gpuMin, reference test_rung");
    printf("PROFILE  %-22s %8.3f s  100.0%%  %7.1f ns/prime\n", "TOTAL accounted", sum,
           np ? 1e9 * sum / np : 0.0);
    printf("PROFILE  CPU tail: %llu primes handed over, %llu rung tests "
           "(%.1f per prime)\n", (unsigned long long)P.ntail,
           (unsigned long long)P.ntail_rungs,
           P.ntail ? (double)P.ntail_rungs / (double)P.ntail : 0.0);
    printf("PROFILE  --- per rung: active positions and wall time ---\n");
    for (int r = 3; r <= 255; r += 4)
      if (P.act_rung[r])
        printf("PROFILE  r=%-3d  active=%-12llu  %7.3f s  %5.1f%%\n", r,
               (unsigned long long)P.act_rung[r], P.per_rung[r],
               sum > 0 ? 100 * P.per_rung[r] / sum : 0.0);
  }

  g_work[0].release();
  g_work[1].release();
  factorize_release();
  sieve_release();
  return 0;
}

// ------------------------------------------------------------------ --verify

namespace {

// Certificate lists are compared as SORTED SETS: emission order is not promised
// (RUNBOOK.md), the emitted set is.
bool same_set(std::vector<uint64_t> a, std::vector<uint64_t> b, const char *what) {
  auto triples = [](std::vector<uint64_t> &v) {
    std::vector<std::array<uint64_t, 3>> t;
    for (size_t i = 0; i + 2 < v.size(); i += 3) t.push_back({v[i], v[i + 1], v[i + 2]});
    std::sort(t.begin(), t.end());
    return t;
  };
  auto ta = triples(a), tb = triples(b);
  if (ta == tb) { printf("  ok   %-30s %zu entries\n", what, ta.size()); return true; }
  printf("  FAIL %-30s gpu %zu vs ref %zu entries\n", what, ta.size(), tb.size());
  for (size_t i = 0; i < std::min(ta.size(), tb.size()); ++i)
    if (ta[i] != tb[i]) {
      printf("       first difference: gpu (%llu %llu %llu) ref (%llu %llu %llu)\n",
             (unsigned long long)ta[i][0], (unsigned long long)ta[i][1],
             (unsigned long long)ta[i][2], (unsigned long long)tb[i][0],
             (unsigned long long)tb[i][1], (unsigned long long)tb[i][2]);
      break;
    }
  return false;
}

int verify_mode(bool hard840) {
  int fails = 0;
  const uint64_t LO = 0, HI = 100000;

  CensusOpts opt;
  opt.rmax = 127; opt.emitRes = true; opt.emitSup = true;
  opt.spanlog = 12; opt.hard840 = hard840;
  opt.deepThreshold = 35; opt.sampleEvery = 0;

  CensusStats gpu;
  DevTables T = upload_tables(HI, hard840);
  census_core(T, LO, HI, opt, gpu, /*stream=*/false);
  const uint64_t gpu_pairpath = ref_pairpath();

  RefRun ref;
  ref_scan_range(LO, HI, hard840, opt.rmax, opt.emitRes, opt.emitSup,
                 opt.deepThreshold, opt.sampleEvery, opt.spanlog, &ref);

  printf("SELF-TEST  range [0, 100000)  %s  GPU vs the reference, in process\n",
         hard840 ? "[--hard840: 6 square classes mod 840]" : "[default: 1 mod 24]");
  auto check = [&](const char *what, uint64_t got, uint64_t want) {
    if (got != want) {
      printf("  FAIL %-30s got %llu want %llu\n", what,
             (unsigned long long)got, (unsigned long long)want);
      ++fails;
    } else printf("  ok   %-30s %llu\n", what, (unsigned long long)got);
  };
  check("primes in class", gpu.nprimes, ref.nprimes);
  check("level J  (Jacobi) failures", gpu.jac, ref.jac);
  check("level RC (other real chi)", gpu.rc, ref.rc);
  check("level S  (beyond real chi)", gpu.sup, ref.sup);
  check("level R  (residual)", gpu.res, ref.res);
  check("escalated", gpu.escalated, ref.escalated);
  check("pairpath", gpu_pairpath, ref.pairpath);
  check("max depth", gpu.maxr, ref.maxr);
  check("max depth prime", gpu.maxr_p, ref.maxr_p);
  int hbad = 0;
  for (int i = 0; i < 256; ++i) if (gpu.hist[i] != ref.hist[i]) ++hbad;
  check("depth histogram (differing rungs)", (uint64_t)hbad, 0);

  if (!same_set(gpu.rc_list, ref.rc_list, "level-RC certificate set")) ++fails;
  if (!same_set(gpu.sup_list, ref.sup_list, "level-S certificate set")) ++fails;
  if (!same_set(gpu.res_list, ref.res_list, "level-R certificate set")) ++fails;
  if (!same_set(gpu.deep_list, ref.deep_list, "deep/escalate set")) ++fails;
  return fails;
}

}  // namespace

int census_verify() {
  int fails = verify_mode(false);
  printf("\n");
  fails += verify_mode(true);
  printf(fails ? "\nSELF-TEST FAILED (%d checks)\n" : "\nSELF-TEST PASSED\n", fails);
  g_work[0].release();
  g_work[1].release();
  factorize_release();
  sieve_release();
  return fails ? 1 : 0;
}

int census_check_threshold(uint64_t lo, uint64_t hi, bool hard840, int spanlog) {
  // The handoff threshold is claimed to be a performance knob. gpuMin = 1 keeps the
  // GPU on the ladder to the bitter end; a huge gpuMin hands everything to the CPU on
  // the first rung. If the claim holds, those two produce identical output.
  const uint32_t LOW = 1, HIGH = 4000000000u;
  CensusStats a, b;
  CensusOpts opt;
  opt.emitRes = true; opt.emitSup = true; opt.deepThreshold = 35;
  opt.spanlog = spanlog; opt.hard840 = hard840;

  DevTables T = upload_tables(hi, hard840);
  opt.gpuMin = LOW;  census_core(T, lo, hi, opt, a, false);
  opt.gpuMin = HIGH; census_core(T, lo, hi, opt, b, false);

  int bad = 0;
  auto cmp = [&](const char *what, uint64_t x, uint64_t y) {
    if (x != y) { printf("  FAIL %-28s gpuMin=%u: %llu  gpuMin=%u: %llu\n", what,
                         LOW, (unsigned long long)x, HIGH, (unsigned long long)y); ++bad; }
  };
  cmp("primes", a.nprimes, b.nprimes);   cmp("jac", a.jac, b.jac);
  cmp("rc", a.rc, b.rc);                 cmp("sup", a.sup, b.sup);
  cmp("res", a.res, b.res);              cmp("escalated", a.escalated, b.escalated);
  cmp("maxr", a.maxr, b.maxr);           cmp("maxr_p", a.maxr_p, b.maxr_p);
  for (int i = 0; i < 256; ++i)
    if (a.hist[i] != b.hist[i]) { printf("  FAIL hist[%d]: %llu vs %llu\n", i,
        (unsigned long long)a.hist[i], (unsigned long long)b.hist[i]); ++bad; }
  if (!same_set(a.rc_list, b.rc_list, "level-RC set")) ++bad;
  if (!same_set(a.sup_list, b.sup_list, "level-S set")) ++bad;
  if (!same_set(a.res_list, b.res_list, "level-R set")) ++bad;
  if (!same_set(a.deep_list, b.deep_list, "deep/escalate set")) ++bad;

  printf("THRESHOLD  [%llu, %llu) %s: gpuMin=%u (all-GPU) vs gpuMin=%u (all-CPU), "
         "%llu primes\n", (unsigned long long)lo, (unsigned long long)hi,
         hard840 ? "--hard840" : "(1 mod 24)", LOW, HIGH,
         (unsigned long long)a.nprimes);
  printf(bad ? "THRESHOLD FAIL (%d)\n" : "THRESHOLD OK -- the handoff point does not "
                                         "affect output\n", bad);
  g_work[0].release();
  g_work[1].release();
  factorize_release();
  sieve_release();
  return bad;
}
