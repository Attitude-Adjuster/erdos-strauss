// gate.cu -- the Task 3 GO/NO-GO gate.
//
// Two jobs, in this order, and the order matters:
//   1. CORRECTNESS. Factor rung 3 for every prime position of one full window on the
//      GPU and diff the result against the reference's factorA() -- trial division
//      plus Miller-Rabin/Pollard, a genuinely different algorithm from the sieve walk
//      the GPU runs, so agreement is evidence and not a tautology. Compared as sorted
//      MULTISETS of (l mod r, e): two distinct primes can share a residue mod r and
//      merging them would change tau(q^2).
//   2. MEASUREMENT. Only once the diff is clean, time stages 1-4 over the same full
//      window with CUDA events, best of 5, first run discarded.
//
// Rung 3 is the worst case on purpose: every prime is still active, so generation
// runs at full width.
#include "common.cuh"
#include "factorize.cuh"
#include "ref_host.h"
#include "tables.cuh"
#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <utility>
#include <vector>

namespace {

struct ClassData {
  uint64_t first = 0, A0 = 0, Amax = 0;
  uint32_t npos = 0, nact = 0;
  std::vector<uint32_t> pjs;
  uint32_t *d_actpos = nullptr;
  uint64_t *d_primebits = nullptr;
  FacR     *d_out = nullptr;
};

// Sorted-multiset comparison of one position's factorization.
bool same_multiset(std::vector<std::pair<int, int>> &a, std::vector<std::pair<int, int>> &b) {
  std::sort(a.begin(), a.end());
  std::sort(b.begin(), b.end());
  return a == b;
}

}  // namespace

int gate(uint64_t lo, uint64_t hi, bool hard840, int spanlog, int reps) {
  const uint32_t r = 3;

  // The window is 840<<spanlog wide at most (840 is the larger of the two wheel
  // moduli), so build the reference's prime tables for at least that far.
  const uint64_t hi_eff = std::max<uint64_t>(hi, lo + (840ull << spanlog));
  DevTables T = upload_tables(hi_eff, hard840);

  const uint64_t mod = ref_mod(), astep = ref_astep();
  const uint64_t SPAN = mod << spanlog;
  const uint64_t wlo = lo, whi = lo + SPAN;
  const uint32_t ncls = ref_nclasses();

  printf("GATE  window [%llu, %llu)  span=%llu (mod %llu << %d)  classes=%u  rung=%u\n",
         (unsigned long long)wlo, (unsigned long long)whi, (unsigned long long)SPAN,
         (unsigned long long)mod, spanlog, ncls, r);

  // ---- host sieve of the whole window, one class at a time, then upload
  std::vector<ClassData> C(ncls);
  uint64_t tot_pos = 0, tot_act = 0;
  for (uint32_t c = 0; c < ncls; ++c) {
    ClassData &cd = C[c];
    uint64_t first = 0, npos = 0;
    cd.nact = ref_sieve_class(wlo, whi, ref_class(c), &first, &npos, cd.pjs);
    cd.first = first;
    cd.npos  = (uint32_t)npos;
    if (!cd.nact) continue;
    cd.A0   = (first + r) / 4;                       // rung_scan.cpp:520
    cd.Amax = cd.A0 + astep * (npos - 1);            // rung_scan.cpp:521
    tot_pos += npos; tot_act += cd.nact;

    // Cheap independent check that the scaffolding sieve is not obviously wrong:
    // Miller-Rabin the first few survivors. (A sieve bug cannot fake a pass -- see
    // ref_host.h -- but it would silently shrink coverage.)
    for (uint32_t k = 0; k < cd.nact && k < 256; ++k)
      if (!ref_is_prime(first + mod * (uint64_t)cd.pjs[k])) {
        printf("GATE FAIL: sieve emitted a composite at j=%u\n", cd.pjs[k]);
        return 1;
      }

    const uint32_t nwords = (cd.npos + 63u) / 64u;
    std::vector<uint64_t> bits(nwords, 0);
    for (uint32_t k = 0; k < cd.nact; ++k) bits[cd.pjs[k] >> 6] |= 1ull << (cd.pjs[k] & 63);

    CUDA_CHECK(cudaMalloc(&cd.d_actpos, (size_t)cd.nact * sizeof(uint32_t)));
    CUDA_CHECK(cudaMalloc(&cd.d_primebits, (size_t)nwords * sizeof(uint64_t)));
    CUDA_CHECK(cudaMalloc(&cd.d_out, (size_t)cd.nact * sizeof(FacR)));
    CUDA_CHECK(cudaMemcpy(cd.d_actpos, cd.pjs.data(), (size_t)cd.nact * sizeof(uint32_t),
                          cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(cd.d_primebits, bits.data(), (size_t)nwords * sizeof(uint64_t),
                          cudaMemcpyHostToDevice));
  }
  printf("GATE  positions=%llu  active(prime)=%llu  (%.1f%%)\n",
         (unsigned long long)tot_pos, (unsigned long long)tot_act,
         tot_pos ? 100.0 * (double)tot_act / (double)tot_pos : 0.0);

  // ---- run once and diff against factorA
  int bad = 0;
  for (uint32_t c = 0; c < ncls; ++c) {
    ClassData &cd = C[c];
    if (!cd.nact) continue;
    factorize_rung(T, cd.A0, astep, cd.npos, cd.Amax, r,
                   cd.d_actpos, cd.nact, cd.d_primebits, cd.d_out, 0);
    CUDA_CHECK(cudaDeviceSynchronize());
    FacStats fs = factorize_last_stats();
    printf("GATE  class %llu: npos=%u nact=%u  facp<=sqrt(Amax)=%u  items=%u "
           "(blockA=%u warpB=%u threadC=%u)  slots=%u hits=%u seg=%u\n",
           (unsigned long long)ref_class(c), cd.npos, cd.nact, fs.nfacp_r, fs.nitems,
           fs.bA, fs.bB - fs.bA, fs.nfacp_r - fs.bB, fs.cap, fs.nhits, fs.nseg);

    std::vector<FacR> got(cd.nact);
    CUDA_CHECK(cudaMemcpy(got.data(), cd.d_out, (size_t)cd.nact * sizeof(FacR),
                          cudaMemcpyDeviceToHost));

    int cbad = 0;
    #pragma omp parallel for schedule(dynamic, 4096) reduction(+ : cbad)
    for (long long ii = 0; ii < (long long)cd.nact; ++ii) {
      const uint32_t i = (uint32_t)ii;
      const uint64_t A = cd.A0 + astep * (uint64_t)cd.pjs[i];
      uint64_t pr[16]; uint8_t ex[16];
      const int k = ref_factorA(A, pr, ex);     // reference, independent algorithm
      std::vector<std::pair<int, int>> want, mine;
      for (int t = 0; t < k; ++t) want.push_back({(int)(pr[t] % r), (int)ex[t]});
      for (int t = 0; t < got[i].k; ++t) mine.push_back({(int)got[i].res[t], (int)got[i].ex[t]});
      if (!same_multiset(want, mine)) {
        if (cbad < 10) {
          printf("FAC mismatch at A=%llu (j=%u): want", (unsigned long long)A, cd.pjs[i]);
          for (auto &e : want) printf(" %d^%d", e.first, e.second);
          printf("  got");
          for (auto &e : mine) printf(" %d^%d", e.first, e.second);
          printf("\n");
        }
        ++cbad;
      }
    }
    bad += cbad;
  }
  printf(bad ? "GATE FAIL: %d mismatches\n" : "GATE: factorizations match\n", bad);
  if (bad) return bad;

  // ---- measure: stages 1-4 over the whole window (all classes), best of `reps`,
  //      first iteration discarded (warm). Uploads and the host sieve are outside the
  //      timed region; those are Task 6's problem, not this pipeline's.
  cudaEvent_t e0, e1;
  CUDA_CHECK(cudaEventCreate(&e0));
  CUDA_CHECK(cudaEventCreate(&e1));
  double best = 1e30, first_ms = 0;
  for (int it = 0; it < reps + 1; ++it) {
    CUDA_CHECK(cudaDeviceSynchronize());
    CUDA_CHECK(cudaEventRecord(e0, 0));
    for (uint32_t c = 0; c < ncls; ++c) {
      ClassData &cd = C[c];
      if (!cd.nact) continue;
      factorize_rung(T, cd.A0, astep, cd.npos, cd.Amax, r,
                     cd.d_actpos, cd.nact, cd.d_primebits, cd.d_out, 0);
    }
    CUDA_CHECK(cudaEventRecord(e1, 0));
    CUDA_CHECK(cudaEventSynchronize(e1));
    float ms = 0;
    CUDA_CHECK(cudaEventElapsedTime(&ms, e0, e1));
    if (it == 0) { first_ms = ms; continue; }     // discard the cold run
    printf("GATE  run %d: %.3f ms\n", it, ms);
    best = std::min(best, (double)ms);
  }
  printf("GATE  cold run (discarded): %.3f ms\n", first_ms);
  printf("GATE STAGES1-4 %.3f ms/window  (best of %d, warm, %u classes, rung %u, "
         "%llu active positions)\n",
         best, reps, ncls, r, (unsigned long long)tot_act);
  const char *verdict = best < 10.0 ? "PROCEED"
                      : best <= 30.0 ? "PROCEED-WITH-RATIONALE" : "STOP";
  printf("GATE VERDICT %s  (<10 proceed, 10-30 rationale, >30 stop)\n", verdict);

  for (uint32_t c = 0; c < ncls; ++c) {
    if (C[c].d_actpos)    cudaFree(C[c].d_actpos);
    if (C[c].d_primebits) cudaFree(C[c].d_primebits);
    if (C[c].d_out)       cudaFree(C[c].d_out);
  }
  factorize_release();
  return 0;
}
