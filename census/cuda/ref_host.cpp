// The whole trick is here: `main` occurs as a bare identifier exactly once in the
// reference (rung_scan.cpp:675) -- the other two occurrences are inside comments,
// which the preprocessor never sees. Renaming it via macro and #include-ing the .cpp
// is zero-modification, full-access reuse of its `static` internals (factorA,
// test_rung, build_base_primes, g_facp, g_invastep, ...). rung_scan.cpp itself is
// never edited, copied, or forked -- see RUNBOOK.md.
#define main rung_scan_main
#include "../rung_scan.cpp"
#undef main

#include "ref_host.h"

void ref_build_tables(uint64_t hi, bool hard840) {
  if (hard840) { g_mod = 840; g_astep = 210; g_classes = {1,121,169,289,361,529};
                 g_stripp = {2,3,5,7}; }
  else         { g_mod = 24;  g_astep = 6;   g_classes = {1};
                 g_stripp = {2,3}; }
  init_phi(); init_rfac();
  build_base_primes(hi);
}

int ref_verify() { return self_test(); }

// Narrow accessors -- see ref_host.h. Pass-throughs only; this is the sanctioned way
// for tables.cu to reach jacobi()/legendre_sym()/g_rfac/g_rnf/g_facp/g_invastep
// without a second #include of rung_scan.cpp.
int      ref_jacobi(uint64_t a, uint64_t n)   { return jacobi(a, n); }
int      ref_legendre(uint64_t a, uint64_t p) { return legendre_sym(a, p); }
int      ref_rnf(int r)                       { return g_rnf[r]; }
int      ref_rfac(int r, int i)               { return g_rfac[r][i]; }
uint32_t ref_facp_count()                     { return (uint32_t)g_facp.size(); }
uint64_t ref_facp(uint32_t i)                 { return g_facp[i]; }
uint64_t ref_invastep(uint32_t i)             { return g_invastep[i]; }
uint32_t ref_base_count()                     { return (uint32_t)g_base.size(); }
uint64_t ref_base(uint32_t i)                 { return g_base[i]; }
uint64_t ref_invmod(uint32_t i)               { return g_invmod[i]; }
uint64_t ref_mod()                            { return g_mod; }
uint64_t ref_astep()                          { return g_astep; }
uint32_t ref_nclasses()                       { return (uint32_t)g_classes.size(); }
uint64_t ref_class(uint32_t i)                { return g_classes[i]; }
bool     ref_is_prime(uint64_t n)             { return is_prime(n); }

int ref_factorA(uint64_t n, uint64_t *pr, uint8_t *ex) {
  Fac f; factorA(n, f);
  for (int i = 0; i < f.k; ++i) { pr[i] = f.pr[i]; ex[i] = f.ex[i]; }
  return f.k;
}

int ref_test_rung(uint64_t p, uint64_t A, uint64_t r) {
  Fac f; factorA(A, f);
  return (int)test_rung(p, A, r, f);
}

uint64_t ref_pairpath()      { return g_pairpath; }
void     ref_reset_pairpath() { g_pairpath = 0; }

void ref_scan_range(uint64_t lo, uint64_t hi, bool hard840, uint64_t rmax,
                    bool emitRes, bool emitSup, uint64_t deepThreshold,
                    uint64_t sampleEvery, int spanlog, RefRun *out) {
  Opts opt;
  opt.rmax = rmax; opt.emitRes = emitRes; opt.emitSup = emitSup;
  opt.deepThreshold = deepThreshold; opt.sampleEvery = sampleEvery;
  opt.spanlog = spanlog; opt.hard840 = hard840;

  const bool saved_stream = g_stream;
  g_stream = false;                 // keep the lists in memory instead of printing
  g_pairpath = 0;
  ref_build_tables(hi, hard840);

  // Same window decomposition and the same per-window Stats + merge() the reference
  // uses, because merge()'s maxr_p tie-break (smaller p wins) is order-sensitive and
  // reproducing the statistic means reproducing the structure, not just the loop.
  const u64 SPAN = g_mod << opt.spanlog;
  Stats total;
  for (u64 w = 0; lo + w * SPAN < hi; ++w) {
    Stats loc;
    process_window(lo + w * SPAN, std::min(hi, lo + (w + 1) * SPAN), opt, loc);
    total.merge(loc);
  }

  out->nprimes = total.nprimes; out->jac = total.jac; out->rc = total.rc;
  out->sup = total.sup;         out->res = total.res;
  out->escalated = total.escalated;
  out->pairpath = g_pairpath;
  out->maxr = total.maxr; out->maxr_p = total.maxr_p;
  for (int i = 0; i <= RCAP; ++i) out->hist[i] = total.hist[i];
  auto flat = [](const std::vector<std::array<u64,3>> &src, std::vector<uint64_t> &dst) {
    for (const auto &e : src) { dst.push_back(e[0]); dst.push_back(e[1]); dst.push_back(e[2]); }
  };
  flat(total.rc_list, out->rc_list);       flat(total.sup_list, out->sup_list);
  flat(total.res_list, out->res_list);     flat(total.deep_list, out->deep_list);
  flat(total.sample_list, out->sample_list);

  g_stream = saved_stream;
}

// See ref_host.h: a host-side copy of the sieve embedded in process_class
// (rung_scan.cpp:482-501), needed because that loop is not separately callable.
// Kept line-for-line identical to the reference so it cannot drift.
uint32_t ref_sieve_class(uint64_t wlo, uint64_t whi, uint64_t cls,
                         uint64_t *first_out, uint64_t *npos_out,
                         std::vector<uint32_t> &pjs) {
  pjs.clear();
  const u64 first = wlo + ((cls + g_mod - wlo % g_mod) % g_mod);
  *first_out = first;
  if (wlo >= whi || first >= whi) { *npos_out = 0; return 0; }
  const u64 npos = (whi - first + g_mod - 1) / g_mod;
  *npos_out = npos;

  std::vector<uint8_t> isp(npos, 1);
  for (size_t i = 0; i < g_base.size(); ++i) {
    u64 l = g_base[i];
    if (g_mod % l == 0) continue;
    if ((u128)l * l >= whi) break;
    u64 j0 = mulmod((l - first % l) % l, g_invmod[i], l);
    u64 jmin = 0;
    if (first < (u128)l * l) jmin = (u64)(((u128)l * l - first + g_mod - 1) / g_mod);
    u64 j = j0;
    if (j < jmin) j += ((jmin - j + l - 1) / l) * l;
    for (; j < npos; j += l) isp[j] = 0;
  }
  for (u64 j = 0; j < npos && first + g_mod * j < 25; ++j) isp[j] = 0;

  pjs.reserve(npos / 8 + 8);
  for (u64 j = 0; j < npos; ++j) if (isp[j]) pjs.push_back((uint32_t)j);
  return (uint32_t)pjs.size();
}
