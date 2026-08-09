// tables.cu -- builds and uploads the device tables declared in tables.cuh (Task 2).
//
// Reference values come exclusively through the ref_host.h accessors
// (ref_jacobi/ref_legendre/ref_rnf/ref_rfac/ref_facp*), never a second
// "#include ../rung_scan.cpp" -- that TU (ref_host.cpp) already owns the reference's
// static internals; duplicating the include would ODR-violate. See RUNBOOK.md and
// ref_host.cpp/.h for the reuse mechanism.
#include "common.cuh"
#include "tables.cuh"
#include "ref_host.h"
#include <cstdio>
#include <cstring>
#include <vector>

static int8_t  h_jac[TABLE_STRIDE * TABLE_STRIDE];
static int8_t  h_leg[TABLE_STRIDE * 4 * TABLE_STRIDE];
static uint8_t h_rnf[TABLE_STRIDE];

// Device allocations, populated by upload_tables(). Task 2 scope: one upload per
// process (rung_scan_cuda has a single call site, same as Task 1); no free-on-exit.
static uint32_t *d_facp     = nullptr;
static uint64_t *d_invastep = nullptr;
static int8_t   *d_jac      = nullptr;
static int8_t   *d_leg      = nullptr;
static uint8_t  *d_rnf      = nullptr;
static uint32_t *d_base     = nullptr;
static uint64_t *d_invmod   = nullptr;

// Populates h_jac/h_leg from the reference's own jacobi()/legendre_sym(), via the
// ref_host accessors, for every odd r == 3 (mod 4), r <= TABLE_RMAX. All other rows
// stay zero.
static void build_host_tables() {
  memset(h_jac, 0, sizeof(h_jac));
  memset(h_leg, 0, sizeof(h_leg));
  memset(h_rnf, 0, sizeof(h_rnf));
  for (int r = 3; r <= TABLE_RMAX; r += 4) {
    for (int x = 0; x < r; ++x)
      h_jac[r * TABLE_STRIDE + x] = (int8_t)ref_jacobi((uint64_t)x, (uint64_t)r);
    const int nf = ref_rnf(r);
    h_rnf[r] = (uint8_t)nf;
    for (int fi = 0; fi < nf; ++fi) {
      const uint64_t pf = (uint64_t)ref_rfac(r, fi);
      for (int x = 0; x < r; ++x) {
        const uint64_t xm = (uint64_t)x % pf;
        const int v = (xm == 0) ? 0 : ref_legendre(xm, pf);
        h_leg[(r * 4 + fi) * TABLE_STRIDE + x] = (int8_t)v;
      }
    }
  }
}

DevTables upload_tables(uint64_t hi, bool hard840) {
  // Free any previous upload rather than leaking it. The original Task 2 note said
  // repeated calls "replace (leak) the previous allocation" on the grounds that there
  // was a single call site; --verify now calls this once per population mode, so that
  // stopped being true.
  for (void *p : {(void *)d_facp, (void *)d_invastep, (void *)d_jac, (void *)d_leg,
                  (void *)d_rnf, (void *)d_base, (void *)d_invmod})
    if (p) cudaFree(p);
  d_facp = nullptr; d_invastep = nullptr; d_jac = nullptr; d_leg = nullptr;
  d_rnf = nullptr; d_base = nullptr; d_invmod = nullptr;

  ref_build_tables(hi, hard840);   // populates g_facp/g_invastep for this hi/mode
  build_host_tables();             // populates h_jac/h_leg (hi/mode-independent)

  const uint32_t nfacp = ref_facp_count();
  std::vector<uint32_t> h_facp(nfacp);
  std::vector<uint64_t> h_invastep(nfacp);
  for (uint32_t i = 0; i < nfacp; ++i) {
    h_facp[i]     = (uint32_t)ref_facp(i);   // all factor primes fit under 2^32, SCALING.md S3.2
    h_invastep[i] = ref_invastep(i);
  }

  CUDA_CHECK(cudaMalloc(&d_jac, sizeof(h_jac)));
  CUDA_CHECK(cudaMalloc(&d_leg, sizeof(h_leg)));
  CUDA_CHECK(cudaMalloc(&d_rnf, sizeof(h_rnf)));
  const uint32_t nbase = ref_base_count();
  std::vector<uint32_t> h_base(nbase);
  std::vector<uint64_t> h_invmod(nbase);
  for (uint32_t i = 0; i < nbase; ++i) {
    h_base[i]   = (uint32_t)ref_base(i);   // <= sqrt(10^18) < 2^32, SCALING.md S3.2
    h_invmod[i] = ref_invmod(i);
  }
  CUDA_CHECK(cudaMalloc(&d_base, (size_t)nbase * sizeof(uint32_t)));
  CUDA_CHECK(cudaMalloc(&d_invmod, (size_t)nbase * sizeof(uint64_t)));
  CUDA_CHECK(cudaMemcpy(d_base, h_base.data(),
                        (size_t)nbase * sizeof(uint32_t), cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(d_invmod, h_invmod.data(),
                        (size_t)nbase * sizeof(uint64_t), cudaMemcpyHostToDevice));

  CUDA_CHECK(cudaMalloc(&d_facp, (size_t)nfacp * sizeof(uint32_t)));
  CUDA_CHECK(cudaMalloc(&d_invastep, (size_t)nfacp * sizeof(uint64_t)));

  CUDA_CHECK(cudaMemcpy(d_jac, h_jac, sizeof(h_jac), cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(d_leg, h_leg, sizeof(h_leg), cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(d_rnf, h_rnf, sizeof(h_rnf), cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(d_facp, h_facp.data(),
                        (size_t)nfacp * sizeof(uint32_t), cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(d_invastep, h_invastep.data(),
                        (size_t)nfacp * sizeof(uint64_t), cudaMemcpyHostToDevice));

  DevTables t;
  t.facp = d_facp; t.invastep = d_invastep; t.nfacp = nfacp;
  t.jac  = d_jac;  t.leg      = d_leg;      t.rnf   = d_rnf;
  t.base = d_base; t.invmod   = d_invmod;   t.nbase = nbase;
  return t;
}

int check_tables() {
  // The r<=TABLE_RMAX Jacobi/Legendre tables are pure number theory on r -- they
  // don't depend on hi/hard840. g_facp/g_invastep do, but this task's required test
  // (per the brief) is jac/leg equality; use the same small range self_test() uses
  // so `--check-tables` stands alone with no --lo/--hi of its own.
  DevTables t = upload_tables(100000, false);

  // Read the DEVICE tables back -- this exercises the actual upload path (host
  // stage -> cudaMemcpy H2D -> cudaMemcpy D2H), not just the host staging arrays,
  // so a copy/alloc bug would show up here too.
  static int8_t  rt_jac[TABLE_STRIDE * TABLE_STRIDE];
  static int8_t  rt_leg[TABLE_STRIDE * 4 * TABLE_STRIDE];
  static uint8_t rt_rnf[TABLE_STRIDE];
  CUDA_CHECK(cudaMemcpy(rt_jac, t.jac, sizeof(rt_jac), cudaMemcpyDeviceToHost));
  CUDA_CHECK(cudaMemcpy(rt_leg, t.leg, sizeof(rt_leg), cudaMemcpyDeviceToHost));
  CUDA_CHECK(cudaMemcpy(rt_rnf, t.rnf, sizeof(rt_rnf), cudaMemcpyDeviceToHost));

  int bad = 0, njac = 0, nleg = 0;
  for (int r = 3; r <= TABLE_RMAX; r += 4) {
    if ((int)rt_rnf[r] != ref_rnf(r)) {
      printf("rnf mismatch r=%d want=%d got=%d\n", r, ref_rnf(r), (int)rt_rnf[r]);
      ++bad;
    }
    for (int x = 0; x < r; ++x) {
      const int want = ref_jacobi((uint64_t)x, (uint64_t)r);   // fresh call, not reused
      const int got  = rt_jac[r * TABLE_STRIDE + x];
      ++njac;
      if (want != got) {
        printf("jac mismatch r=%d x=%d want=%d got=%d\n", r, x, want, got);
        ++bad;
      }
    }
    const int nf = ref_rnf(r);
    for (int fi = 0; fi < nf; ++fi) {
      const uint64_t pf = (uint64_t)ref_rfac(r, fi);
      for (int x = 0; x < r; ++x) {
        const uint64_t xm = (uint64_t)x % pf;
        const int want = (xm == 0) ? 0 : ref_legendre(xm, pf);   // fresh call
        const int got  = rt_leg[(r * 4 + fi) * TABLE_STRIDE + x];
        ++nleg;
        if (want != got) {
          printf("leg mismatch r=%d fi=%d x=%d want=%d got=%d\n", r, fi, x, want, got);
          ++bad;
        }
      }
    }
  }
  printf("checked %d jac + %d leg = %d pairs against the reference's own "
         "jacobi()/legendre_sym()\n", njac, nleg, njac + nleg);
  printf(bad ? "TABLES FAIL (%d)\n" : "TABLES OK\n", bad);
  return bad;
}
