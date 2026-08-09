#pragma once
#include <cstdint>
#include <vector>
int  rung_scan_main(int argc, char **argv);   // the reference's own main, renamed
void ref_build_tables(uint64_t hi, bool hard840);
int  ref_verify();

// Narrow accessors onto the reference's `static` internals (jacobi, legendre_sym,
// g_rfac, g_rnf, g_facp, g_invastep), for other translation units -- e.g.
// cuda/tables.cu -- that need reference values but must NOT #include
// "../rung_scan.cpp" a second time (that would ODR-violate: see RUNBOOK.md and
// ref_host.cpp). Each one is a direct pass-through, no logic duplicated.
int      ref_jacobi(uint64_t a, uint64_t n);     // wraps jacobi(a, n)
int      ref_legendre(uint64_t a, uint64_t p);   // wraps legendre_sym(a, p)
int      ref_rnf(int r);                         // wraps g_rnf[r]
int      ref_rfac(int r, int i);                 // wraps g_rfac[r][i]
uint32_t ref_facp_count();                       // g_facp.size()
uint64_t ref_facp(uint32_t i);                   // g_facp[i]
uint64_t ref_invastep(uint32_t i);               // g_invastep[i]
// The PRIMALITY sieve's base list -- primes <= sqrt(hi)+2, longer than g_facp, with
// g_invmod[i] = g_mod^{-1} mod g_base[i] (0 for the primes dividing g_mod).
uint32_t ref_base_count();                       // g_base.size()
uint64_t ref_base(uint32_t i);                   // g_base[i]
uint64_t ref_invmod(uint32_t i);                 // g_invmod[i]

// Wheel parameters, set by ref_build_tables(): p steps by ref_mod(), A = (p+r)/4
// steps by ref_astep(), and the population is the ref_nclasses() residue classes.
uint64_t ref_mod();
uint64_t ref_astep();
uint32_t ref_nclasses();
uint64_t ref_class(uint32_t i);
bool     ref_is_prime(uint64_t n);               // wraps is_prime(n)

// Wraps the reference's factorA() -- trial division by primes <= 65536 then
// Miller-Rabin/Pollard. A genuinely DIFFERENT algorithm from the sieve walk the CUDA
// pipeline uses, which is what makes it worth diffing against. Writes at most 16
// (prime, exponent) pairs; returns the count.
int ref_factorA(uint64_t n, uint64_t *pr, uint8_t *ex);

// Wraps the reference's test_rung() -- the classifier the CUDA one is diffed against.
// Factors A internally with the reference's own factorA, so the returned level depends
// on nothing the GPU produced. Return values are the reference's `enum Level`
// (rung_scan.cpp:214): 0 HIT, 1 J, 2 RC, 3 S, 4 R -- the same numbering as DevLevel.
int ref_test_rung(uint64_t p, uint64_t A, uint64_t r);

// The reference's g_pairpath counter -- how many times test_rung took the gcd(q,r) != 1
// 2-D pair path. SUMMARY prints it, and it must be 0. The GPU census reports the
// reference's counter because its own CPU-tail handoff calls test_rung directly.
uint64_t ref_pairpath();
void     ref_reset_pairpath();

// One run of the reference scanner, in process, with emission captured instead of
// printed. This is what --verify diffs the GPU census against: comparing to the
// reference's live numbers is strictly stronger than comparing to literals copied out
// of it, and it costs nothing extra. Certificate lists are flattened (p, A, r) triples.
struct RefRun {
  uint64_t nprimes = 0, jac = 0, rc = 0, sup = 0, res = 0, escalated = 0;
  uint64_t pairpath = 0, maxr = 0, maxr_p = 0;
  uint64_t hist[256] = {0};                 // indexed by rung; RCAP is 255
  std::vector<uint64_t> rc_list, sup_list, res_list, deep_list, sample_list;
};
void ref_scan_range(uint64_t lo, uint64_t hi, bool hard840, uint64_t rmax,
                    bool emitRes, bool emitSup, uint64_t deepThreshold,
                    uint64_t sampleEvery, int spanlog, RefRun *out);

// The primality sieve over one residue class of one window. This mirrors
// rung_scan.cpp:482-501, which lives inline inside process_class() and is not
// separately callable; it is host-side scaffolding for the --gate harness only (and
// for Task 6, which moves the sieve to the device). It is NOT what --gate tests: the
// factorization comparison is GPU-vs-factorA at whatever positions this hands over,
// so an error here changes which A are checked, it cannot manufacture a pass.
// Fills pjs with the ascending positions j whose p = first + ref_mod()*j is prime.
uint32_t ref_sieve_class(uint64_t wlo, uint64_t whi, uint64_t cls,
                         uint64_t *first_out, uint64_t *npos_out,
                         std::vector<uint32_t> &pjs);
