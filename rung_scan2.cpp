// rung_scan2.cpp -- the OPTIMIZED second CPU scanner. rung_scan.cpp stays the frozen
// reference; this binary earns trust the same way the CUDA port did: identical
// statistical SUMMARY fields and an identical sorted certificate set on every range
// where both run (tests/diff2.sh), plus the same --verify self-test against the
// Python/SymPy-derived constants.
//
// It is an independent implementation of the same algorithm with four optimizations
// back-ported from the CUDA route (GPU_SCALE.md, CPU annex):
//
//   1. INCREMENTAL FIRST-HIT.  The reference recomputes, per rung, one mulmod per
//      factor prime: j0 = -A0 * astep^{-1} (mod l).  But stepping r -> r+4 increments
//      A0 by exactly 1, so j0 shifts by -astep^{-1} (mod l): a subtract and a
//      conditional add replace the mulmod for every prime after the first rung.
//   2. BITSET DP for r <= 63.  The divisor-reachability DP lives in one uint64
//      instead of a bool[256] array.  Rungs above 63 keep a byte-array DP with the
//      reference's exact semantics.
//   3. LEVEL-S CONFINEMENT.  The support-subgroup closure runs only at rungs where a
//      level-S failure is possible at all: since 4A = p + r the support subgroup
//      always contains 4, which confines level S to {51,119,123,187,195,219,255}
//      below RCAP (derived by es_levels.py; verified against the 10^13 and 10^15
//      censuses -- all observed level-S failures are at r = 51).  Everywhere else a
//      failure past level RC is level R directly.  This cannot change output -- it
//      skips a test that provably returns false -- and diff2.sh checks anyway.
//   4. JACOBI/LEGENDRE TABLES.  All certificate symbols have arguments reduced mod
//      r <= 255; both are table lookups here.  This is the "known, deliberately
//      unexploited" optimization RUNBOOK.md reserves away from the reference.
//
// Everything else mirrors the reference's behavior exactly: wheel setup, window
// decomposition, BULK_MIN handoff, Pollard-Brent with the cycle-abort fix, emission
// tags and order-of-streaming, SUMMARY format, --shard arithmetic, self-test.
// --occupancy is NOT supported here (diagnostic mode; use the reference).
//
// BUILD  make rung_scan2     TEST  ./tests/diff2.sh 0 100000000 --hard840
// Public domain / CC0.  No warranty.

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <vector>
#include <array>
#include <string>
#include <algorithm>
#include <chrono>
#include <numeric>
#ifdef _OPENMP
#include <omp.h>
#endif

using namespace std;
typedef uint64_t u64;
typedef __uint128_t u128;

static const int RCAP = 255;

// ------------------------------------------------------------------ arithmetic

static inline u64 mulmod(u64 a, u64 b, u64 m) { return (u64)((u128)a * b % m); }

static u64 powmod(u64 b, u64 e, u64 m) {
  u64 r = 1; b %= m;
  while (e) { if (e & 1) r = mulmod(r, b, m); b = mulmod(b, b, m); e >>= 1; }
  return r;
}

static bool is_prime(u64 n) {                        // deterministic for n < 3.3e24
  if (n < 2) return false;
  for (u64 sp : {2ull,3ull,5ull,7ull,11ull,13ull,17ull,19ull,23ull,29ull,31ull,37ull}) {
    if (n == sp) return true;
    if (n % sp == 0) return false;
  }
  u64 d = n - 1; int s = 0;
  while (!(d & 1)) { d >>= 1; ++s; }
  for (u64 a : {2ull,3ull,5ull,7ull,11ull,13ull,17ull,19ull,23ull,29ull,31ull,37ull}) {
    u64 x = powmod(a, d, n);
    if (x == 1 || x == n - 1) continue;
    bool comp = true;
    for (int i = 1; i < s; ++i) { x = mulmod(x, x, n); if (x == n - 1) { comp = false; break; } }
    if (comp) return false;
  }
  return true;
}

// Brent's variant, WITH the cycle-abort fix (SCALING.md S1.1): when y == x the cycle
// has closed on a degenerate c -- abort this c instead of substituting 1 into the
// product and burning the whole r budget noticing nothing.
static u64 pollard_brent(u64 n) {
  if ((n & 1) == 0) return 2;
  for (u64 c = 1; ; ++c) {
    u64 y = 2, m = 128, g = 1, r = 1, qacc = 1, x = 0, ys = 0;
    auto f = [&](u64 v) { return (mulmod(v, v, n) + c) % n; };
    while (g == 1) {
      x = y;
      for (u64 i = 0; i < r; ++i) y = f(y);
      for (u64 k = 0; k < r && g == 1; k += m) {
        ys = y;
        u64 lim = min(m, r - k);
        for (u64 i = 0; i < lim; ++i) {
          y = f(y);
          if (y == x) { g = n; break; }
          u64 diff = x > y ? x - y : y - x;
          qacc = mulmod(qacc, diff, n);
        }
        if (g == n) break;
        g = std::gcd(qacc, n);
      }
      r <<= 1;
      if (r > (1ull << 26)) break;
    }
    if (g == n || g == 1) {
      g = 1;
      while (g == 1) {
        ys = f(ys);
        u64 diff = x > ys ? x - ys : ys - x;
        g = std::gcd(diff ? diff : n, n);
        if (ys == x) break;
      }
    }
    if (g != 1 && g != n) return g;
  }
}

static int jacobi(u64 a, u64 n) {
  a %= n;
  int t = 1;
  while (a) {
    while ((a & 1) == 0) { a >>= 1; u64 m8 = n & 7; if (m8 == 3 || m8 == 5) t = -t; }
    u64 tmp = a; a = n; n = tmp;
    if ((a & 3) == 3 && (n & 3) == 3) t = -t;
    a %= n;
  }
  return n == 1 ? t : 0;
}

static inline int legendre_sym(u64 a, u64 p) {       // p an odd prime, gcd(a,p)=1
  a %= p;
  if (a == 0) return 0;
  return powmod(a, (p - 1) / 2, p) == 1 ? 1 : -1;
}

// ------------------------------------------------------------------ factoring

static const int MAXF = 16;
struct Fac {
  u64 pr[MAXF]; uint8_t ex[MAXF]; uint8_t k = 0;
  inline void add(u64 p, int e) {
    for (int i = 0; i < k; ++i) if (pr[i] == p) { ex[i] = (uint8_t)(ex[i] + e); return; }
    if (k >= MAXF) { fprintf(stderr, "FATAL: more than %d distinct prime factors\n", MAXF); abort(); }
    pr[k] = p; ex[k] = (uint8_t)e; ++k;
  }
};

static vector<u64> g_small;
static vector<u64> g_base;
static u64 g_mod   = 24;
static u64 g_astep = 6;
static vector<u64> g_classes = {1};
static vector<u64> g_stripp  = {2, 3};
static vector<u64> g_invmod;
static vector<u64> g_facp;
static vector<u64> g_invastep;
static const u64 TD_BOUND = 65536;

static u64 isqrt_u64(u64 n) {
  u64 s = (u64)sqrtl((long double)n);
  while (s > 0 && s * s > n) --s;
  while ((s + 1) * (s + 1) <= n) ++s;
  return s;
}

static void factor_hard(u64 n, Fac &f) {
  if (n == 1) return;
  if (is_prime(n)) { f.add(n, 1); return; }
  u64 s = isqrt_u64(n);
  if (s * s == n) { Fac g; factor_hard(s, g); for (int i = 0; i < g.k; ++i) f.add(g.pr[i], 2 * g.ex[i]); return; }
  u64 d = pollard_brent(n);
  factor_hard(d, f);
  factor_hard(n / d, f);
}

static void factorA(u64 n, Fac &f) {
  f.k = 0;
  for (u64 sp : g_small) {
    if (sp * sp > n) break;
    if (n % sp == 0) { int e = 0; while (n % sp == 0) { n /= sp; ++e; } f.add(sp, e); }
  }
  if (n > 1) {
    if (n <= TD_BOUND * TD_BOUND || is_prime(n)) f.add(n, 1);
    else factor_hard(n, f);
  }
}

// ------------------------------------------------------------------ rung tables

enum Level { LV_HIT = 0, LV_J = 1, LV_RC = 2, LV_S = 3, LV_R = 4 };

static u64 g_pairpath = 0;

static int g_rfac[RCAP + 1][4];
static int g_rnf[RCAP + 1];
static int g_phi[RCAP + 1];

static void init_phi() {
  for (int r = 1; r <= RCAP; ++r) {
    int n = r, res = r;
    for (int d = 2; d * d <= n; ++d)
      if (n % d == 0) { while (n % d == 0) n /= d; res -= res / d; }
    if (n > 1) res -= res / n;
    g_phi[r] = res;
  }
}

static void init_rfac() {
  for (int r = 3; r <= RCAP; r += 2) {
    int n = 0, m = r;
    for (int d = 3; d * d <= m; d += 2)
      if (m % d == 0) { g_rfac[r][n++] = d; while (m % d == 0) m /= d; }
    if (m > 1) g_rfac[r][n++] = m;
    g_rnf[r] = n;
  }
}

// Optimization 4: per-rung symbol tables. jt[r][x] = Jacobi (x/r); lt[r][fi][x] =
// Legendre of x at the fi-th distinct prime factor of r. Arguments in the certificate
// path are always reduced mod r first, so x < r <= 255 covers every call.
static int8_t g_jt[RCAP + 1][RCAP + 1];
static int8_t g_lt[RCAP + 1][4][RCAP + 1];

static void init_symtabs() {
  for (int r = 3; r <= RCAP; r += 2) {
    for (int x = 0; x < r; ++x) g_jt[r][x] = (int8_t)jacobi((u64)x, (u64)r);
    for (int fi = 0; fi < g_rnf[r]; ++fi) {
      const u64 pf = (u64)g_rfac[r][fi];
      for (int x = 0; x < r; ++x) {
        const u64 xm = (u64)x % pf;
        g_lt[r][fi][x] = (int8_t)(xm == 0 ? 0 : legendre_sym(xm, pf));
      }
    }
  }
}

// Optimization 3: the rungs that can carry a level-S failure at all. Derived by
// brute-force subgroup enumeration (es_levels.py): a rung qualifies only if some
// H <= (Z/r)* has 4 in H, -1 not in H, and (-1)H a square in the quotient -- and
// 4 in H is forced by 4A = p + r. Everywhere else, past-RC failures are level R.
static bool g_sadm[RCAP + 1];
static void init_sadm() {
  memset(g_sadm, 0, sizeof(g_sadm));
  for (int r : {51, 119, 123, 187, 195, 219, 255}) g_sadm[r] = true;
}

// ------------------------------------------------------------------ rung test

// `res` are the residues mod r of p and of the primes of A -- already reduced by
// the caller (test_rung computes them once, Barrett-fast, for the DP bases).
static int realchar_mask(const int *res, int nres, u64 r, int target) {
  const int nf = g_rnf[r];
  auto bvec = [&](int x) {                            // x already reduced mod r
    int b = 0;
    for (int i = 0; i < nf; ++i)
      if (g_lt[r][i][x] == -1) b |= (1 << i);
    return b;
  };
  int bt = bvec(target);
  int bl[24], nb = 0;
  for (int i = 0; i < nres; ++i) bl[nb++] = bvec(res[i]);
  for (int S = 1; S < (1 << nf); ++S) {
    bool triv = true;
    for (int i = 0; i < nb && triv; ++i) triv = (__builtin_popcount(S & bl[i]) & 1) == 0;
    if (triv && (__builtin_popcount(S & bt) & 1) == 1) return S;
  }
  return 0;
}

static bool support_blocks(const int *res, int nres, u64 r, int target) {
  const int m = (int)r;
  u64 gens[24]; int ng = 0;
  for (int i = 0; i < nres; ++i) gens[ng++] = (u64)res[i];
  bool inH[RCAP + 1]; memset(inH, 0, sizeof(bool) * m);
  int stack[RCAP + 2], sp = 0, one = 1 % m;
  inH[one] = true; stack[sp++] = one;
  while (sp) {
    int x = stack[--sp];
    for (int i = 0; i < ng; ++i) {
      int y = (int)((u64)x * gens[i] % m);
      if (!inH[y]) { inH[y] = true; stack[sp++] = y; }
    }
  }
  return !inH[target];
}

// Optimization 5 (branchless/dep-chain pass): all DP products are < m*m <= 3969 for
// the uint64 path, so `x % m` is a 4 KB lookup table rebuilt only when the rung
// changes. The hardware div this replaces is ~25 cycles, unpipelined, and sits on
// the DP's serial dependency chain; the table load is 4-5 cycles from L1 and the
// table is hot for the entire rung (every active prime shares it). Verified in the
// generated assembly: test_rung compiled to 47 div instructions before this.
static thread_local int g_modm = 0;
static thread_local uint8_t g_modtab[4096];
static thread_local uint8_t g_coptab[RCAP + 1];   // gcd(x, m) == 1, x < m
static thread_local u64 g_barM = 0;               // floor(2^64 / m), for Barrett

// Optimization 7 (Barrett): x % m for 64-bit x via one mulhi + one multiply + one
// conditional subtract, ~5 cycles against idiv's ~18 unpipelined. With
// M = floor(2^64/m) the estimate q = mulhi(x, M) is Q or Q-1, so a single
// correction bounds the remainder. m is odd >= 3, never a power of two, so
// ~0ull/m equals floor(2^64/m). Rebuilt with the mod table, once per rung.
static inline u64 bmod(u64 x, u64 m) {
  u64 q = (u64)(((u128)x * g_barM) >> 64);
  u64 rr = x - q * m;
  if (rr >= m) rr -= m;
  return rr;
}

static inline void ensure_modtab(int m) {
  if (g_modm == m) return;
  for (int x = 0; x < 4096; ++x) g_modtab[x] = (uint8_t)(x % m);
  for (int x = 0; x < m; ++x) {
    int a = x, b = m;
    while (b) { int t = a % b; a = b; b = t; }
    g_coptab[x] = (uint8_t)(a == 1);
  }
  g_barM = ~0ull / (u64)m;
  g_modm = m;
}

static Level test_rung(u64 p, u64 A, u64 r, const Fac &fA) {
  const int m = (int)r;
  ensure_modtab(m);                                  // mod table + coprime + Barrett M
  const u64 pm = bmod(p, m), Am = bmod(A, m);
  const u64 qmod = bmod(pm * Am, m);                 // product < m^2 <= 65025 << 2^64
  const int target = (int)(qmod ? (u64)m - qmod : 0);

  u64 base[24]; int expo[24]; int nb = 0;
  int res[24];                                       // the residues, computed ONCE
  base[nb] = pm; res[nb] = (int)pm; expo[nb] = 2; ++nb;
  for (int i = 0; i < fA.k; ++i) {
    const u64 fm = bmod(fA.pr[i], m);
    base[nb] = fm; res[nb] = (int)fm; expo[nb] = 2 * fA.ex[i]; ++nb;
  }

  bool hit;
  if (g_coptab[qmod]) {
    if (m <= 63) {
      // Optimization 2: the reachable-residue set is one uint64. The transition
      // T = union over e of S * base^e is a bit permutation applied per exponent.
      // Optimization 5: every `% m` here is a table read (see ensure_modtab).
      uint64_t S = 1ull << (1 % m);
      for (int i = 0; i < nb; ++i) {
        uint64_t T = 0;
        int pw = 1 % m;
        const int bi = (int)base[i];               // < m after the reduction below
        for (int e = 0; e <= expo[i]; ++e) {
          uint64_t s = S;
          while (s) {
            const int a = __builtin_ctzll(s);
            s &= s - 1;
            T |= 1ull << g_modtab[a * pw];
          }
          pw = g_modtab[pw * bi];
        }
        S = T;
      }
      hit = (S >> target) & 1ull;
    } else {
      bool S[RCAP + 1], T[RCAP + 1];
      memset(S, 0, sizeof(bool) * m);
      S[1 % m] = true;
      for (int i = 0; i < nb; ++i) {
        memset(T, 0, sizeof(bool) * m);
        int pw = 1;
        for (int e = 0; e <= expo[i]; ++e) {
          for (int a = 0; a < m; ++a) if (S[a]) T[(int)bmod((u64)a * pw, m)] = true;
          pw = (int)bmod((u64)pw * base[i], m);
        }
        memcpy(S, T, sizeof(bool) * m);
      }
      hit = S[target];
    }
  } else {
    // Unreachable for r < 3p; retained as a guard, usage reported (as the reference).
#ifdef _OPENMP
#pragma omp atomic
#endif
    ++g_pairpath;
    static thread_local vector<uint8_t> S, T;
    S.assign((size_t)m * m, 0); T.assign((size_t)m * m, 0);
    S[(size_t)(1 % m) * m + (1 % m)] = 1;
    for (int i = 0; i < nb; ++i) {
      fill(T.begin(), T.end(), 0);
      int E = expo[i];
      if (E > 126) E = 126;
      int pw[128]; pw[0] = 1;
      for (int e = 1; e <= E; ++e) pw[e] = (int)((u64)pw[e - 1] * base[i] % m);
      for (int a = 0; a < m; ++a) for (int b = 0; b < m; ++b) if (S[(size_t)a * m + b])
        for (int e = 0; e <= E; ++e)
          T[(size_t)((u64)a * pw[e] % m) * m + (int)((u64)b * pw[E - e] % m)] = 1;
      S.swap(T);
    }
    hit = S[(size_t)target * m + target] != 0;
  }
  if (hit) return LV_HIT;

  if (g_coptab[qmod]) {
    // Optimization 4 in action: every symbol below is a table read over the
    // already-reduced residues.
    bool allQR = true;
    for (int i = 0; allQR && i < nb; ++i) allQR = (g_jt[r][res[i]] == 1);
    if (allQR && g_jt[r][target] == -1) return LV_J;
    if (realchar_mask(res, nb, r, target)) return LV_RC;
    if (g_sadm[r] && support_blocks(res, nb, r, target)) return LV_S;   // optimization 3
  }
  return LV_R;
}

// ------------------------------------------------------------------ statistics

struct Stats {
  u64 nprimes = 0;
  u64 hist[RCAP + 1] = {0};
  u64 jac = 0, rc = 0, sup = 0, res = 0, escalated = 0;
  u64 maxr = 0, maxr_p = 0;
  vector<array<u64,3>> res_list, rc_list, sup_list, deep_list, sample_list;
  void merge(const Stats &o) {
    nprimes += o.nprimes; jac += o.jac; rc += o.rc; sup += o.sup; res += o.res;
    escalated += o.escalated;
    for (int i = 0; i <= RCAP; ++i) hist[i] += o.hist[i];
    if (o.maxr > maxr || (o.maxr == maxr && o.maxr_p && (!maxr_p || o.maxr_p < maxr_p)))
      { maxr = o.maxr; maxr_p = o.maxr_p; }
    res_list.insert(res_list.end(), o.res_list.begin(), o.res_list.end());
    rc_list.insert(rc_list.end(), o.rc_list.begin(), o.rc_list.end());
    sup_list.insert(sup_list.end(), o.sup_list.begin(), o.sup_list.end());
    deep_list.insert(deep_list.end(), o.deep_list.begin(), o.deep_list.end());
    sample_list.insert(sample_list.end(), o.sample_list.begin(), o.sample_list.end());
  }
};

struct Opts {
  u64 rmax = 127;
  bool emitRes = false, emitSup = true;
  u64 deepThreshold = 0, sampleEvery = 0;
  int spanlog = 20;
  u64 progress = 0;
  bool hard840 = false;
};

static bool g_stream = true;

static void flush_emits(Stats &st) {
  if (!g_stream) return;
  if (st.rc_list.empty() && st.sup_list.empty() && st.res_list.empty()
      && st.deep_list.empty() && st.sample_list.empty()) return;
#ifdef _OPENMP
#pragma omp critical(emitout)
#endif
  {
    for (auto &e : st.rc_list)
      printf("REALCHAR %llu %llu %llu\n", (unsigned long long)e[0], (unsigned long long)e[1], (unsigned long long)e[2]);
    for (auto &e : st.sup_list)
      printf("SUPPORT %llu %llu %llu\n", (unsigned long long)e[0], (unsigned long long)e[1], (unsigned long long)e[2]);
    for (auto &e : st.res_list)
      printf("RESIDUAL %llu %llu %llu\n", (unsigned long long)e[0], (unsigned long long)e[1], (unsigned long long)e[2]);
    for (auto &e : st.deep_list)
      printf("%s %llu %llu %llu\n", e[1] ? "DEEP" : "ESCALATE",
             (unsigned long long)e[0], (unsigned long long)e[1], (unsigned long long)e[2]);
    for (auto &e : st.sample_list)
      printf("SAMPLE %llu %llu %llu\n", (unsigned long long)e[0], (unsigned long long)e[1], (unsigned long long)e[2]);
  }
  st.rc_list.clear(); st.sup_list.clear(); st.res_list.clear();
  st.deep_list.clear(); st.sample_list.clear();
}

static inline void record_hit(u64 p, u64 A, u64 r, const Opts &opt, Stats &st) {
  st.hist[r]++;
  if (r > st.maxr) { st.maxr = r; st.maxr_p = p; }
  if (opt.deepThreshold && r >= opt.deepThreshold) st.deep_list.push_back({p, A, r});
  if (opt.sampleEvery && (p % opt.sampleEvery) < 24) st.sample_list.push_back({p, A, r});
}

static inline void record_fail(Level lv, u64 p, u64 A, u64 r, const Opts &opt, Stats &st) {
  if (lv == LV_J) st.jac++;
  else if (lv == LV_RC) { st.rc++; if (opt.emitSup) st.rc_list.push_back({p, A, r}); }
  else if (lv == LV_S) { st.sup++; if (opt.emitSup) st.sup_list.push_back({p, A, r}); }
  else { st.res++; if (opt.emitRes) st.res_list.push_back({p, A, r}); }
}

// ------------------------------------------------------------------ window scan

static void process_class(u64 wlo, u64 whi, u64 cls, const Opts &opt, Stats &st) {
  if (wlo >= whi) return;
  u64 first = wlo + ((cls + g_mod - wlo % g_mod) % g_mod);
  if (first >= whi) return;
  const u64 npos = (whi - first + g_mod - 1) / g_mod;

  vector<uint8_t> isp(npos, 1);
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

  vector<uint32_t> pidx(npos, 0), pjs;
  pjs.reserve(npos / 8 + 8);
  for (u64 j = 0; j < npos; ++j)
    if (isp[j]) { pidx[j] = (uint32_t)pjs.size(); pjs.push_back((uint32_t)j); }
  const u64 npr = pjs.size();
  if (!npr) return;
  st.nprimes += npr;

  vector<uint8_t> act(npr, 1);
  vector<Fac> facs(npr);
  vector<u64> remA(npr);
  u64 nact = npr;
  const u64 BULK_MIN = 64;

  // Optimization 6 (read-mostly walk): liveness is FOLDED INTO isp -- a position's
  // byte dies when its prime first hits -- so the walk's filter is a single byte
  // load with no dependent pidx/act chase for the ~90% of touches that reject. The
  // survivors are gathered BRANCHLESSLY (unconditional store, guarded increment: no
  // mispredict on the quasi-random prime pattern) into `hitbuf`, and the expensive
  // part -- pidx chase, remA division, Fac append, i.e. all the read-modify-write --
  // runs afterward over a short dense list. The walk itself reads and never writes:
  // stores stop breaking the load pipeline, and dead positions cost one load.
  // Sized once for the smallest walked prime (>= 5 in either wheel): the gather can
  // never exceed npos/5 + 1 survivors-plus-slop, and never resizing again means no
  // zero-fill churn inside the rung loop.
  vector<uint32_t> hitbuf(npos / 5 + 2);

  // Optimization 1: per-prime first-hit offsets, carried across rungs. joff[i] holds
  // j0 for the UPCOMING rung for primes i < jvalid; primes past the watermark (first
  // rung, or the l*l <= Amax prefix grew) initialize with the mulmod, exactly the
  // computation the reference performs every rung for every prime.
  vector<uint32_t> joff(g_facp.size());
  size_t jvalid = 0;

  u64 r = 3;
  for (; r <= opt.rmax && nact >= BULK_MIN; r += 4) {
    const u64 A0 = (first + r) / 4;
    const u64 Amax = A0 + g_astep * (npos - 1);
    for (u64 k = 0; k < npr; ++k) if (act[k]) {
      u64 A = A0 + g_astep * pjs[k];
      facs[k].k = 0;
      for (u64 sp : g_stripp) {
        int e = 0; while (A % sp == 0) { A /= sp; ++e; }
        if (e) facs[k].add(sp, e);
      }
      remA[k] = A;
    }
    size_t walked = 0;
    for (size_t i = 0; i < g_facp.size(); ++i) {
      const u64 l = g_facp[i];
      if ((u128)l * l > Amax) break;
      walked = i + 1;
      if (g_astep % l == 0) continue;
      u64 j0;
      if (i < jvalid) {
        j0 = joff[i];
      } else {
        j0 = mulmod((l - A0 % l) % l, g_invastep[i], l);
      }
      {  // shift to the next rung's offset: A0 -> A0 + 1
        const u64 inv = g_invastep[i];
        joff[i] = (uint32_t)(j0 >= inv ? j0 - inv : j0 + l - inv);
      }
      // pass 1: read-only, branchless gather of live positions this prime divides
      uint32_t nh = 0;
      uint32_t *hb = hitbuf.data();
      for (u64 j = j0; j < npos; j += l) {
        hb[nh] = (uint32_t)j;
        nh += isp[j];
      }
      // pass 2: the RMW work, dense over the few survivors
      for (uint32_t t = 0; t < nh; ++t) {
        const uint32_t k = pidx[hb[t]];
        u64 v = remA[k];
        if (v % l == 0) { int ee = 0; do { v /= l; ++ee; } while (v % l == 0); remA[k] = v; facs[k].add(l, ee); }
      }
    }
    jvalid = walked;
    for (u64 k = 0; k < npr; ++k) if (act[k]) {
      if (remA[k] > 1) facs[k].add(remA[k], 1);
      const u64 p = first + g_mod * pjs[k], A = A0 + g_astep * pjs[k];
      Level lv = test_rung(p, A, r, facs[k]);
      if (lv == LV_HIT) {
        record_hit(p, A, r, opt, st);
        act[k] = 0; isp[pjs[k]] = 0;   // optimization 6: dead position, dead byte
        --nact;
      }
      else record_fail(lv, p, A, r, opt, st);
    }
  }

  for (u64 k = 0; k < npr; ++k) if (act[k]) {
    const u64 p = first + g_mod * pjs[k];
    bool done = false;
    for (u64 rr = r; rr <= opt.rmax; rr += 4) {
      const u64 A = (p + rr) / 4;
      Fac f; factorA(A, f);
      Level lv = test_rung(p, A, rr, f);
      if (lv == LV_HIT) { record_hit(p, A, rr, opt, st); done = true; break; }
      record_fail(lv, p, A, rr, opt, st);
    }
    if (!done) { st.escalated++; st.deep_list.push_back({p, 0, 0}); }
  }
}

static void process_window(u64 wlo, u64 whi, const Opts &opt, Stats &st) {
  for (u64 c : g_classes) process_class(wlo, whi, c, opt, st);
  flush_emits(st);
}

static void build_base_primes(u64 hi) {
  u64 lim = isqrt_u64(hi) + 2;
  vector<uint8_t> c(lim + 1, 0);
  for (u64 i = 2; i * i <= lim; ++i) if (!c[i]) for (u64 j = i * i; j <= lim; j += i) c[j] = 1;
  g_base.clear();
  for (u64 i = 2; i <= lim; ++i) if (!c[i]) g_base.push_back(i);
  g_invmod.assign(g_base.size(), 0);
  for (size_t i = 0; i < g_base.size(); ++i)
    if (g_mod % g_base[i]) g_invmod[i] = powmod(g_mod % g_base[i], g_base[i] - 2, g_base[i]);
  g_small.clear();
  for (u64 sp : g_base) { if (sp > TD_BOUND) break; g_small.push_back(sp); }
  u64 flim = isqrt_u64(hi / 4 + 256) + 2;
  g_facp.clear(); g_invastep.clear();
  for (u64 q : g_base) { if (q > flim) break; g_facp.push_back(q); }
  if (g_facp.empty() || g_facp.back() < flim) {
    for (u64 q = (g_facp.empty() ? 2 : g_facp.back() + 1); q <= flim; ++q)
      if (is_prime(q)) g_facp.push_back(q);
  }
  g_invastep.assign(g_facp.size(), 0);
  for (size_t i = 0; i < g_facp.size(); ++i)
    if (g_astep % g_facp[i]) g_invastep[i] = powmod(g_astep % g_facp[i], g_facp[i] - 2, g_facp[i]);
}

// ------------------------------------------------------------------ self-test

static int self_test_mode(bool hard840, int &fails);

static int self_test() {
  int fails = 0;
  self_test_mode(false, fails);
  printf("\n");
  self_test_mode(true, fails);
  printf(fails ? "\nSELF-TEST FAILED (%d checks)\n" : "\nSELF-TEST PASSED\n", fails);
  return fails ? 1 : 0;
}

static int self_test_mode(bool hard840, int &fails) {
  Opts opt; opt.rmax = 127; opt.emitRes = true; opt.emitSup = true;
  g_stream = false;
  g_pairpath = 0;
  if (hard840) { g_mod = 840; g_astep = 210; g_classes = {1,121,169,289,361,529}; g_stripp = {2,3,5,7}; }
  else         { g_mod = 24;  g_astep = 6;   g_classes = {1};                     g_stripp = {2,3}; }
  Stats st;
  init_phi(); init_rfac(); init_symtabs(); init_sadm();
  build_base_primes(100000);
  process_window(0, 100000, opt, st);

  auto check = [&](const char *what, u64 got, u64 want) {
    if (got != want) { printf("  FAIL %-30s got %llu want %llu\n", what,
                              (unsigned long long)got, (unsigned long long)want); ++fails; }
    else printf("  ok   %-30s %llu\n", what, (unsigned long long)got);
  };
  printf("SELF-TEST  range [0, 100000)  %s  vs Python/SymPy reference\n",
         hard840 ? "[--hard840: 6 square classes mod 840]" : "[default: 1 mod 24]");
  check("primes in class", st.nprimes, hard840 ? 273 : 1181);
  struct RN { u64 r, n; };
  static const RN expect24[]  = {{3,575},{7,475},{11,83},{15,11},{19,16},{23,15},{27,1},{31,5}};
  static const RN expect840[] = {{3,87}, {7,55}, {11,83},{15,11},{19,16},{23,15},{27,1},{31,5}};
  const RN *expect = hard840 ? expect840 : expect24;
  for (int i = 0; i < 8; ++i) {
    char buf[64]; snprintf(buf, sizeof buf, "depth r = %llu", (unsigned long long)expect[i].r);
    check(buf, st.hist[expect[i].r], expect[i].n);
  }
  check("level J  (Jacobi) failures", st.jac, hard840 ? 414 : 834);
  check("level RC (other real chi)", st.rc, 0);
  check("level S  (beyond real chi)", st.sup, 0);
  check("level R  (residual)", st.res, 20);
  check("total failures", st.jac + st.rc + st.sup + st.res, hard840 ? 434 : 854);
  check("max depth", st.maxr, 31);
  check("escalations", st.escalated, 0);
  check("pair-DP path uses", g_pairpath, 0);

  static const u64 RES20[20][3] = {
    {1201,303,11},{2521,633,11},{2521,635,19},{14401,3603,11},{21169,5297,19},
    {28921,7233,11},{28921,7235,19},{31081,7773,11},{31249,7817,19},{35809,8957,19},
    {37489,9377,19},{60601,15153,11},{64849,16217,19},{67369,16847,19},{67369,16849,27},
    {74209,18557,19},{83689,20927,19},{87481,21877,27},{94441,23613,11},{99961,24995,19}};
  sort(st.res_list.begin(), st.res_list.end());
  bool ok = st.res_list.size() == 20;
  for (size_t i = 0; ok && i < 20; ++i)
    ok = st.res_list[i][0] == RES20[i][0] && st.res_list[i][1] == RES20[i][1]
      && st.res_list[i][2] == RES20[i][2];
  if (!ok) { printf("  FAIL level-R list does not match reference\n"); ++fails; }
  else printf("  ok   level-R list                all 20 match (p, A, r)\n");

  return fails;
}

// ------------------------------------------------------------------ main

int main(int argc, char **argv) {
  if (argc >= 2 && !strcmp(argv[1], "--verify")) return self_test();
  if (argc < 3) {
    fprintf(stderr,
      "usage: %s LO HI [options]   (optimized scanner; flags as rung_scan, minus\n"
      "       %s --verify           --occupancy -- use the reference for that)\n",
      argv[0], argv[0]);
    return 1;
  }
  u64 lo = strtoull(argv[1], 0, 10), hi = strtoull(argv[2], 0, 10);
  Opts opt;
  u64 shard = 0, nshard = 1;
  for (int i = 3; i < argc; ++i) {
    if (!strcmp(argv[i], "--shard") && i + 1 < argc) {
      unsigned long long a = 0, b = 1;
      if (sscanf(argv[++i], "%llu/%llu", &a, &b) == 2) { shard = a; nshard = b; }
    }
    else if (!strcmp(argv[i], "--rmax") && i + 1 < argc) opt.rmax = strtoull(argv[++i], 0, 10);
    else if (!strcmp(argv[i], "--emit-residual")) opt.emitRes = true;
    else if (!strcmp(argv[i], "--no-emit-support")) opt.emitSup = false;
    else if (!strcmp(argv[i], "--emit-deep") && i + 1 < argc) opt.deepThreshold = strtoull(argv[++i], 0, 10);
    else if (!strcmp(argv[i], "--sample") && i + 1 < argc) opt.sampleEvery = strtoull(argv[++i], 0, 10);
    else if (!strcmp(argv[i], "--spanlog") && i + 1 < argc) opt.spanlog = atoi(argv[++i]);
    else if (!strcmp(argv[i], "--hard840")) opt.hard840 = true;
    else if (!strcmp(argv[i], "--progress") && i + 1 < argc) opt.progress = strtoull(argv[++i], 0, 10);
    else if (!strcmp(argv[i], "--occupancy")) {
      fprintf(stderr, "rung_scan2: --occupancy is a diagnostic mode of the reference; use ./rung_scan\n");
      return 2;
    }
  }
  if (opt.rmax > RCAP) opt.rmax = RCAP;
  if (opt.hard840) {
    g_mod = 840; g_astep = 210;
    g_classes = {1, 121, 169, 289, 361, 529};
    g_stripp  = {2, 3, 5, 7};
  }
  if (nshard == 0) nshard = 1;
  if (nshard > 1) {
    u64 span = (hi - lo + nshard - 1) / nshard;
    u64 a = lo + shard * span, b = min(hi, a + span);
    lo = a; hi = b;
  }
  if (lo >= hi) {
    printf("SUMMARY lo=%llu hi=%llu primes=0 jac=0 rc=0 sup=0 res=0 escalated=0 "
           "pairpath=0 maxr=0 maxr_p=0 hist= sieve_s=0.00 scan_s=0.00 ns_per_prime=0\n",
           (unsigned long long)lo, (unsigned long long)hi);
    return 0;
  }

  auto t0 = chrono::steady_clock::now();
  init_phi();
  init_rfac();
  init_symtabs();
  init_sadm();
  build_base_primes(hi);
  auto t1 = chrono::steady_clock::now();

  if (opt.spanlog < 12) opt.spanlog = 12;
  if (opt.spanlog > 27) opt.spanlog = 27;
  const u64 SPAN = g_mod << opt.spanlog;
  u64 nwin = (hi - lo + SPAN - 1) / SPAN;
  Stats total;
  u64 g_done = 0;
  auto tstart = chrono::steady_clock::now();

#ifdef _OPENMP
#pragma omp parallel
  {
    Stats loc;
#pragma omp for schedule(dynamic, 1) nowait
    for (long long w = 0; w < (long long)nwin; ++w) {
      process_window(lo + (u64)w * SPAN, min(hi, lo + ((u64)w + 1) * SPAN), opt, loc);
      if (opt.progress) {
#pragma omp critical(progress)
        {
          if (++g_done % opt.progress == 0) {
            double el = chrono::duration<double>(chrono::steady_clock::now() - tstart).count();
            double f = (double)g_done / nwin;
            fprintf(stderr, "[rung_scan2] %llu/%llu windows (%.1f%%)  %.0fs elapsed  ~%.0fs left\n",
                    (unsigned long long)g_done, (unsigned long long)nwin, 100 * f, el, el * (1 / f - 1));
          }
        }
      }
    }
#pragma omp critical
    total.merge(loc);
  }
#else
  for (u64 w = 0; w < nwin; ++w) {
    process_window(lo + w * SPAN, min(hi, lo + (w + 1) * SPAN), opt, total);
    if (opt.progress && (++g_done % opt.progress == 0)) {
      double el = chrono::duration<double>(chrono::steady_clock::now() - tstart).count();
      double f = (double)g_done / nwin;
      fprintf(stderr, "[rung_scan2] %llu/%llu windows (%.1f%%)  %.0fs elapsed  ~%.0fs left\n",
              (unsigned long long)g_done, (unsigned long long)nwin, 100 * f, el, el * (1 / f - 1));
    }
  }
#endif

  auto t2 = chrono::steady_clock::now();
  double ts = chrono::duration<double>(t1 - t0).count();
  double tc = chrono::duration<double>(t2 - t1).count();

  printf("SUMMARY lo=%llu hi=%llu primes=%llu jac=%llu rc=%llu sup=%llu res=%llu escalated=%llu "
         "pairpath=%llu maxr=%llu maxr_p=%llu hist=",
         (unsigned long long)lo, (unsigned long long)hi, (unsigned long long)total.nprimes,
         (unsigned long long)total.jac,
         (unsigned long long)total.rc, (unsigned long long)total.sup, (unsigned long long)total.res,
         (unsigned long long)total.escalated, (unsigned long long)g_pairpath,
         (unsigned long long)total.maxr, (unsigned long long)total.maxr_p);
  for (u64 r = 3; r <= (u64)RCAP; r += 4) if (total.hist[r])
    printf("%llu:%llu,", (unsigned long long)r, (unsigned long long)total.hist[r]);
  printf(" sieve_s=%.2f scan_s=%.2f ns_per_prime=%.0f\n", ts, tc,
         total.nprimes ? 1e9 * tc / total.nprimes : 0.0);
  return 0;
}
