// rung_scan3.cpp -- the OPTIMIZED verification scanner. cover_scan.cpp stays the frozen
// reference; this binary earns trust the way rung_scan2 and the CUDA port did:
// identical SUMMARY fields and an identical sorted certificate set on every range where
// both run (tests/diff3.sh), plus a self-test pinning the same constants.
//
// It is an INDEPENDENT implementation. It includes rung_scan.cpp (for factorA, as
// cover_scan does) and NOTHING from cover_scan.cpp -- two independent witnesses are
// what make the differential gate mean something.
//
// BUILD  make rung_scan3      TEST  ./rung_scan3 --verify      GATE  make check-diff3
// Public domain / CC0.  No warranty.
#define main rung_scan_main
#include "rung_scan.cpp"
#undef main

#include <unordered_map>
#include <numeric>
#include <cinttypes>
#include <mutex>       // call_once for the fast-factorization tables   // PRIu64 et al: uint64_t is NOT unsigned long long (it is
                       // unsigned long on LP64), so "%llu" would need a cast at every
                       // call site. The format macros take the types as they are.

static const uint32_t UCERT_RMAX = 63, UCERT_UMAX = 64;

// ------------------------------------------------------- the stage-E ceiling
//
// Stage E factors n = p + 4a^2 d in u64, and the bound on n is NOT 2^63 -- that was a
// guess, and it threw away exactly half the usable range. The real bound is a property
// of the frozen reference and not a choice:
//
//   isqrt_u64 ends with `while ((s + 1) * (s + 1) <= n) ++s` in u64 arithmetic. Once
//   s reaches 2^32 - 1, (s+1)*(s+1) is 2^64, which wraps to 0; `0 <= n` is true and the
//   loop runs away. s reaches 2^32 - 1 exactly when n reaches (2^32 - 1)^2. factorA needs
//   isqrt_u64 through factor_hard's perfect-square test, and census/rung_scan.cpp is
//   frozen, so this is inherited rather than fixable here.
//
// Everything else in the path clears that bound comfortably: mulmod and the (u128)c*k
// product are 128-bit; m + 1 is safe because m divides n; and pollard_brent's
// `mulmod(v,v,n) + c` has 2^33 - 1 of headroom for c, which never exceeds a handful.
//
// a, d <= 8 so n <= p + 4*8*8*8 = p + 2048, which fixes the bound on p itself.
static const u64 DIRECT_NMAX = 18446744065119617025ull;          // (2^32 - 1)^2
static const u64 DIRECT_PMAX = DIRECT_NMAX - 2048 - 1;           // 1.8446744e19

// THIS DIVERGES FROM sieve/cover_scan.cpp, DELIBERATELY, AND ONLY ABOVE 2^63.
// The frozen reference still carries the old `nn >> 63`, and it cannot be edited. So the
// two agree byte for byte everywhere below 2^63 -- which is every range tests/diff3.sh
// runs, and every range cover_scan is fast enough to run at all -- and above it this
// scanner is strictly MORE complete: it solves primes the reference would decline. That
// is a safe direction to diverge in, because declining is not unsoundness (an unsolved
// prime is reported as a SURVIVOR, never accepted); it is incompleteness. Above 2^63,
// cover_scan is no longer a valid reference for stage E, and `make check-direct` is what
// stands in for it -- exact-rational re-derivation instead of a second implementation.
// Class-table digests, pinned PER MODULUS. A wheel with no entry here runs unpinned
// and says so on stderr -- that is what --wheel experiments need -- but the published
// moduli must match or the binary refuses to run.
struct ClassPin { u64 M; u64 fnv64; const char *sha256; };
static const ClassPin CLASS_PINS[] = {
  {120120,  0xdbc4ab9b16d73f76ull,
   "154e82672d9da66cff97d48079047b8b0b0a31fa049ac62fb05b5c82ad540f60"},
  {2042040, 0xb9b5d1408da3428eull,
   "a77b2187201ff96ff3ecf822150f4152d841941deeb0cf616238bcbb0a648bbc"},
};
static const ClassPin *class_pin(u64 M) {
  for (const ClassPin &p : CLASS_PINS) if (p.M == M) return &p;
  return nullptr;
}

// ------------------------------------------------------------------ arithmetic

static u64 gcd3(u64 a, u64 b) { while (b) { u64 t = a % b; a = b; b = t; } return a; }

// Extended Euclid. The strides are generally composite, so Fermat is invalid here.
static u64 inv3(u64 a, u64 m) {
  if (m == 1) return 0;
  int64_t t = 0, nt = 1, r = (int64_t)m, nr = (int64_t)(a % m);
  while (nr) {
    int64_t q = r / nr;
    int64_t tmp = t - q * nt; t = nt; nt = tmp;
    tmp = r - q * nr; r = nr; nr = tmp;
  }
  if (r != 1) { fprintf(stderr, "FATAL: inv3(%" PRIu64 ",%" PRIu64 ") not coprime\n",
                        a, m); abort(); }
  return (u64)(t < 0 ? t + (int64_t)m : t);
}

// ------------------------------------------------------------------ F1 covers

// Every p ≡ res (mod mod) with p >= pmin is solved by
// 4/p = 1/(ade) + 1/(acdp) + 1/(cdep), with m = 4acd-1, e = ck-a, k = (p+4a^2d)/mod.
struct Cover { u64 mod, res, pmin; uint32_t a, c, d; };

static vector<Cover> gen_covers(u64 mmax) {
  unordered_map<u64, unordered_map<u64, Cover>> best;
  const u64 nmax = (mmax + 1) / 4;
  for (u64 a = 1; a <= nmax; ++a)
    for (u64 d = 1; a * d <= nmax; ++d)
      for (u64 c = 1; a * d * c <= nmax; ++c) {
        u128 mm = (u128)4 * a * c * d - 1, bb = (u128)4 * a * a * d;
        u64 m = (u64)mm, res = (u64)(mm - bb % mm) % m;
        u64 pmin = (u64)((u128)m * (a / c + 1) - bb);
        auto &slot = best[m];
        auto it = slot.find(res);
        if (it == slot.end() || pmin < it->second.pmin)
          slot[res] = Cover{m, res, pmin, (uint32_t)a, (uint32_t)c, (uint32_t)d};
      }
  vector<Cover> out;
  for (auto &kv : best) for (auto &kv2 : kv.second) out.push_back(kv2.second);
  sort(out.begin(), out.end(), [](const Cover &x, const Cover &y) {
    return x.mod != y.mod ? x.mod < y.mod : x.res < y.res; });
  return out;
}

// Covers whose modulus DIVIDES M -- the only ones build_wheel can use.
//
// gen_covers(M) enumerates every cover with modulus <= M, which at M = 2,042,040 means
// acd <= 510,510: ~4.4e7 triples and ~33 s, essentially all of it discarded. Driving the
// enumeration from the divisors of M instead costs milliseconds.
//
// m ≡ 3 (mod 4) is not a filter of convenience: F1's modulus is 4N-1 always (see the
// header), so no other divisor can host a cover at all.
//
// Output order and dedup rule match gen_covers exactly -- ascending (mod, res), keeping
// the witness with the smallest pmin -- and rung_scan3 --verify asserts the two agree
// tuple for tuple at M = 120120, where the full enumeration is still affordable.
static vector<Cover> gen_wheel_covers(u64 M) {
  vector<u64> divs{1};
  u64 n = M;
  for (u64 q = 2; q * q <= n; ++q)
    if (n % q == 0) {
      int e = 0;
      while (n % q == 0) { n /= q; ++e; }
      const size_t base = divs.size();
      u64 pe = 1;
      for (int i = 1; i <= e; ++i) { pe *= q; for (size_t j = 0; j < base; ++j) divs.push_back(divs[j] * pe); }
    }
  if (n > 1) { const size_t base = divs.size(); for (size_t j = 0; j < base; ++j) divs.push_back(divs[j] * n); }
  sort(divs.begin(), divs.end());

  unordered_map<u64, unordered_map<u64, Cover>> best;
  for (u64 m : divs) {
    if (m % 4 != 3) continue;
    const u64 t = (m + 1) / 4;                       // acd = t
    for (u64 a = 1; a <= t; ++a) {
      if (t % a) continue;
      const u64 ta = t / a;
      for (u64 c = 1; c <= ta; ++c) {
        if (ta % c) continue;
        const u64 d = ta / c;
        const u128 bb = (u128)4 * a * a * d;
        const u64 res = (u64)(((u128)m - bb % m) % m);
        const u64 pmin = (u64)((u128)m * (a / c + 1) - bb);
        auto &slot = best[m];
        auto it = slot.find(res);
        if (it == slot.end() || pmin < it->second.pmin)
          slot[res] = Cover{m, res, pmin, (uint32_t)a, (uint32_t)c, (uint32_t)d};
      }
    }
  }
  vector<Cover> out;
  for (auto &kv : best) for (auto &kv2 : kv.second) out.push_back(kv2.second);
  sort(out.begin(), out.end(), [](const Cover &x, const Cover &y) {
    return x.mod != y.mod ? x.mod < y.mod : x.res < y.res; });
  return out;
}

// ------------------------------------------------------------------ F2 certificates

// A rung r plus a fixed witness u whose two conditions -- u | q^2 and u ≡ -q (mod r),
// q = pA, A = (p+r)/4 -- are forced by p mod L, L = 4 lcm(r,u). gcd(u,r) = 1 is required
// so u is invertible mod r, which is what makes the third denominator integral.
//
// Necessary, not convenient: reparameterizing F1 by N = acd, alpha = c^2 d shows its
// modulus is 4N-1 ≡ 3 (mod 4) ALWAYS, so F1 can never constrain a residue mod 8.
struct Ucert { u64 mod, res; uint32_t r, u; };

static vector<Ucert> gen_ucerts(uint32_t rmax, uint32_t umax) {
  unordered_map<u64, unordered_map<u64, Ucert>> best;
  for (uint32_t r = 1; r <= rmax; ++r)
    for (uint32_t u = 1; u <= umax; ++u) {
      if (gcd3(u, r) != 1) continue;
      const u64 L = 4ull * ((u64)r * u / gcd3(r, u));
      for (u64 rho = 0; rho < L; ++rho) {
        if ((rho + r) % 4) continue;
        bool ok = true;
        for (int lift = 0; ok && lift < 2; ++lift) {
          const u64 p = rho + (u64)lift * L, A = (p + r) / 4, q = p * A;
          if ((q % u) * (q % u) % u || (q + u) % r) ok = false;
        }
        if (!ok) continue;
        auto &slot = best[L];
        auto it = slot.find(rho);
        if (it == slot.end() || r < it->second.r || (r == it->second.r && u < it->second.u))
          slot[rho] = Ucert{L, rho, r, u};
      }
    }
  vector<Ucert> out;
  for (auto &kv : best) for (auto &kv2 : kv.second) out.push_back(kv2.second);
  sort(out.begin(), out.end(), [](const Ucert &x, const Ucert &y) {
    return x.mod != y.mod ? x.mod < y.mod : x.res < y.res; });
  return out;
}

// ------------------------------------------------------------------ wheel

struct WheelResult { vector<u64> classes; u64 max_pmin; u64 n_f1_only; };

// ONLY certificates whose modulus divides M may be used here. One with L not dividing M
// kills a sub-progression inside a lane -- that is stage B's job, and applying it to the
// wheel would be a soundness bug.
static WheelResult build_wheel(u64 M, const vector<Cover> &cov, const vector<Ucert> &uc) {
  vector<uint8_t> dead(M, 0);
  WheelResult w; w.max_pmin = 0; w.n_f1_only = 0;
  for (const Cover &f : cov) {
    if (M % f.mod) continue;
    w.max_pmin = max(w.max_pmin, f.pmin);
    for (u64 x = f.res; x < M; x += f.mod) dead[x] = 1;
  }
  for (u64 x = 1; x < M; ++x) if (!dead[x] && gcd3(x, M) == 1) ++w.n_f1_only;
  for (const Ucert &c : uc) {
    if (M % c.mod) continue;
    for (u64 x = c.res; x < M; x += c.mod) dead[x] = 1;
  }
  for (u64 x = 1; x < M; ++x) if (!dead[x] && gcd3(x, M) == 1) w.classes.push_back(x);
  return w;
}

static u64 fnv64_classes(const vector<u64> &cls) {
  u64 h = 14695981039346656037ull;
  char buf[24];
  for (size_t i = 0; i < cls.size(); ++i) {
    int n = snprintf(buf, sizeof buf, i ? ",%" PRIu64 : "%" PRIu64, cls[i]);
    for (int j = 0; j < n; ++j) { h ^= (unsigned char)buf[j]; h *= 1099511628211ull; }
  }
  return h;
}

// ------------------------------------------------- fast stage-D arithmetic
//
// perf on the CUDA port's host tail (2026-08-18, single thread for attribution):
// __umodti3 -- SOFTWARE 128-bit long division, emitted because gcc cannot prove the
// quotient of (u128)a*b % m fits 64 bits -- was 48.5% of tail CPU, and factorA's
// hardware-divide trial division another 25.3%. Microbenchmarked on the same box:
// Montgomery multiplication 3.83 ns against the frozen mulmod's 34.72 (9.1x), and
// magic-inverse divisibility 0.45 ns/prime against `n % sp`'s 8.71 (19x).
//
// This section reimplements the FACTORIZATION only, in this editable file -- the frozen
// census/rung_scan.cpp is untouched, and its factorA remains the reference that
// --check-factor diffs against. Swapping it under stage D/E is output-invariant by
// construction: a factorization is unique, min_div_res takes a minimum over divisors
// (order-independent), and divisors3 sorts. The gates prove it rather than this comment:
// --check-factor (canonical factorizations, products, primality of factors),
// check-diff3 (byte-identity against the frozen cover_scan end to end), check-direct
// (exact rationals either side of 2^63).

// Montgomery arithmetic for odd n, valid for ALL odd n < 2^64. The redc is the careful
// two-word form: for n >= 2^63 the textbook (t + m*n) sum can overflow u128, so the high
// halves are added in u128 with the low-half carry folded in explicitly. The low 64 bits
// of t + m*n are zero by construction, so that carry is simply (low64(t) != 0).
struct Mg {
  u64 n, ninv, r2, one;
  static inline u64 inv_pow2(u64 a) {              // a odd: a^{-1} mod 2^64, Newton
    u64 x = a;
    for (int i = 0; i < 5; ++i) x *= 2 - a * x;
    return x;
  }
  explicit Mg(u64 n_) : n(n_) {
    ninv = ~inv_pow2(n_) + 1;                      // -n^{-1} mod 2^64
    one = (u64)(((u128)1 << 64) % n);              // to(1) = 2^64 mod n; one software
    r2 = (u64)((u128)one * one % n);               // mod per MODULUS, not per multiply
  }
  inline u64 redc(u128 t) const {
    const u64 m = (u64)t * ninv;
    const u128 mn = (u128)m * n;
    u128 res = (t >> 64) + (mn >> 64) + ((u64)t != 0);
    if (res >= n) res -= n;
    return (u64)res;
  }
  inline u64 mul(u64 a, u64 b) const { return redc((u128)a * b); }
  inline u64 add(u64 a, u64 b) const { const u64 s = a + b; return s >= n || s < a ? s - n : s; }
  inline u64 to(u64 a) const { return mul(a % n, r2); }
};

// The frozen isqrt_u64 runs away for n >= (2^32-1)^2 -- its (s+1)*(s+1) probe wraps.
// (Unreachable there for inputs <= DIRECT_NMAX, by the accident that the one dangerous
// value (2^32-1)^2 is divisible by 3 and so never survives trial division; this version
// is safe for every u64 by clamping the probe, and costs the same.)
static u64 isqrt_fast(u64 n) {
  u64 s = (u64)sqrtl((long double)n);
  if (s > 0xFFFFFFFFull) s = 0xFFFFFFFFull;
  while (s > 0 && s * s > n) --s;
  while (s < 0xFFFFFFFFull && (s + 1) * (s + 1) <= n) ++s;
  return s;
}

// The reference's 12-base deterministic Miller-Rabin, with every modular multiply in
// Montgomery form. Same prologue, same bases, same verdict for every u64.
static bool is_prime_fast(u64 n) {
  if (n < 2) return false;
  for (u64 sp : {2ull, 3ull, 5ull, 7ull, 11ull, 13ull, 17ull, 19ull, 23ull, 29ull, 31ull, 37ull}) {
    if (n == sp) return true;
    if (n % sp == 0) return false;
  }
  const Mg M(n);
  const u64 mone = n - M.one;                      // to(n-1)
  u64 d = n - 1;
  int r = 0;
  while (!(d & 1)) { d >>= 1; ++r; }
  for (u64 a : {2ull, 3ull, 5ull, 7ull, 11ull, 13ull, 17ull, 19ull, 23ull, 29ull, 31ull, 37ull}) {
    u64 x = M.one, b = M.to(a), e = d;
    while (e) {
      if (e & 1) x = M.mul(x, b);
      b = M.mul(b, b);
      e >>= 1;
    }
    if (x == M.one || x == mone) continue;
    bool composite = true;
    for (int i = 1; i < r; ++i) {
      x = M.mul(x, x);
      if (x == mone) { composite = false; break; }
    }
    if (composite) return false;
  }
  return true;
}

// Brent's rho, the frozen pollard_brent's exact structure, iterated in Montgomery form.
// The map x -> x^2*R^{-1} + c still factors through arithmetic mod p for every p | n, so
// cycle detection is as valid as for the reference's map -- but it is a DIFFERENT map,
// so the factor found may differ. That is why --check-factor compares canonicalized
// factorizations rather than Fac entry order. gcd works on domain values directly:
// gcd(a*2^64 mod n, n) = gcd(a, n) for odd n.
static u64 pollard_fast(u64 n) {
  if ((n & 1) == 0) return 2;
  const Mg M(n);
  for (u64 c = 1;; ++c) {
    const u64 cm = M.to(c);
    u64 y = M.to(2), m = 128, g = 1, r = 1, qacc = M.one, x = 0, ys = 0;
    auto f = [&](u64 v) { return M.add(M.mul(v, v), cm); };
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
          qacc = M.mul(qacc, diff);
        }
        if (g == n) break;
        g = __gcd(qacc, n);
      }
      r <<= 1;
      if (r > (1ull << 26)) break;
    }
    if (g == n || g == 1) {
      g = 1;
      while (g == 1) {
        ys = f(ys);
        u64 diff = x > ys ? x - ys : ys - x;
        g = __gcd(diff ? diff : n, n);
        if (ys == x) break;
      }
    }
    if (g != 1 && g != n) return g;
  }
}

static void factor_hard_fast(u64 n, Fac &f) {
  if (n == 1) return;
  if (is_prime_fast(n)) { f.add(n, 1); return; }
  const u64 s = isqrt_fast(n);
  if (s * s == n) {
    Fac g;
    factor_hard_fast(s, g);
    for (int i = 0; i < g.k; ++i) f.add(g.pr[i], 2 * g.ex[i]);
    return;
  }
  const u64 d = pollard_fast(n);
  factor_hard_fast(d, f);
  factor_hard_fast(n / d, f);
}

// Trial division by magic inverse: for odd sp, sp | n  <=>  n * inv(sp) mod 2^64 is at
// most floor((2^64-1)/sp) -- one multiply and one compare, no divide, and successive
// primes are INDEPENDENT operations where the reference's `n % sp` chain serializes on
// the divider. The tables are self-contained (primes <= TD_BOUND sieved here, once,
// under call_once) so this path never depends on build_base_primes having run, or on
// what `hi` it was built for -- g_small is TRUNCATED below TD_BOUND when hi is small,
// which changes the frozen path's work split but not the factorization, and depending
// on it here would import that subtlety for nothing.
static vector<u64> g_ff_p, g_ff_inv, g_ff_lim, g_ff_p2;
static std::once_flag g_ff_once;

static void init_factor_fast() {
  vector<uint8_t> c(TD_BOUND + 1, 0);
  for (u64 i = 2; i * i <= TD_BOUND; ++i)
    if (!c[i]) for (u64 j = i * i; j <= TD_BOUND; j += i) c[j] = 1;
  for (u64 i = 3; i <= TD_BOUND; i += 2) {
    if (c[i]) continue;
    g_ff_p.push_back(i);
    g_ff_inv.push_back(Mg::inv_pow2(i));
    g_ff_lim.push_back(~0ull / i);
    g_ff_p2.push_back(i * i);
  }
}

static void factorA_fast(u64 n, Fac &f) {
  std::call_once(g_ff_once, init_factor_fast);
  f.k = 0;
  if (n && !(n & 1)) {
    int e = 0;
    while (!(n & 1)) { n >>= 1; ++e; }
    f.add(2, e);
  }
  for (size_t i = 0; i < g_ff_p.size(); ++i) {
    if (g_ff_p2[i] > n) break;                     // same invariant as the reference:
    if (n * g_ff_inv[i] <= g_ff_lim[i]) {          // past here the cofactor is prime
      const u64 sp = g_ff_p[i];
      int e = 0;
      while (n % sp == 0) { n /= sp; ++e; }        // divides are rare: hits only
      f.add(sp, e);
    }
  }
  if (n > 1) {
    if (n <= TD_BOUND * TD_BOUND || is_prime_fast(n)) f.add(n, 1);
    else factor_hard_fast(n, f);
  }
}

// --------------------------------------------------- the fast-arithmetic gate
//
// factorA_fast must produce THE factorization -- not "a compatible" one -- for every
// input the scanners can hand it. A factorization is unique, so the check is exact:
// canonicalize both results (Fac entry order is legitimately implementation-dependent,
// because pollard may split a composite at a different factor) and compare prime for
// prime, exponent for exponent. Two further checks are independent of the frozen
// comparison and therefore catch the both-wrong-identically case: the product of the
// prime powers must reconstruct n exactly, and every reported factor must pass the
// reference's own is_prime.
//
// The battery is not uniform random. Uniform u64 values are almost all easy -- a small
// factor falls out of trial division immediately -- so the battery force-feeds the hard
// shapes: primes (the remainder path), hard semiprimes p*q with both factors > TD_BOUND
// (the pollard path, including products above 2^63, where Montgomery reduction is at
// its overflow-prone worst), perfect squares of large primes (the isqrt path), smooth
// numbers (deep trial division), and the exact boundary values TD_BOUND^2 +- 1 and
// 2^63 +- 1.
static void ck_canon(Fac &f) {
  for (int i = 1; i < f.k; ++i)
    for (int j = i; j > 0 && f.pr[j - 1] > f.pr[j]; --j) {
      swap(f.pr[j - 1], f.pr[j]);
      swap(f.ex[j - 1], f.ex[j]);
    }
}

static int check_factor(u64 nper) {
  // The FROZEN path's implicit dependency, learned by running this gate without it:
  // factorA trial-divides by g_small, which build_base_primes fills only to
  // min(TD_BOUND, isqrt(hi)) -- and with it empty, factorA's remainder rule
  // `n <= TD_BOUND^2 -> prime` is false and it emits composite "factors". 2^34 makes
  // isqrt(hi) exceed TD_BOUND, so g_small is complete. factorA_fast has no such
  // dependency by construction; that asymmetry is exactly why its tables are its own.
  build_base_primes(1ull << 34);
  vector<u64> vals;
  u64 seed = 0x9e3779b97f4a7c15ull;
  auto rnd = [&]() { seed ^= seed << 13; seed ^= seed >> 7; seed ^= seed << 17; return seed; };
  // uniform per decade, 1e6 .. DIRECT_NMAX
  for (u64 dec = 1000000ull; dec < DIRECT_NMAX / 10; dec *= 10)
    for (u64 i = 0; i < nper; ++i) vals.push_back(dec + rnd() % (9 * dec));
  // hard semiprimes p*q, p,q in (TD_BOUND, 2^32): the pollard path, both sides of 2^63
  for (u64 i = 0; i < nper; ++i) {
    u64 a = 3037000000ull + rnd() % 1257000000ull;   // ~3.0e9 .. 4.3e9
    u64 b = 2147483648ull + rnd() % 2147000000ull;   // ~2.1e9 .. 4.3e9
    while (!is_prime(a)) ++a;
    while (!is_prime(b)) ++b;
    if ((u128)a * b <= (u128)DIRECT_NMAX) vals.push_back(a * b);
  }
  // perfect squares of large primes: the isqrt path
  for (u64 i = 0; i < nper; ++i) {
    u64 a = 2147483648ull + rnd() % 2147000000ull;
    while (!is_prime(a)) ++a;
    vals.push_back(a * a);
  }
  // smooth numbers: deep trial division
  for (u64 i = 0; i < nper; ++i) {
    u64 v = 1;
    while (v < DIRECT_NMAX / 65536) v *= 2 + rnd() % 65535;
    vals.push_back(v);
  }
  // primes: the remainder path at height
  for (u64 i = 0; i < nper / 4 + 1; ++i) {
    u64 a = ((1ull << 62) + rnd() % (1ull << 62)) | 1ull;
    while (!is_prime(a)) a += 2;
    vals.push_back(a);
  }
  // boundaries
  const u64 B = TD_BOUND * TD_BOUND;
  const u64 edge[] = {B - 1, B, B + 1, (1ull << 63) - 1, 1ull << 63, (1ull << 63) + 1,
                      DIRECT_NMAX - 2, DIRECT_NMAX - 1, DIRECT_NMAX, 1, 2};
  for (u64 v : edge) vals.push_back(v);

  u64 bad_cmp = 0, bad_prod = 0, bad_prime = 0;
  for (u64 n : vals) {
    Fac a, b;
    factorA(n, a);                 // the frozen reference's path
    factorA_fast(n, b);            // the path under test
    ck_canon(a);
    ck_canon(b);
    bool same = a.k == b.k;
    for (int i = 0; same && i < a.k; ++i) same = a.pr[i] == b.pr[i] && a.ex[i] == b.ex[i];
    if (!same) {
      if (++bad_cmp <= 5) fprintf(stderr, "  MISMATCH n=%" PRIu64 "\n", n);
      continue;
    }
    u128 prod = 1;
    for (int i = 0; i < b.k; ++i) {
      for (int e = 0; e < b.ex[i]; ++e) prod *= b.pr[i];
      if (!is_prime(b.pr[i])) ++bad_prime;
    }
    if (prod != (u128)n) ++bad_prod;
  }
  const u64 bad = bad_cmp + bad_prod + bad_prime;
  printf("%s check-factor %zu values -- mismatches: %" PRIu64 ", bad products: %" PRIu64
         ", composite factors: %" PRIu64 "\n",
         bad ? "FAIL" : "ok  ", vals.size(), bad_cmp, bad_prod, bad_prime);
  return bad ? 1 : 0;
}

// ------------------------------------------------------------------ stage D

// Smallest divisor u of q^2 with u ≡ target (mod r) and u <= q, or 0 if none.
// The prime powers of q^2 are p^2 together with each prime power of A doubled: p is
// prime and A = (p+r)/4 < p for r < 3p, so p never divides A.
//
// Minimum-per-residue is optimal at every stage: multiplying by a positive factor
// preserves order, so a larger partial product can never overtake a smaller one with
// the same residue. Bounding partial products by q loses nothing -- every prefix of a
// divisor <= q is itself <= q -- and a witness <= q always EXISTS, because if u | q^2
// with u ≡ -q then v = q^2/u satisfies v ≡ -q too (as (-q)(-q) ≡ q^2) and one of u, v
// is <= q. That bound is what keeps every intermediate inside u128.
static u128 min_div_res(u64 p, const Fac &fA, u128 q, u64 r, u64 target) {
  const u128 INF = ~(u128)0;
  vector<u128> cur(r, INF), nxt(r, INF);
  cur[1 % r] = 1;
  auto absorb = [&](u64 pr, int e) {
    fill(nxt.begin(), nxt.end(), INF);
    const u64 prm = pr % r;
    for (u64 s = 0; s < r; ++s) {
      if (cur[s] == INF) continue;
      u128 v = cur[s]; u64 rr = s;
      for (int j = 0; ; ++j) {
        if (v < nxt[rr]) nxt[rr] = v;
        if (j == e || v > q / pr) break;       // next multiply would exceed the bound
        v *= pr; rr = (u64)((u128)rr * prm % r);
      }
    }
    cur.swap(nxt);
  };
  absorb(p, 2);
  for (int i = 0; i < fA.k; ++i) absorb(fA.pr[i], 2 * fA.ex[i]);
  return cur[target] == INF ? 0 : cur[target];
}

static void app_u128(string &s, u128 v) {
  char buf[48]; int n = 0;
  if (!v) buf[n++] = '0';
  while (v) { buf[n++] = (char)('0' + (int)(v % 10)); v /= 10; }
  while (n--) s.push_back(buf[n]);
}

// Emission contract: smallest rung r ≡ -p (mod 4), then the smallest witness u at that
// rung. Rungs with gcd(q, r) > 1 are SKIPPED -- u ≡ -q (mod r) need not be invertible
// mod r there, which the third denominator's integrality requires. The reference notes
// that path is unreachable for r < 3p; anything it would catch falls through to stage E.
static bool solve_rung3(u64 p, string &out) {
  const u64 r0 = (4 - p % 4) % 4 ? (4 - p % 4) % 4 : 4;
  for (u64 r = r0; r <= (u64)RCAP; r += 4) {
    const u64 A = (p + r) / 4;
    if (!A) continue;
    const u128 q = (u128)p * A;
    const u64 qm = (u64)(q % r);
    if (r > 1 && gcd3(qm, r) != 1) continue;
    const u64 target = qm ? r - qm : 0;
    // factorA_fast, not the frozen factorA: identical factorization (--check-factor),
    // 9.1x cheaper mulmod and 19x cheaper trial division under it. min_div_res's result
    // is a min over divisors, so Fac entry order -- the one thing that can differ --
    // cannot reach the certificate.
    Fac f; factorA_fast(A, f);
    const u128 u = min_div_res(p, f, q, r, target);
    if (!u) continue;
    char head[80];   // 3 u64s are at most 60 digits; 64 was tight enough to warn
    snprintf(head, sizeof head, "RUNG %" PRIu64 " %" PRIu64 " %" PRIu64 " ",
             p, A, r);
    out = head;
    app_u128(out, u);
    out.push_back('\n');
    return true;
  }
  return false;
}

// ------------------------------------------------------------------ stage E

static vector<u64> divisors3(const Fac &f) {
  vector<u64> ds{1};
  for (int i = 0; i < f.k; ++i) {
    const size_t base = ds.size();
    u64 pe = 1;
    for (int e = 1; e <= f.ex[i]; ++e) {
      pe *= f.pr[i];
      for (size_t j = 0; j < base; ++j) ds.push_back(ds[j] * pe);
    }
  }
  sort(ds.begin(), ds.end());
  return ds;
}

// An F1 instance for this single p: p = m k - 4a^2 d with m = 4acd-1 means m | n and
// k = n/m for n = p + 4a^2 d, with c = (m+1)/(4ad) a positive integer and e = ck-a >= 1.
// Canonical: smallest a, then d, then m.
static bool solve_direct3(u64 p, string &out) {
  for (u64 a = 1; a <= 8; ++a)
    for (u64 d = 1; d <= 8; ++d) {
      const u128 nn = (u128)p + (u128)4 * a * a * d;
      // Unreachable in a scan -- main refuses HI above DIRECT_PMAX -- but solve_direct3
      // is also called directly by --check-direct, so it guards itself.
      if (nn > (u128)DIRECT_NMAX) continue;
      const u64 n = (u64)nn, q4 = 4 * a * d;
      Fac f; factorA_fast(n, f);   // divisors3 sorts, so entry order cannot matter here

      for (u64 m : divisors3(f)) {
        if (m % q4 != q4 - 1) continue;
        const u64 c = (m + 1) / q4, k = n / m;
        if (!c || !k || (u128)c * k <= a) continue;
        char buf[128];
        snprintf(buf, sizeof buf, "SOLVED %" PRIu64 " %" PRIu64 " %" PRIu64 " %" PRIu64 " %" PRIu64 "\n",
                 p, a,
                 c, d, k);
        out = buf;
        return true;
      }
    }
  return false;
}

// ------------------------------------------------------------------ engine

struct Scan3Stats {
  u64 positions = 0, covered = 0, sieved = 0, mr_calls = 0, composite = 0,
      rung_solved = 0, direct_solved = 0, survivors = 0;
  void merge(const Scan3Stats &o) {
    positions += o.positions; covered += o.covered; sieved += o.sieved;
    mr_calls += o.mr_calls;
    composite += o.composite; rung_solved += o.rung_solved;
    direct_solved += o.direct_solved; survivors += o.survivors;
  }
};

// Primes p <= P with p not dividing M. A prime dividing M can never divide a position
// coprime to M, so including it would mark nothing and cost bookkeeping.
static vector<u64> small_primes(u64 P, u64 M) {
  vector<uint8_t> comp(P + 1, 0);
  vector<u64> out;
  for (u64 i = 2; i <= P; ++i) {
    if (!comp[i]) {
      if (M % i) out.push_back(i);
      for (u64 j = i * i; j <= P; j += i) comp[j] = 1;
    }
  }
  return out;
}

// Base-2 strong probable prime. Nearly every survivor is composite (at mmax = 10^4
// every composite survivor measured was a perfect square), so this rejects almost all
// of them for 1/12 of the cost of the full 12-base test. A FAIL is conclusive; a pass
// falls through to is_prime, so the composite count is identical either way.
static bool sprp2(u64 n) {
  if (n < 2) return false;
  if (!(n & 1)) return n == 2;
  u64 d = n - 1; int s = 0;
  while (!(d & 1)) { d >>= 1; ++s; }
  u64 x = powmod(2, d, n);
  if (x == 1 || x == n - 1) return true;
  for (int i = 1; i < s; ++i) { x = mulmod(x, x, n); if (x == n - 1) return true; }
  return false;
}

// Sweep one lane over [lo, hi) in cache-resident segments.
//
// Index positions by the GLOBAL index J = (p - rho)/M. Cover (mod, res) hits at
// J ≡ j0 (mod step) with g = gcd(M, mod), step = mod/g -- a pattern fixed for the whole
// lane, so extended Euclid runs once per (lane, cover) per shard and the offsets are
// then simply carried from segment to segment. `rhs % g` is the ONLY solvability
// condition: per prime the minimum exponent goes wholly into g, so M/g and mod/g are
// automatically coprime.
//
// No pmin advance is needed: main refuses to start unless lo exceeds every filter's
// pmin, so every position in the sweep is already past validity.
//
// STAGE B' (the small-prime sieve) is a SEPARATE phase, applied strictly after the
// covers, never interleaved. `mods`/`ress` hold covers in [0, ncov) and small primes in
// [ncov, end); a prime p is just (mod = p, res = 0), because p | first + M*J has the
// same congruence shape as a cover. A popcount between the phases separates the two
// counters exactly, which matters because they mean entirely different things: a cover
// proves 4/p SOLVABLE and is part of the certificate chain, while a small prime only
// proves the position is not prime and certifies nothing. Conflating them would corrupt
// `covered`, which is the counter the published claim rests on.
static void sweep_lane(u64 rho, u64 lo, u64 hi, u64 M,
                       const vector<u64> &mods, const vector<u64> &ress, size_t ncov,
                       u64 seg, Scan3Stats &st, vector<string> &emit) {
  const u64 first = lo + ((rho + M - lo % M) % M);
  if (first >= hi) return;
  const u64 npos = (hi - first + M - 1) / M;
  const size_t nall = mods.size();          // covers [0,ncov) + small primes [ncov,nall)

  vector<uint32_t> off(nall), step(nall);
  for (size_t i = 0; i < nall; ++i) {
    const u64 mod = mods[i];
    const u64 g = gcd3(M, mod);
    const u64 rhs = (ress[i] + mod - first % mod) % mod;
    if (rhs % g) { step[i] = 0; continue; }        // this cover misses the lane entirely
    const u64 s = mod / g;
    step[i] = (uint32_t)s;
    off[i] = s == 1 ? 0 : (uint32_t)((u128)(rhs / g) * inv3((M / g) % s, s) % s);
  }

  vector<u64> bits((seg + 63) / 64);
  vector<uint32_t> surv;
  surv.reserve(seg / 8 + 64);

  for (u64 base = 0; base < npos; base += seg) {
    const u64 n = min(seg, npos - base);
    const size_t nw = (size_t)((n + 63) / 64);
    fill(bits.begin(), bits.begin() + nw, ~0ull);
    if (n & 63) bits[nw - 1] = (1ull << (n & 63)) - 1;

    auto mark = [&](size_t from, size_t to) {
      for (size_t i = from; i < to; ++i) {
        const uint32_t s = step[i];
        if (!s) continue;
        uint32_t j = off[i];
        for (; j < n; j += s) bits[j >> 6] &= ~(1ull << (j & 63));
        off[i] = (uint32_t)(j - n);                // carry into the next segment
      }
    };
    auto alive = [&]() {
      u64 c = 0;
      for (size_t w = 0; w < nw; ++w) c += (u64)__builtin_popcountll(bits[w]);
      return c;
    };

    mark(0, ncov);                                 // stage B  -- certified covers
    const u64 after_covers = ncov == mods.size() ? 0 : alive();
    mark(ncov, mods.size());                       // stage B' -- small primes

    surv.clear();
    for (size_t w = 0; w < nw; ++w) {
      u64 x = bits[w];
      while (x) { surv.push_back((uint32_t)(w * 64 + __builtin_ctzll(x))); x &= x - 1; }
    }
    st.positions += n;
    // `covered` counts ONLY certified-cover kills, with or without the sieve, so the
    // SUMMARY contract with the frozen cover_scan is preserved exactly.
    if (ncov == mods.size()) {
      st.covered += n - surv.size();
    } else {
      st.covered += n - after_covers;
      st.sieved  += after_covers - surv.size();
    }
    for (uint32_t j : surv) {
      const u64 p = first + M * (base + j);
      ++st.mr_calls;
      if (!sprp2(p) || !is_prime(p)) { ++st.composite; continue; }
      string line;
      if (solve_rung3(p, line))   { ++st.rung_solved;   emit.push_back(line); continue; }
      if (solve_direct3(p, line)) { ++st.direct_solved; emit.push_back(line); continue; }
      ++st.survivors;
      emit.push_back("SURVIVOR " + to_string(p) + "\n");
    }
  }
}

// --------------------------------------------------------------- stage E probe

// STAGE E HAS NEVER FIRED IN A SCAN. `direct=0` across 1.13e12 positions up to 10^15,
// because stage D resolves essentially everything that survives the covers. That left it
// the one path in the pipeline with no coverage at all -- and it is exactly the path the
// old 2^63 guard sat on, where it did not fail loudly but silently declined to solve,
// turning a numeric limit into what reads as an unsolvable prime. Calling it directly is
// the only way to exercise it, and running it either side of 2^63 is the only way to show
// the ceiling moved.
//
// Certificates go to STDOUT so verify_covers.py can re-derive them in exact rationals;
// status goes to stderr, so the two do not mix.
static int check_direct(u64 lo, u64 N) {
  // factorA needs only g_small -- primes up to TD_BOUND = 65536 -- and everything above
  // that goes through Pollard, so the base-prime table needs isqrt(bound) >= 65536 and
  // nothing more. Building to sqrt(2^63) would cost ~3 GB and minutes and change no
  // answer. The exact-arithmetic re-derivation downstream is what actually proves
  // factorA did its job, rather than this comment.
  init_phi(); init_rfac(); build_base_primes(1ull << 34);
  u64 found = 0, solved = 0;
  for (u64 p = lo | 1ull; found < N; p += 2) {
    if (!is_prime(p)) continue;
    ++found;
    string line;
    if (solve_direct3(p, line)) { ++solved; fputs(line.c_str(), stdout); }
    else fprintf(stderr, "  stage E DECLINED p=%" PRIu64 "\n", p);
  }
  fprintf(stderr, "%s check-direct %" PRIu64 "/%" PRIu64 " primes at or above %" PRIu64
          " solved by stage E (ceiling %" PRIu64 ")\n",
          solved == N ? "ok  " : "FAIL", solved, N, lo, DIRECT_PMAX);
  return solved == N ? 0 : 1;
}

// ------------------------------------------------------------------ self-test

static int scan3_self_test() {
  int fails = 0;
  auto check = [&](const char *what, u64 got, u64 want) {
    if (got != want) { printf("  FAIL %-34s got %" PRIu64 " want %" PRIu64 "\n", what,
                              got, want); ++fails; }
    else printf("  ok   %-34s %" PRIu64 "\n", what, got);
  };

  // The divisor-driven wheel generator must produce EXACTLY what gen_covers produces
  // once restricted to m | M. This assertion is the entire correctness argument for
  // that change, and it is cheap to make at M = 120120 where gen_covers is affordable.
  {
    vector<Cover> slow;
    for (const Cover &f : gen_covers(120120)) if (120120 % f.mod == 0) slow.push_back(f);
    vector<Cover> fast = gen_wheel_covers(120120);
    u64 diff = slow.size() != fast.size();
    for (size_t i = 0; !diff && i < slow.size(); ++i)
      diff = !(slow[i].mod == fast[i].mod && slow[i].res == fast[i].res &&
               slow[i].pmin == fast[i].pmin && slow[i].a == fast[i].a &&
               slow[i].c == fast[i].c && slow[i].d == fast[i].d);
    check("wheel covers: divisor == full", diff, 0);
    check("wheel covers at 120120", fast.size(), slow.size());
  }

  vector<Ucert> uc = gen_ucerts(UCERT_RMAX, UCERT_UMAX);
  vector<Cover> cov840 = gen_covers(840);
  WheelResult w840 = build_wheel(840, cov840, uc);
  check("wheel(840) F1 only", w840.n_f1_only, 24);
  check("wheel(840) classes", w840.classes.size(), 6);
  check("wheel(840) max pmin", w840.max_pmin, 34);
  static const u64 REF840[6] = {1, 121, 169, 289, 361, 529};
  bool same = w840.classes.size() == 6;
  for (int i = 0; same && i < 6; ++i) same = w840.classes[i] == REF840[i];
  if (!same) { printf("  FAIL wheel(840) != {1,121,169,289,361,529}\n"); ++fails; }
  else printf("  ok   wheel(840) == hard classes (DERIVED, not assumed)\n");

  vector<Cover> cov120 = gen_covers(120120);
  WheelResult w120 = build_wheel(120120, cov120, uc);
  check("wheel(120120) F1 only", w120.n_f1_only, 920);
  check("wheel(120120) classes", w120.classes.size(), 220);
  check("wheel(120120) max pmin", w120.max_pmin, 15014);
  u64 h = fnv64_classes(w120.classes);
  if (h != class_pin(120120)->fnv64) {
    printf("  FAIL wheel(120120) fnv64 %016" PRIx64 " want dbc4ab9b16d73f76\n",
           h); ++fails;
  } else printf("  ok   wheel(120120) fnv64            dbc4ab9b16d73f76\n");

  // The new default wheel, derived and pinned the same way.
  WheelResult w204 = build_wheel(2042040, gen_wheel_covers(2042040), uc);
  check("wheel(2042040) classes", w204.classes.size(), 2308);
  u64 h2 = fnv64_classes(w204.classes);
  if (h2 != class_pin(2042040)->fnv64) {
    printf("  FAIL wheel(2042040) fnv64 %016" PRIx64 " want %016" PRIx64 "\n",
           h2, class_pin(2042040)->fnv64); ++fails;
  } else printf("  ok   wheel(2042040) fnv64           %016" PRIx64 "\n", h2);
  {
    u64 strays = 0;
    for (u64 c : w204.classes) {
      bool hard = false;
      for (u64 hc : REF840) if (c % 840 == hc) hard = true;
      if (c % 24 != 1 || !hard) ++strays;
    }
    check("wheel(2042040) outside hard class", strays, 0);
  }

  // The derivation is independent of Mordell, so it must REPRODUCE Mordell.
  u64 strays = 0;
  for (u64 c : w120.classes) {
    bool hard = false;
    for (u64 hc : REF840) if (c % 840 == hc) hard = true;
    if (c % 24 != 1 || !hard) ++strays;
  }
  check("survivors outside hard class", strays, 0);

  printf(fails ? "\nSELF-TEST FAILED (%d checks)\n" : "\nSELF-TEST PASSED\n", fails);
  return fails ? 1 : 0;
}

// ------------------------------------------------------------------ main

// Layer L1 as a finite object: the surviving classes, then a NAMED killing certificate
// for every other unit class. Byte-identical to cover_scan --dump-classes and to
// verify_covers.py --class-table --write; three independent derivations, one artifact.
static int dump_classes(u64 M) {
  // gen_wheel_covers, not gen_covers: only m | M can reach the wheel, and the marking
  // below has no per-cover divisibility guard -- feeding it covers with m not dividing
  // M would let them claim classes they cannot legally kill.
  const vector<Cover> cov = gen_wheel_covers(M);
  const vector<Ucert> uc = gen_ucerts(UCERT_RMAX, UCERT_UMAX);
  const WheelResult w = build_wheel(M, cov, uc);
  u64 ncop = 0;
  for (u64 x = 1; x < M; ++x) if (gcd3(x, M) == 1) ++ncop;
  printf("# certified class table mod %" PRIu64 ": every unit class is killed by a named\n"
         "# certificate or survives.  F1: covers m=4acd-1 | M, kill rho ≡ -4a^2d (mod m).\n"
         "# F2: uniform rung certificates (r,u), r <= %" PRIu32 " u <= %" PRIu32 ", kill rho ≡ res (mod L).\n",
         M, UCERT_RMAX, UCERT_UMAX);
  printf("# unit_classes=%" PRIu64 " classes=%zu kills=%" PRIu64 " fnv64=%016" PRIx64 " sha256=%s\n",
         ncop, w.classes.size(),
         (ncop - w.classes.size()),
         fnv64_classes(w.classes), class_pin(M) ? class_pin(M)->sha256 : "unpinned");
  for (u64 c : w.classes) printf("CLASS %" PRIu64 "\n", c);
  // Canonical killer: F1 by ascending (m,res) first, then F2 by ascending (L,res).
  // Both generators emit in that order, so "first hit wins" is the rule.
  //
  // Computed by MARKING rather than by scanning every filter per class. Scanning is
  // O(phi(M) * ncov), which was tolerable at M = 120120 (23,040 classes) and is not at
  // M = 2,042,040 (368,640): it ran at ~320 lines/s, i.e. twenty minutes. Marking is
  // O(M * sum 1/m) and gives the IDENTICAL answer, because writing only where unmarked
  // reproduces exactly "first in canonical order wins".
  vector<int32_t> who(M, -1);                      // index into cov, then into uc
  for (size_t i = 0; i < cov.size(); ++i)
    for (u64 x = cov[i].res; x < M; x += cov[i].mod)
      if (who[x] < 0) who[x] = (int32_t)i;
  const int32_t F2BASE = (int32_t)cov.size();
  for (size_t i = 0; i < uc.size(); ++i) {
    if (M % uc[i].mod) continue;
    for (u64 x = uc[i].res; x < M; x += uc[i].mod)
      if (who[x] < 0) who[x] = F2BASE + (int32_t)i;
  }
  for (u64 rho = 1; rho < M; ++rho) {
    if (gcd3(rho, M) != 1 || who[rho] < 0) continue;
    if (who[rho] < F2BASE) {
      const Cover &f = cov[who[rho]];
      printf("KILL %" PRIu64 " F1 %" PRIu64 " %" PRIu64 " %" PRIu32 " %" PRIu32 " %" PRIu32 "\n", rho,
             f.mod, f.res, f.a, f.c, f.d);
    } else {
      const Ucert &c = uc[who[rho] - F2BASE];
      printf("KILL %" PRIu64 " F2 %" PRIu64 " %" PRIu64 " %" PRIu32 " %" PRIu32 "\n", rho,
             c.mod, c.res, c.r, c.u);
    }
  }
  return 0;
}

// Guarded so this TU can be #included by a harness (tests/bench_staged.cpp) to profile
// the real functions rather than a copy. The #define-main trick used on rung_scan.cpp
// does NOT work here: the #undef main inside this file would cancel the includer's
// rename before this point is reached.
#ifndef RUNG_SCAN3_NO_MAIN
int main(int argc, char **argv) {
  if (argc >= 2 && !strcmp(argv[1], "--verify")) return scan3_self_test();
  if (argc >= 4 && !strcmp(argv[1], "--check-direct"))
    return check_direct(strtoull(argv[2], 0, 10), strtoull(argv[3], 0, 10));
  if (argc >= 2 && !strcmp(argv[1], "--check-factor"))
    return check_factor(argc >= 3 ? strtoull(argv[2], 0, 10) : 2000);
  if (argc >= 2 && !strcmp(argv[1], "--dump-classes")) {
    u64 dm = 2042040;
    for (int i = 2; i + 1 < argc; ++i)
      if (!strcmp(argv[i], "--wheel")) dm = strtoull(argv[i + 1], 0, 10);
    return dump_classes(dm);
  }
  if (argc < 3) {
    fprintf(stderr,
      "usage: %s LO HI [--mmax N] [--wheel M] [--seg N] [--threads N]\n"
      "                 [--shard i/n] [--progress N] [--filters FILE] [--sieve P]\n"
      "                 [--no-emit-survivors]\n"
      "       %s --verify | --dump-classes\n"
      "Optimized verification sieve: certifies 4/p solvable for every prime in\n"
      "[LO, HI). Must agree byte for byte with cover_scan (tests/diff3.sh).\n",
      argv[0], argv[0]);
    return 1;
  }
  u64 lo = strtoull(argv[1], 0, 10), hi = strtoull(argv[2], 0, 10);
  u64 M = 2042040, mmax = 10000, seg = 262144, shard = 0, nshard = 1, progress = 0;
  int threads = 0;                       // 0 = leave OpenMP's default alone
  const char *filter_file = nullptr;
  // Stage B' defaults OFF because it MEASURED AS A LOSS: 2.135 s at --sieve 0 against
  // 2.247 s at --sieve 1000 over [1e12, 1.5e12). It works as designed -- it removes 47%
  // of survivors -- but it can only remove COMPOSITE ones, and those are the cheap
  // survivors (~800 ns, one SPRP). The expensive ones are the primes reaching stage D,
  // which it cannot touch by construction. Kept as a profiling knob, and because the
  // economics invert where modexp is dear relative to marking, i.e. on a GPU.
  // See docs/design/2026-08-14-small-prime-sieve-design.md S6.
  u64 sieve_p = 0;
  bool emit_surv = true;
  for (int i = 3; i < argc; ++i) {
    if (!strcmp(argv[i], "--mmax") && i + 1 < argc) mmax = strtoull(argv[++i], 0, 10);
    else if (!strcmp(argv[i], "--wheel") && i + 1 < argc) M = strtoull(argv[++i], 0, 10);
    else if (!strcmp(argv[i], "--seg") && i + 1 < argc) seg = strtoull(argv[++i], 0, 10);
    else if (!strcmp(argv[i], "--progress") && i + 1 < argc) progress = strtoull(argv[++i], 0, 10);
    else if (!strcmp(argv[i], "--threads") && i + 1 < argc) threads = atoi(argv[++i]);
    else if (!strcmp(argv[i], "--filters") && i + 1 < argc) filter_file = argv[++i];
    else if (!strcmp(argv[i], "--sieve") && i + 1 < argc) sieve_p = strtoull(argv[++i], 0, 10);
    else if (!strcmp(argv[i], "--shard") && i + 1 < argc) {
      unsigned long long a = 0, b = 1;
      if (sscanf(argv[++i], "%llu/%llu", &a, &b) == 2) { shard = a; nshard = b; }
    }
    else if (!strcmp(argv[i], "--no-emit-survivors")) emit_surv = false;
    else {
      // Refuse unknown options rather than ignoring them. A silently-dropped flag is a
      // real footgun here: a typo'd --mmax would quietly scan at the default and the
      // output would look entirely plausible.
      fprintf(stderr, "FATAL: unknown option '%s'\n", argv[i]);
      return 1;
    }
  }
  if (!seg) seg = 262144;

  // Shard arithmetic mirrors rung_scan.cpp exactly, so the two tile ranges the same
  // way and merge.py-style coverage checks carry over. An empty shard still prints its
  // SUMMARY, so a coverage check sees the full tiling rather than a gap.
  if (nshard == 0) nshard = 1;
  if (nshard > 1) {
    const u64 span = (hi - lo + nshard - 1) / nshard;
    const u64 a = lo + shard * span, b = min(hi, a + span);
    lo = a; hi = max(a, b);
  }

  const vector<Cover> covers = gen_covers(mmax);
  // The wheel needs covers with m | M up to M itself, independent of --mmax --
  // otherwise a small mmax would silently weaken the pinned class table.
  const vector<Cover> wcov = gen_wheel_covers(M);   // only m | M can reach the wheel
  const vector<Ucert> ucerts = gen_ucerts(UCERT_RMAX, UCERT_UMAX);
  const WheelResult wr = build_wheel(M, wcov, ucerts);
  const vector<u64> &wheel = wr.classes;
  u64 cover_pmax = 0;
  for (const Cover &f : covers) cover_pmax = max(cover_pmax, f.pmin);

  // The class table is the whole of layer L1. A binary whose derivation drifted from
  // the published table must REFUSE to run, not warn -- same rule as make check.
  if (const ClassPin *pin = class_pin(M)) {
    if (fnv64_classes(wheel) != pin->fnv64) {
      fprintf(stderr, "FATAL: class table digest mismatch at M=%" PRIu64 " -- derived "
              "fnv64 %016" PRIx64 ", expected %016" PRIx64 ". Regenerate "
              "class_table_%" PRIu64 ".txt and re-verify before trusting this binary.\n",
              M, fnv64_classes(wheel), pin->fnv64, M);
      return 1;
    }
  } else {
    fprintf(stderr, "[rung_scan3] WARNING: no pinned digest for M=%" PRIu64
            " -- running UNPINNED\n", M);
  }
  // Symmetric to the pmin floor below: refuse ceilings stage E cannot reach. Silently
  // skipping it instead -- which is what the old `nn >> 63` did above 2^63 -- turns a
  // NUMERIC limit into what reads as a mathematical result, because an unsolved prime is
  // then reported as a SURVIVOR. Refusing to run says which of the two it is.
  if (hi > DIRECT_PMAX) {
    fprintf(stderr, "FATAL: HI must not exceed %" PRIu64 ", the stage-E ceiling: above it "
            "isqrt_u64 in the frozen reference overflows, and a prime stage D misses "
            "would be reported as a survivor rather than solved\n", DIRECT_PMAX);
    return 1;
  }
  // The identity only certifies p >= pmin: refuse floors the table cannot cover.
  if (lo <= max(wr.max_pmin, cover_pmax)) {
    fprintf(stderr, "FATAL: LO must exceed max pmin of the filter table (%" PRIu64 "); "
            "the region below is settled by the existing censuses\n",
            max(wr.max_pmin, cover_pmax));
    return 1;
  }

  init_phi();
  init_rfac();
  build_base_primes(hi);

  // Flattened filter set. Parallel arrays: the inner marking loop touches only mod and
  // res, so keeping a/c/d out of the way is worth it. Covers with m | M are excluded --
  // those classes are already gone from the wheel, and excluding them is what makes
  // this set identical to cover_scan's.
  vector<u64> mods, ress;
  if (filter_file) {
    // A pruned table replaces the generated set. Soundness is MONOTONE in this set --
    // dropping filters can only leave more survivors, never manufacture a false
    // certificate -- so loading a smaller set is safe by construction, and the nesting
    // check in the gate is what proves the implementation honours that.
    FILE *fp = fopen(filter_file, "r");
    if (!fp) { fprintf(stderr, "FATAL: cannot open %s\n", filter_file); return 1; }
    char line[256];
    while (fgets(line, sizeof line, fp)) {
      unsigned long long mod, res;
      if (sscanf(line, "FILTER %llu %llu", &mod, &res) == 2) {
        mods.push_back((u64)mod); ress.push_back((u64)res);
      }
    }
    fclose(fp);
    if (mods.empty()) {
      fprintf(stderr, "FATAL: no FILTER lines in %s\n", filter_file); return 1;
    }
  } else {
    mods.reserve(covers.size()); ress.reserve(covers.size());
    for (const Cover &f : covers)
      if (M % f.mod) { mods.push_back(f.mod); ress.push_back(f.res); }
  }
  // Stage B': small primes appended AFTER the covers. ncov is the boundary the sweep
  // uses to keep the two kill-counters apart.
  const size_t ncov = mods.size();
  if (sieve_p) {
    for (u64 q : small_primes(sieve_p, M)) { mods.push_back(q); ress.push_back(0); }
  }

  // FNV-1a over the loaded set, reported in SUMMARY. cover_scan/rung_scan3 implement no
  // SHA-256 by design; Python pins the file's SHA-256 and regenerates it independently
  // (verify_covers.py --diff-pruned), and this digest ties the binary to what it read.
  u64 fhash = 14695981039346656037ull;
  for (size_t i = 0; i < ncov; ++i) {              // certified filters only
    char b[48];
    int n = snprintf(b, sizeof b, "%" PRIu64 ",%" PRIu64 ";", mods[i], ress[i]);
    for (int j = 0; j < n; ++j) { fhash ^= (unsigned char)b[j]; fhash *= 1099511628211ull; }
  }

  fprintf(stderr, "[rung_scan3] wheel M=%" PRIu64 " lanes=%zu (spacing %.0f)  covers=%zu "
          "(mmax=%" PRIu64 ")  seg=%" PRIu64 "  wheel_pmin=%" PRIu64 "\n"
          "[rung_scan3] filters=%zu fnv64=%016" PRIx64 " source=%s sieve=%" PRIu64 " (+%zu primes)\n",
          M, wheel.size(), (double)M / wheel.size(),
          ncov, mmax, seg,
          wr.max_pmin, ncov, fhash, filter_file ? filter_file : "generated",
          sieve_p, mods.size() - ncov);

  // Lanes are independent: each owns its bitset, its offsets, its stats and its
  // emission buffer, and writes nothing shared. Determinism does NOT depend on the
  // schedule -- it comes from merging strictly in lane order after the parallel region,
  // so a dynamic schedule is free to reorder execution without reordering output.
  vector<Scan3Stats> lst(wheel.size());
  vector<vector<string>> lem(wheel.size());
#ifdef _OPENMP
  if (threads > 0) omp_set_num_threads(threads);
#pragma omp parallel for schedule(dynamic, 1)
#endif
  for (size_t i = 0; i < wheel.size(); ++i) {
    sweep_lane(wheel[i], lo, hi, M, mods, ress, ncov, seg, lst[i], lem[i]);
    // --progress goes to stderr ONLY: stdout carries the certificate stream the gate
    // diffs, and a progress line landing in it would corrupt the comparison.
    if (progress && (i + 1) % progress == 0)
      fprintf(stderr, "[rung_scan3] lane %zu/%zu done\n", i + 1, wheel.size());
  }

  Scan3Stats st;
  for (size_t i = 0; i < wheel.size(); ++i) {          // strictly in lane order
    st.merge(lst[i]);
    for (const string &s : lem[i])
      if (emit_surv || s.compare(0, 9, "SURVIVOR ") != 0) fputs(s.c_str(), stdout);
  }

  printf("SUMMARY lo=%" PRIu64 " hi=%" PRIu64 " wheelM=%" PRIu64 " lanes=%zu mmax=%" PRIu64 " ucert_rmax=%" PRIu32 " "
         "ucert_umax=%" PRIu32 " wheel_pmin=%" PRIu64 " classes_sha256=%s positions=%" PRIu64 " "
         "covered=%" PRIu64 " sieved=%" PRIu64 " mr=%" PRIu64 " composite=%" PRIu64 " rung=%" PRIu64 " direct=%" PRIu64 " survivors=%" PRIu64 "\n",
         lo, hi, M,
         wheel.size(), mmax, UCERT_RMAX, UCERT_UMAX,
         wr.max_pmin, class_pin(M) ? class_pin(M)->sha256 : "unpinned", st.positions,
         st.covered, st.sieved, st.mr_calls,
         st.composite, st.rung_solved,
         st.direct_solved, st.survivors);
  return st.survivors ? 2 : 0;
}
#endif  // RUNG_SCAN3_NO_MAIN
