// bench_staged.cpp -- profile stage D, the per-prime rung walk.
//
// Stage D is the expensive survivor path: measured indirectly, primes reaching it cost
// far more than the composites, and the small-prime sieve failed precisely because it
// could not touch them. This measures WHERE that cost is, on the real code rather than
// a copy: rung_scan3.cpp is included with its main renamed, the same trick it uses on
// rung_scan.cpp, so solve_rung3/min_div_res/factorA are the production functions.
//
// BUILD  g++ -O3 -march=native -std=c++17 -Icensus -o bench_staged tests/bench_staged.cpp
// RUN    ./bench_staged PRIMEFILE     (a file of `RUNG p ...` lines, or bare primes)
#define RUNG_SCAN3_NO_MAIN
#include "../sieve/rung_scan3.cpp"

#include <chrono>

using Clock = std::chrono::steady_clock;
static double secs(Clock::time_point a, Clock::time_point b) {
  return std::chrono::duration<double>(b - a).count();
}

int main(int argc, char **argv) {
  if (argc < 2) { fprintf(stderr, "usage: %s PRIMEFILE\n", argv[0]); return 1; }

  vector<u64> ps;
  {
    FILE *fp = fopen(argv[1], "r");
    if (!fp) { fprintf(stderr, "cannot open %s\n", argv[1]); return 1; }
    char line[512];
    while (fgets(line, sizeof line, fp)) {
      unsigned long long p;
      if (sscanf(line, "RUNG %llu", &p) == 1 || sscanf(line, "%llu", &p) == 1)
        ps.push_back((u64)p);
    }
    fclose(fp);
  }
  if (ps.empty()) { fprintf(stderr, "no primes in %s\n", argv[1]); return 1; }
  printf("primes: %zu   (%" PRIu64 " .. %" PRIu64 ")\n", ps.size(), ps.front(), ps.back());

  u64 hi = 0;
  for (u64 p : ps) hi = max(hi, p);
  init_phi(); init_rfac(); build_base_primes(hi + 1000);
  printf("g_small (trial-division primes <= 65536): %zu\n\n", g_small.size());

  // ---- 1. whole of stage D, as the scanner calls it
  auto t0 = Clock::now();
  u64 hits = 0;
  for (u64 p : ps) { string s; hits += solve_rung3(p, s); }
  const double t_all = secs(t0, Clock::now());
  printf("stage D total        %8.3f s   %8.1f us/prime   (%" PRIu64 " solved)\n",
         t_all, t_all / ps.size() * 1e6, hits);

  // ---- 2. how many rung attempts each prime needs, and what an attempt costs
  u64 attempts = 0;
  for (u64 p : ps) {
    const u64 r0 = (4 - p % 4) % 4 ? (4 - p % 4) % 4 : 4;
    for (u64 r = r0; r <= (u64)RCAP; r += 4) {
      ++attempts;
      const u64 A = (p + r) / 4;
      const u128 q = (u128)p * A;
      const u64 qm = (u64)(q % r);
      if (r > 1 && gcd3(qm, r) != 1) continue;
      Fac f; factorA(A, f);
      if (min_div_res(p, f, q, r, qm ? r - qm : 0)) break;
    }
  }
  printf("rung attempts        %8" PRIu64 "     %8.2f per prime\n",
         attempts, (double)attempts / ps.size());

  // ---- 3. factorA alone, on exactly the A values stage D asks for
  vector<u64> As;
  for (u64 p : ps) {
    const u64 r0 = (4 - p % 4) % 4 ? (4 - p % 4) % 4 : 4;
    As.push_back((p + r0) / 4);
  }
  t0 = Clock::now();
  u64 sink = 0;
  for (u64 A : As) { Fac f; factorA(A, f); sink += f.k; }
  const double t_fac = secs(t0, Clock::now());
  printf("factorA alone        %8.3f s   %8.1f us/call    (sink %" PRIu64 ")\n",
         t_fac, t_fac / As.size() * 1e6, sink);

  // ---- 4. min_div_res alone, given a precomputed factorization
  vector<Fac> fs(As.size());
  for (size_t i = 0; i < As.size(); ++i) factorA(As[i], fs[i]);
  t0 = Clock::now();
  u128 sink2 = 0;
  for (size_t i = 0; i < ps.size(); ++i) {
    const u64 p = ps[i], A = As[i], r = (4 - p % 4) % 4 ? (4 - p % 4) % 4 : 4;
    const u128 q = (u128)p * A;
    const u64 qm = (u64)(q % r);
    sink2 += min_div_res(p, fs[i], q, r, qm ? r - qm : 0);
  }
  const double t_dp = secs(t0, Clock::now());
  printf("min_div_res alone    %8.3f s   %8.1f us/call    (sink %" PRIu64 ")\n",
         t_dp, t_dp / ps.size() * 1e6, (u64)(sink2 & 0xffffffff));

  printf("\nshare of stage D: factorA %.0f%%, min_div_res %.0f%%, other %.0f%%\n",
         100 * t_fac * attempts / ps.size() / t_all,
         100 * t_dp * attempts / ps.size() / t_all,
         100 * (1 - (t_fac + t_dp) * attempts / ps.size() / t_all));
  return 0;
}
