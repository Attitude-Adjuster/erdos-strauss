#!/usr/bin/env bash
# height_sweep2.sh -- the parts height_sweep.sh could not afford, at a sane width.
#
# height_sweep.sh's ranges came out 1000x wider than intended: ~40 BILLION primes per
# height instead of ~40 million. That made section 2 a better measurement than planned
# (it is census-scale) but made its CPU control unaffordable -- the reference over 4e13
# of width is ~11 hours per row.
#
# So: same questions, ~40M primes per row, and CPU and GPU always at the SAME width so
# the ratio is apples to apples. Cost per prime depends on width (per-window and
# table-build costs amortize), so a ratio taken across different widths is meaningless.
set -u
cd "$(dirname "$0")/.."
export PATH=/usr/local/cuda/bin:$PATH
GPU=./rung_scan_cuda
CPU=../rung_scan

say() { printf '%s\n' "$*"; }
summarize() {
  local label="$1"; shift
  local s
  s=$("$@" 2>/dev/null | grep '^SUMMARY ' | head -1)
  if [[ -z "$s" ]]; then say "$label | NO SUMMARY"; return 1; fi
  printf '%-30s primes=%-11s ns/prime=%-6s scan_s=%-8s tables_s=%s\n' "$label" \
    "$(sed -E 's/.* primes=([0-9]+) .*/\1/' <<<"$s")" \
    "$(sed -E 's/.* ns_per_prime=([0-9]+).*/\1/' <<<"$s")" \
    "$(sed -E 's/.* scan_s=([0-9.]+) .*/\1/' <<<"$s")" \
    "$(sed -E 's/.* sieve_s=([0-9.]+) .*/\1/' <<<"$s")"
}

say "=== sweep2 started $(date -u '+%Y-%m-%dT%H:%M:%SZ') ==="
say ""
say "--- CPU vs GPU at the same width (~40M primes), by height ---"
summarize "10^13  GPU" $GPU  10000000000000  10038000000000 --hard840
summarize "10^13  CPU x12" $CPU 10000000000000 10038000000000 --hard840 --spanlog 22
summarize "10^14  GPU" $GPU 100000000000000 100041000000000 --hard840
summarize "10^14  CPU x12" $CPU 100000000000000 100041000000000 --hard840 --spanlog 22
summarize "10^15  GPU" $GPU 1000000000000000 1000044000000000 --hard840
summarize "10^15  CPU x12" $CPU 1000000000000000 1000044000000000 --hard840 --spanlog 22
say ""
say "--- does --spanlog 24 still hold at 10^15? ---"
for L in 22 24 25 26; do
  summarize "10^15  GPU --spanlog $L" $GPU 1000000000000000 1000044000000000 \
            --hard840 --spanlog "$L"
done
say ""
say "--- phase mix at 10^15: does the sieve come back as pi(sqrt X) grows? ---"
$GPU 1000000000000000 1000044000000000 --hard840 --profile 2>/dev/null \
  | grep -E 'PROFILE  (---|host|H2D|factor|classi|D2H|CPU|TOTAL)'
say ""
say "=== sweep2 finished $(date -u '+%Y-%m-%dT%H:%M:%SZ') ==="
