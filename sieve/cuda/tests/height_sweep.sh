#!/usr/bin/env bash
# height_sweep.sh [SPAN] -- cost per integer against height, at a FIXED span so the rows
# are directly comparable.
#
# The standing rule on this project is that a cost model must never be carried across a
# change of regime, and it has been paid for twice. So this measures rather than projects,
# and it takes ninety seconds.
#
# READ THE COMPONENTS, NOT `wall`. Wall grows ~4x from 10^12 to 10^16 and that growth is
# ENTIRELY per-process startup: build_base_primes sieves to sqrt(hi), which is 2.4e6 at
# 10^12 and 1e8 at 10^16. The actual work -- offsets + launch + device + host tail -- is
# flat to 3% across those four decades, because two effects cancel: survivor density
# FALLS with height (so device work falls) while stage D costs more per prime (so the
# tail rises). A run long enough to matter amortises the startup away entirely.
set -euo pipefail
here="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
BIN="$here/sieve/cuda/sieve_cuda"
TAB="$here/tables/filters/f_M2042040_mmax2000_gpu.txt"
SPAN=${1:-5000000000000}
[[ -x "$BIN" ]] || { echo "FAIL: sieve_cuda missing" >&2; exit 1; }

printf '%-20s %8s %8s %9s %14s %12s %10s %10s\n' \
       lo wall work startup ns/int_work survivors primes us/prime
for lo in 1000000000000 10000000000000 100000000000000 1000000000000000 10000000000000000; do
  hi=$((lo + SPAN))
  err=$(mktemp); out=$(mktemp)
  "$BIN" "$lo" "$hi" --filters "$TAB" --profile --no-emit-survivors 2>"$err" >"$out" || true
  g() { grep -o "$1=[0-9.]*" "$err" | head -1 | cut -d= -f2; }
  s() { grep -o "$1=[0-9]*" "$out" | head -1 | cut -d= -f2; }
  awk -v lo="$lo" -v span="$SPAN" -v w="$(g wall)" -v o="$(g offsets)" -v l="$(g launch)" \
      -v d="$(g device-wait)" -v t="$(g host-tail)" -v mr="$(s mr)" -v ru="$(s rung)" \
      'BEGIN{work=o+l+d+t;
       printf "%-20s %8.3f %8.3f %9.3f %14.5f %12d %10d %10.1f\n",
              lo, w, work, w-work, work/span*1e9, mr, ru, (ru>0? t/ru*1e6 : 0)}'
  rm -f "$err" "$out"
done
