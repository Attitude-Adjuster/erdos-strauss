#!/usr/bin/env bash
# diff_cuda.sh LO HI [args...] -- the CUDA sieve must agree with rung_scan3 exactly:
# every SUMMARY counter identical, certificate SET identical. Emission ORDER is not
# required to match (it already differs between cover_scan and rung_scan3), so the sets
# are sorted before comparing -- but they are ALSO compared unsorted, because this port
# emits in lane order by construction and a change that broke that would be worth
# knowing about even though it is not a correctness failure.
#
# Runs at --sieve 0: stage B' changes mr and composite by design, so a diff with it on
# would be comparing two different definitions of the same counter.
set -euo pipefail
here="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
LO=$1; HI=$2; shift 2
[[ -x "$here/rung_scan3" ]] || { echo "FAIL: rung_scan3 missing -- make -C $here" >&2; exit 1; }
[[ -x "$here/sieve/cuda/sieve_cuda" ]] || { echo "FAIL: sieve_cuda missing" >&2; exit 1; }
tmp=$(mktemp -d); trap 'rm -rf "$tmp"' EXIT

# rung_scan3 exits 2 when it leaves survivors, which is a legitimate outcome for a
# range, not a failure of the run -- `set -e` must not treat it as one.
"$here/rung_scan3"            "$LO" "$HI" "$@" --sieve 0 > "$tmp/cpu.out" 2>/dev/null || true
"$here/sieve/cuda/sieve_cuda" "$LO" "$HI" "$@" --sieve 0 > "$tmp/gpu.out" 2>/dev/null || true

if ! diff <(grep '^SUMMARY ' "$tmp/cpu.out") <(grep '^SUMMARY ' "$tmp/gpu.out"); then
  echo "FAIL: SUMMARY differs [$LO,$HI) $*" >&2; exit 1
fi
if ! diff <(grep -v '^SUMMARY ' "$tmp/cpu.out" | sort) \
          <(grep -v '^SUMMARY ' "$tmp/gpu.out" | sort); then
  echo "FAIL: certificate sets differ [$LO,$HI) $*" >&2; exit 1
fi
order=same
cmp -s <(grep -v '^SUMMARY ' "$tmp/cpu.out") <(grep -v '^SUMMARY ' "$tmp/gpu.out") \
  || order=DIFFERENT

# Determinism: repeated runs must be BYTE-identical, not merely set-identical. This is
# the check that would catch an atomicAdd sneaking into survivor placement.
"$here/sieve/cuda/sieve_cuda" "$LO" "$HI" "$@" --sieve 0 > "$tmp/gpu2.out" 2>/dev/null || true
cmp -s "$tmp/gpu.out" "$tmp/gpu2.out" \
  || { echo "FAIL: repeated GPU runs differ" >&2; exit 1; }

echo "DIFF_CUDA CLEAN [$LO,$HI) $* ($(grep -c '^RUNG \|^SOLVED ' "$tmp/gpu.out") certificates, emission order $order)"
