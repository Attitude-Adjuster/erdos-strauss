#!/usr/bin/env bash
# diff_range.sh LO HI [scanner args...]
# The project's primary test: CPU and CUDA scanners must agree exactly.
set -euo pipefail
here="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
LO=$1; HI=$2; shift 2

if [[ ! -x "$here/../../rung_scan" ]]; then
  echo "FAIL: $here/../../rung_scan not found or not executable -- run 'make' in the repo root first" >&2
  exit 1
fi

tmp=$(mktemp -d); trap 'rm -rf "$tmp"' EXIT

"$here/../../rung_scan"    "$LO" "$HI" "$@" > "$tmp/cpu.out"
"$here/rung_scan_cuda"  "$LO" "$HI" "$@" > "$tmp/gpu.out"

# SUMMARY carries wall-clock timing (sieve_s, scan_s, ns_per_prime) after the
# statistics. That's measurement noise, not a correctness signal -- it varies
# run-to-run even for the identical binary, and once real GPU kernels exist a
# faster ns_per_prime on the GPU side is the whole point, not a diff failure.
# Strip it so the comparison is over the deterministic statistics only.
strip_timing() { sed -E 's/ sieve_s=[0-9.]+ scan_s=[0-9.]+ ns_per_prime=[0-9.]+$//'; }
if ! diff <(grep '^SUMMARY ' "$tmp/cpu.out" | strip_timing) \
          <(grep '^SUMMARY ' "$tmp/gpu.out" | strip_timing); then
  echo "FAIL: SUMMARY differs" >&2; exit 1
fi
if ! diff <(grep -v '^SUMMARY ' "$tmp/cpu.out" | sort) \
          <(grep -v '^SUMMARY ' "$tmp/gpu.out" | sort); then
  echo "FAIL: certificate sets differ" >&2; exit 1
fi
echo "DIFF CLEAN [$LO,$HI) $*"
