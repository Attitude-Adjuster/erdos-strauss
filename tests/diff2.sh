#!/usr/bin/env bash
# diff2.sh LO HI [scanner args...] -- rung_scan2 must agree with the reference
# exactly: statistical SUMMARY fields byte-identical, certificate SET identical
# (emission order may differ; sort before comparing). Same bar as the CUDA port.
set -euo pipefail
here="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
LO=$1; HI=$2; shift 2
for b in rung_scan rung_scan2; do
  [[ -x "$here/$b" ]] || { echo "FAIL: $here/$b missing -- make $b first" >&2; exit 1; }
done
tmp=$(mktemp -d); trap 'rm -rf "$tmp"' EXIT
"$here/rung_scan"  "$LO" "$HI" "$@" > "$tmp/ref.out"
"$here/rung_scan2" "$LO" "$HI" "$@" > "$tmp/opt.out"
strip_timing() { sed -E 's/ sieve_s=[0-9.]+ scan_s=[0-9.]+ ns_per_prime=[0-9]+$//'; }
if ! diff <(grep '^SUMMARY ' "$tmp/ref.out" | strip_timing) \
          <(grep '^SUMMARY ' "$tmp/opt.out" | strip_timing); then
  echo "FAIL: SUMMARY differs [$LO,$HI) $*" >&2; exit 1
fi
if ! diff <(grep -v '^SUMMARY ' "$tmp/ref.out" | sort) \
          <(grep -v '^SUMMARY ' "$tmp/opt.out" | sort); then
  echo "FAIL: certificate sets differ [$LO,$HI) $*" >&2; exit 1
fi
echo "DIFF2 CLEAN [$LO,$HI) $*"
