#!/usr/bin/env bash
# cover_smoke.sh LO HI [cover_scan args...] -- the end-to-end gate for the reference.
# Everything cover_scan emits must re-derive in exact arithmetic, the class table must
# match the checked-in artifact byte for byte, and no survivor may remain.
#
# Run this per machine and per flag set, for the same reason `make check` is run that
# way: it guards against miscompilation and unsafe optimization flags, not just logic.
set -euo pipefail
here="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
LO=$1; HI=$2; shift 2
[[ -x "$here/cover_scan" ]] || { echo "FAIL: cover_scan missing -- make cover_scan" >&2; exit 1; }
tmp=$(mktemp -d); trap 'rm -rf "$tmp"' EXIT

"$here/cover_scan" --verify > "$tmp/verify.out" || { cat "$tmp/verify.out"; exit 1; }
grep -q '^SELF-TEST PASSED' "$tmp/verify.out" || { echo "FAIL: self-test" >&2; exit 1; }

# Layer L1: the binary's own derivation vs the published artifact vs Python's.
"$here/cover_scan" --dump-classes > "$tmp/classes.txt"
diff -q "$tmp/classes.txt" "$here/tables/class_table_120120.txt" \
  || { echo "FAIL: class table differs from the checked-in artifact" >&2; exit 1; }
python3 "$here/sieve/verify_covers.py" --class-table --check "$here/tables/class_table_120120.txt" >/dev/null \
  || { echo "FAIL: python class table disagrees" >&2; exit 1; }

# Layers L2/L3: every emitted certificate re-derived in exact rationals, nothing left.
"$here/cover_scan" "$LO" "$HI" "$@" > "$tmp/scan.out"
python3 "$here/sieve/verify_covers.py" "$tmp/scan.out" \
  || { echo "FAIL: emitted certificates do not verify" >&2; exit 1; }
grep '^SUMMARY ' "$tmp/scan.out" | grep -q 'survivors=0' \
  || { echo "FAIL: survivors remain" >&2; grep '^SUMMARY ' "$tmp/scan.out" >&2; exit 1; }

# Free falsifiable check: windowing may not touch arithmetic. Same range, different
# spanlog, must produce an identical SUMMARY and an identical certificate SET.
"$here/cover_scan" "$LO" "$HI" "$@" --spanlog 14 > "$tmp/scan2.out"
diff <(grep '^SUMMARY ' "$tmp/scan.out") <(grep '^SUMMARY ' "$tmp/scan2.out") \
  || { echo "FAIL: SUMMARY depends on --spanlog" >&2; exit 1; }
diff <(grep -v '^SUMMARY ' "$tmp/scan.out" | sort) \
     <(grep -v '^SUMMARY ' "$tmp/scan2.out" | sort) \
  || { echo "FAIL: certificate set depends on --spanlog" >&2; exit 1; }

echo "COVER SMOKE CLEAN [$LO,$HI) $*"
