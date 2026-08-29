#!/usr/bin/env bash
# diff3.sh LO HI [args...] -- rung_scan3 must agree with the FROZEN cover_scan exactly:
# every SUMMARY counter identical, certificate SET identical. Emission ORDER differs by
# construction (cover_scan is window-major, rung_scan3 is lane-major), so sort first --
# the set may not differ.
#
# Run with the DEFAULT filter set only. Pruning and --f2-covers deliberately change the
# filter set, so they change `covered` and `mr`; those modes are checked by the nesting
# property in Task 5, not here.
set -euo pipefail
here="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
LO=$1; HI=$2; shift 2
for b in cover_scan rung_scan3; do
  [[ -x "$here/$b" ]] || { echo "FAIL: $here/$b missing -- make $b first" >&2; exit 1; }
done
tmp=$(mktemp -d); trap 'rm -rf "$tmp"' EXIT
"$here/cover_scan"  "$LO" "$HI" "$@" > "$tmp/ref.out" 2>/dev/null
"$here/rung_scan3"  "$LO" "$HI" "$@" > "$tmp/opt.out" 2>/dev/null
# `sieved=` is a rung_scan3-only field (stage B' does not exist in the frozen
# reference), so it is removed before the comparison -- the same precedent as
# diff2.sh's strip_timing. Its VALUE is checked separately at the bottom of this
# file; here only the fields the two implementations share are compared, and those
# must be byte-identical.
#
# `classes_sha256=` is also removed, for a different reason: cover_scan is FROZEN with a
# single pinned digest, so under `--wheel M` for any M other than 120120 it prints that
# constant regardless of the table it actually derived. rung_scan3 pins per modulus and
# prints the right one. Dropping the field costs nothing, because the class table is
# verified far more strongly elsewhere -- `--dump-classes` must be byte-identical to the
# checked-in artifact, which verify_covers.py regenerates independently.
strip_sieved() { sed -E 's/ sieved=[0-9]+//; s/ classes_sha256=[0-9a-f]+//'; }
if ! diff <(grep '^SUMMARY ' "$tmp/ref.out" | strip_sieved) \
          <(grep '^SUMMARY ' "$tmp/opt.out" | strip_sieved); then
  echo "FAIL: SUMMARY differs [$LO,$HI) $*" >&2; exit 1
fi
if ! diff <(grep -v '^SUMMARY ' "$tmp/ref.out" | sort) \
          <(grep -v '^SUMMARY ' "$tmp/opt.out" | sort); then
  echo "FAIL: certificate sets differ [$LO,$HI) $*" >&2; exit 1
fi
# Free falsifiable check: the segment size is a cache-tuning knob and may not touch
# arithmetic. Same range at a different --seg must give an identical SUMMARY and an
# identical certificate set.
"$here/rung_scan3" "$LO" "$HI" "$@" --seg 4096 > "$tmp/opt2.out" 2>/dev/null
if ! diff <(grep '^SUMMARY ' "$tmp/opt.out") <(grep '^SUMMARY ' "$tmp/opt2.out"); then
  echo "FAIL: SUMMARY depends on --seg" >&2; exit 1
fi
if ! diff <(grep -v '^SUMMARY ' "$tmp/opt.out" | sort) \
          <(grep -v '^SUMMARY ' "$tmp/opt2.out" | sort); then
  echo "FAIL: certificate set depends on --seg" >&2; exit 1
fi

# Shards must tile the range exactly: the union of 4 shards equals the whole run, and
# the per-stage counters must add up. This is what makes a crashed shard recomputable.
: > "$tmp/shards.out"
for i in 0 1 2 3; do
  "$here/rung_scan3" "$LO" "$HI" "$@" --shard $i/4 >> "$tmp/shards.out" 2>/dev/null
done
if ! diff <(grep -v '^SUMMARY ' "$tmp/opt.out" | sort) \
          <(grep -v '^SUMMARY ' "$tmp/shards.out" | sort); then
  echo "FAIL: shard union != whole range" >&2; exit 1
fi
for f in positions covered mr composite rung direct survivors; do
  whole=$(grep '^SUMMARY ' "$tmp/opt.out" | tr ' ' '\n' | sed -n "s/^$f=//p")
  # awk, not bc: bc is not on the Debian cloud image. Counters here are far below
  # 2^53, so double accumulation is exact.
  sum=$(grep '^SUMMARY ' "$tmp/shards.out" | tr ' ' '\n' | sed -n "s/^$f=//p" \
        | awk '{s+=$1} END{printf "%.0f\n", s}')
  [[ "$whole" == "$sum" ]] || { echo "FAIL: $f does not reconcile ($whole vs $sum)" >&2; exit 1; }
done

# Determinism: thread count may not affect output, and repeated runs must be identical.
# This is cmp, not sort|diff -- BYTE-identical, which is achievable because lanes are
# merged in a fixed order regardless of the schedule, and is a much stronger check.
for t in 1 4 8; do
  "$here/rung_scan3" "$LO" "$HI" "$@" --threads $t > "$tmp/t$t.out" 2>/dev/null
done
for t in 4 8; do
  cmp -s "$tmp/t1.out" "$tmp/t$t.out" \
    || { echo "FAIL: output differs at --threads $t (not just reordered)" >&2; exit 1; }
done
"$here/rung_scan3" "$LO" "$HI" "$@" --threads 8 > "$tmp/rep.out" 2>/dev/null
cmp -s "$tmp/t8.out" "$tmp/rep.out" \
  || { echo "FAIL: repeated 8-thread runs differ" >&2; exit 1; }

# Stage B' (the small-prime sieve) makes a DIFFERENT claim from the scanner, so it gets
# a different test. Everything above ran at --sieve 0 and is unweakened by its existence.
# Here the sieve is ON, and three things must hold: the certificate set is identical, no
# survivor remains, and `covered` -- the counter the published claim rests on -- has not
# moved. (1) is what fails if the sieve ever removes a PRIME; (3) is what fails if the
# cover and sieve phases interleave and cross-contaminate their counters.
"$here/rung_scan3" "$LO" "$HI" "$@" --sieve 1000 > "$tmp/sv.out" 2>/dev/null
if ! diff <(grep -v '^SUMMARY ' "$tmp/opt.out" | sort) \
          <(grep -v '^SUMMARY ' "$tmp/sv.out" | sort); then
  echo "FAIL: --sieve changed the certificate set (it may only remove composites)" >&2
  exit 1
fi
grep '^SUMMARY ' "$tmp/sv.out" | grep -q 'survivors=0' \
  || { echo "FAIL: survivors remain with --sieve on" >&2; exit 1; }
cov0=$(grep '^SUMMARY ' "$tmp/opt.out" | tr ' ' '\n' | sed -n 's/^covered=//p')
cov1=$(grep '^SUMMARY ' "$tmp/sv.out"  | tr ' ' '\n' | sed -n 's/^covered=//p')
[[ "$cov0" == "$cov1" ]] \
  || { echo "FAIL: --sieve changed the covered count ($cov0 vs $cov1)" >&2; exit 1; }
sieved=$(grep '^SUMMARY ' "$tmp/sv.out" | tr ' ' '\n' | sed -n 's/^sieved=//p')
[[ -n "$sieved" && "$sieved" != "0" ]] \
  || { echo "FAIL: --sieve 1000 sieved nothing -- the check would be vacuous" >&2; exit 1; }

echo "DIFF3 CLEAN [$LO,$HI) $* ($(grep -c '^RUNG \|^SOLVED ' "$tmp/opt.out") certificates, $sieved sieved)"
