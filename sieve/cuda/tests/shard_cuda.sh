#!/usr/bin/env bash
# shard_cuda.sh LO HI N [args...] -- N shards must TILE [LO,HI) exactly and their union
# must equal the unsharded run.
#
# WHY THIS EXISTS. --shard is how a production run is actually driven, and until now it
# was the one path in sieve_cuda with no coverage at all: implemented by mirroring
# rung_scan3's arithmetic and never executed by any gate. A tiling bug does not crash --
# it drops or double-counts a slice of the range, and the output still looks entirely
# plausible. `positions` is what catches it: it must SUM to the unsharded total exactly,
# because every position is either in exactly one shard or the tiling is wrong.
#
# The certificate SET is compared too, and cross-checked against rung_scan3 sharded the
# same way, so the two implementations are shown to agree on the tiling and not merely
# to produce the same grand total.
set -euo pipefail
here="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
LO=$1; HI=$2; N=$3; shift 3
GPU="$here/sieve/cuda/sieve_cuda"
CPU="$here/rung_scan3"
[[ -x "$GPU" ]] || { echo "FAIL: sieve_cuda missing" >&2; exit 1; }
[[ -x "$CPU" ]] || { echo "FAIL: rung_scan3 missing -- make -C $here" >&2; exit 1; }
tmp=$(mktemp -d); trap 'rm -rf "$tmp"' EXIT

# Both scanners exit 2 when a range leaves survivors, which is a legitimate outcome and
# must not be read as a failure by `set -e`.
run() { "$@" 2>/dev/null || true; }

FIELDS="positions covered sieved mr composite rung direct survivors"

# THE BOUNDARY CHECK IS THE ONE THAT MATTERS, and summing counters is not a substitute.
# A floor-instead-of-ceil span drops only (HI-LO) mod N integers -- at most N-1 of them --
# and at one wheel position per 885 integers a handful of dropped integers almost always
# contain none at all. The first version of this gate summed `positions`, passed with the
# tail-dropping bug deliberately compiled in, and would have shipped as decoration.
# Comparing the shard boundaries catches any off-by-one immediately, whether or not a
# position happens to fall in the gap.
check_tiling() {   # check_tiling FILE WHO -- shards must tile [LO,HI) with no gap/overlap
  local prev=$LO first=1
  while read -r l h; do
    [[ "$l" == "$h" ]] && continue                 # empty shard: legitimate, carries no range
    if [[ $first == 1 ]]; then
      [[ "$l" == "$LO" ]] || { echo "FAIL: $2 first shard starts at $l, not $LO" >&2; exit 1; }
      first=0
    elif [[ "$l" != "$prev" ]]; then
      echo "FAIL: $2 tiling $( [[ $l -gt $prev ]] && echo gap || echo overlap ) at $prev -> $l" >&2
      exit 1
    fi
    prev=$h
  done < <(grep -o 'lo=[0-9]* hi=[0-9]*' "$1" | sed 's/lo=//;s/hi=//')
  [[ "$prev" == "$HI" ]] || { echo "FAIL: $2 shards end at $prev, not $HI" >&2; exit 1; }
}

sum_field() {   # sum_field FILE KEY -- add up one SUMMARY counter across all shards
  grep -o "$2=[0-9]*" "$1" | cut -d= -f2 | paste -sd+ - | bc
}

for who in gpu cpu; do
  bin=$GPU; [[ $who == cpu ]] && bin=$CPU
  run "$bin" "$LO" "$HI" "$@" > "$tmp/$who.full"
  : > "$tmp/$who.shards"
  for ((i = 0; i < N; ++i)); do
    run "$bin" "$LO" "$HI" "$@" --shard "$i/$N" >> "$tmp/$who.shards"
  done
  grep '^SUMMARY ' "$tmp/$who.full"   > "$tmp/$who.full.sum"
  grep '^SUMMARY ' "$tmp/$who.shards" > "$tmp/$who.shards.sum"
  for f in $FIELDS; do
    want=$(sum_field "$tmp/$who.full.sum" "$f")
    got=$(sum_field "$tmp/$who.shards.sum" "$f")
    if [[ "$want" != "$got" ]]; then
      echo "FAIL: $who $f -- unsharded $want, $N shards summed to $got" >&2; exit 1
    fi
  done
  # An empty shard still prints its SUMMARY, so a coverage check sees the full tiling
  # rather than a gap. Exactly N of them, always.
  n=$(wc -l < "$tmp/$who.shards.sum")
  [[ "$n" -eq "$N" ]] || { echo "FAIL: $who emitted $n SUMMARY lines for $N shards" >&2; exit 1; }
  check_tiling "$tmp/$who.shards.sum" "$who"
  if ! diff <(grep -v '^SUMMARY ' "$tmp/$who.full"   | sort) \
            <(grep -v '^SUMMARY ' "$tmp/$who.shards" | sort) >/dev/null; then
    echo "FAIL: $who certificate set differs between sharded and unsharded" >&2; exit 1
  fi
done

# And the two implementations must tile identically, not merely total identically.
for f in $FIELDS; do
  a=$(sum_field "$tmp/gpu.shards.sum" "$f"); b=$(sum_field "$tmp/cpu.shards.sum" "$f")
  [[ "$a" == "$b" ]] || { echo "FAIL: sharded $f gpu=$a cpu=$b" >&2; exit 1; }
done
if ! diff <(grep -v '^SUMMARY ' "$tmp/gpu.shards" | sort) \
          <(grep -v '^SUMMARY ' "$tmp/cpu.shards" | sort) >/dev/null; then
  echo "FAIL: sharded certificate sets differ between sieve_cuda and rung_scan3" >&2; exit 1
fi

pos=$(sum_field "$tmp/gpu.full.sum" positions)
echo "SHARD_CUDA CLEAN [$LO,$HI) x$N $* ($pos positions tiled exactly, $(grep -c '^RUNG \|^SOLVED ' "$tmp/gpu.shards") certificates)"
