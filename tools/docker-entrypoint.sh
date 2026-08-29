#!/usr/bin/env bash
# Build on THIS machine, self-test, then census.  Usage: LO HI [NSHARDS]
#
# Every knob run8.sh understands is passed through the environment unchanged
# (HARD840, JOBS, THREADS, SPANLOG, DEEP, PROGRESS, EMIT_RESIDUAL, NSHARDS,
# MACHINES, MACHINE, NUMA). The defaults below are the 10^15 production settings,
# not the run8.sh defaults.
set -euo pipefail

LO=${1:?usage: LO HI [NSHARDS]}
HI=${2:?usage: LO HI [NSHARDS]}

# Default the shard count to ~8x the job count so a core that finishes early picks
# up more work instead of idling through the tail, and so resume granularity stays
# fine on a run measured in hours.
JOBS=${JOBS:-$(nproc)}
NSHARDS=${3:-${NSHARDS:-$(( JOBS * 8 ))}}

export HARD840=${HARD840:-1}          # the population the hypotheses are about
export SPANLOG=${SPANLOG:-22}         # 22 above ~10^14; re-tune per hardware
export DEEP=${DEEP:-71}               # collect the whole deep tail, not just the tip
export PROGRESS=${PROGRESS:-5}
export EMIT_RESIDUAL=${EMIT_RESIDUAL:-0}
export JOBS THREADS=${THREADS:-1}

OUTDIR=${OUTDIR:-/out/census}
mkdir -p "$OUTDIR"

echo "== build (this machine, NATIVE=${NATIVE:-1}) =="
make -C /erdos clean >/dev/null
make -C /erdos NATIVE="${NATIVE:-1}"

# Hard gate. run8.sh self-tests too, but failing here means we never touched OUTDIR.
echo
echo "== self-test =="
/erdos/rung_scan --verify

# A 10^15 census is hours of compute; measure this machine before spending them.
if [[ "${PILOT:-1}" == "1" ]]; then
  echo
  echo "== pilot: ns/prime at the top of the range =="
  PLO=$(( HI - HI / 1000 ))
  OMP_NUM_THREADS=1 /erdos/rung_scan "$PLO" "$HI" --hard840 \
      --rmax 255 --spanlog "$SPANLOG" 2>/dev/null | grep '^SUMMARY' \
      | tr ' ' '\n' | grep -E 'ns_per_prime|primes=' || true
  echo "   (multiply ns_per_prime by the hard-class prime count under HI for core-seconds)"
fi

echo
echo "== census [$LO, $HI) into $OUTDIR =="
echo "   jobs=$JOBS shards=$NSHARDS hard840=$HARD840 spanlog=$SPANLOG"
cd /erdos
exec ./census/run8.sh "$LO" "$HI" "$NSHARDS" "$OUTDIR"
