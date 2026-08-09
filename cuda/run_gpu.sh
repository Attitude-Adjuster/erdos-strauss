#!/usr/bin/env bash
# run_gpu.sh -- the GPU census driver: sharded, resumable, gate-first.
#
#   bash cuda/run_gpu.sh LO HI NSHARDS OUTDIR
#   bash cuda/run_gpu.sh 0 1000000000000000 400 c15
#
# Mirrors run8.sh's contract so merge.py consumes the output unchanged: OUTDIR gets
# shard_<i>.out files, each containing that shard's streamed certificates plus its
# SUMMARY line. Shards run SEQUENTIALLY -- one GPU, and the scanner already saturates
# it -- so sharding here buys exactly one thing: CHECKPOINTING. The box this runs on
# has terminated spontaneously before (zone maintenance); with ~400 shards a kill
# costs ~30 seconds of work, and rerunning the script skips every completed shard.
#
# A shard is COMPLETE iff its .out contains a SUMMARY line; a file without one is a
# casualty of an interruption and is redone. merge.py independently re-checks the
# tiling, so a subtly truncated shard cannot slip through -- but the SUMMARY test
# means we never *resume into* a half-written file.
#
# Census flags match the published 10^13 run (RUN_1e13.md): hard class, deep
# threshold 71, sample stride 10^9. --spanlog 24 is the GPU's measured optimum.
set -euo pipefail
cd "$(dirname "$0")"
export PATH=/usr/local/cuda/bin:$PATH

LO=${1:?usage: run_gpu.sh LO HI NSHARDS OUTDIR}
HI=${2:?}
NSHARDS=${3:?}
OUT=${4:?}
DEEP=${DEEP:-71}
SAMPLE=${SAMPLE:-1000000000}
# 255, matching run8.sh -- NOT the binary's default of 127. Omitting this cost a
# 400-shard census 13 escalations (primes deeper than 127, including the depth
# record), caught by merge.py's must-be-zero check. Only the affected shards needed
# rerunning -- a shard with no escalations is identical under either bound.
RMAX=${RMAX:-255}

mkdir -p "../$OUT"

# ---- the gate, every run, no exceptions (see RUNBOOK.md). A census produced by a
# binary that has not passed the full check on THIS machine is worthless.
make -s rung_scan_cuda
make -s -C .. >/dev/null 2>&1 || true
echo "[run_gpu] make check (the gate) ..."
make -s check > "../$OUT/gate.log" 2>&1 || {
  echo "[run_gpu] GATE FAILED -- see $OUT/gate.log. No census." >&2
  exit 1
}
grep -cE 'DIFF CLEAN|OK$|PASSED' "../$OUT/gate.log" | \
  xargs -I{} echo "[run_gpu] gate green ({} checks)"

# Execute a COPY of the binary (run8.sh's lesson, spec T0): a rebuild mid-census
# rewrites the file in place and kills any process running the old image.
cp rung_scan_cuda "../$OUT/rung_scan_cuda.run"

t0=$(date +%s)
done_n=0
for ((i = 0; i < NSHARDS; ++i)); do
  f="../$OUT/shard_$i.out"
  if [[ -f $f ]] && grep -q '^SUMMARY ' "$f"; then
    continue                                        # complete from a previous attempt
  fi
  "../$OUT/rung_scan_cuda.run" "$LO" "$HI" --shard "$i/$NSHARDS" --hard840 --spanlog 24 \
      --rmax "$RMAX" --emit-deep "$DEEP" --sample "$SAMPLE" > "$f.tmp" 2> "../$OUT/shard_$i.err"
  grep -q '^SUMMARY ' "$f.tmp" || { echo "[run_gpu] shard $i: no SUMMARY" >&2; exit 1; }
  mv "$f.tmp" "$f"                                  # atomic completion marker
  done_n=$((done_n + 1))
  if (( done_n % 10 == 0 )); then
    el=$(( $(date +%s) - t0 ))
    echo "[run_gpu] $((i + 1))/$NSHARDS shards, ${el}s elapsed"
  fi
done

echo "[run_gpu] all $NSHARDS shards complete; merging"
python3 ../merge.py "../$OUT"          # writes MERGED.txt + interesting.txt, validates
echo "[run_gpu] done: $OUT/MERGED.txt, $OUT/interesting.txt"
echo "[run_gpu] verify on a DIFFERENT machine: python3 verify_cert.py $OUT/interesting.txt"
