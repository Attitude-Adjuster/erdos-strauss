#!/usr/bin/env bash
# run8.sh -- shard an Erdos-Straus certificate census across cores, sockets, machines.
#
#   ./run8.sh LO HI [NSHARDS] [OUTDIR]
#
#   ./run8.sh 0 1000000000000                      # 0 -> 10^12, 8 shards, 8 cores
#   JOBS=4 THREADS=2 ./run8.sh 0 1000000000000     # 4 processes x 2 threads
#   NUMA=1 ./run8.sh 0 10000000000000 16           # pin processes round-robin to NUMA nodes
#   MACHINES=4 MACHINE=2 NSHARDS=32 ./run8.sh 0 1e15-ish ...   # box 2 of a 4-box cluster
#
# PARALLELISM MODEL
#   Two independent axes, and you choose the split:
#     JOBS     concurrent processes on this machine   (default 8)
#     THREADS  OpenMP threads inside each process     (default 1)
#   Cores used = JOBS x THREADS. Default 8 x 1.
#
#   Prefer PROCESSES (THREADS=1) below ~10^14: shards are independent and individually
#   recomputable, a crash costs one shard rather than the run, and there is no lock
#   contention at all.
#   Prefer THREADS only when the shared prime tables get large, which is ~10^18
#   (1.2 GB shared -> ~10 GB duplicated across 8 processes). At 10^15 they are only
#   ~45 MB, so processes still win. Hybrid (JOBS=4 THREADS=2) can help locality on
#   two sockets.
#
# MULTIPLE MACHINES
#   Set NSHARDS to the GLOBAL shard count and give each box its index:
#     box 0:  MACHINES=4 MACHINE=0 NSHARDS=32 ./run8.sh LO HI 32 out
#     box 1:  MACHINES=4 MACHINE=1 NSHARDS=32 ./run8.sh LO HI 32 out
#   Box k runs shards k, k+4, k+8, ... Shards share nothing; collect the .out files
#   onto one host at the end and run merge.py over the union.
#
# Other design notes, each a lesson paid for:
#   * The binary is COPIED into OUTDIR and the copy is executed, so rebuilding the
#     source tree mid-run cannot kill the run (the linker rewrites the file in place
#     and any process running the old image dies).
#   * RESUMABLE: a shard whose .out already ends in a SUMMARY line is skipped.
#   * Residual lines are off by default: at 10^12 there are ~24 million of them.
#   * HARD840=1 restricts to the 6 SQUARE classes mod 840 -- by Mordell every other
#     class is settled by a classical identity. 4.3x fewer primes, and empirically it
#     keeps every level-R failure and every deep ladder. It DOES change the population,
#     so shares are not comparable with a (1 mod 24) census.

set -euo pipefail

LO=${1:?usage: run8.sh LO HI [NSHARDS] [OUTDIR]}
HI=${2:?usage: run8.sh LO HI [NSHARDS] [OUTDIR]}
NSHARDS=${3:-${NSHARDS:-8}}
OUTDIR=${4:-census_out}

JOBS=${JOBS:-8}                 # concurrent processes on this machine
THREADS=${THREADS:-1}           # OpenMP threads per process
MACHINES=${MACHINES:-1}         # total machines cooperating
MACHINE=${MACHINE:-0}           # index of this machine, 0..MACHINES-1
NUMA=${NUMA:-0}                 # 1 = pin processes round-robin across NUMA nodes

RMAX=${RMAX:-255}
# Which scanner binary to shard. rung_scan (default) is the frozen reference and the
# trust anchor; rung_scan2 is the optimized scanner, ~2.3x faster at its own optimum
# (--spanlog 20 -- note the reference's optimum is 22), output-identical by
# tests/diff2.sh. For big CPU-only censuses: BIN=rung_scan2 SPANLOG=20 ./run8.sh ...
BIN=${BIN:-rung_scan}
SPANLOG=${SPANLOG:-22}          # 20 below ~10^14, 22 above; re-tune per hardware
DEEP=${DEEP:-91}
EMIT_RESIDUAL=${EMIT_RESIDUAL:-0}
PROGRESS=${PROGRESS:-25}
HARD840=${HARD840:-0}      # 1 = only the 6 square classes mod 840 (4.3x fewer primes)

here="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
[[ -x "$here/$BIN" ]] || { echo "error: $here/$BIN not built. Run 'make $BIN'." >&2; exit 1; }

echo "== self-test (must pass before anything else) =="
"$here/$BIN" --verify || { echo "SELF-TEST FAILED -- do not use this build." >&2; exit 1; }
echo

mkdir -p "$OUTDIR"
cp -f "$here/$BIN" "$OUTDIR/rung_scan.frozen"
chmod +x "$OUTDIR/rung_scan.frozen"

EXTRA=(--rmax "$RMAX" --spanlog "$SPANLOG" --emit-deep "$DEEP" --progress "$PROGRESS")
[[ "$EMIT_RESIDUAL" == "1" ]] && EXTRA+=(--emit-residual)
[[ "$HARD840" == "1" ]] && EXTRA+=(--hard840)

# NUMA nodes, if pinning was asked for and numactl exists
numa_nodes=0
if [[ "$NUMA" == "1" ]]; then
  if command -v numactl >/dev/null 2>&1; then
    numa_nodes=$(numactl --hardware 2>/dev/null | awk '/^available:/{print $2}')
    echo "NUMA pinning enabled across ${numa_nodes} node(s)"
  else
    echo "NUMA=1 but numactl not found; continuing unpinned" >&2
  fi
fi

echo "== census [$LO, $HI) =="
echo "   shards(global)=$NSHARDS   this machine=$MACHINE of $MACHINES"
echo "   jobs=$JOBS x threads=$THREADS = $((JOBS * THREADS)) cores   spanlog=$SPANLOG"
echo "   outdir=$OUTDIR   emit-residual=$EMIT_RESIDUAL   hard840=$HARD840"
echo

# which global shard indices belong to this machine
mine=()
for ((i = MACHINE; i < NSHARDS; i += MACHINES)); do mine+=("$i"); done
echo "   this machine owns shards: ${mine[*]}"
echo

running=0
pids=()
launch() {
  local i=$1
  local out="$OUTDIR/shard_$i.out" log="$OUTDIR/shard_$i.log"
  if [[ -s "$out" ]] && grep -q '^SUMMARY ' "$out"; then
    echo "  shard $i: already complete, skipping"; return
  fi
  local pin=()
  if [[ "$numa_nodes" =~ ^[0-9]+$ ]] && (( numa_nodes > 1 )); then
    pin=(numactl --cpunodebind=$(( i % numa_nodes )) --preferred=$(( i % numa_nodes )))
  fi
  OMP_NUM_THREADS="$THREADS" nohup "${pin[@]}" "$OUTDIR/rung_scan.frozen" "$LO" "$HI" \
      --shard "$i/$NSHARDS" "${EXTRA[@]}" > "$out" 2> "$log" &
  pids+=($!)
  echo "  shard $i: pid ${pids[-1]}  ->  $out"
}

fail=0
for i in "${mine[@]}"; do
  # keep at most JOBS processes in flight
  while (( $(jobs -rp | wc -l) >= JOBS )); do wait -n || fail=1; done
  launch "$i"
done
for pid in "${pids[@]}"; do
  if ! wait "$pid"; then echo "  !! shard pid $pid exited non-zero" >&2; fail=1; fi
done
(( fail )) && echo "WARNING: a shard failed; re-run this identical command to retry just it." >&2

echo
if (( MACHINES > 1 )); then
  echo "== this machine done. Collect all shard_*.out onto one host, then: =="
  echo "   python3 merge.py $OUTDIR"
else
  echo "== merging =="
  python3 "$here/merge.py" "$OUTDIR"
fi
