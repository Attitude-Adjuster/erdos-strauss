#!/usr/bin/env bash
# run19.sh -- the verification sieve over [255255, 10^19) on one L4, restartably.
#
# DESIGN. 250 shards via the gated --shard arithmetic (tests/shard_cuda.sh), each ~1 h:
# a failure or restart loses at most one shard, and a finished shard is never touched
# again. Certificates are NOT kept -- at 10^19 the stream is ~3 TB, far past the disk --
# and they do not need to be: the stream is deterministic (byte-identical reruns are part
# of the gate), so each shard's meta records its sha256, its SUMMARY, every SURVIVOR
# line verbatim, and the provenance (binary commit, table digest), which makes any shard
# re-derivable and checkable later at the cost of re-running it. The raw stream lives on
# disk only transiently, one shard at a time.
#
# RESTART. Idempotent: a shard with a meta file containing a SUMMARY line is skipped, so
# rerunning this script -- by hand, or from the @reboot cron that tools/the sync driver's launch
# instructions install -- resumes where it stopped. flock guards double-starts. A shard
# that fails 3 attempts is recorded as FAILED and skipped, so one bad shard cannot stall
# the other 249; failures are listed at the end for a manual re-run.
#
# EXIT CODES from sieve_cuda: 0 = clean, 2 = SURVIVORS PRESENT -- a legitimate result
# and the single most important thing this run could produce; recorded loudly, never
# retried (it is deterministic), never treated as a crash. Anything else is a crash and
# retries.
set -u
# Production values; a smoke test can override all three (plus OUT) positionally:
#   tools/run19.sh [LO HI N [OUTDIR]]
LO=${1:-255255}
HI=${2:-10000000000000000000}
N=${3:-250}
here="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
TABLE="$here/tables/filters/f_M2042040_mmax2000_gpu.txt"
BIN="$here/sieve/cuda/sieve_cuda"
OUT="${4:-$HOME/run19}"
THREADS=12

mkdir -p "$OUT"
exec 9>"$OUT/.lock"
flock -n 9 || { echo "run19: already running"; exit 0; }

[[ -x "$BIN" ]] || { echo "run19: FATAL sieve_cuda missing" | tee -a "$OUT/run.log"; exit 1; }
# Refuse to start on missing config rather than burning every shard's retry budget on
# it -- the smoke test did exactly that when the table was absent from the mirror.
[[ -r "$TABLE" ]] || { echo "run19: FATAL filter table missing: $TABLE" | tee -a "$OUT/run.log"; exit 1; }
nvidia-smi -L >/dev/null 2>&1 || { echo "run19: FATAL no GPU visible" | tee -a "$OUT/run.log"; exit 1; }
TSHA=$(sha256sum "$TABLE" | cut -d' ' -f1)
GITC=$(cat "$here/.git-commit" 2>/dev/null || echo unknown)

log() { echo "[$(date -u '+%F %T')] $*" >> "$OUT/run.log"; }
log "start: [$LO,$HI) x$N table=$TSHA commit=$GITC"

t_work_sum=0; n_done_now=0
for ((i = 0; i < N; ++i)); do
  meta="$OUT/shard_$i.meta"
  grep -q '^SUMMARY ' "$meta" 2>/dev/null && continue
  [[ -f "$OUT/shard_$i.FAILED" ]] && continue

  ok=0
  for attempt in 1 2 3; do
    raw="$OUT/shard_$i.raw"; err="$OUT/shard_$i.err"
    t0=$(date +%s)
    "$BIN" "$LO" "$HI" --shard "$i/$N" --filters "$TABLE" --threads "$THREADS" \
      > "$raw" 2> "$err"
    rc=$?
    t1=$(date +%s); wall=$((t1 - t0))
    if [[ $rc -eq 0 || $rc -eq 2 ]]; then
      # Completion is judged by CONTENT, not just rc: the SUMMARY line is printed last,
      # so its presence proves the scan ran to the end and the stream above it is whole.
      if ! grep -q '^SUMMARY ' "$raw"; then
        log "shard $i attempt $attempt: rc=$rc but no SUMMARY -- treating as crash"
      else
        {
          echo "# shard $i/$N  rc=$rc  wall=${wall}s  attempt=$attempt  $(date -u '+%F %T')"
          echo "# binary_commit=$GITC table_sha256=$TSHA threads=$THREADS"
          echo "certs=$(grep -c '^RUNG \|^SOLVED ' "$raw")"
          echo "cert_sha256=$(grep -v '^SUMMARY ' "$raw" | sha256sum | cut -d' ' -f1)"
          grep '^SURVIVOR ' "$raw"          # verbatim: the alarm, if it ever rings
          grep '^SUMMARY ' "$raw"           # last: its presence marks the meta complete
        } > "$meta.tmp" && mv "$meta.tmp" "$meta"
        rm -f "$raw"
        [[ $rc -eq 2 ]] && log "shard $i: *** SURVIVORS PRESENT *** see $meta"
        t_work_sum=$((t_work_sum + wall)); n_done_now=$((n_done_now + 1))
        left=0; for ((j = i + 1; j < N; ++j)); do
          grep -q '^SUMMARY ' "$OUT/shard_$j.meta" 2>/dev/null || left=$((left + 1)); done
        eta="?"
        [[ $n_done_now -gt 0 ]] && eta="$(( left * (t_work_sum / n_done_now) / 3600 ))h"
        log "shard $i done rc=$rc wall=${wall}s $(grep '^certs=' "$meta") left=$left eta~$eta"
        ok=1
      fi
    else
      log "shard $i attempt $attempt: rc=$rc -- $(tail -c 200 "$err" | tr '\n' ' ')"
    fi
    [[ $ok -eq 1 ]] && break
    sleep 30
  done
  if [[ $ok -eq 0 ]]; then
    date -u > "$OUT/shard_$i.FAILED"
    log "shard $i FAILED after 3 attempts -- skipping, re-run manually"
  fi
done

nfail=$(ls "$OUT"/shard_*.FAILED 2>/dev/null | wc -l)
ndone=$(grep -lc '^SUMMARY ' "$OUT"/shard_*.meta 2>/dev/null | wc -l)
log "pass complete: $ndone/$N shards done, $nfail FAILED"
if [[ $nfail -eq 0 && $ndone -eq $N ]]; then
  log "ALL SHARDS COMPLETE -- shutting down in 10 minutes (metas persist on disk)"
  # The laptop is expected to be closed for most of this run; an unattended finish must
  # not idle-bill. A guest shutdown stops the instance; the disk -- and every meta --
  # persists, and `tools/the sync driver up` brings it back to collect them.
  sudo shutdown -h +10 2>/dev/null || true
fi
