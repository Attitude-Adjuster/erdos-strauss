# How to run the 10¹³ census

Everything below assumes the delivered kit in one directory: `rung_scan.cpp`,
`Makefile`, `run8.sh`, `merge.py`, `verify_cert.py`.

## The short answer

```bash
make && make check                       # must print SELF-TEST PASSED (both modes)

HARD840=1 NSHARDS=64 PROGRESS=5 DEEP=71 \
  ./run8.sh 0 10000000000000 64 c13
```

**~53 min on 8 cores.** Only four things change from the defaults, and only two of them
matter.

## What changes, and why

| setting | default | for 10¹³ | reason |
|---|---|---|---|
| `HARD840` | 0 | **1** | The population the hypotheses are about. Exactly 4× cheaper, and it loses nothing — every non-Jacobi failure and every deep ladder found so far lies in the hard class. |
| `NSHARDS` | 8 | **64** | More shards than cores. `JOBS=8` keeps 8 in flight and a core that finishes early picks up the next, so no core idles at the tail. Also gives 8× finer resume granularity. |
| `PROGRESS` | 25 | **5** | 2,838 windows total ÷ 64 shards ≈ 44 per shard, so a 25-window heartbeat would print twice. At 5 you get ~9 updates per shard. |
| `DEEP` | 91 | **71** | Collects the whole deep tail (~7,000 lines, trivial) instead of just the extreme tip. H3 wants the tail as a population, not a single record. |

Everything else stays: `SPANLOG=22`, `RMAX=255`, `JOBS=8`, `THREADS=1`.

### Why *not* to change the others

- **`SPANLOG=22`** is right here. At 10¹³ the per-window fixed cost is ~23 ns/prime at
  spanlog 22 versus ~91 at 20 — both negligible against a ~2,300 ns baseline — while
  memory goes 29 MB → 116 MB → 466 MB per process at 20/22/24. 22 is the comfortable
  middle at ~0.9 GB across 8 processes. Since the window now scales with the wheel
  modulus, **do not raise it to compensate for `--hard840`**; that is handled.
- **`THREADS=1`** — the shared prime tables at 10¹³ are ~5 MB. Duplicating them across
  8 processes costs nothing, and processes give crash isolation and clean resume.
- **`RMAX=255`** — the deepest ladder known is 107. Escalations are counted and
  reported, never silently dropped, so if something exceeds 255 you will see it.

### The one real decision: `EMIT_RESIDUAL`

Left at 0, output is a few MB (deep primes, the rare non-Jacobi certificates, per-shard
summaries). That is enough for the headline numbers — the level counts and the depth
histogram are in the `SUMMARY` lines.

Set to 1 and you get **~110 million RESIDUAL lines, ≈ 3.3 GB**. Worth it only if you
want the per-miss dataset (τ(q²), ω(A) distributions — the divisor-poverty analysis).
If you do want that, consider taking it from a sub-range instead:

```bash
HARD840=1 EMIT_RESIDUAL=1 ./run8.sh 9000000000000 10000000000000 32 c13_top
```

which gives the same distributional information at the top decade for a tenth of the
disk.

## Before committing 53 minutes: a 2-minute pilot

Measure `ns_per_prime` on *your* hardware rather than trusting my extrapolation from a
2-core VM:

```bash
OMP_NUM_THREADS=1 ./rung_scan 9999000000000 10000000000000 \
    --hard840 --rmax 255 --spanlog 22 | grep SUMMARY
```

Take `ns_per_prime` from that line and multiply by 1.08 × 10¹⁰ (the number of hard-class
primes below 10¹³) to get core-seconds. If it disagrees with ~2,300 ns by more than
about 30%, re-time `--spanlog 20` and `24` before the full run — the optimum tracks
L2/L3 size and mine is probably not yours.

## What you get, and what to look at

`merge.py` runs automatically and writes `c13/MERGED.txt` and `c13/interesting.txt`.
The four things worth checking immediately:

1. **Level S** — is it still *only* at r = 51? That is 47/47 through 10¹¹ with no
   exception; 10¹³ is where that either becomes hard to dismiss or breaks.
2. **The residual share** — J / RC / S / R in `MERGED.txt`. This is the hard-class
   share, which is the one H1 is actually about, and it should be roughly double the
   (1 mod 24) numbers in the paper. The sequence to extend is 2.34% → 1.05% → 0.71% →
   0.60% (those are the *diluted* values; the hard-class series starts at 4.61%).
3. **Max depth** — 107 at p = 8,803,369 has stood since 8.8 × 10⁶. Two more decades is
   a real test of it.
4. **Escalations** — must be 0. Anything else means a ladder exceeded r = 255 and needs
   investigating before the number is trusted.

Then verify the interesting claims in exact arithmetic, ideally on another machine:

```bash
python3 verify_cert.py c13/interesting.txt
```

## If it is interrupted

Re-run the identical command. Shards with a complete `SUMMARY` line are skipped; only
the unfinished ones restart. And `make` during a run is safe — the harness executes a
frozen copy of the binary.
