# Runbook — Erdős–Straus certificate census on 8 cores

Five files: `rung_scan.cpp` (scanner), `Makefile`, `run8.sh` (shard harness),
`merge.py` (merge + validate), `verify_cert.py` (exact-arithmetic spot checks).

## Start here

```bash
make            # builds with -O3 -march=native -fopenmp
make check      # MUST print "SELF-TEST PASSED" before you trust anything
```

`make check` recomputes the entire *p* < 10⁵ census and compares all four certificate
counts, the full depth histogram, the maximum depth, and the exact list of the 20
residual failures against an independent Python/SymPy reference. It guards against
miscompilation and unsafe flags, not just logic bugs — so **run it on every machine,
compiler, and flag set**, not just once.

Then:

```bash
./run8.sh 0 1000000000000            # 0 → 10¹², 8 shards, into ./census_out
```

On 8 cores that is roughly 25 minutes. When it finishes it prints the merged census
and writes `census_out/MERGED.txt` and `census_out/interesting.txt`.

## Parallelism: two independent axes

The scanner is **both** multithreaded and multi-process, and you pick the split:

| | mechanism | knob |
|---|---|---|
| threads | OpenMP over windows, `schedule(dynamic,1)`, per-thread stats merged once at the end | `THREADS` |
| processes | independent contiguous shards, `--shard i/n`, sharing nothing | `JOBS` |

Cores used = `JOBS × THREADS`. **Default is `JOBS=8 THREADS=1`** — 8 single-threaded
processes.

That default is deliberate, and it holds further than I first claimed. Processes win
almost everywhere: shards are independent and individually recomputable, a crash costs
one shard instead of the run, and there is no lock contention at all. The only reason to
switch is duplicated memory for the *shared prime tables*, and those stay small for a
long time — π(√X), not √X:

| height | shared tables | ×8 processes |
|---|---|---|
| 10¹² | 2 MB | 16 MB |
| 10¹⁵ | 45 MB | 0.4 GB |
| 10¹⁸ | 1.2 GB | 9.9 GB |

So the threads-vs-processes crossover is around **10¹⁸**, not 10¹⁵. Below that, stay
with processes. (An earlier version of this file said 10¹⁵; it had confused √X with
π(√X).) On two sockets a hybrid can still help for locality:

```bash
JOBS=4 THREADS=2 ./run8.sh 0 1000000000000     # 4 processes x 2 threads
NUMA=1 JOBS=8 ./run8.sh 0 10000000000000 16    # pin processes across NUMA nodes
```

`NUMA=1` pins each process to a node with `numactl --cpunodebind/--preferred` (silently
skipped if `numactl` isn't installed), which matters on multi-socket boxes because the
sieve is memory-bandwidth-bound.

### Multiple machines

Set `NSHARDS` to the **global** shard count and give each box its index. Box *k* runs
shards *k, k+MACHINES, k+2·MACHINES, …*:

```bash
# on box 0..3 of a 4-machine cluster, 32 global shards
MACHINES=4 MACHINE=0 ./run8.sh 0 100000000000000 32 out
MACHINES=4 MACHINE=1 ./run8.sh 0 100000000000000 32 out
```

Shards share nothing and need no coordination. Collect all `shard_*.out` onto one host
and run `merge.py` over the union — it will refuse to report if the shards don't tile
the range exactly.

## What else the harness does

- Shards are independent and individually recomputable, so a crash costs one shard,
  not the run.
- **Freezes the binary.** It copies `rung_scan` into the output directory and runs the
  copy, so `make` during a run can't kill it. (The linker rewrites the binary in place;
  any process running the old image dies. This cost me a 40-minute run.)
- **Resumable.** A shard whose `.out` already ends in a `SUMMARY` line is skipped —
  re-run the identical command after an interruption and it continues.
- **Validates on merge.** `merge.py` checks that shards tile the range with no gap or
  overlap, that each shard's histogram reconciles with its own counters, that no shard
  escalated (hit `--rmax` without finding a rung), and that the unreachable pair-DP
  path was never taken.

## The 4× win: `--hard840`

By Mordell, 4/n is solvable for every n **except** possibly n a square mod 840. There
are exactly six such classes — {1, 121, 169, 289, 361, 529} — and all six lie inside
(1 mod 24), which is why that coarser class is the usual statement. But (1 mod 24)
contains **24** reduced classes mod 840, so sieving all of it processes **4× more
primes than the hard class actually needs**.

```bash
./run8.sh 0 1000000000000        # default: 1 mod 24
HARD840=1 ./run8.sh 0 1000000000000   # 6 square classes mod 840, ~4.3x faster
```

The restriction is empirically free. Over p < 10⁵ it keeps:

| | 1 mod 24 | squares mod 840 |
|---|---|---|
| primes examined | 1,181 | **273** (4.33× fewer) |
| depth ≥ 11 counts | 83, 11, 16, 15, 1, 5 | **identical** |
| max depth | 31 at p = 21169 | **identical** |
| level-R failures | 20 | **the same 20** |
| depth 3 / 7 | 575 / 475 | 87 / 55 |

It drops only primes that hit at rung 3 or 7 — exactly the ones a classical identity
already resolves. Every level-S prime and every deep-ladder prime found below 10¹⁰ is a
square mod 840, so nothing scientifically interesting is lost.

**But it changes the denominator.** The residual share among failed rungs is 20/854 =
2.34% on (1 mod 24) and 20/434 = 4.61% on the square classes. Shares from the two
populations are *not* comparable, and H1 has to be stated for one or the other. Arguably
the square classes are the more meaningful population — including the identity-resolved
primes inflates the level-J count with rungs that were never in doubt. Pick one and say
which.

`--verify` tests **both** modes and has a separate reference for each.

**`--spanlog` means positions per residue class**, and the window width scales with the
wheel modulus (24 or 840) so the same value means the same thing in both modes. This
matters: the scanner walks the factor-prime list once *per class*, so `--hard840` pays
that fixed cost 6× over 4× fewer primes. Without the scaling, at 10¹⁵ that overhead
alone would be ~7.9 µs/prime against a ~4.3 µs baseline — it would have **more than
cancelled** the 4× win. With it, the overhead is ~0.2 µs/prime and the 4× is real. Keep
`SPANLOG` the same when you switch modes; don't compensate by hand.

## Knobs

| variable | default | when to change |
|---|---|---|
| `SPANLOG` | 22 | 20 below ~10¹⁴; 22 above. Bigger windows amortize the per-window prime loop but lose cache locality — re-tune on your hardware, the optimum tracks L2/L3. |
| `RMAX` | 255 | Leave it. Deepest ladder known is 107; escalations are counted, never silently dropped. |
| `DEEP` | 91 | Lower to collect more of the deep tail. |
| `EMIT_RESIDUAL` | 0 | Set to 1 for the full residual dataset. **At 10¹² that is ~24 million lines**, so budget disk. |
| `PROGRESS` | 25 | Heartbeat interval in windows; `tail -f census_out/shard_0.log`. |
| `HARD840` | 0 | 1 restricts to the 6 square classes mod 840 — 4.3× faster, different population (see above). |

```bash
SPANLOG=20 EMIT_RESIDUAL=1 ./run8.sh 0 100000000000 8 c11
```

## Budget on 8 cores

Measured single-core ns/prime through 10¹⁵; the rest extrapolated.

Two populations. `--hard840` is exactly 4× cheaper (6 of the 24 reduced classes mod 840
inside 1 mod 24) and is the one the hypotheses are about.

| target | \(1 \bmod 24\): primes / 8 cores | **hard class**: primes / 8 cores / 48 cores |
|---|---|---|
| 10¹² | 4.7 × 10⁹ · 25 min | **1.2 × 10⁹ · ~6 min · ~1 min** |
| 10¹³ | 4.3 × 10¹⁰ · 3.5 h | **1.1 × 10¹⁰ · ~53 min · ~9 min** |
| 10¹⁴ | 4.0 × 10¹¹ · 1.9 d | **1.0 × 10¹¹ · ~11 h · ~1.9 h** |
| 10¹⁵ | 3.7 × 10¹² · 23 d | **9.3 × 10¹¹ · ~5.8 d · ~1.2 d** |

10¹⁶ and beyond needs a bucket sieve first (see `SCALING.md` §3.1) — past that point
the per-window loop over the prime list dominates, and no amount of cores fixes it.

## Verifying results

The merge checks internal consistency. For external verification, re-derive the
interesting claims in exact arithmetic — ideally on a different machine:

```bash
python3 verify_cert.py census_out/interesting.txt
```

This re-does each claim by explicit divisor enumeration with Python bignums (a
different algorithm from the scanner's residue DP), and prints an actual decomposition
4/p = 1/A + 1/B + 1/C, checking 4ABC = p(AB+BC+CA) over the integers.

For a publishable census also: run ~1% of shards twice on different hardware and diff
the `SUMMARY` lines, and publish the per-shard summaries rather than only the merged
total, so anyone can reproduce a single line.

## What changed for production

- **Pollard–Brent cycle detection fixed.** The old guard masked the `x == y` test, so a
  degenerate parameter ran to the internal iteration cap. On one census value,
  A = 9,997,728,493 = 83,987 × 119,039: 268 million iterations / 11 s, versus 1,700
  iterations / 0.1 ms after the fix.
- **Bulk sieve-factorization.** A = (p+r)/4 = A₀ + 6j is an arithmetic progression, so
  ℓ | A is one too. One strided walk per ℓ replaces ~6,500 trial divisions per A. Since
  the walk uses every prime to √A_max, the leftover cofactor is prime by construction —
  no primality test and no Pollard on the main path at all. 7–20× faster, and the
  per-prime cost is now nearly flat in height.
- **Occupancy instrumentation gated** behind `--occupancy`; it was costing 43%.
- **Emitted lines streamed per window** instead of accumulated — at 10¹² the residual
  list alone would have been ~600 MB of RAM. Output is consequently unsorted;
  `merge.py` sorts it.
- **`Fac` packed to 16 slots** (14 is the true maximum below 10¹⁸) with a loud abort on
  overflow instead of silently dropping a factor.
- Startup prints a per-thread memory estimate on stderr, so a bad `SPANLOG` fails
  visibly before it fails by swapping.
- **Fixed a data race**: `g_pairpath` (the diagnostic counter for the provably
  unreachable pair-DP path) was incremented from inside the OpenMP parallel region
  without synchronization. Now atomic. It only ever mattered in `THREADS>1` mode, and
  only for a counter that should read zero — but a race that corrupts your only
  tripwire is exactly the wrong race to have.
