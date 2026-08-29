# DESIGN — `rung_scan3`: a self-contained certified verification sieve

Status: **design, approved**. Successor spec to `DESIGN_cover_scan.md`, which stays
valid for the mathematics of the Type II family and is superseded on scope, tuning
defaults, and file plan.

Every number in this document was measured or proved in the session that produced it.
Nothing is projected from another platform's scaling law.

---

## 0. Scope, product, and non-goals

`rung_scan3` is a **verification** scanner. Its product is the claim

> for every prime `p ∈ [lo, hi)`, `4/p = 1/x + 1/y + 1/z` has a solution in positive
> integers — with every layer of the argument reproducible from published, checksummed
> artifacts and nothing taken on authority.

It is **not** a census. `rung_scan.cpp` remains the frozen reference and the census's
J/RC/S/R statistics, depth histograms, and certificate levels are untouched. Removing
primes by constructive identity before classification would change those statistics;
the two pipelines stay separate, exactly as `DESIGN_cover_scan.md` §0 requires.

The two pipelines cannot substitute for one another, and it is worth stating why in the
same language. In the Type II identity `p + k = 4ade`, put `A = ade`: then `4A = p + k`,
so **`k` is a rung** — and `k ≡ 3 (mod 4)` follows from `m ≡ 3 (mod 4)` and
`p ≡ 1 (mod 4)`, so covers land on legal rungs. But `k = (p + 4a²d)/m ≈ p/m`, so a
cover with `m ≤ 10⁴` certifies a hit at rung `~10⁹`, nine orders of magnitude above the
census's `RCAP = 255`. `rung_scan` enumerates by small `r` (huge `m`, found by factoring
`A`); `rung_scan3` enumerates by small `m` (huge `r`, found by progression). Same family,
opposite ends. This is why covers make verification ~10⁵× cheaper and do nothing at all
for the census.

Cost scales with **range**, not prime count. The scanner never enumerates primes.

Population: all primes. The reduction to the 220 hard lanes mod 120120 is derived by
the tool itself (§2), not cited.

### Staged goal

1. **P0–P4, CPU:** `[0, 10¹⁷)` on `cloud-scan`, as proof of concept and as a complete,
   publishable artifact.
2. **P5, CUDA:** port for the record, past 10¹⁸. Every data structure and stage boundary
   in this design is chosen to be GPU-shaped from the start: window-independent state,
   lane-parallel with no shared mutable state, fixed-size per-lane workspaces, survivor
   handoff as a compacted list rather than a branchy tail.

---

## 1. The claim, in four checkable layers

| layer | certifies | cost | artifact |
|---|---|---|---|
| **L0** composite → prime | if `d \| n` and `4/d` is solvable then `4/n` is | argument | this spec; `verify_covers.py --sweep` separately bounds the family's non-square exceptions `{288, 336, 4545}` below 10⁵, all composite |
| **L1** class reduction | every unit class mod 120120 except **220** is solvable outright | one-time | `class_table_120120.txt` (220 classes + 22,820 kill witnesses) |
| **L2** covering progressions | almost every `p` in those 220 lanes | the sieve | pruned filter table + digest |
| **L3** survivors | the handful L2 misses, individually | rare | `RUNG` / `SOLVED` lines |

`φ(120120) = 23,040`, of which 220 survive; the other 22,820 each get a named killing
certificate, so L1 is a finite object a reader checks line by line rather than a
generator a reader must trust.

---

## 2. Three certificate families

### 2.1 F1 — Type II covers (exists; `cover_scan.cpp`, `verify_covers.py`)

For `a, c, d, k ≥ 1` with `ck > a`, put `m = 4acd − 1`, `e = ck − a`,
`p = mk − 4a²d`. Then `p + k = 4ade` and

```
  4/p = 1/(ade) + 1/(acdp) + 1/(cdep)
```

so **every** integer `p ≡ −4a²d (mod m)` with `p ≥ pmin = m(⌊a/c⌋+1) − 4a²d` is solved.
Self-certifying: the progression is the certificate, no per-`p` witness exists or is
needed.

**F1 is complete, and `m ≡ 3 (mod 4)` is forced.** Reparameterize by `N = acd`,
`α = c²d`. The general Type II construction is: choose `n₁, n₂, k` with
`A = k·n₁n₂/(n₁+n₂)` integral and `p = 4A − k`. Taking `n₁ = N` fixed and `n₂` linear in
`k` (the only shape that yields a progression in `p` rather than a progression of
multiples) gives `n₂ = αk − N`, hence

```
  A = Nk − N²/α ,      p = k(4N − 1) − 4N²/α ,      m = 4N − 1 ,
  res ≡ −4N²/α (mod m) ,   valid for αk > N ,   any α | N² .
```

Every `α | N²` is reachable by some `(a, c, d)`: per prime `q`, with `n = v_q(N)` and
`s = v_q(α) ≤ 2n`, take `x = max(0, s−n)` and `y = s − 2x`; then `2x + y = s` and
`x + y ≤ n`, so `c = q^x`, `d = q^y`, `a = N/(cd)` are integral. Verified
computationally: the `(N, α|N²)` enumeration and the `(a,c,d)` generator produce
**identical** `(m, res)` sets *with identical `pmin`* at `mmax = 200` (435 pairs) and
`mmax = 840` (2,961 pairs).

Consequence: `m = 4N − 1 ≡ 3 (mod 4)` always, so **F1 can never constrain a residue
mod 8**. F2 is mathematically necessary, not a convenience.

### 2.2 F2 — uniform rung certificates (new)

A rung `r` together with a fixed witness `u` such that *both* rung conditions —
`u | q²` and `u ≡ −q (mod r)`, where `A = (p+r)/4` and `q = pA` — are forced by `p`'s
residue class alone. The solution is then the repo's standard triple

```
  x = A ,   y = (q + u)/r ,   z = q(q + u)/(r u) .
```

Two instances close the gap:

| class | `r` | `u` | note |
|---|---|---|---|
| `p ≡ 3 (mod 4)` | 1 | 1 | `r = 1` makes the congruence vacuous |
| `p ≡ 13 (mod 24)` | 3 | 2 | `A` even forces `2 \| q²`; `p ≡ 1 (mod 3)` forces `q ≡ 1 (mod 3)` |

The second is exactly the classical `4/13 = 1/4 + 1/18 + 1/468`.

**Measured this session:**

```
F1 alone, all unit classes mod 120120        : 920 survivors
  + F2 (4, 3)   r=1 u=1                      : 460
  + F2 (24, 13) r=3 u=2                      : 230   ← all ≡ 1 (mod 24)
  + the rest of the search to r ≤ 63, u ≤ 64 : 220   ← 6 by (r=11,u=13), 4 by (r=11,u=39)
```

and 58,333 instances of the two F2 certificates re-checked in exact rationals with
**0 failures**. The 920 survivors of F1 alone split as exactly 230 in each of the
classes 1, 7, 13, 19 mod 24 — a 4-fold symmetry that is a direct consequence of
`m ≡ 3 (mod 4)`.

F2 **marks nothing at runtime**. It is a class-level argument consumed once, when the
wheel is built. The scan walks 220 lanes; the reduction costs zero scan work.

**The class count is a parameter, not a constant.** 230 is what the two certificates
above give on their own; carrying the whole search to `r ≤ 63, u ≤ 64` kills ten more
classes and lands on **220**, each kill verified on real primes in exact rationals. The
lane count is therefore a function of `(rmax, umax)` exactly as the filter table is of
`mmax`, and all three are published together. Since F2 costs nothing at runtime, a
smaller lane count is a permanent proportional saving: 220 vs 230 is 4.3% off every
scan. Measured ladder at M = 120120: 224 lanes at `r ≤ 15`, 224 at `r ≤ 31`, 220 at
`r ≤ 47` and at `r ≤ 63`.

**Only certificates with `L | M` belong to the wheel.** Of the 5,807 generated at the
default bounds, 390 have `L | M` and are spent here; the other 5,417 kill a
sub-progression *inside* a lane, which is stage B's job — they are extra dynamic covers,
not class kills, and applying one to the wheel would be a soundness bug.

The generator **searches** small `(r, u)` mechanically and reports which classes each
pair certifies, rather than hard-coding the two above. The search is the artifact; the
two instances are its output. Mordell's reduction to
`{1, 121, 169, 289, 361, 529} (mod 840)` becomes a cross-check we reproduce, not a
dependency.

### 2.3 F3 — per-`p` certificates

- **Stage D, rung walk.** `A = (p+r)/4`, factor `A`, divisor-class DP for `u | q²` with
  `u ≡ −q (mod r)`; emit `RUNG p A r u`. Code is reused from the frozen reference via
  the sanctioned include trick (`#define main rung_scan_main` / `#include
  "../rung_scan.cpp"` / `#undef main`). One piece is genuinely new: `test_rung` returns
  a `Level` and discards the witness `u`, so a witness-extracting variant is required.
  It is small, and every line it emits is re-derived independently by
  `verify_covers.py`, so the new code is checked per use, not per review.
- **Stage E, direct parametric solver.** For `a, d` in a small box, factor `p + 4a²d`,
  scan divisors `m ≡ −1 (mod 4ad)` with `ck > a`; emit `SOLVED p a c d k`.
- Anything still standing: `SURVIVOR p`, nonzero exit.

### 2.4 Soundness is monotone in the filter set

Dropping any filter can only leave more survivors; it can never manufacture a false
certificate. Every survivor is then resolved individually by stages C/D/E or reported.
This is what makes the aggressive pruning of §4 safe: **the worst case of a pruning bug
is a slower run, not a wrong claim.**

---

## 3. Engine: window ≠ segment, and offsets that never need recomputing

The skeleton measures ~10 ns per mark, which is one cache miss per mark. Two structural
changes remove it.

**3.1 Split the amortization unit from the locality unit.** The skeleton uses one size
for both.

- **Window** `SPAN = M << spanlog`, `spanlog ≈ 24` — how often fixed costs are paid.
- **Segment** `SEG = 2¹⁸` lane positions = a **32 KB** bitset — cache residency.

`rung_scan2` learned the same lesson from the other side (`spanlog 22` for L3
residency); the covers' binding constraint is L1 and it is a different number. Tune the
two independently.

**3.2 Index by global position, not by window offset.** Let `J = (p − ρ)/M` be the
lane-`ρ` global position index. Cover `i` hits lane `ρ` exactly at
`J ≡ j0ᵢ (mod stepᵢ)`, `stepᵢ = mᵢ / gcd(M, mᵢ)` — a pattern fixed for the entire scan.
So the extended-Euclid setup runs **once per (lane, cover) per shard**, not once per
window, and offsets carry forward across windows unchanged. Per segment a cover costs
one subtract, plus `SEG/stepᵢ` bit-clears when `stepᵢ < SEG`.

`inv_coprime` must stay extended Euclid — `step` is generally composite, so a Fermat
inverse is invalid. The inverse always exists: per prime the minimum exponent goes
wholly into `g = gcd(M, m)`, so `M/g` and `m/g` are coprime, which is why `rhs % g == 0`
is the only solvability test.

> **Measured 2026-08-14, after building it.** §3.1 was right and §3.2 was not.
> The segment prediction holds exactly — `2¹⁸`, a 32 KB bitset, is the measured optimum.
> But `cover_scan` turns out to be **flat across `--spanlog` 20/22/24**, so per-window
> setup was never a bottleneck and the global-`J` amortization of §3.2 buys nothing
> measurable. The engine's whole 1.33× is cache locality. Keep the global-`J` indexing —
> it is simpler, it is what makes lane-major parallelism and shard-resume clean, and it
> is the right shape for the GPU — but do not budget any speed to it. See §6.

**3.3 The offset state is a cache, not a dependency.** It is recomputable by egcd at any
boundary from `J` alone. Shard-resume and window-independence survive intact — the exact
property RUNBOOK.md flags that bucket sieves destroy.

**3.4 Per-lane working set:** 32 KB bitset + `4·ncov` bytes of offsets (~20 KB at 5,000
pruned covers). Both L1-resident, and both sized for a CUDA block's shared memory.

---

## 4. Filter selection: pruning by unique kill

Cost of keeping cover `i`, per lane position:

```
  Cᵢ  =  1/stepᵢ        (bit-clears)
      +  1/SEG           (per-segment bookkeeping, paid whether it hits or not)
```

The second term is why `mmax = 50,000` measured slow: 448,204 covers against
`SEG = 2¹⁸` is **1.7 bookkeeping ops per position**, more than the marking itself. It
also says large-`m` covers are cheap but never free, and that mid-size redundant covers
are the real waste.

Redundancy is large and non-uniform. At `mmax = 840`: 11.4 marks per position, measured
survivor density `3.0·10⁻⁴`. Independent marks at that rate would give `1.1·10⁻⁵`, so
the covers are ~27× correlated — most marks land on positions another cover already
killed, and a position only needs killing once.

**Procedure** (deterministic, reproducible, published):

1. Evaluate on a fixed reference domain: all 220 lanes, `J ∈ [0, 2²²)`. Height-
   independent, because every `pmin` sits far below any legal scan floor, so a cover's
   hit pattern on a lane depends only on `(ρ mod m, J mod step)`.
2. Build a saturating kill-count array plus a single-killer-id array; count each cover's
   **unique** kills `Uᵢ`.
3. Drop covers whose `Uᵢ` fails to pay for `Cᵢ` in a common currency of nanoseconds,
   using the measured per-mark and per-survivor costs.
4. Recompute (uniqueness changes when covers are dropped) and iterate to a fixed point.

**Two hard constraints ride along:**

- survivor density ≤ `3·10⁻⁴` of lane positions — keeps stage C under ~5% of runtime;
- **prime**-survivor density ≤ `10⁻¹¹` per integer of range. This is a *publication*
  constraint, not a speed one: at `1.75·10⁻⁹` (what `mmax = 2000` measured) a 10¹⁷ run
  emits `1.75·10⁸` certificates, ~8 GB. At `10⁻¹¹` it emits ≤ 10⁶, ~40 MB. If the
  constraint proves unreachable, fall back to per-shard gzip plus a published digest.

Output: the surviving `(m, res, a, c, d, pmin)` list, its SHA-256, and the pruning
parameters `(M, mmax, SEG, reference domain, prune_threshold)`. The binary regenerates the
table at startup, compares to the checked-in digest, and **refuses to run on mismatch** —
the same trust pattern as `make check`.

`DESIGN_cover_scan.md`'s default `mmax = 10⁴` was chosen when stages D and E were stubs,
so any prime survivor was a run failure. With stage D implemented, the trade reopens:
the measured 21 prime survivors at `mmax = 2000` cost a rung walk each (~40 µs) against
the 2.2 s that `mmax = 10⁴` spends to remove them. `mmax` becomes an output of the
pruning cost model, not a hand-set default.

---

## 5. Survivor pipeline and parallelism

Per segment: word-scan the bitset with `ctz` into a **compacted survivor list** (the
GPU-shaped handoff), then

```
  base-2 SPRP  →  12-base MR  →  stage D rung walk  →  stage E direct  →  SURVIVOR
```

The base-2 pre-filter matters because nearly every survivor is composite — at
`mmax = 10⁴` every composite survivor measured was a perfect square — and it costs 1/12
of the full 12-base test. A segmented prime sieve loses at these densities (measured in
`DESIGN_cover_scan.md` §2.C); revisit only if pruning pushes survivor density up by
orders of magnitude.

OpenMP over windows, `--shard i/n` on the `run8.sh` model, per-thread `Stats` merged at
the end, `--progress`, streamed emission per window. Determinism by construction:
emission is keyed `(shard, window, lane, position)` and merged in that order, so a
parallel run is byte-identical to a serial one — the contract `diff2.sh` and
`diff_range.sh` already enforce elsewhere in the repo.

---

## 6. Cost model — MEASURED, superseding the original estimate

**This section was rewritten on 2026-08-14 after the engine was built and measured. The
original version was wrong in a way worth recording, because it set a false expectation
that shaped a whole plan.**

Per integer of range: `(marks/pos + ncov/SEG) × ns_per_mark / 546.0`, where
`546.0 = 120120/220` integers per lane position, plus the survivor term.

### What the original estimate got wrong

It inferred **10.4 ns/mark** for the skeleton, by dividing `DESIGN_cover_scan.md`'s
"0.5 ns per integer of range" by ~25 marks/pos, and concluded that ~8× was sitting in
the marking loop waiting for cache-resident segments. That inference was invalid: the
0.5 ns/integer figure was measured at `mmax = 10⁴` over a 1.2·10¹⁰ range, a regime where
**per-lane cover setup dominates and marking barely features**. It was a setup
measurement being read as a marking measurement.

Measured directly on `cloud-dev` (`c4-highcpu-8`), `[10¹², 2·10¹²)`, `mmax = 2000`,
single-threaded, net of 0.7 s startup:

| | ns/mark | note |
|---|---|---|
| `cover_scan` (reference) | 0.68 | flat across `--spanlog` 20/22/24 — 17.83/17.70/17.56 s |
| `rung_scan3` (segmented) | **0.51** | optimum at `--seg 262144`; 14.25/13.35/16.14/16.67 s at 64K/256K/1M/4M |

Two conclusions, both of which change how the remaining work should be planned:

- **The reference was already near-optimal at marking.** A strided bit-clear is a
  dependent load-modify-store; 0.5–0.7 ns is close to the floor. The engine rewrite buys
  **1.33×**, not 8×.
- **Setup amortization buys nothing.** `cover_scan` is flat in `--spanlog`, so the
  "extended Euclid once per lane per shard instead of once per window" argument of §3.2
  — the centrepiece of the original design — is not where time goes. The entire 1.33× is
  cache locality from the segment. §3.1's segment prediction was right (`2¹⁸`, a 32 KB
  bitset, is the measured optimum); §3.2's amortization prediction was not.

### The corrected table

| | marks/pos | ns/integer | 10¹⁵ | 10¹⁶ | 10¹⁷ | 10¹⁸ |
|---|---|---|---|---|---|---|
| `cover_scan`, the reference | 13.59 | 0.0169 | 24 min | 3.9 h | 1.6 d | 16 d |
| `rung_scan3`, no pruning **(measured)** | 13.59 | **0.0127** | 18 min | 2.9 h | **1.2 d** | 12 d |
| `rung_scan3` + pruning (target) | ~3 | 0.0028 | 4 min | 39 min | **6.5 h** | 2.7 d |

Wall times at **12×**, from 12 physical cores. The measured row is a single-threaded
number on a 4-physical-core dev box, scaled by core count only — same `c4` family, so
per-core speed should carry, but it is a projection until run on the real box.

The original table predicted 0.026 ns/integer for this row and 0.008 with pruning. The
measured value is **better than the first and within 1.6× of the second, before any
pruning or threading** — the pessimism came entirely from the bad baseline. The goal of
§0 (10¹⁷ on CPU) is therefore comfortably in reach; what remains is threading (Task 4 of
the phase-2 plan) and pruning (Task 5), which together target the last ~4.5×.

For P5, the census CUDA port measured ~30× against the same-box CPU. That number is a
**projection to be measured on the box**, never assumed: the standing rule is that a CPU
scaling law must never be carried onto the port, and measuring takes ninety seconds. The
lesson of this section is the same rule applied one level down — do not carry a cost
model across a change of regime without re-measuring it.

---

## 7. Trust artifacts and gates

Published per run: pruned filter table + digest; `class_table_120120.txt` + digest; a
`SUMMARY` line carrying every parameter (`M, mmax, prune_threshold, SEG, spanlog, lo,
hi, shard`) and
all per-stage counters (wheel-killed, cover-killed, MR-composite, rung-solved,
direct-solved, survivors); and the `RUNG` / `SOLVED` / `SURVIVOR` certificate stream.

| gate | what it catches |
|---|---|
| `./cover_scan --verify` | identity, `wheel(840) = {1,121,169,289,361,529}` derived (24→12→6), 920→460→220 at M=120120, pinned class FNV, CRT vs brute force, survivors inside the hard class |
| `./rung_scan3 --verify` | the same constants, **plus** segmented engine vs naive marking on a random window, **plus** filter-table digest enforcement |
| `tests/diff3.sh lo hi` | byte-identical `SUMMARY` and byte-identical *sorted* certificate set, `rung_scan3` vs frozen `cover_scan` |
| `verify_covers.py` (+ `--uniform`, `--kill-witnesses`) | independent exact-arithmetic re-derivation of every table and every emitted line — **run on a different machine than produced it** |
| `make check` | all of the above on a small range, per machine and per flag set |

Free falsifiable checks, in the spirit of the level-S confinement:

- every `RUNG` line satisfies `r ≡ 3 (mod 4)` and `4A = p + r`;
- every surviving class reduces into `{1, 121, 169, 289, 361, 529} (mod 840)`;
- statistics are **bit-identical across `spanlog` and `SEG`** — windowing may not touch
  arithmetic;
- pruned vs unpruned on a shared range: survivor sets **nest**
  (`unpruned ⊆ pruned`) and the set of `p` ultimately certified is **identical**. This is
  what makes a pruning bug visible instead of merely slow.

The scanner refuses to start unless `lo` exceeds the maximum `pmin` of every filter in
use (15,014 for the default wheel). The region below is settled by the existing censuses
and by `verify_covers.py --sweep`.

---

## 8. File plan

| file | role |
|---|---|
| `cover_scan.cpp` | finished, then **frozen**: the naive, obviously-correct reference. Stages D/E reuse `rung_scan.cpp` via the include trick |
| `rung_scan3.cpp` | the optimized scanner: pruned filters, segmented marking, global-`J` offsets, OpenMP, `--shard` |
| `verify_covers.py` | extended with F2 search regeneration, kill-witness verification, pruned-table diff |
| `filters/` | published pruned filter tables + checksums |
| `class_table_120120.txt` | 220 classes + 22,820 kill witnesses; supersedes `wheel_120120_hard.txt` |
| `tests/diff3.sh` | the differential gate |
| `Makefile` | `rung_scan3` target; `make check` additions |
| `DESIGN_cover_scan.md` | updated for the reference/optimized split, the F1 completeness result, and the revised `mmax` guidance |

`cover_scan.cpp` and `rung_scan3.cpp` stand in exactly the relationship `rung_scan.cpp`
and `rung_scan2.cpp` do: a frozen reference whose worth is that it is simple and
unchanged, and an optimized implementation that must agree with it byte for byte. The
reference also gives the eventual CUDA port a CPU oracle.

---

## 9. Milestones

- **P0 — F2 and kill witnesses.** Mechanical `(r, u)` search, exact-arithmetic verifier,
  `class_table_120120.txt`. No scanner change. *Gate:* all 23,040 unit classes
  accounted for, exactly 220 survive, exact checks clean.
- **P1 — finish and freeze `cover_scan.cpp`.** Stage D with witness extraction, stage E,
  exit contract. *Gate:* `--verify`, a small-range run with zero survivors,
  `verify_covers.py` clean.
- **P2 — `rung_scan3` engine.** Global-`J` offsets, window/segment split, OpenMP,
  `--shard`, progress. *Gate:* `diff3.sh` byte-identical over
  `[10¹², 10¹² + 10¹⁰)`; `spanlog`/`SEG` invariance.
- **P3 — pruning.** Deterministic fixed point, published table, digest enforcement.
  *Gate:* nesting and identical certified set; measured marks/pos and both survivor
  densities inside budget.
- **P4 — measure and run.** Tuning sweep on `cloud-scan` (`SEG`, `spanlog`, `mmax`,
  `prune_threshold`), record measured ns/integer, then the sharded 10¹⁷ run.
- **P5 — CUDA port.** Its own spec.

---

## 10. Risks

| risk | absorbed by |
|---|---|
| pruning underdelivers | 10¹⁷ still lands in ~2.5 days on the engine fix alone |
| bookkeeping wall as `ncov` grows | the bucket lever — documented, deferred; it is Wall 1 from the GPU-sieve audit and it trades away the window-independence that resume and double-buffering lean on |
| new witness-extraction code in stage D | every emitted line independently re-derived by `verify_covers.py` |
| certificate emission volume | prime-survivor constraint inside the pruning objective; per-shard gzip + digest as fallback |
| F2 search finds no further families if L1 ever needs widening | not needed for this design — the two known instances close L1 exactly, measured |

---

## 11. Alternatives considered

- **Integrate covers into `rung_scan` as a pre-filter** — rejected: contaminates the
  census statistics, and couples two trust stories that are cleaner apart.
- **Cite Mordell for the class reduction** — rejected in favour of self-derivation, which
  F2 makes free (zero extra lanes). Mordell is retained as a cross-check.
- **Copy Salez's or Terzi's 192/198-class table** — rejected: known-errors risk, and
  regeneration is cheaper than transcription.
- **Bigger static wheel (2,042,040+)** — rejected as default: dynamic covers already beat
  its mean spacing by orders of magnitude on lane positions, and lane bookkeeping grows
  11×.
- **Keep the plain `mmax` filter rule** — rejected: leaves ~4× redundancy on the table,
  which costs the stretch target. Pruning is sound by §2.4 and fully reproducible.
- **Single binary with a `--naive` mode instead of a separate reference** — rejected: the
  "reference" would share the binary's own cover generator and lane arithmetic, so a bug
  there would be invisible to the differential test.
- **Retire `cover_scan.cpp`, let Python be the only oracle** — rejected: Python can only
  cross-check tiny ranges, leaving the marking engine — where the subtle bugs live —
  without a full-range independent check.
