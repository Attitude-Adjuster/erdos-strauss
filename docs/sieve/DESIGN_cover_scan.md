# DESIGN — `cover_scan`: a certified covering-progression verification sieve

Companion design for extending this repository with a Salez-style presieve,
per the correspondent's suggestions. Status: **design + skeleton**. Every number in
this document was measured or proven in this session; the scripts that produced them
are folded into `verify_covers.py` so they can be reproduced with one command.

## 0. Scope and non-goals

`cover_scan` is a **separate verification sieve**, not a change to `rung_scan.cpp`.
The census's product is the *classification* (J/RC/S/R populations, depth
histograms). Removing primes by constructive identities before classification would
change the paper's statistics; the classifier must keep seeing the full hard-class
population. `cover_scan` answers a different question — "is every prime in `[lo,hi)`
solvable?" — at orders of magnitude higher throughput, the way Salez covered long
intervals. It earns trust the way `rung_scan2` and the CUDA port did: independent
implementation, self-test, exact-arithmetic re-verification of emitted claims, and a
published, checksummed filter table.

Population: primes only, restricted to the hard class (`--hard840` lanes, all
≡ 1 mod 24). Squares and the small full-`n` exceptions are composite, so they never
arise (§1.3).

## 1. Mathematical basis

### 1.1 The covering identity (proved, then verified exhaustively)

For positive integers `a, c, d, k` with `ck > a`, put

```
m = 4acd − 1,   e = ck − a  (≥ 1),   p = m·k − 4a²d .
```

Then `p + k = 4ade`, and

```
  4/p = 1/(ade) + 1/(acdp) + 1/(cdep)          [the boxed decomposition]
```

*Proof.* `p + k = (4acd−1)k − 4a²d + k = 4ad(ck − a) = 4ade`. Over the common
denominator `acdep` the right side has numerator `cp + e + a`; since
`cp = 4acde − ck` and `ck = e + a`, this is `4acde`. ∎

All three denominators are positive exactly when `e ≥ 1`, i.e. `ck > a` — this is
where the threshold comes from. Nothing about primality is used: **every** integer

```
  p ≡ −4a²d (mod m),   p ≥ p_min(a,c,d) = m·(⌊a/c⌋+1) − 4a²d
```

is solved by the triple above. Each `(a,c,d)` therefore certifies an entire
arithmetic progression; no per-`p` certificate is ever needed (the family is
self-certifying). `verify_covers.py --identity` re-checks the identity in exact
rational arithmetic on an exhaustive small grid plus 20,000 random large tuples
(31,641 instances this session, 0 failures).

This is Salez's parameterization derived from the Rosati equations; the repo should
cite Salez §4 and Rosati, and *derive* everything from the identity rather than
copying any published class table (Terzi's table is reported to contain errors —
see §3.3).

### 1.2 What the family cannot cover

Odd squares are a genuine structural exception (the classical quadratic obstruction
to Type II solutions), and there are finitely many small non-square stragglers. The
correspondent recalled "288, 336 and I believe another one around 9000. 9545 or so."
Measured exactly this session — `verify_covers.py --sweep 100000` enumerates the
family **exhaustively** for `p ≤ 10⁵` (reparameterize by `(a,d,e,k)` with
`c = (a+e)/k ∈ ℤ`; then `4ade = p + k ≤ p + a + e` bounds the whole search, so the
result is a statement about the *unbounded* family below 10⁵):

> survivors below 100,000 = the 315 squares ∪ **{288, 336, 4545}**.

So the third exception is **4545 = 3²·5·101**, not 9545. All three are composite
(288 = 2⁵3², 336 = 2⁴·3·7), so a prime-only scanner never meets them. Beyond 10⁵
this remains an *observed bounded sweep*, not a theorem — phrase it that way
anywhere it appears, and never use the unrestricted claim as a regression fixture.

## 2. Pipeline

```
  window [w, w+SPAN)                                    per-stage counters
  ── A. static wheel: 220 certified lanes mod 120120    (positions never touched)
  ── B. dynamic covers: (m,r,p_min), m ≤ mmax, CRT      COVERED++
  ── C. deterministic MR on survivors                   COMPOSITE++
  ── D. rung walk on surviving primes                   RUNG p A r u
  ── E. direct parametric solver for anything left      SOLVED p a c d k / SURVIVOR p
```

### 2.A Certified static wheel (generated, never copied)

A residue class ρ (mod M) dies iff some cover with `m | M` has `ρ ≡ −4a²d (mod m)`;
then *every* element ≥ p_min of the class is solved. Generation at startup costs
milliseconds. Measured ladder for the hard class (survivors ≡ 1 mod 24), single
family, all `m | M`, versus the counts Salez reports for his full filter system:

| M          | covers (m,r) | survivors here | mean spacing | Salez     | max p_min |
|------------|-------------|----------------|--------------|-----------|-----------|
| 840        | 13          | **6**          | 140          | 6         | 34        |
| 9,240      | 39          | 36             | 257          | 34        | 1,154     |
| 120,120    | 120         | **230**        | 522          | 192       | 15,014    |

Those counts are F1 alone, restricted to `ρ ≡ 1 (mod 24)`. Adding the F2 uniform rung
certificates (see the newer spec) both *derives* that restriction instead of assuming
it and takes M = 120,120 to **220** lanes, mean spacing 546.
| 2,042,040  | 303         | 2,612          | 782          | 1,507     | 255,254   |
| 38,798,760 | 892         | 28,298         | 1,371        | 13,380    | 1,616,614 |

Three facts worth noting. First, at M = 840 the generator reproduces **exactly**
`rung_scan.cpp`'s `{1,121,169,289,361,529}` — a free end-to-end validation of the
generator. Second, the single family already beats the naive 336-class
prime-by-prime construction at 120,120 (230 classes, before F2) because composite `m | M`
("shortened filters") are included automatically. Third, the gap to Salez's 192
(and Terzi's 198) grows with M: Salez used additional criteria families beyond this
identity. Do **not** conflate the three counts; publish ours as "single-family
certified" with its own checksum. The gap is unimportant in practice because stage B
recovers it (measured below) — the static wheel's only job is to cut the touched
positions by ~546× before the per-position work starts.

Default `M = 120120`. Larger wheels give steeply diminishing returns against
dynamic covers and complicate the lane bookkeeping (2,612 lanes at 2,042,040).

### 2.B Dynamic covering progressions

Generate covers `(m = 4acd−1, r = −4a²d mod m, p_min)` for `m ≤ mmax`, dedup by
`(m, r)` keeping the smallest `p_min`, drop `m | M` (their classes are already
gone), sort by ascending `m`. For a lane `p_j = first + M·j`:

```
g = gcd(M, m);  rhs = (r − first) mod m
if g ∤ rhs: no hits.  step = m/g;  j0 = (rhs/g)·(M/g)⁻¹ (mod step)
advance j0 by multiples of step until first + M·j0 ≥ p_min   [tiny at height]
clear bits j0, j0+step, …
```

Two correctness points the sketch in the correspondence already flagged, plus one
it didn't:

- `(M/g)⁻¹ (mod step)` must use **extended Euclid**: `step` is generally composite,
  so the Fermat inverse used for the prime sieve moduli is invalid here.
- All of `4acd−1`, `4a²d`, and residue reductions are computed in `unsigned
  __int128` before narrowing.
- The inverse always exists: for each prime, the minimum exponent goes entirely
  into `g`, so `M/g` and `m/g = step` are automatically coprime. (This is why the
  `rhs % g == 0` test is the *only* solvability condition.)

Measured two ways. First a Python simulation at height 10¹²–3.3·10¹³ (8 survivor
lanes × 2¹⁸ positions, application cross-checked bit-for-bit against a brute-force
per-position test — 0 mismatches):

| mmax   | covers  | Σ 1/m (marks/pos) | surviving fraction of lane positions |
|--------|---------|-------------------|--------------------------------------|
| 840    | 2,961   | 11.4              | 3.0·10⁻⁴                             |
| 2,000  | 8,849   | 15.7              | 6.7·10⁻⁵                             |
| 50,000 | 448,204 | 41.4              | 6.7·10⁻⁵                             |

Then the skeleton itself over `[10¹², 10¹² + 1.2·10¹⁰)` — 23,000,000 lane
positions, single thread, startup included:

| mmax   | wall  | MR calls | composite            | prime survivors           |
|--------|-------|----------|----------------------|---------------------------|
| 2,000  | 3.7 s | 1,290    | 1,269                | **21** (all fall to the rung walk at r ≤ 27) |
| 5,000  | ~5 s  | 1,149    | 1,149                | 0                         |
| 10,000 | 5.9 s | 1,148    | **1,148 = exactly the perfect squares in the lanes** | 0 |
| 50,000 | 25.2 s| 1,148    | 1,148                | 0                         |

So the apparent plateau is squares plus a *thin* non-square tail (~9·10⁻⁷ of lane
positions at mmax = 2,000 — invisible in the 2.1M-position simulation) that is gone
by mmax ≈ 5,000 at this height. The residue at mmax = 10⁴ is exactly the structural
one: every composite survivor is a perfect square, and every prime in the range is
certified by wheel + covers alone. Past ~10⁴ the extra covers only re-kill dead
positions. Default `mmax = 10,000`; the correspondent's instinct to "restrict one
or two of the variables … so that only few possible exceptions remain" is
confirmed, with the effective restriction being on `m` itself (bounding the modulus
bounds the cost directly — covers with expected < 1 hit per window are pure
overhead and can be dropped, which only shifts work to stages C/D, never loses
correctness). Keep `mmax` adaptive: if the per-window survivor counter rises above
a budget, raise it.

### 2.C Primality on survivors

Survivor density ≈ 5·10⁻⁵ of lane positions (≈ 1 MR call per 10⁷ integers of
range at mmax = 10⁴ — nearly all of them perfect squares) makes the choice easy:
the repo's deterministic 12-base Miller–Rabin per survivor. A segmented sieve pays
only if stages A–B are weakened drastically (correspondent's own criterion;
measured, MR wins by a wide margin). An `isqrt` pre-filter for squares is optional
and not worth the code.

### 2.D / 2.E Survivor resolution

Expected calls at height: zero at mmax = 10⁴ (measured over 23M positions), ~2 per
10⁹ integers of range at mmax = 2,000 — and those 21 measured stragglers all fell
to the rung walk at r ≤ 27 (distribution 8·r3, 5·r7, 3·r11, 1·r15, 2·r19, 1·r23,
1·r27). Both stages still must exist and be correct:

- **D. rung walk** — the exact solver the repo already trusts: `A = (p+r)/4`,
  factor `A`, subset-product DP for a divisor `u | q²` with `u ≡ −q (mod r)`; emit
  `RUNG p A r u`. Lift `factorA`/`test_rung`-style code from `rung_scan.cpp`
  verbatim; per-A factoring is fine at this call rate.
- **E. direct parametric solver** — the correspondent's "direct solver, including
  factoring (p+i)/4": for `a, d` in a small box, factor `p + 4a²d`, look for a
  divisor `m ≡ −1 (mod 4ad)` with `ck > a`; emit `SOLVED p a c d k`. Anything still
  standing is emitted as `SURVIVOR p` and fails the run's exit status.

## 3. Trust and publication artifacts

Per the correspondent's checklist, the run publishes:

1. the generated filter table (`FILTER m r a c d pmin` lines, `--dump-filters`);
2. its SHA-256 (the 220-class table: `154e8267…`, full digest and the 22,820 kill
   witnesses in `class_table_120120.txt`);
3. the parameter bounds (`M`, `mmax`, `pmin_cap`) in the SUMMARY line;
4. a verifier for the boxed decomposition (`verify_covers.py`: identity in exact
   arithmetic, table regeneration + byte-diff, RUNG/SOLVED certificate checks);
5. survivor lists and their direct certificates;
6. per-stage candidate counts (wheel-killed, cover-killed, MR-composite,
   rung-solved, direct-solved, survivors).

The scanner refuses to start unless `lo > max p_min` of every filter in use
(15,014 for the default wheel; the region below is settled by the existing censuses
and by `--sweep`).

### 3.3 Provenance rules

Generate everything; hard-code nothing. Terzi's 198-class table reportedly contains
small errors, and Elsholtz–Tao note the residue classes were sound while the
checked-prime list was not. Our 220-class table is *derived* at startup from the
identity and cross-checked against a checked-in checksum — the same trust pattern as
`make check`.

## 4. Cost model and expected throughput

Measured on the unoptimized skeleton (single thread, gcc -O2, startup included):
`[10¹², 10¹² + 1.2·10¹⁰)` in 5.9 s at mmax = 10⁴, i.e. **≈ 0.5 ns per integer of
range** (≈ 260 ns per lane position, dominated by per-lane cover setup and
cache-hostile strided clears). Headroom before it matters: OpenMP across windows,
the incremental-j0 trick from `rung_scan2` across adjacent windows, cover
setup amortization via larger `spanlog`, and dropping covers with expected < 1
hit/window. Even at the unoptimized 0.5 ns/integer, 10¹⁵ costs ~6 core-days and
10¹⁷ ~1.6 core-years; the census scanner's job at 10¹⁵ (932·10⁹ hard-class primes ×
820 ns) is ~9 core-days for comparison — but cover_scan's number scales with the
*range*, not the prime count, and parallelizes embarrassingly. The number that
matters for tuning is the survivor counter per window.

Windowing, sharding, OpenMP, streaming emission: copy `rung_scan.cpp`'s structure
(`SPAN = M << spanlog`, `--shard i/n`, per-thread stats merged at the end).

## 5. Alternatives considered

- **Integrate into `rung_scan` as a pre-filter** — rejected: contaminates the
  census statistics (§0); also couples two trust stories that are cleaner apart.
- **Copy Salez's/Terzi's 192/198-class table** — rejected: known-errors risk,
  and regeneration is cheaper than transcription (§3.3).
- **Bigger static wheel (2,042,040+)** — rejected as default: dynamic covers at
  mmax ≈ 2,000 already outperform even the 892,371,480-wheel's mean spacing
  (6,056) by orders of magnitude on lane positions (§2.B); lane-count bookkeeping
  grows 11×.
- **Segmented prime sieve before survivors** — rejected at default tuning
  (density 10⁻⁴ → MR wins); revisit only if `mmax` is cut very low.
- **Emitting per-prime COVER certificates** — rejected: billions of lines that
  all instantiate one two-line identity; the filter table + verifier is the
  certificate (this is what "self-certifying" buys).

## 6. File plan

| file | role |
|---|---|
| `cover_scan.cpp` | the sieve -- COMPLETE and FROZEN as the reference implementation |
| `verify_covers.py` | identity proof check, table regeneration + diff, certificate verification, exhaustive small-n sweep |
| `class_table_120120.txt` | checked-in 220-class table + 22,820 kill witnesses + SHA-256 (regenerated + diffed by both binaries) |
| `tests/diff_cover.sh` | (future) cross-check cover_scan survivors against rung_scan first-hits on shared ranges |

## 7. Status

**Superseded** for scope, tuning defaults, and file plan by
`docs/design/2026-08-13-rung-scan3-design.md`. `cover_scan.cpp` is now
COMPLETE and FROZEN as the reference implementation; the optimized scanner will be
`rung_scan3.cpp`, gated against this one by `tests/diff3.sh`.

The mathematics in §1 and the measurements in §2 remain valid. Three things the newer
spec adds, all measured after this document was written:

- **F1 is complete, and its modulus is always `≡ 3 (mod 4)`.** Reparameterized by
  `N = acd`, `α = c²d`, the family is exactly `{(N, α) : α | N²}` with `m = 4N − 1`;
  every `α | N²` is reachable. So F1 can never constrain a residue mod 8, which is why
  the classes 7, 13, 19 (mod 24) survive it and why a second family is *necessary*.
- **F2, uniform rung certificates**, closes that gap: a rung `r` plus a fixed witness
  `u` whose two conditions are forced by `p`'s residue class alone. The reduction to
  `p ≡ 1 (mod 24)` is therefore **derived**, not cited — `wheel(840)` reproducing
  `{1,121,169,289,361,529}` is the check. Mordell becomes a cross-check.
- **The lane count is 220, not 230**, and it is a *parameter*: 230 is what the two
  minimal F2 certificates give, while the full search to `r ≤ 63, u ≤ 64` kills ten
  more classes. The table is published with `(rmax, umax)` alongside `mmax`.
  `wheel_120120_hard.txt` (230 classes, F1 only, restricted to `ρ ≡ 1 mod 24`) has been
  **retired** in favour of `class_table_120120.txt`, and `verify_covers.py --wheel`
  removed with it: two class tables with different counts in one repo is exactly the
  kind of confusion this project should not publish.

Stage D (rung walk, emitting the witness `u`) and stage E (direct parametric solver)
are implemented and exercised by `make check-cover`. Still open, and belonging to
`rung_scan3.cpp` rather than here: OpenMP, `--shard`, progress plumbing,
`tests/diff3.sh`, and benchmark-driven `spanlog`/`SEG` tuning.
