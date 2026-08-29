# Scaling the minimal-rung census: 10¹⁰ → 10¹² → 10¹⁸

Measured on this machine (2 cores, `-O3 -march=native`). Every number below is either
measured or explicitly labelled as extrapolated. `rung_scan2.cpp` is the optimized
scanner; it produces **bit-identical output** to the verified `rung_scan.cpp` on every
range tested, and passes the same self-test.

---

## 1. What changed, and why it was worth it

### 1.1 A latent bug that would have wrecked a large run

The original scanner's Pollard–Brent had this inner loop:

```c
u64 diff = x > y ? x - y : y - x;
qacc = mulmod(qacc, diff ? diff : 1, n);     // <-- BUG
```

Substituting `1` when `x == y` silently **destroys Brent's cycle detection**. When a
choice of `c` produces a degenerate rho sequence, the algorithm no longer notices the
cycle has closed; it keeps iterating until the `r > 2^26` safety cap and only then
tries the next `c`.

Measured effect, on a real value from the census:

```
A = 9,997,728,493 = 83,987 × 119,039
  buggy : 268,436,004 rho iterations, 11,070 ms   (c=1 exhausted, c=2 then succeeded)
  fixed :          ~1,700 iterations,    0.1 ms
```

One such value per window is enough to dominate everything: the range
`[3.999×10¹⁰, 4.0×10¹⁰)` measured **236 µs/prime** against ~20 µs for its neighbours,
and the whole 236 µs was four primes waiting on one factorization. The fix is to abort
the current `c` the moment `y == x`. This matters more as the search deepens, because
rough cofactors — the ones that reach Pollard — get commoner with height.

### 1.2 The real optimization: bulk sieve-factorization

The old scanner factored each `A = (p+r)/4` by trial division: ~6,500 divisions per
`A`, ~1.5 factorizations per prime. That is what made cost grow so steeply with height.

The structural observation is that **A is an arithmetic progression in the window
index**. Writing `p = first + 24j`,

```
A = (p + r)/4 = A₀ + 6j,        A₀ = (first + r)/4,
```

so for any prime `ℓ ≥ 5`,

```
ℓ | A   ⟺   j ≡ −A₀ · 6⁻¹  (mod ℓ),
```

which is again an arithmetic progression in `j`. So the whole window can be factored
with one strided walk per `ℓ` — total cost `Σ_ℓ npos/ℓ ≈ npos · log log` touches —
instead of `π(B)` divisions per `A`. Three consequences:

- **`ℓ = 2, 3` are free.** Since `6j` is divisible by both, they divide either every
  `A` in the window or none; they get stripped directly, per prime.
- **No primality test, no Pollard, in the hot path.** The walk uses every prime up to
  `√(A_max)`, so whatever remains of `A` afterwards is *prime by construction*. Pollard
  survives only in the deep-tail fallback (~1% of primes), where the fixed bug now also
  makes it safe.
- **The ladder thins for free.** Once fewer than `BULK_MIN = 64` primes are still
  walking, the strided walk stops paying for itself and the tail falls back to per-`A`
  factoring.

### 1.3 Measured speedup

Same range, same output, single-threaded except where noted:

| height | v2 ns/prime | v3 ns/prime | speedup |
|---|---|---|---|
| 10¹⁰ | 12,864 | 1,767 | **7.3×** |
| 10¹¹ | 24,864 | 1,942 | **12.8×** |
| 10¹² | 37,933 | 1,872 | **20.3×** |
| 10¹³ | 1,123,304 | 2,529 | **444×** |

The 10¹³ figure is where the Pollard bug detonates; the honest "algorithmic" speedup is
the 7–20× band. Note v3's cost is nearly **flat** across three decades — that is the
sieve amortizing.

End-to-end: the full 10¹⁰ census went from **272 s → 82 s** on 2 cores, with
byte-identical output. As a demonstration, the census was then extended a decade:
**0 → 10¹¹, all 514,742,404 primes, in 12.4 minutes on 2 cores** (1,448 ns/prime wall).
That run is what produced the strengthened level-S result in the paper (47 cases, every
one at r = 51) and corrected the earlier "isolated outlier" reading of the depth
record.

### 1.4 Window tuning

`--spanlog L` sets the window to `24·2^L` integers (default `L = 20`). Larger windows
amortize the per-window loop over factor primes, but lose cache locality. At 10¹⁵:

| `--spanlog` | window | ns/prime |
|---|---|---|
| 20 | 25 M | 4,844 |
| 22 | 101 M | **4,311** |
| 24 | 403 M | 6,268 |

**Use `--spanlog 22` above ~10¹⁴.** Re-tune on your hardware; the optimum tracks L2/L3
size.

---

## 2. Runtime estimates

Cumulative cost of a full census `0 → X`, single-core-equivalent. Rows to 10¹⁵ use
measured ns/prime; 10¹⁶–10¹⁸ are extrapolated (see §3 for why they are optimistic
without further work).

| height | primes ≡ 1 (mod 24) | ns/prime/core | core-hours | wall on 1,000 cores |
|---|---|---|---|---|
| 10¹² | 4.70 × 10⁹ | 2,221 *(meas.)* | **3** | seconds |
| 10¹³ | 4.33 × 10¹⁰ | 2,332 *(meas.)* | **28** | 2 min |
| 10¹⁴ | 4.01 × 10¹¹ | 3,294 *(meas.)* | **367** | 22 min |
| 10¹⁵ | 3.73 × 10¹² | 4,311 *(meas.)* | **4,467** | 4.5 h |
| 10¹⁶ | 3.49 × 10¹³ | 5,500 *(extrap.)* | 53,000 | 2.2 d |
| 10¹⁷ | 3.28 × 10¹⁴ | 7,400 *(extrap.)* | 674,000 | 28 d |
| 10¹⁸ | 3.09 × 10¹⁵ | 11,600 *(extrap.)* | 9,960,000 | 415 d |

With a bucket sieve (§3.1): 10¹⁶ ≈ 49,000 core-h; 10¹⁷ ≈ 537,000 core-h;
10¹⁸ ≈ 6.0 × 10⁶ core-h ≈ **690 core-years**.

**10¹² is now a rounding error — 3 core-hours.** Run it today. 10¹⁵ is a comfortable
cluster job. 10¹⁸ is a genuine multi-hundred-core-year project.

---

## 2b. The GPU port, and why the height law above does not apply to it

Everything in §2 is the CPU scanner. The CUDA port (`census/cuda/`, `RUNBOOK.md`) has a
different cost structure, and — this is the part worth writing down — **a different
height law**. Measured on one NVIDIA L4, `--hard840`, ~40M hard-class primes per row,
CPU and GPU at the same width so the ratio means something
(`census/cuda/tests/height_sweep2.sh`):

| height | L4 ns/prime | same-box CPU ×12 | ratio |
|---|---|---|---|
| 10¹³ | 12 | 418 | 35× |
| 10¹⁴ | 13 | 437 | 34× |
| 10¹⁵ | **14** | 650 | 46× |
| 10¹⁶ | 18 *(census-width run)* | — | — |

*(Table measured before the final optimization round; after kernel fusion, tail
overlap, and double-buffered windows the current figure is **9 ns/prime at both 10¹³
and 10¹⁵** — a full 10¹⁵ hard-class census in ~2.3 h on one L4.)*

**Cost per prime is nearly flat: ~1.17× per decade, against the CPU's ~1.25×.** The
reason is visible in the phase breakdown at 10¹⁵ — the primality sieve, the part that
grows with π(√X), is **7%** of GPU runtime, while `factorize` (per-prime work) is
**62%**. The CPU's height growth comes from exactly the term the GPU has made cheap.

Consequence for the table in §2: **do not scale a GPU measurement by the CPU's
factor.** Earlier estimates in this project multiplied the 10¹³ GPU number by the
1.85× of §2 and were wrong by 2×. The sweep costs ninety seconds; run it.

Full 10¹⁵ hard-class census (9.33 × 10¹¹ primes) at the measured 14 ns/prime:
**~3.6 hours on one L4**, against ~38 hours projected for a 24-vCPU CPU box.

Two caveats that are honest rather than decorative:

- 10¹⁶ is measured but the code has never been *verified* above it. `diff_range`
  against the CPU reference is clean at 10¹⁴, 10¹⁵ and 10¹⁶; beyond that, nothing.
- §3.1's bucket-sieve wall has not been hit **because the sieve is only 7% of GPU
  time.** It will still arrive, just much later and worth much less than it is for
  the CPU.

---

## 3. Is 10¹² → 10¹⁸ "just distribution and parallelism"?

**Mostly, but not entirely — there are two real algorithmic walls.** The distribution
part is genuinely solved: `--shard i/n` splits into contiguous, independent,
recomputable ranges with zero coordination, and merging is adding histograms. Scaling
to 10,000 cores needs no new code. But:

### 3.1 Wall 1 — the per-window fixed cost (bites around 10¹⁶)

Both sieves iterate a prime list per window to compute start positions, whether or not
that prime hits the window. The lists grow fast:

| height | factor primes `π(√(X/4))` | sieve base primes `π(√X)` |
|---|---|---|
| 10¹² | 41,538 | 78,498 |
| 10¹⁵ | 1.0 × 10⁶ | 3.2 × 10⁷ |
| 10¹⁸ | 2.6 × 10⁷ | 5.1 × 10⁷ |

At 10¹⁸ with `--spanlog 22` there are ~304 k primes per window but ~77 M start
positions to compute — roughly **250 wasted modular operations per prime**, which
dominates everything else. The standard fix is a **bucket sieve**: for each large
prime, store the next window it hits in a per-window bucket, so each prime costs O(1)
per *hit* rather than O(1) per *window*. This is well-understood engineering (it is what
every fast segmented sieve does) but it is real work, and it is the main reason 10¹⁸ is
not a pure scale-out.

Raising `--spanlog` is a partial mitigation and costs memory linearly.

### 3.2 Wall 2 — memory and bandwidth

At 10¹⁸ the shared prime tables are ~400 MB (store them as `uint32_t`, not `uint64_t`
— all fit under 2³²) and per-thread window state grows with `--spanlog`. The `Fac`
array is the largest per-thread structure; shrinking it to 16 slots × `uint32_t`
prime-index would roughly halve it. None of this is hard; all of it needs doing before
a 10¹⁸ run.

### 3.3 What does *not* break

Checked explicitly: `p < 10¹⁸ < 2⁶⁰` and `A < 2.5 × 10¹⁷ < 2⁵⁸` both fit in `uint64_t`;
`q = pA ≈ 10³⁵` is **never materialized** (only `q mod r`); `(u128)ℓ·ℓ` is safe to
`ℓ ≈ 5 × 10⁸`; all rung arithmetic is on residues `< r ≤ 255`. No bignum is needed
anywhere at 10¹⁸. `--rmax 255` has ample headroom: the deepest ladder found below 10¹⁰
is 107, and escalations are counted and reported rather than silently dropped.

---

## 4. Recommended plan

1. **Now:** `./rung_scan2 0 1000000000000 --rmax 255 --spanlog 22` — 3 core-hours,
   extends the census two decades. Verify with `--verify` on every distinct
   machine first, and re-check a few percent of shards on different hardware.
2. **This week:** 10¹⁴ (367 core-hours) or 10¹⁵ (4,467 core-hours) sharded across the
   cluster. This is where the open questions actually live — whether level-S failures
   keep occurring only at `r = 51`, whether the residual share keeps falling
   (2.3% → 1.05% → 0.71% so far), and whether the depth record 107 at
   *p* = 8,803,369 is ever beaten.
3. **Before going past 10¹⁶:** implement the bucket sieve. Without it you will pay
   ~2× at 10¹⁷ and ~1.7× at 10¹⁸ for nothing.

### One caution worth repeating

A full census to 10¹⁸ **adds no verification value** — Erdős–Straus is already verified
to 10¹⁸ by other means. Its only product is statistical, and the statistics converge
long before the census does. A **stratified design** — full census to 10¹³, then
fixed-width windows of, say, 10¹¹ at each higher decade — gives essentially the same
distributional information for a ~10⁻⁴ fraction of the cost, and each window is
independently comparable. If the goal is the science rather than the record, spend the
cluster on 10¹³ complete plus deep decade windows, not on grinding 10¹⁸.
