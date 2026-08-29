# CUDA rung_scan — design

**Date:** 2026-08-07
**Status:** approved design, not yet planned or implemented
**Target hardware:** H100 in production; L4 (`wm-trainer`, `g2-standard-8`) as the
proof-of-concept box
**Reference implementation:** `erdos/rung_scan.cpp`, unchanged and authoritative

---

## 1. Goal

Build a CUDA implementation of the Erdős–Straus certificate census that makes deep
windows at arbitrary height cheap, and can later run a complete census.

**Order of delivery: the windowing engine first, complete-census capability second.**
The two want different things — a complete census wants sustained throughput, resume,
and shard-tiling proofs; a windowing engine wants to start cheaply at an arbitrary
offset. The shard model already supports both, so this is a sequencing decision rather
than an architectural fork.

### What success means

The proof of concept succeeds if it produces **an honest measurement of what a GPU buys
on this algorithm**. A result of "3×, and here is precisely why not more" is an
acceptable and useful outcome. What is *not* acceptable is a fast scanner whose output
cannot be checked.

There is no throughput target. There is a hard correctness gate (§3).

### Non-goals

- **The bucket sieve** (`SCALING.md` §3.1). It is the wall at 10¹⁶ and beyond; at 10¹⁵
  the per-window fixed cost is still tolerable. Building it before we know whether the
  GPU pipeline pays would be premature. Revisit after this engine is measured.
- **Optimizing `rung_scan.cpp`.** See §7.
- **Multi-GPU as new design.** N GPUs is N processes with `--shard i/n`, exactly as with
  the CPU version. `run8.sh` drives it unchanged.

---

## 2. The measurement this design is built on

Before designing anything we separated sieve cost from ladder cost, by sweeping `--rmax`
over one range (`[9999000000000, 10^13)`, `--hard840`, `--spanlog 22`, single-threaded,
best of 3) on a `c4-highcpu-24` (Xeon Platinum 8581C, 260 MB L3):

| rmax | 3 | 7 | 11 | 15 | 19 | 23 | 27 | 31 | 51 | 255 |
|---|---|---|---|---|---|---|---|---|---|---|
| `scan_s` | 0.31 | 0.46 | 0.57 | 0.63 | 0.68 | 0.72 | 0.76 | 0.80 | 0.80 | 0.80 |
| marginal | — | .15 | .11 | .06 | .05 | .04 | .04 | .04 | **0** | **0** |

The marginals decay but flatten at **~0.04 s per rung** even after all but 0.06% of
primes have hit. That floor is the active-count-independent strided walk. Fitting
`marginal = W + P·active` gives `W ≈ 0.04`, `P ≈ 0.27`, predicting the whole curve
within ~20%:

- **strided walk — 8 rungs × 0.04 ≈ 0.32 s — 40%**
- **per-active-prime work — 0.27 × Σactive(1.67) ≈ 0.45 s — 56%**
- **primality sieve — 0.05 s — 6%** (reported separately as `sieve_s`)

**A prior estimate that the scatter was ~90% of runtime was wrong.** It was derived by
counting touches and assuming ~3 cycles each. The measurement says the walk runs at
**~0.74 cycles per touch**, because the entire window working set is L3-resident on this
machine, the loop is simple, and the skip branch is well-predicted. Recording this
because it is the single most important number in the design: **the CPU is unexpectedly
good at the part we assumed would be its weakness.**

### Amdahl consequences

| offload | ceiling |
|---|---|
| `test_rung`-ish per-prime work only | 2.3× |
| strided walk only | 1.7× |
| everything except the primality sieve | 17× |
| everything | no structural floor |

Both halves are embarrassingly parallel with **uniform `r` across all threads**, so a
full port has no serial floor. Partial offloads do. Hence: full port.

**Everything past rung 31 is free** (marginal 0.00 at every rmax tested). The deep tail
that produced the depth-131 record costs nothing.

### Per-window quantities (spanlog 22, `--hard840`, 10¹³)

| quantity | value |
|---|---|
| positions per class (`npos`) | 4.19 × 10⁶ |
| window width | 3.52 × 10⁹ |
| primes per class / per window | ~61 K / ~368 K |
| walk touches per window | 3.5 × 10⁸ |
| *useful* hits (prime **and** active) | ~1 × 10⁶ — **0.3% of touches** |
| CPU time per window | 0.28 s single-core |

---

## 3. Correctness bar (non-negotiable)

**The CUDA scanner must produce byte-identical statistical fields of the `SUMMARY` line
and byte-identical sorted certificate sets to `rung_scan` on every range where both can
run, and must pass `--verify` in both population modes on the GPU itself.**

*Amended 2026-08-07, during Task 1.* This first read "byte-identical `SUMMARY` lines",
which is not a satisfiable bar: the line ends with `sieve_s=`, `scan_s=` and
`ns_per_prime=`, measured wall-clock that jitters ±2% between runs of the same binary
and that a GPU port is *meant* to change. The bar applies to
`primes jac rc sup res escalated pairpath maxr maxr_p hist` — every field that describes
the mathematics — and explicitly not to the timing suffix.

Emission *order* may differ — `merge.py` already sorts, and the CPU scanner is already
unsorted because it streams per window — but the emitted *set* must match exactly.

This makes the CPU scanner the permanent reference implementation, not a bring-up
crutch. The rationale is that the entire value of this census is that its numbers are
checkable; the paper's verification protocol depends on it, and a GPU path that could
not be diffed would undermine it.

Nothing here is at risk from floating-point associativity: the computation is integer
throughout, and `q = pA` is never materialised.

---

## 4. Architecture

### 4.1 Host/device split

The host parses the CLI (mirroring `rung_scan` plus `--device`, `--batch`), builds the
two prime tables once, uploads them, then loops over windows launching device work and
draining output. Table sizes are comfortable: π(√X) = 1.9 × 10⁶ base primes and
π(√A_max) = 1.0 × 10⁶ factor primes at 10¹⁵ — 12 MB as `uint32_t`; 307 MB at 10¹⁸.
All values fit under 2³².

Per (window, class) the device runs:

1. segmented primality sieve over `p = first + mod·j` → bitmap
2. stream-compact survivors (`cub::DeviceSelect`, stable)
3. **rung wavefront** — per rung: generate hits, sort, segment-reduce, classify, compact
4. emit certificates to a device ring buffer
5. reduce depth histogram and level counters

Two or three CUDA streams overlap window *i*'s drain with window *i+1*'s sieve.

### 4.2 The deep tail stays on the CPU, permanently

Once the active set falls below a threshold, survivors are copied back and run through
the existing `factorA` + `test_rung` fallback (`rung_scan.cpp:557-572`) unchanged.

Three reasons: it costs *zero* (measured); it is the only divergent code in the program,
being where Pollard–Brent lives; and it is where the scientifically interesting rare
events are, so level-S and deep-ladder certificates keep coming out of the already
verified C++ path.

**Consequence worth stating explicitly: the bulk/fallback threshold is a pure
performance knob with no effect on output.** Both paths compute the same `Level` from
the same `Fac`; `BULK_MIN` only decides which one does it. We may tune the handoff
freely without touching the diff bar.

### 4.3 Statistics determinism

Per-window totals reduce on device (integer sums, order-independent), then accumulate on
the host **in window order**. The `SUMMARY` line is therefore reproducible run to run
regardless of block scheduling.

---

## 5. Data layout

### 5.1 Factorizations are carried as 8-bit residues mod `r`

Every consumer of `Fac` in `test_rung` (`rung_scan.cpp:314-371`) — the DP bases,
`jacobi(fA.pr[i] % r, r)`, `legendre_sym` against `r`'s prime factors, the
support-subgroup generators, `tau_q2` — uses either `ℓ mod r` or the exponent. **The
value of ℓ is never needed.** With `r ≤ 255`, a factorization is 16 pairs of
`(uint8 residue, uint8 exponent)`: **32 bytes instead of 144.**

This also disposes of the wide-cofactor problem `SCALING.md` §3.2 raises: the leftover
prime is reduced mod `r` on the way out and never stored wide.

Layout is structure-of-arrays throughout, one entry per **active prime** rather than per
position: position index, `p mod r`, `A mod r`, residue factorization.

### 5.2 The wavefront

An active list of `uint32` position indices, rebuilt after each rung with
`cub::DeviceSelect::Flagged` (stable — survivors keep relative order, so the list is a
deterministic function of results, not of scheduling). It shrinks 100% → 42% → 18% →
4.6% → 2.1%.

**Where compaction pays, precisely:** hit *generation* is a walk over factor primes, so
its cost does not shrink with the active set — that is the 0.04 s floor, and compaction
alone will not remove it. What compaction enables is **filtering hit pairs against the
active bitmap before the sort**, so the expensive stages scale with active hits even
though generation does not.

### 5.3 Batching, because tail rungs starve the GPU

At rung 15 only 4.6% of ~61 K primes per class are active — a few thousand threads,
which will not fill an H100. The unit of work is therefore **all classes of a window
batched together** (6 in `--hard840`), with **several windows in flight across streams**.

### 5.4 One deliberate non-optimization

For late rungs, abandoning the sieve and trial-dividing each active `A` directly is a
trap on GPU: at rung 15, ~2,800 active values per class × 6,542 trial divisions to
`TD_BOUND` is ~1.8 × 10⁷ divisions per class-rung, and it worsens as √A grows.
Sieve-plus-filter stays better until the active count is genuinely tiny — which is the
handoff point to the CPU deep tail.

---

## 6. The factorization pipeline (generate–sort–segment)

For rung `r`, `A_j = A₀ + astep·j` with `A₀ = (first + r)/4` and `astep = mod/4` (210 in
`--hard840`, 6 otherwise). Prime ℓ divides `A_j` exactly when
`j ≡ −A₀·astep⁻¹ (mod ℓ)`. Primes dividing `astep` (2, 3, 5, 7) are stripped per-prime.

**Four stages, no atomics anywhere:**

1. **Count** — one thread per factor prime computes its hit count in `[0, npos)` by
   closed form, no walking.
2. **Prefix-sum** the counts (`cub::DeviceScan`) giving every ℓ its own exclusive output
   slice. *This is what buys determinism by construction:* each prime writes only into
   its own slice at a computed index, so the output array is a pure function of the
   inputs regardless of scheduling.
3. **Generate + filter** — each ℓ walks its progression, emitting `(j, ℓ)` only where
   position `j` is both prime and still active.
4. **Sort + segment-reduce** — radix-sort by `j`, run-length-encode into segments, then
   one thread per active position walks its ≤14 factors, computes each exponent by
   division, divides them out of `A_j`, and reduces the leftover cofactor mod `r`.
   Output is the 32-byte residue struct.

**The filter in stage 3 is the whole ballgame.** Of 3.5 × 10⁸ touches per window, only
~10⁶ produce useful pairs (0.3%), because a touch matters only if position `j` holds a
prime that is still active. So the sort never sees more than a few million keys and
costs well under a millisecond. Approach C's feared weakness — pushing 10⁸ pairs through
a sort — does not materialise once the filter precedes it.

**Load balancing by ℓ magnitude**, since progressions are wildly uneven: block-per-prime
with strided threads for small ℓ, warp-per-prime for the middle, thread-per-prime for
large ℓ.

---

## 7. The classify kernel

Every thread in a launch shares the same `r`.

**The DP state fits in one 64-bit register.** The CPU keeps `S ⊆ ℤ/r` as `bool S[255]`.
The GPU runs only the bulk rungs; measurement puts the natural handoff near `r = 31`,
and `r = 63` covers 99.99% of primes, so **the kernel is capped at `m ≤ 64`** — chosen
to leave headroom above the expected handoff rather than to sit exactly on it. `S` is
then a `uint64_t` bitmask in registers, never touching local memory. Any rung above 63
is by construction the CPU's, which §4.2 establishes costs nothing. The DP iterates
set bits with `__ffsll`; for `r = 3, 7, 11` (95% of primes) the mask has ≤11 live bits.

**Precomputed tables turn the certificate path into lookups.** Failures are common —
7.28 × 10⁹ against 1.08 × 10¹⁰ primes, ~0.67 per prime — and each failed rung runs the
Jacobi loop over `p` and every prime factor of `A` (`rung_scan.cpp:364-365`), up to 14
iterative reciprocations. Since `r ≤ 63` and all arguments are already reduced,
**`jacobi` becomes a 64-entry table lookup**, and the Legendre symbols against `r`'s
prime factors likewise. With ~16 distinct bulk rungs the tables total a few KB, computed
on the host at startup and resident in constant/shared memory.

`realchar_mask` (≤15 subset iterations) and `support_blocks` (subgroup closure over ≤64
elements) run **only** when a failure is not level J — 0.02% of failures. The rest take
the table lookup and exit.

**Deliberate restraint:** the same table trick would speed up the CPU scanner. We will
not apply it. `rung_scan.cpp` is the reference the diff bar is defined against, and its
present value is that it is verified, self-testing, and unchanged since it produced the
published census. Optimizing the reference to chase the thing being benchmarked against
it would be a bad trade. Recorded here as a known, deliberately unexploited opportunity.

---

## 8. Verification and testing

**Three diff tests:**

1. **Self-test on device** — `--verify` passes in both population modes, run on the GPU,
   per the Makefile's per-machine/per-flag-set discipline. Small (p < 10⁵) but it
   exercises the classifier and the GPU→CPU handoff.
2. **Range regression** — both binaries to 10¹⁰, then `diff` the `SUMMARY` lines and the
   sorted certificate sets. Seconds to run; runs on every change.
3. **High-offset acceptance** — one window at 10¹⁵, diffed once. Catches anything that
   appears only when √A_max grows and the factor-prime list lengthens.

**Determinism tests**, because determinism is designed-for rather than free: run twice
and byte-compare; run at two block sizes and batch widths and compare; run one range as
1 shard and as 8 shards and compare. With count-then-prefix-sum allocation these should
pass by construction — a failure means an atomic crept in.

---

## 9. Bring-up order

Arranged so there is always something diffable.

- **(a) Scaffolding, no GPU code.** CLI mirror, prime tables, window loop, output path,
  with the CPU's own routines doing the work. Diffs clean trivially; establishes the
  harness before any kernel exists.
- **(b) GO/NO-GO GATE.** Factorization pipeline (§6 stages 1–4) for **rung 3 only** —
  the worst case, since every prime is still active and generation is at full width.
  Diff its residue factorizations against the CPU's for the same window, and measure
  wall time for stages 1–4 over one full window (all 6 classes).

  The CPU spends ~0.04 s per rung on the walk plus ~0.27 s on rung 3's per-prime work.
  Stages 1–4 replace the walk and the factorization half of that. Decision rule:

  | measured | verdict |
  |---|---|
  | **< 10 ms** | proceed to (c) |
  | **10–30 ms** | judgment call — proceed only with a written rationale for where the remaining stages will find their margin |
  | **> 30 ms** | stop, and write up why |

  A negative result here is a valid deliverable under §1, not a failure of the project.
  Diff correctness is a precondition in every case: a fast gate that does not diff clean
  is a failed gate.
- **(c) Classify kernel**, diffed per-rung against CPU `Level` decisions.
- **(d) Full wavefront** with inter-rung compaction and CPU handoff.
- **(e) Primality sieve** on device — deliberately last, since it is only 6% and the CPU
  can carry it while everything else is proven.
- **(f) Batching, streams, tuning.**

---

## 10. Integration

- `merge.py`, `verify_cert.py`, `analyze_census.py` — **unchanged**.
- `run8.sh` — one small change to select the binary (currently hardcodes
  `rung_scan.frozen`).
- `Dockerfile` — swap `debian:12-slim` for a CUDA devel image and `g++` for `nvcc`,
  keeping the compile-on-host-then-self-test discipline. Fat binary for `sm_89` (L4) and
  `sm_90` (H100).

---

## 11. Risks

| risk | assessment |
|---|---|
| **The gate at 9(b) fails.** | Real. The CPU's 0.74 cycles/touch on a 260 MB L3 is the reason it might. This is why the gate exists and why it is early. |
| **Throughput estimate is wrong.** | ~10 ms/window against 280 ms/core → ~28×/core. Carries maybe a 3× error bar in either direction; one prior estimate in this design was already wrong by 2×. |
| **28×/core is only ~2.3× against a 12-core box.** | A *positive* gate does not prove the engine beats CPU shards economically. It proves it is worth finishing and measuring — which is the stated goal. |
| **L4 ≠ H100.** | L4 is `sm_89`/24 GB/~300 GB/s; H100 is `sm_90`/80 GB/~3 TB/s. L4 results are indicative only; the scatter-heavy stages will look disproportionately bad there. |

---

## 12. Open questions

- Where exactly to put the GPU→CPU handoff. It is a free parameter (§4.2) and should be
  tuned by measurement once (d) works, not guessed now.
- Whether batching several *windows* (not just classes) is needed to fill an H100 at the
  tail rungs, or whether 6 classes suffices. Measure at (f).
