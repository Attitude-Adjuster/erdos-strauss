# SCALING_COVER.md — measured cost of the verification sieve

Every number here was measured on **`cloud-dev`** (`c4-highcpu-8`, europe-west3-c,
Debian 12, gcc 12.2, `-O3 -march=native`), 2026-08-14. That box has **4 physical cores**
plus SMT. **These are not publication numbers** — the census box in `RUNBOOK.md` is where
those get made. Everything below is a dev-box measurement plus an explicitly labelled
projection.

Unless stated otherwise: range `[10¹², 1.5·10¹²)`, `mmax = 2000`, `--seg 262144`.

---

## 1. Cost constants and the stage-D profile

**Superseded twice; this is the measured version.** The pipeline's cost is
`t = marks·c_m + survivors·c_s`, but `c_s` is not one number — a prime survivor costs
~28× a composite one, and conflating them has now caused two wrong decisions.

Budget over `[10¹², 1.5·10¹²)` with the published table, single-thread equivalent:

| | time | share | rate |
|---|---|---|---|
| marking (8.02·10⁹ marks) | 6.13 s | **90.9%** | **0.76 ns/mark** |
| stage D (23,182 prime survivors) | 0.53 s | 7.8% | **22.7 µs/prime** |
| composite survivors (112,386) | 0.09 s | 1.3% | ~0.8 µs each |

**Marking dominates at ~91%.** Earlier versions of this file claimed survivors were 61%;
that came from feeding the two-equation solve with survivor densities taken from
`prune_filters.py`'s unanchored evaluation domain, which overstated them 6× (see §5).

### Stage D, profiled directly (`tests/bench_staged.cpp`)

On the 465 primes an actual scan hands it:

| | |
|---|---|
| stage D total | 22.7 µs/prime |
| rung attempts | 2.40 per prime |
| `factorA` | **10.6 µs/call — essentially 100% of stage D** |
| `min_div_res` (the divisor DP) | 0.1 µs/call — **1%** |

`factorA` trial-divides by every prime ≤ `TD_BOUND = 65536` — 6,542 divisions — and its
`sp*sp > n` early exit never fires, because `65536² = 4.3·10⁹` is far below
`A ≈ 2.5·10¹¹`. The per-rung `vector<u128>` allocations inside `min_div_res`, which
looked like the obvious suspect, cost 1%.

**But stage D is only 7.8% of runtime**, so even eliminating `factorA` entirely caps out
at ~8%. It is a clean target, not a transformative one.

### Measurement discipline this cost

The box's absolute speed drifts by ~20% between sessions: the unpruned configuration
measured 2.19 s on one day and 2.645 s on another, with nothing changed. **Only
within-session comparisons are valid**, and every table in this file should be read as a
set of ratios measured together, never as absolute times comparable across sections.

## 2. The wheel modulus

Moved from M = 120,120 to **M = 2,042,040** (spec `2026-08-15-bigger-wheel-design.md`).
The wheel is the only lever that cuts marking and survivor work together, because it
cuts the positions both are proportional to.

| M | lanes (F1+F2) | spacing | positions vs 120120 | unit classes |
|---|---|---|---|---|
| 120,120 | 220 | 546 | 1.00× | 23,040 |
| **2,042,040** | **2,308** | **885** | **1.62×** | 368,640 |
| 4,084,080 | 4,616 | 885 | 1.62× | 737,280 |
| 38,798,760 | 23,887 | 1,624 | 2.97× | 6,635,520 |

Measured, `[10¹², 1.5·10¹²)`, 8 threads, three repeats each. All four rows measured
together, so the ratios are sound:

| configuration | rep1 | rep2 | rep3 | mean | vs baseline |
|---|---|---|---|---|---|
| M = 120,120, `mmax 2000` | 1.644 | 1.656 | 1.661 | 1.654 s | 1.00× |
| M = 2,042,040, `mmax 2000` | 1.137 | 1.145 | 1.146 | 1.143 s | 1.447× |
| M = 2,042,040, pruned (4,163 filters) | 0.900 | 0.902 | 0.907 | **0.903 s** | **1.83×** |

The re-pruned table for this wheel is worth a further **1.266×** on top of the wheel, and
**1.83× in total** against the previous configuration.

**1.447×**, against 1.62× fewer positions — the gap is the surviving lanes being
pre-filtered harder, so marks and survivor density *per position* both rise slightly.
This is measured on a **short** range, where the per-lane setup tax (one extended-Euclid
per filter per lane, ~1.0 s at 2,308 lanes against ~0.1 s at 220) is at its worst, so
**1.447× is a lower bound on the production figure**; at 10¹⁷ there are 4.9·10¹⁰
positions per lane and the tax vanishes.

38,798,760 was rejected as the default despite its 2.97×: the kill-witness table has one
row per unit class, so it would be ~250 MB against 14 MB here, and layer L1's value is
being a finite object a reader can check line by line. It remains reachable via
`--wheel`.

## 3. Segment size

`--seg` is a pure cache knob; the gate proves it cannot change arithmetic.

| `--seg` | bitset | time |
|---|---|---|
| 4,096 | 0.5 KB | 12.44 s |
| 16,384 | 2 KB | 8.50 s |
| 65,536 | 8 KB | 7.40 s |
| 131,072 | 16 KB | 7.11 s |
| **262,144** | **32 KB** | **6.97 s** ← default |
| 524,288 | 64 KB | 7.61 s |
| 1,048,576 | 128 KB | 8.36 s |
| 4,194,304 | 512 KB | 8.59 s |

The optimum is a 32 KB bitset, exactly as the spec predicted. The small-`seg` end
confirms the cost model's *structure* too: at `seg = 4096` with 8,741 filters the
`ncov/SEG` bookkeeping term is 2.13 ops per position and swamps the marking.

## 4. Threads

| threads | time | speedup |
|---|---|---|
| 1 | 6.956 s | — |
| 2 | 3.756 s | 1.85× |
| 4 | 2.199 s | 3.16× |
| 8 | 2.223 s | 3.13× |

**SMT buys nothing**, and the obvious explanation is wrong. Two hyperthreads per core
each holding a 32 KB bitset against a 48 KB L1 would predict that halving `--seg`
unlocks it; measured, it does not (8 threads at `seg = 131072` gives 2.166 s against 4
threads at 2.189 s, and shrinking further is worse). The plateau is the physical-core
limit. **Project with physical cores, at ~79% efficiency.**

## 5. Engine vs the frozen reference

| | ns/mark | note |
|---|---|---|
| `cover_scan` | 0.68 | flat across `--spanlog` 20/22/24 — 17.83/17.70/17.56 s |
| `rung_scan3` | 0.51 | 1.33× |

Two things this retires. `cover_scan` was **already near-optimal at marking** — a
strided bit-clear is a dependent load-modify-store and 0.5–0.7 ns is close to the floor,
so the ~8× the spec expected from cache-resident segments was never there. And
**setup amortization buys nothing**: `cover_scan` is flat in `--spanlog`, so the
"extended Euclid once per lane per shard instead of once per window" argument — the
centrepiece of spec §3.2 — is not where time goes. The global-`J` indexing is kept
because it is simpler and is the right shape for lane-major parallelism, shard-resume
and the GPU, but no speed is budgeted to it.

## 6. Pruning

Candidates: 8,741 F1 covers (`m ≤ 2000`, `m ∤ M`) + 5,417 F2 stage-B certificates
(`L ∤ M`) = 14,158. Of these, 6,401 hit at least one lane.

Measured together, three repeats each, so the ratios are sound:

| filter set | filters | marks/pos | wall (8 threads), mean of 3 |
|---|---|---|---|
| unpruned | 14,158 | 14.93 | 2.645 s |
| **published, corrected domain** | **4,533** (4,344 F1 + 189 F2) | **8.17** | **2.146 s** |
| previous table, unanchored domain | 4,585 (4,385 F1 + 200 F2) | 8.75 | 2.153 s |
| greedy, placeholder constants | 226 | 4.00 | 12.611 s (earlier session) |

The two 4,500-filter tables are **identical within noise**, so the 6× domain error did
not materially change the chosen table — but the calibration behind it is now honest.

The published table is the greedy's own stopping point; forcing it further with a
density constraint only adds poor-ratio filters and costs time. Repeat runs of the same
configuration vary by ~7% on this shared VM, so the top two rows are within noise of
each other and the choice between them is not load-bearing.

**Pruning is worth 1.23–1.27×, not the 3–4× the spec hoped**, and the reason is §1: the
survivor term dominates, so the marks you are allowed to give up are limited. Both
pruned tables are verified correct — `survivors=0`, every emitted certificate re-derived
in exact rationals, and no prime certified by the full set is lost.

**F2 stage-B certificates earn their place**: 200 of the 4,585 chosen filters are F2.
They entered as candidates, not assumptions, and the cost model kept them.

### The evaluation domain must be anchored at a real scan floor

`prune_filters.py` originally indexed positions from `J = 0`, i.e. from `p = ρ`, on the
theory that a filter's hit pattern depends only on `(ρ mod mod, J mod step)` and so any
window is as representative as another. That is true of **marks per position** — just
`Σ npos/step`, and it measured correct — and false of the union's **survivor density**,
which is phase-independent only over `lcm(steps)`, astronomically larger than 2¹⁶.
Anchoring at a tiny `p` aligned every filter's phase against it:

| domain | marks/pos | survivor density |
|---|---|---|
| `J` from 0 | 8.75 | 9.40·10⁻⁴ |
| `J` from `LO_REF = 10¹²` | 8.75 | 1.58·10⁻⁴ |
| scanner, measured | — | 1.48·10⁻⁴ |

`LO_REF` is now a published parameter in the table header.

### Nesting checks only apply between genuinely nested filter sets

A pruned table is *not* automatically a subset of the set it was pruned from. The
`M = 2,042,040` table keeps 157 F2 stage-B certificates, which the F1-only `--mmax 2000`
set does not contain — so it kills positions that set cannot, and three primes certified
by stage D in the `mmax 2000` run never reach stage D in the pruned run (they are killed
by F2 filters `(3132, 901)` and `(636, 421)`). That looks exactly like a nesting
violation and is not one.

The invariant that always holds is the weaker and more useful one: **both runs end at
`survivors = 0`, and every emitted certificate verifies in exact rationals.** Check that.
Reach for nesting only when one filter set is literally a subset of the other.

### Two algorithm errors worth not repeating

- **Unique-kill scoring is the wrong metric here.** Scoring each filter by the positions
  only it kills, and dropping those that do not pay, dropped all 14,158 at once and took
  survivor density to 1.0. With ~15 overlapping covers per position almost nothing is
  killed by exactly one filter, so every individual score is ≈0 even though the set is
  collectively essential. Marginal analysis cannot see that. Forward greedy set cover
  can.
- **Order by gain/cost, not by gain.** Raw-gain ordering picks expensive small-stride
  filters first and then stops when the highest-*gain* filter fails its cost test, while
  far cheaper filters would still pay. That produced a table 2.3× slower than doing
  nothing.

### The publication constraint binds harder than the cost model

Pruning does not lose primes — it **moves** them from being covered by a progression
(self-certifying, no output) to being certified individually by stage D. So the
certificate stream grows as the filter set shrinks:

| filter set | certificates over `[10¹², 10¹²+2·10⁸)` | per integer | projected at 10¹⁷ |
|---|---|---|---|
| unpruned | 0 | — | — |
| 4,585 | 13 | 6.5·10⁻⁸ | 6.5·10⁹ |
| 226 | 841 | 4.2·10⁻⁶ | 4.2·10¹¹ |

The spec's publication budget is **10⁻¹¹ per integer** (≤10⁶ certificates at 10¹⁷,
~40 MB). Every table above misses it by two to five orders of magnitude. This is a real
open decision for P4, not a bug: either raise `mmax` (the design doc's `mmax = 10⁴`
measured zero prime survivors over 1.2·10¹⁰), or accept a large certificate stream with
per-shard gzip and a published digest. **The certificates are the valuable output**, so
the second option is not obviously worse — but it must be chosen deliberately.

## 7. Projections — labelled as such

At the published configuration on 4 physical cores, M = 2,042,040: 1.143 s for `5·10¹¹`
integers = **2.3·10⁻¹² s per integer** with the generated `mmax = 2000` filter set, and
**1.8·10⁻¹² s per integer** with the pruned table (0.903 s).

Scaling by core count only — same `c4` family, so per-core speed should carry, but this
is a projection until it runs on the real box:

With the pruned table (1.81·10⁻¹² s/integer):

| | 4 cores (measured box) | 12 physical cores (projected) |
|---|---|---|
| 10¹⁵ | 30 min | 10 min |
| 10¹⁶ | 5.0 h | 1.7 h |
| 10¹⁷ | **2.1 d** | **17 h** |
| 10¹⁸ | 21 d | **7.0 d** |

These are *lower bounds* in one specific sense: the per-lane setup tax that inflates the
short benchmark disappears at production range, and the pruned table for this wheel is
not included.

The spec's §0 goal — `[0, 10¹⁷)` on CPU — is comfortably in reach.

**Height dependence is not measured yet.** Cost per integer should be nearly flat, since
this scanner never enumerates primes and the only height-dependent term is the `mulmod`
cost inside the survivor tests, but that is a prediction. Measure it before quoting it;
the standing rule that a cost model must not be carried across a change of regime has
now been paid for twice in this project.

---

## 8. The CUDA port

Measured on `cloud-gpu` (`g2-standard-12` + one NVIDIA L4, `sm_89`, CUDA 12.9), 2026-08-17.
**Every number in this section is from that one session on that one box**, and every ratio
names its control. Do not compare these against the `cloud-dev` numbers in §1–§7: that is a
different machine, and this box's CPU is much the weaker of the two.

### 8.1 What moved and what did not

Marking, survivor compaction and primality run on the device. **Stages D and E stay on the
host**, multi-threaded and overlapped with the next round's device work. That split is the
single fact the rest of this section follows from.

### 8.2 Segment and block size

One segment per lane, all lanes at once, `M = 2042040`, `mmax = 10000`, best of 5, Gpos/s:

| `--seg` | 32768 | 65536 | 131072 | 262144 | 393216 | 524288 |
|---|---|---|---|---|---|---|
| (block 256) | 0.94 | 1.58 | 2.30 | 2.26 | — | 0.94 |

| `--block` | 128 | 256 | 384 | 512 | 768 | 1024 |
|---|---|---|---|---|---|---|
| (seg 262144) | 1.54 | 2.26 | 2.45 | **2.51** | 2.17 | 1.40 |

Defaults: `--seg 262144 --block 512`. The segment optimum coincides with the CPU's for an
unrelated reason — the CPU's is L1 residency, the device's is amortizing the per-segment
fixed cost against shared-memory occupancy. 524288 needs 64 KB of shared memory per block,
collapses to one block per SM, and is the cliff on the right.

### 8.3 The offset setup was 79% of the port, and it is a host cost

The first `--profile` of the finished port, `[10¹², 1.5·10¹²)` at the default wheel:

```
wall 4.56 s  =  offsets 3.60  +  device 0.22  +  host tail 0.001
```

`sweep_lane` computes, per (lane, filter), `g = gcd(M,mod)`, `s = mod/g` and
`inv3((M/g) % s, s)`. At production settings that is 2,308 × 64,708 = **149 million
extended Euclids** — and none of them depends on the lane. Only `rhs` does. Hoisting them
out, and then rewriting `first % mod` as `(lo % mod + (first − lo)) % mod` so the remainder
is 32-bit:

| | offsets | wall |
|---|---|---|
| as written | 3.60 | 4.67 |
| extended Euclid hoisted | 1.26 | 2.33 |
| + 32-bit remainder | **1.05** | **2.13** |
| same-box CPU, 12 threads | — | 5.11 |

`--check-mark` computes both the naive and the hoisted form and compares them, so this is a
gated hoist and not a hopeful one. **`rung_scan3` has the identical loop-invariant and does
not exploit it**, so part of the 2.40× above is a host optimization the CPU could also take.

### 8.4 Device cost constants

Fitted from four filter sets at `[10¹², 1.5·10¹²)`, solving `t = t₀ + marks·c_m +
survivors·c_s` by least squares. All four rows reproduce within 2%:

| `mmax` | marks | survivors | measured | model |
|---|---|---|---|---|
| 500 | 3.71·10⁹ | 1,047,744 | 0.155 s | 0.155 |
| 1000 | 5.43·10⁹ | 118,591 | 0.139 | 0.137 |
| 3000 | 8.93·10⁹ | 41,071 | 0.169 | 0.172 |
| 10000 | 1.40·10¹⁰ | 40,573 | 0.229 | 0.228 |

**`c_m` = 0.01095 ns/mark, `c_s` = 39.8 ns/survivor on the device, `t₀` = 73 ms.** Against
the CPU's 0.76 and 4,545: marking got **69× cheaper**, device-side survivor handling **114×**.

There is a third term the CPU model has no equivalent of. The offset setup is **7.03 ns per
(lane, filter) per shard** — per filter, 2,308 × 7.03 ns ≈ 16 µs of setup no matter how few
positions it ever marks. It amortizes with *shard width*, not with segment count.

### 8.5 The small-prime sieve prediction is REFUTED

The spec predicted stage B′ would invert on the device, where it is a 5% loss on the CPU.
`[10¹², 1.5·10¹²)`, device time:

| `--sieve` | 0 | 100 | 1000 | 10000 |
|---|---|---|---|---|
| device | **0.229 s** | 0.248 | 0.254 | 0.252 |
| survivors | 40,573 | 27,127 | 16,441 | 16,175 |

It works exactly as designed — 60% of survivors removed — and it is a **~10% loss**, not a
win. The measured constants say why: the added marks cost 6.7 ms and the removed survivors
save 1.0 ms. Stage B′ can only remove *composites*, and on the device a composite survivor
costs 39.8 ns. The prediction's premise — that emulated modexp makes survivors dear
relative to marking — is simply false here: Miller–Rabin is embarrassingly parallel with no
memory traffic and maps beautifully to the hardware, while irregular `atomicAnd` into shared
memory is where the device's advantage is *smaller*.

### 8.6 The filter set: the device wants a LARGER one, for a reason the spec did not give

`[10¹², 6·10¹²)`, all rows this session. `rung` is the count of surviving **primes**, which
is what the host tail actually costs:

| filter set | marks | survivors | primes | device | host tail | wall |
|---|---|---|---|---|---|---|
| `--mmax 1000` | 5.43·10¹⁰ | 1,050,258 | 126,508 | 0.06 | 1.33 | 2.11 s |
| `--mmax 1500` | 6.62·10¹⁰ | 404,334 | 22,322 | 1.09 | 0.25 | **2.05 s** |
| `--mmax 2000` | 7.53·10¹⁰ | 299,589 | 5,773 | 1.32 | 0.07 | 2.17 s |
| `--mmax 3000` | 8.93·10¹⁰ | 266,697 | 732 | 1.51 | 0.02 | 2.45 s |
| **published pruned table, 4,163 filters** | 4.59·10¹⁰ | 1,242,946 | 229,299 | 0.01 | 2.37 | **3.03 s** |

Three things to take from this table.

**The CPU-pruned table is the worst row on the device — 48% slower than plain
`--mmax 1500`.** It was chosen where a mark costs 0.76 ns, so it trades marks away for
survivors aggressively. On the device a mark costs 0.011 ns and that trade is badly wrong.

**The host tail is linear in PRIMES, not survivors**: 126,508 primes → 1.328 s and 229,299 →
2.371 s, both **10.35 µs of wall per prime** on 12 threads. So a survivor's true cost on this
port is 39.8 ns on the device plus, if it is prime, 10.35 µs on the host — and the survivor:
mark ratio is **177,600 against the CPU's 5,980, thirty times higher.** The spec's conclusion
that the device wants a larger set is *confirmed*; its stated mechanism is not. Marking got
69× cheaper while the part of a survivor that dominates never left the CPU at all.

**A note on the prime/composite split, and a correction.** Going `mmax` 1000 → 3000 cuts
survivors 3.9× but primes **173×** — past a certain coverage depth the survivors are
essentially the perfect squares, which covers structurally cannot touch (a cover
`m = 4acd−1` kills `p ≡ −4a²d`, and `−4a²d` is a square mod `m` only when `−d` is a
quadratic residue there). An earlier revision of this section concluded from that
`prune_filters.py` was mis-scoring, and that the fix was to score candidates on surviving
primes rather than surviving positions. **That was wrong, and §8.8 records the
measurements that settled it.** The collapse is real but lives far deeper than any pruned
table reaches; in the pruner's own operating regime the surviving prime fraction is flat
at ~0.186. The reason the CPU-pruned table is bad on the device is the plain one above —
a mark costs 0.011 ns there instead of 0.76.

### 8.7 End to end, against a same-box control

`[10¹², 6·10¹²)`, `rung_scan3 --threads 12` on the same box, same session:

| | CPU ×12 | L4 | ratio |
|---|---|---|---|
| `--mmax 1000` | 10.15 s | 2.16 s | 4.69× |
| `--mmax 10000` | 24.62 s | 4.18 s | 5.89× |
| published pruned table | 10.42 s | 3.16 s | 3.29× |
| each at its own best | 10.15 s | 2.05 s | **4.95×** |

**About 5×, and the control is stated: 12 threads of a `g2-standard-12`, not `cloud-scan`.**
That box's cores are much stronger, so this ratio would be smaller there. Note also that the
GPU rows carry the §8.3 host hoist and the CPU rows do not.

The port is **tail-bound, not device-bound**, at every filter set that is any good — which
is the opposite of the failure mode the plan was written to avoid, and means the next
worthwhile work is on stage D, not on the kernel.

### 8.8 Scoring the pruner on surviving primes — tried, measured, rejected

§8.6 originally argued that `prune_filters.py` should credit each filter with the
surviving **primes** it kills rather than the positions, since a prime survivor costs
28× a composite on the CPU and 260× on the device. The machinery to do it exactly is
cheap — a position is `first + M·J`, so "q divides it" is the same congruence the filter
marking already computes, and sieving the domain by every prime up to `isqrt(max
position)` is exact, 10 s at `M = 120120` and ~2 min at `M = 2042040`. It is implemented
(`prime_masks`, verified position by position against Miller–Rabin) and the result is
unambiguous:

| | prime density of survivors, training domain (`lo = 10¹²`) | unseen domain (`lo = 2·10¹²`) |
|---|---|---|
| published table / shared fraction | 17.2% | 18.7% |
| scored on per-filter primes | **6.0%** | **13.9%** |

**The 6.0% is fitted, not real**, and on the unseen domain the "improved" table is
*worse* than the published one. Confirmed on the scanner over `[10¹², 1.5·10¹²)`: the
per-filter table leaves **50,060** surviving primes against the published table's
**39,298**, for no change in wall at all — 1.183 s vs 1.186 s on `cloud-dev`, 8 threads,
mean of 3.

The mechanism is worth keeping, because it is a trap the domain's own design invites. A
filter's **hit set** is phase-structured, which is exactly why an anchored window predicts
survivor density at all (§6). **Which of those hits are prime is not a congruence
property**, so per-filter prime credit describes this stretch of the number line and no
other. The greedy then takes the best of 6,401 candidates 4,533 times over — a winner's
curse that compounds, and that bites hardest at the end, where `remaining` is small, the
per-filter prime counts are tiny, and the stopping decision is actually made.

What survives is the weaker, sound form: weight position kills by the **aggregate** prime
fraction of the live domain. Being identical for every candidate it cannot bias the choice
between filters; it moves only where the greedy stops. Measured, that fraction is **flat
at ~0.186** across the whole operating regime, so it **reproduces the published table byte
for byte** — 4,533 filters, 4,344 F1 + 189 F2, 8.17 marks/pos — and the old guessed 0.171
was right to within 8%. The change is therefore not a speedup; it replaces an assumed
constant with a measured one and makes the pruner track the collapse at depths where it
does happen.

### 8.9 The device-tuned filter table — `tables/filters/f_M2042040_mmax2000_gpu.txt`

Produced by `prune_filters.py --wheel 2042040 --mmax 2000 --device`, with the §8.4
constants and the live prime fraction of §8.8. **4,231 filters (4,117 F1 + 114 F2)**,
9.52 marks/pos, survivor density 8.71·10⁻⁵, prime density 6.25·10⁻⁶.

Gated first, measured second. `diff_cuda.sh` at `--seg 4096` over `[10¹², 1.05·10¹²)`:
byte-identical `SUMMARY` against `rung_scan3`, identical certificate set, byte-identical
repeated GPU runs, `survivors=0`. It emits **374** certificates where the CPU-tuned table
emits **2,208** — 5.9× less stage-D work — and that row is now in `make -C sieve/cuda check`.

`[10¹², 6·10¹²)` on the L4, best of 3, all rows the same session:

| filter set | marks | survivors | primes | offsets | device | host tail | wall |
|---|---|---|---|---|---|---|---|
| **device-tuned, 4,231** | 5.38·10¹⁰ | 392,243 | 44,475 | 0.07 | 0.67 | 0.48 | **1.883 s** |
| `--mmax 1500` (§8.6's best) | 6.62·10¹⁰ | 404,334 | 22,322 | 0.10 | 1.10 | 0.25 | 2.108 s |
| CPU-tuned, 4,163 | 4.59·10¹⁰ | 1,242,946 | 229,299 | 0.07 | 0.01 | 2.37 | 3.088 s |

**1.64× over the CPU-tuned table and 1.12× over the best hand-picked `--mmax`.** The
interesting column is not `wall` but the pair beside it: the device table is the only one
that BALANCES the two overlapped resources (0.67 device against 0.48 tail). `--mmax 1500`
is device-heavy, the CPU-tuned table is tail-heavy by 200×. Nothing in the additive cost
model asks for balance — it falls out of pricing a survivor at what it actually costs on
this port, and it is the strongest evidence that the §8.4 constants are right.

End to end against the same-box control, same session: `rung_scan3 --threads 12` takes
**9.203 s** with this table and 10.422 s with the CPU-tuned one, so the L4 is **4.89×**
its host — with the §8.3 caveat that the GPU side carries the host hoist and the CPU side
does not.

**The CPU table is not wrong; it is right for a different machine.** On `cloud-dev`, the
`c4-highcpu-8` its constants were measured on, over `[10¹², 2·10¹²)` at 8 threads, the two
tables tie and the CPU-tuned one is 1.6% ahead (1.554 s against 1.579 s, mean of 3). The
device table wins on the `g2-standard-12`'s CPU too (9.20 s against 10.42 s) because that
box's cores are much weaker, so a stage-D prime costs relatively more there and the optimum
moves toward more filters. The optimum is a property of the machine, which is the whole
reason `--device` exists rather than a single published table.

### 8.10 The height curve — measured, and it is FLAT

`sieve/cuda/tests/height_sweep.sh`, fixed span of 5·10¹² integers at each height so the
rows are directly comparable, device table, one L4:

| lo | wall | work | startup | ns/int (work) | survivors | primes | µs/prime |
|---|---|---|---|---|---|---|---|
| 10¹² | 1.914 | 1.360 | 0.554 | 0.00027 | 392,243 | 44,475 | 10.9 |
| 10¹³ | 1.923 | 1.368 | 0.555 | 0.00027 | 291,939 | 43,173 | 12.5 |
| 10¹⁴ | 2.270 | 1.387 | 0.883 | 0.00028 | 228,656 | 40,942 | 15.3 |
| 10¹⁵ | 3.536 | 1.390 | 2.146 | 0.00028 | 205,733 | 38,154 | 18.7 |
| 10¹⁶ | 7.720 | 1.404 | 6.316 | 0.00028 | 197,999 | 35,550 | 22.8 |

**Work grows 3.0% across four decades.** `wall` grows 4×, and every bit of that growth is
per-process startup: `build_base_primes` sieves to `√hi`, which is 2.4·10⁶ at 10¹² and
10⁸ at 10¹⁶. A run long enough to matter amortises it away completely.

Two effects cancel, which is why the work is flat and not merely slowly growing:

- **Survivor density falls with height** — 392,243 → 197,999 over the same span — so the
  device half gets *cheaper* (`device-wait` 0.668 → 0.384 s).
- **Stage D costs more per prime** — 10.9 → 22.8 µs, i.e. **1.20× per decade** — so the
  host tail gets dearer (0.486 → 0.811 s), while the primes reaching it thin out as
  `1/ln p`.

**Do not carry the census port's 1.17×/decade onto this one.** That factor describes a
different product with a different bottleneck. Here the correct figure for cost per
integer is ~1.00×/decade, and it was ninety seconds of measurement.

### 8.11 A verification run to 10¹⁵

`[255255, 10¹⁵)` — 255,254 is the largest `pmin` in the filter table, and below it the
range is settled by the existing censuses. Twenty shards, device table, one L4,
**4.9 minutes wall** (13.7 s to 15.4 s per shard, rising with height exactly as §8.10
predicts).

| | |
|---|---|
| positions | 1,130,242,306,420 |
| covered by certified covers | 1,130,198,943,629 (99.99616%) |
| reaching primality | 43,362,791 |
| composite | 35,520,569 |
| **primes, all solved at stage D** | **7,842,222** |
| stage E (`direct`) | 0 — never needed |
| **survivors** | **0** |

What that establishes: every prime in `[255255, 10¹⁵)` lying in one of the 2,308 hard
classes mod 2,042,040 is certified solvable, and every other unit class is killed outright
by the class table (layer L1, itself pinned and independently re-derived). Primes below
255,255 are settled by the existing censuses. **This is not a new bound on
Erdős–Straus** — 10¹⁸ stands by other means — it is this pipeline verified end to end at
a scale worth quoting.

How it was checked, rather than asserted:

- **Tiling exact.** The twenty shards chain `lo → hi` with no gap or overlap from 255,255
  to 10¹⁵, checked on the boundaries and not just on summed counters (§ the shard gate).
- **`survivors = 0` in every shard**, so nothing fell through unresolved.
- **Every one of the 7,842,222 certificates re-derived in exact rational arithmetic** by
  `sieve/verify_covers.py` — 0 bad — **on the laptop, a different machine than produced
  them**, which is the standing rule for this project.
- **Diffed against the independent CPU scanner at 10¹⁵**, not only at 10¹², over
  `[9.99·10¹⁴, 10¹⁵)`: byte-identical `SUMMARY`, identical certificate set, 7,690
  certificates. That row is now permanent in `make -C sieve/cuda check`.

The height curve predicted this run to within 4% (307 s predicted, 296 s actual), which
is the first end-to-end confirmation that the model in §8.4 and §8.10 extrapolates.

### 8.12 Projection to 10¹⁹ — now with a measured height curve under it

At the §8.10 rate, extrapolating stage D at its measured 1.20×/decade and the prime count
at `1/ln p` three decades past the last measurement:

**~39 days on one L4** (2.8–3.4·10⁻¹³ s per integer × 10¹⁹). Per-shard startup grows to
~200 s at `√10¹⁹ = 3.16·10⁹` and stays negligible against that, though it needs ~3.2 GB
for the bytemap plus ~2.4 GB of prime and inverse tables.

Two things stand in the way that are not speed:

- ~~10¹⁹ exceeds 2⁶³.~~ **Fixed 2026-08-18 — see §8.13.** The ceiling is now
  1.8446744·10¹⁹, comfortably above 10¹⁹.
- `GPUS_ALL_REGIONS` is 1, so this is one L4 and cannot be spread across GPUs without a
  quota increase.

The run would also want its filter table re-pruned at height: at 10¹⁹ the tail is ~70% of
the work, so the optimum moves further toward more filters than the 10¹²-tuned table has.

### 8.13 The stage-E ceiling was 2⁶³ and should have been 2× that

`solve_direct3` guarded with `if (nn >> 63) continue`, silently skipping stage E for
`n = p + 4a²d ≥ 2⁶³`. **2⁶³ was a guess, and it threw away exactly half the usable range.**

The real bound is a property of the frozen reference, not a choice. `isqrt_u64` ends with
`while ((s + 1) * (s + 1) <= n) ++s` in `u64`: once `s` reaches `2³² − 1`, `(s+1)²` is
`2⁶⁴`, which wraps to 0, `0 <= n` is true, and the loop runs away. `s` reaches `2³² − 1`
exactly when `n` reaches `(2³² − 1)²`. `factorA` needs `isqrt_u64` through `factor_hard`'s
perfect-square test, and `census/rung_scan.cpp` cannot be edited, so the bound is
inherited:

    DIRECT_NMAX = (2³² − 1)² = 18,446,744,065,119,617,025
    DIRECT_PMAX = DIRECT_NMAX − 2049      (a, d ≤ 8 ⟹ n ≤ p + 2048)
                = 18,446,744,065,119,614,976  ≈ 1.8446744·10¹⁹

Everything else on the path clears that comfortably: `mulmod` and `(u128)c*k` are
128-bit, `m + 1` is safe because `m | n`, and `pollard_brent`'s `mulmod(v,v,n) + c` has
`2³³ − 1` of headroom for a `c` that never exceeds a handful.

**The silent skip was the actual bug, not the bound.** Above the ceiling stage E declined
without saying so, and a prime stage D missed was then reported as a `SURVIVOR` — which
reads as a mathematical result when it is a numeric limit. Both scanners now **refuse to
start** above `DIRECT_PMAX`, symmetric to the existing `pmin` floor refusal.

**This diverges from the frozen `sieve/cover_scan.cpp`, deliberately and only above 2⁶³**,
where the reference keeps the old guard because it cannot be edited. The two remain
byte-identical everywhere below — every range `diff3.sh` runs, and every range
`cover_scan` is fast enough to run at all — and above it `rung_scan3` is strictly *more*
complete. That is a safe direction: declining is incompleteness, never unsoundness.

#### Stage E had never been executed, by anything

`direct = 0` across 1.13·10¹² positions to 10¹⁵ (§8.11), because stage D resolves
essentially everything that survives the covers. So the path carrying the bug had no
coverage at all, and no scan could give it any. `rung_scan3 --check-direct LO N` calls it
directly, and `make check-direct` runs four rows — below 2⁶³, at 2⁶³, near 10¹⁹, and just
under the ceiling — piping the certificates into `verify_covers.py`:

    ok  check-direct 20/20 primes at or above 9223372036854770000  solved by stage E
    ok  check-direct 20/20 primes at or above 9223372036854775808  solved by stage E
    ok  check-direct 20/20 primes at or above 9999999999999990000  solved by stage E
    ok  check-direct 20/20 primes at or above 18446744065119000000 solved by stage E
    verified: 0 RUNG, 80 SOLVED; 0 bad, 0 survivors

Those are the first `SOLVED` certificates this project has produced from a scanner, and
they are re-derived in exact rational arithmetic on a different machine. With the old
guard restored the same probe reports `0/20` and names every declined prime, so the gate
is sensitive to exactly the bug it exists for.

### 8.14 Profiling the port: `k_mark` is 99% of the GPU, and its words were twice too wide

`nsys` on a production-config run, GPU kernel time:

| kernel | share |
|---|---|
| `k_mark` | **99.1%** |
| `k_primality` | 0.6% |
| `k_compact` | 0.3% |
| `k_scan_counts`, `k_seg_n` | 0.0% |

Host-to-device copies are 16 ms of one-off offset upload; device-to-host is 51 µs total.
So "the device half" *is* marking, and nothing else on the GPU is worth looking at.

`ncu` on `k_mark` said it was not stalled on anything obvious — DRAM throughput **0.55%**
(everything is in shared memory), warps active **98.1%** (occupancy is not the problem) —
but SM throughput only **52.6%**, at **8.02·10⁹ instructions for 5.87·10⁸ shared atomics**,
i.e. ~13.7 instructions per mark, with **3.22·10⁸ bank conflicts** — 0.55 per atomic.

That points at the word width, and a microbenchmark in exactly this access pattern
settled it before anything was touched:

| | block 256 | block 512 | block 1024 |
|---|---|---|---|
| 64-bit `atomicAnd` | 90.5 ms | 81.9 ms | 140.0 ms |
| **32-bit `atomicAnd`** | **36.2 ms** | 44.4 ms | 72.2 ms |

**2.27×**, identical results. Two reasons compound: a 64-bit shared access spans two banks
so it conflicts twice as often, and halving the word halves the segment's shared
footprint, which lifts occupancy.

Converted throughout — shared bitset, copy-out, compaction and the host replay. The gate
is what makes that safe: `--check-mark` compares the bitset bit for bit against the CPU
scanner, `--check-compact` compares the candidate list element for element, and all four
diff rows plus three shard rows stay byte-identical.

**Marking: 2.51 → 6.51 Gpos/s (2.59×).** `--seg 262144 --block 512` remains optimal.

End-to-end, work per 5·10¹² integers (startup excluded):

| height | before | after | |
|---|---|---|---|
| 10¹² | 1.360 | **0.576** | 2.36× |
| 10¹⁴ | 1.387 | **0.714** | 1.94× |
| 10¹⁶ | 1.404 | **0.897** | 1.57× |

The gain shrinks with height because the host tail is unchanged and now dominates — and
that is the important structural change. **The port is no longer device-bound.** At 10¹⁶
with 12 threads `device-wait` is **0.003 s** against a tail of 0.788: the GPU is entirely
hidden. The cost curve is also no longer flat (§8.10) — it is ~1.12×/decade now, because
what remains is stage D, which grows.

#### What that makes the next lever

Host cores, which were worth nothing before this change and are worth a lot now. The tail
scales cleanly: 1.920 s at 4 threads, 1.061 at 8, 0.788 at 12. **`g2-standard-32` carries
the same single L4 with 32 vCPU instead of 12** — 16 physical cores against 6 — which
should take the tail to ~0.35 and make the device the pole again at ~0.48, for roughly
another 1.5×. It exactly consumes `CPUS_ALL_REGIONS = 32`, so nothing else may run.

After that the filter table wants re-pruning again: marking got 2.6× cheaper, so the
device constants in §8.4 are stale and the balance has moved toward affording *more*
filters, which buys fewer surviving primes and hence less tail.

**Revised 10¹⁹ projection: ~29 days on the current box, and ~18 days on a
`g2-standard-32` with a re-pruned table** — against 39 days before this section.

### 8.15 The fast host tail: Montgomery + magic-inverse, gate-first

§8.14's perf profile said the tail was 48.5% `__umodti3` (software 128-bit division under
every `mulmod`, emitted because gcc cannot prove the quotient of `(u128)a*b % m` fits
64 bits) and 25.3% hardware-divide trial division. Microbenchmarks on the box: Montgomery
multiplication **9.1×** the frozen `mulmod` (3.83 vs 34.72 ns), magic-inverse
divisibility **19×** the `n % sp` loop (0.45 vs 8.71 ns/prime).

The factorization is reimplemented in `sieve/rung_scan3.cpp` — the frozen
`census/rung_scan.cpp` untouched, its `factorA` retained as the reference — and swapped
under stages D and E. Output-invariance is by construction (a factorization is unique;
`min_div_res` takes an order-independent minimum; `divisors3` sorts), and by gates:

- **`make check-factor`, built before the implementation**: canonical factorizations
  diffed against the frozen `factorA` on 32,512 adversarial values — hard semiprimes both
  sides of 2⁶³, prime squares, smooth numbers, the `TD_BOUND²` and `DIRECT_NMAX`
  boundaries — plus factor-primality and product-reconstruction checks that are
  independent of the frozen comparison. Its first run caught a real setup bug (the frozen
  path mis-factors when `g_small` is unpopulated — an implicit dependency the fast path's
  self-contained tables deliberately do not have). Fault-injected both ways: a corrupted
  trial-division inverse fails with 1,650 mismatches; an unconverged Montgomery inverse
  hangs Pollard, rc 124. A `lim±1` fault is NOT caught — hit probability ~2⁻⁶⁴ — which is
  worth recording as the gate's known blind spot.
- `make check-diff3`: byte-identity against the frozen `cover_scan` end to end.
- `make check-direct`: 80 stage-E certificates either side of 2⁶³, exact rationals.
- The full `sieve/cuda` gate, all rows.

Also fixed while there: the fast path uses an `isqrt` that is safe for every u64 (the
frozen one's `(s+1)²` probe wraps at `(2³²−1)²` — unreachable below `DIRECT_NMAX` only by
the accident that 3 divides `2³²−1`), and `solve_rung3`'s certificate buffer was one
worst-case digit too small (`head[64]` → 80; benign at our bounds, `-Wformat-truncation`).

Measured on the L4 box, same session:

| | before (§8.14) | after | |
|---|---|---|---|
| tail, 10¹⁶, 1 thread | 7.478 s | **1.164 s** | 6.4× |
| tail, 10¹⁶, 12 threads | 0.788 s | **0.154 s** | 5.1× |
| µs per prime (12 threads) | 22.8 | **4.3** | 5.3× |
| work per 5·10¹² at 10¹⁶ | 0.897 s | **0.432 s** | 2.08× |
| `[10¹², 6·10¹²)` end to end | 1.883 s | **0.96 s** | 2.0× |

Same-box control, same session: `rung_scan3 --threads 12` at 10.33 s → the L4 is
**10.7×** its host. (The CPU scanner gains little from the fast tail — its cost is 91%
marking.)

**The height curve is flat again**: work is 0.430–0.432 s per 5·10¹² across all four
decades, because stage D — the only term that grew — is now a sliver. And the port is
back in balance: at 10¹⁶ with 12 threads, `device-wait` 0.172 vs tail 0.154. Neither
resource is the pole, which also means **`g2-standard-32` is no longer the next lever**;
re-pruning the filter table with re-measured constants is (both sides got cheaper, but by
different factors, so the §8.4 constants are doubly stale).

**10¹⁹ projection: 8.64·10⁻¹⁴ s per integer, flat → ~10.0 days on this box** — from 39
days two sections ago, via the 32-bit bitset (§8.14) and this.

### 8.16 Production hardening for the 10¹⁹ run

Three changes, each forced by arithmetic rather than taste:

- **Certificates are streamed, not buffered.** The driver used to hold every certificate
  in per-lane buffers and print after the last round — fine at 10¹⁵ (390k lines/shard),
  fatal at 10¹⁹, where a shard carries ~10⁸ lines and the buffers alone exceed the box's
  48 GB. Emission now happens per round in lane order: equally deterministic (reruns stay
  byte-identical), but round-major rather than the CPU scanner's lane-major, so
  `diff_cuda.sh` reports `emission order DIFFERENT` on multi-round rows — the sets still
  match, which is what the gate requires. The `data/sieve15` digests predate this and are
  set-comparable only (`sort | sha256sum`).
- **The certificate stream is hashed, not stored.** At 10¹⁹ it is ~5·10¹⁰ lines ≈ 3 TB
  against a 150 GB disk. Each shard's meta records the stream's sha256, its SUMMARY, any
  SURVIVOR lines verbatim, and provenance (binary commit, table digest); determinism makes
  any shard re-derivable and checkable later at the cost of re-running it (~1 h).
- **`tools/run19.sh`**: 250 shards over `[255255, 10¹⁹)` via the gated `--shard`
  arithmetic, ~1 h each against ~2 min average startup (the √hi base-prime sieve is 188 s
  at full height, measured; N=250 keeps it ~4%). Idempotent restart — a shard with a
  SUMMARY in its meta is never touched again — with 3 retries, FAILED markers that skip
  rather than stall, rc=2 (survivors) treated as a *result* to record loudly, never a
  crash, and completion judged by content (SUMMARY present), not exit code. Smoke-tested
  for run/skip/selective-resume, and its missing-table refusal exists because the smoke
  test watched it burn every shard's retry budget on exactly that.

Preflight at height: a real slice at `[10¹⁹ − 5·10¹⁰, 10¹⁹)` runs clean — `survivors=0`,
1,497 certificates, RSS 6.7 GB, startup 188 s.
