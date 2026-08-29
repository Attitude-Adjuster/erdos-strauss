# DESIGN — stage B′: the small-prime sieve for `rung_scan3`

Status: **design, approved**. Extends
`docs/design/2026-08-13-rung-scan3-design.md`; changes nothing about the
claim, the certificate families, or the class table.

---

## 0. Why

Measured on `cloud-dev` at the published 4,585-filter configuration, per lane position:

| | cost | share |
|---|---|---|
| marking (8.75 marks × 0.274 ns) | 2.40 ns | 39% |
| survivor tests (9.23·10⁻⁴ × 4,090 ns) | 3.78 ns | **61%** |

> **This table is WRONG and is kept only to show what the change was argued from.**
> Both inputs were bad: the survivor density came from `prune_filters.py`'s unanchored
> domain and was 6× too high, and `c_s` averaged prime and composite survivors whose
> costs differ ~28×. Measured properly (`SCALING_COVER.md` §1): marking is **90.9%**,
> stage D **7.8%**, composite survivors **1.3%**. The sieve could only ever touch that
> last 1.3% while adding ~10% more marking, so it never had a chance — which the
> measurement in §6 confirmed before this correction was understood.

**Survivors, not marking, are the bottleneck**, and about three-quarters of them are
composite. Each one currently costs a full base-2 SPRP (~800 ns) to reject — a very
expensive way to discover that a number is divisible by 7.

The cost constants come from solving `t = marks·c_m + survivors·c_s` across two filter
sets with very different balances (full `mmax = 2000`: 14.93 marks/pos, 8.571·10⁻⁴
density, 2.208 s; 226 filters: 4.00 marks/pos, 1.037·10⁻² density, 12.611 s, both over
`[10¹², 1.5·10¹²)` at 8 threads). The resulting model predicts 1.79 s for the published
table against 1.71–1.83 s measured, so it is trustworthy.

Note this corrects `SCALING_COVER.md`, which reports `c_m = 0.51 ns` from dividing total
time by marks — that silently attributes *all* runtime to marking. The **ratio**
(~14,900) is unchanged, so the pruning result stands; only the breakdown was wrong.

## 1. What it is

A small prime `p` kills lane position `J` exactly when `p | first + M·J`. That is the
same congruence shape as a cover with `res = 0`, so small primes drop into the existing
`mods`/`ress`/`off`/`step` machinery unchanged — offset setup, segment carry, and the
marking loop all work as they stand.

Primes dividing `M` are skipped: they can never divide a position coprime to `M`.

## 2. What it is NOT — the distinction the design turns on

A **cover** proves `4/p` is solvable. It is part of the certificate chain, and `covered`
is the counter the published claim rests on.

A **small prime** proves the position is not prime at all. It certifies nothing. It is
pure optimization bookkeeping.

Conflating the two would corrupt `covered`, break the `SUMMARY` contract with the frozen
`cover_scan`, and blur exactly the distinction the trust story depends on. So the sieve
is a **separate stage with its own counter**, applied strictly after the covers.

**Soundness:** the sieve can only clear genuinely composite positions, and a composite is
not a prime in need of a certificate. It cannot remove a prime, so it cannot weaken the
claim. The failure mode is not "wrong answer" but "wrong counter", which is what the
gate below is for.

## 3. Pipeline

Per segment, strictly sequential — never interleaved:

```
  fill bitset                     all positions alive
  mark covers        (stage B)    popcount ⇒ covered   [certified solvable]
  mark small primes  (stage B′)   popcount ⇒ sieved    [proven composite]
  scan survivors     (stage C)    SPRP → MR → D → E
```

Two popcounts per segment separate the counters exactly: ~2 operations per 262,144
positions, free.

Implementation: one `mods`/`ress` array with a boundary index `ncov`; entries `[0, ncov)`
are covers and `[ncov, end)` are small primes. The marking loop is unchanged; only the
popcount between the two ranges is new.

## 4. Interface

- `--sieve P` — sieve by primes `p ≤ P` with `p ∤ M`. `P = 0` disables the stage
  entirely, which is the mode the differential gate uses.
- `SUMMARY` gains one field, `sieved=`, after `covered=`.

Extending `SUMMARY` is deliberate here and not a repeat of the earlier mistake: this is
a genuinely new pipeline stage. But the field's **presence** breaks byte-identity with
`cover_scan` even when its value is zero, so `diff3.sh` strips `sieved=` before
comparing — the same precedent as `diff2.sh`'s `strip_timing`. Only the fields the two
implementations share are compared, and those stay byte-identical; the value of
`sieved` is checked separately by the sieve-on assertions in §5. The field is placed
after `covered=` so the two kill-counters read together.

## 5. Gating — both claims, separately

The sieve makes a *different* claim from the scanner, so it gets a different test.

- **`diff3.sh` keeps running at `--sieve 0`.** Byte-identical `SUMMARY` and byte-identical
  sorted certificate set against the frozen `cover_scan`, exactly as today. **Unweakened.**
- **A new check runs the same range with the sieve on** and asserts three things:
  1. the certificate set is **identical** to the `--sieve 0` run,
  2. `survivors = 0`,
  3. `covered` is **unchanged**.

Together those say the sieve removed only composites and touched nothing else. (1) is
what would fail if it ever removed a prime; (3) is what would fail if the phases were
interleaved and the counters cross-contaminated.

`--seg` invariance, shard tiling, and thread determinism all extend to the sieve for
free, since it uses the same marking machinery.

## 6. Result — MEASURED, and it is a LOSS

Implemented, gated, and measured on `cloud-dev` over `[10¹², 1.5·10¹²)` with the
published 4,585-filter table at 8 threads:

| `--sieve P` | wall |
|---|---|
| 0 (off) | **2.135 s** |
| 30 | 2.155 s |
| 100 | 2.202 s |
| 300 | 2.245 s |
| 1000 | 2.247 s |
| 3000 | 2.264 s |
| 10000 | 2.280 s |
| 30000 | 2.312 s |

**Monotonically slower. No bound wins.** The stage works exactly as designed — over
`[10¹², 1.01·10¹²)` it removes 1,275 of 2,716 survivors (47%) and, by construction,
**not one prime**:

| | survivors | of which prime | sieved |
|---|---|---|---|
| `--sieve 0` | 2,716 | 465 | — |
| `--sieve 1000` | 1,441 | 465 | 1,275 |

### Why the estimate was wrong

§0 valued each removed survivor at `c_s ≈ 4,090 ns`. That is the **average** over all
survivors, and it is dominated by the primes, which pay a 12-base Miller–Rabin, a
stage-D rung walk with Pollard factoring and a divisor DP, and a certificate emission.
A *composite* survivor pays one base-2 SPRP, ~800 ns.

The sieve can only ever remove composites. So it removes exactly the **cheap** survivors
and leaves every expensive one untouched. Valued correctly at ~800 ns, 0.23 ns/position
of extra marking buys back ~0.06 ns — a net loss of ~0.17 ns/position, which is the ~5%
slowdown observed.

**The rule this cost:** when a change removes a *subset* of a population, cost it at that
subset's price, never at the population average. The average was built from the very
items the change cannot touch.

### What this redirects to

The expensive survivors are the primes reaching stage D. Only two things reduce that
cost: **more covers** (so fewer primes survive at all — the unpruned `mmax = 2000` set
reaches `rung = 0`), or **a faster stage D** (it currently allocates two `vector<u128>`
per rung attempt inside `min_div_res`, and walks rungs from 3 upward). Neither is this
change.

## 7. Disposition

`--sieve P` is **kept and defaults to 0 (off)**. It is correct, gated both ways, and
costs nothing when off. Two reasons not to delete it: it is the cleanest available knob
for isolating survivor cost when profiling, and its economics invert wherever modexp is
expensive relative to marking — which is precisely the GPU, where 64-bit modmul is
emulated and strided marking is the hardware's strongest operation. Re-measure it there
before assuming either way.

## 8. Out of scope

The bigger wheel (`M = 2,042,040`, ~1.5× by shrinking positions) is the other measured
lever and is deliberately a separate change: it moves the class table, the published
artifact, and every pinned digest, whereas this one moves none of them.
