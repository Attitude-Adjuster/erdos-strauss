# DESIGN — moving the wheel to M = 2,042,040

Status: **design, approved**. Extends
`docs/design/2026-08-13-rung-scan3-design.md`. Changes the wheel modulus, the
class-table artifact and its digests, and the pruned filter table. Changes nothing about
the claim, the certificate families, the engine, or the survivor pipeline.

---

## 0. Why

Marking is **90.9%** of runtime (`SCALING_COVER.md` §1). The wheel is the only lever that
reduces marking and survivor work together, because it reduces the *positions* both are
proportional to.

Measured over all unit classes, applying F1 covers with `m | M` and F2 certificates with
`L | M`:

| M | lanes (F1 only) | lanes (F1+F2) | spacing | positions vs today | unit classes |
|---|---|---|---|---|---|
| 120,120 (today) | 920 | 220 | 546 | 1.00× | 23,040 |
| **2,042,040** (×17) | 10,448 | **2,308** | **885** | **1.62×** | 368,640 |
| 4,084,080 (×2) | 20,896 | 4,616 | 885 | 1.62× | 737,280 |
| 38,798,760 (×17×19) | 113,192 | 23,887 | 1,624 | 2.97× | 6,635,520 |

Multiplying by 2 buys nothing — positions in a lane coprime to `M` are already odd.

**2,042,040 is chosen over 38,798,760** despite the smaller speedup. The kill-witness
table has one row per unit class, so it grows with `φ(M)`: ~14 MB here against ~250 MB
there. Layer L1's whole value is being *a finite object a reader can check line by
line*; a 250 MB table is a thing you regenerate and hope, which is what the design set
out to avoid. The larger wheel remains available via `--wheel` for anyone who wants the
2.97× and will regenerate the table locally.

Independently confirmed three ways: Python here, and the **frozen `cover_scan`
unmodified** (`--wheel 2042040`, 33 s, reports `lanes=2308`).

## 1. The one nontrivial code change

`build_wheel` uses only covers with `m | M`, but `gen_covers(M)` enumerates **every**
cover with modulus `≤ M`. At M = 2,042,040 that is `acd ≤ 510,510` — roughly 4.4·10⁷
triples — and it is essentially the whole of the 33 s measured above. Per shard that is
unacceptable.

Replace it, for the wheel path only, with a divisor-driven enumeration:

```
  for each divisor m | M with m ≡ 3 (mod 4):        # 2,042,040 has 256 divisors
      for each factorization 4acd = m + 1:
          emit (m, res = -4a²d mod m, pmin = m(⌊a/c⌋+1) - 4a²d)
```

`m ≡ 3 (mod 4)` is not a heuristic — F1's modulus is `4N − 1` always, proved in the
parent spec, so no other divisor can host a cover.

**This must produce a provably identical set**, and that is testable rather than assumed:
at M = 120,120 the existing `gen_covers(120120)` is cheap, so assert the two agree
exactly — same `(m, res, pmin, a, c, d)` tuples, same order. That assertion is the whole
correctness argument for the change.

`cover_scan` stays **frozen** and keeps the slow path. 33 s is fine for a gate that runs
on small ranges, and its unmodified agreement is worth more than its speed.

## 2. Artifacts and digests

- New `class_table_2042040.txt`: 2,308 classes + 366,332 kill witnesses, ~14 MB.
- `class_table_120120.txt` **stays**. The frozen `cover_scan` pins it, and it remains a
  valid smaller wheel.
- `rung_scan3` pins digests **per modulus** — a small `{M → fnv64}` table replacing the
  single `CLASS_FNV64` constant — and still refuses to run on mismatch. An `M` with no
  pinned entry runs unpinned and says so on stderr, which is what `--wheel` experiments
  need.
- The pruned filter table is re-derived against the new lanes and gets its own file; the
  120120 table stays for reproducing earlier runs.

## 3. Pruning re-runs

Covers with `m | 2,042,040` are absorbed into the wheel, so the dynamic filter set
shrinks and the greedy must re-select against the new lane set. `LO_REF` anchoring
(fixed 2026-08-14) applies unchanged — the domain must be anchored at a real scan floor,
or survivor density is overstated several-fold.

## 4. Gates

Everything existing carries over untouched: `--seg` invariance, shard tiling and counter
reconciliation, byte-identical output across threads, and both small-prime-sieve checks.
Added:

- **`diff3` at M = 2,042,040** against the frozen `cover_scan`, alongside the existing
  M = 120,120 run. Byte-identical `SUMMARY` (less the `sieved=` field) and byte-identical
  sorted certificate set, as now.
- **Generator equivalence at M = 120,120**: divisor-driven enumeration vs `gen_covers`,
  asserted equal in `--verify`.
- **Class-table cross-check at the new modulus**: `rung_scan3 --dump-classes --wheel
  2042040` byte-identical to `class_table_2042040.txt`, which `verify_covers.py`
  regenerates independently.

## 5. Expected result, and why the benchmark will understate it

1.62× fewer positions. The net speedup will likely be **less**, because the surviving
lanes are pre-filtered harder — marks and survivor density *per position* may both rise.
Expect ~1.4–1.6×; measure, do not claim.

Per-lane setup is one extended-Euclid per filter, so it rises with the lane count: ~0.1 s
at 220 lanes, ~1.0 s at 2,308, **per shard**. At production range that is invisible
(10¹⁷/2,042,040 = 4.9·10¹⁰ positions per lane); on a short benchmark range it is a
visible tax. So benchmarks must use a range large enough to amortize it, and any measured
speedup on a short range is a **lower bound** on the production figure.

Two standing measurement rules apply: only within-session timings are comparable (the box
drifts ~20% between sessions), and every reported speedup states what it was measured
against.

## 6. Out of scope

- **M = 38,798,760.** Available via `--wheel`, not the default, for the artifact-size
  reason in §0.
- **A stride-aware cost model.** The pruning greedy costs every mark equally, but a
  stride-17 mark and a stride-2000 mark have very different locality. That is a real
  modelling flaw and a separate change.
- **`factorA`'s 6,542 trial divisions.** Worth ~8% (`SCALING_COVER.md` §1); separate.
