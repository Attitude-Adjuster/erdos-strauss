# GPU scale plan: 10¹⁵ → 10¹⁸⁺ — the ns ledger

INTERNAL (not in the publish whitelist). Written 2026-08-08, while the 10¹⁵ census
runs. Current baseline: **9 ns/prime on one L4** (hard class, spanlog 24, double
buffered). Everything below is priced against that.

## Why every ns counts up there

| height | hard-class primes | 1 ns/prime costs | census at ~12 ns, one L4 |
|---|---|---|---|
| 10¹⁵ | 9.3 × 10¹¹ | 15 min / ~$0.2 | 3.1 h |
| 10¹⁶ | 8.7 × 10¹² | 2.4 h / ~$1.7 | ~29 h |
| 10¹⁷ | 8.2 × 10¹³ | ~23 h / ~$16 | ~11 days |
| 10¹⁸ | 7.7 × 10¹⁴ | **~9 days / ~$150** | ~107 days → a fleet job |
| 10¹⁹ | 7.3 × 10¹⁵ | **~84 days / ~$1,400** | uint64 edge; see hard limits |

A 10¹⁸ run on one L4 is ~107 days, so the real shape is N GPUs × shards (already
embarrassingly parallel via `--shard`; `GPUS_ALL_REGIONS = 1` must be raised first).
16 L4s ≈ 1 week ≈ $1.8k. Every ns shaved is ~$150 and ~14 fleet-hours at 10¹⁸.

## The walls, in order of arrival

### Wall A (~10¹⁶): window count × per-window fixed cost

SPAN = 840·2²⁴ ≈ 1.4 × 10¹⁰, so 10¹⁸ is **7.1 × 10⁷ windows**. At ~30 launches +
2 stream syncs per (class, rung), launch/sync overhead alone reaches wall-clock
scale (~2 × 10⁹ launches × ~5 µs ≈ 10⁴ s). Mitigations, cheapest first:
1. Raise `--spanlog` beyond the current **hard clamp of 27** (revisit the clamp,
   the u32 position arithmetic holds to 2³¹, and re-sweep the optimum — but note
   the byte sieve map leaves L2 at spanlog ≥ 26 and FacR is 33 B × npos).
2. **CUDA graphs** for the per-rung kernel chain (~10× cheaper replay than launch).
3. Persistent-kernel ladder (biggest rewrite, last resort).

### Wall B (~10¹⁶, dominant by 10¹⁷): the sieve's per-window prime scan

π(√X): 1.9M at 10¹⁵ → 50.8M at 10¹⁸ (26.7×). We pay one emulated 64-bit modmul per
base prime per window-class regardless of hits. Sieve share ~8% today → dominant at
10¹⁷⁺. Fix: **Oliveira e Silva bucket sieve** (CUDASieve proves it on GPU; ~10 B
state/prime; ~0.5 GB at 10¹⁸ — fits). Friction, recorded in the audit: bucket state
couples consecutive windows, which breaks the window-independence that double
buffering and shard-resume rely on — each shard/stream rebuilds bucket state once at
startup (one full scan, amortized over the shard).

### Wall C (same era, bigger): factorize's per-rung prime walk

The same scan runs **per rung per class** — 96×/window vs the sieve's 6× — over
π(√(X/4)) factor primes: 1.0M at 10¹⁵ → 26.4M at 10¹⁸. Two-step fix:
1. **Incremental first-hit** (the audit's find): r → r+4 increments A0 by exactly 1,
   so j0 ← (j0 − invastep) mod l — subtract replaces modmul; persist per-prime
   offsets across rungs (~8 MB/class at 10¹⁵, ~210 MB at 10¹⁸). Ceiling ~5–10%
   today; much more at height. **Do first, measurable at 10¹⁵.**
2. **Lower the walk bound from √Amax to ~npos** and let the classify-tally kernel
   handle cofactors with device MR + Pollard (both already exist from the
   `--check-factor` path). At 10¹⁸ this deletes ~25M of 26.4M walked primes per
   rung-class at the price of one MR per active position. The trade flips somewhere
   between 10¹⁶ and 10¹⁷ — measure, don't guess; the direct-path lesson (28 µs/pos
   when occupancy is starved) says the crossover is real but empirical.

### Wall D: the CPU tail — NOT a wall

Handed-over fraction is ~5 × 10⁻⁶ of primes, height-independent to first order, so
the tail stays ~4% of wall at any height (it parallelizes with the window threads).
If it creeps: the device direct path can absorb it.

## Hard limits (know before 10¹⁹)

- p < 1.8 × 10¹⁹ (uint64) — 10¹⁹ fits, barely. A < 2.5 × 10¹⁸ < 2⁶². OK.
- Factor primes ≤ √(2.5 × 10¹⁸) ≈ 1.6 × 10⁹ < 2³² — the u32 prime tables hold. OK.
- Base primes ≤ √10¹⁹ ≈ 3.2 × 10⁹ < 2³² — holds, with ~25% headroom only.
- `is_prime_dev` base set proven to 3.3 × 10²⁴. OK everywhere.
- Beyond 10¹⁹: u32 prime tables break first. Do not plan past it without widening.

## Micro-inventory (measured shares at 10¹⁵, worth revisiting when the walls fall)

| item | share now | note |
|---|---|---|
| `k_gen_warp` inner loop | ~34% | ballot/popc per 32 multiples; the honest core. Try 64-wide iterations, `__ldg` on the filter, warp-level batching of small-l primes |
| `k_reduce` | ~8% | binary search per segment; fine |
| classify+tally (fused) | ~9% | DP bit-loop is serial per set bit; acceptable |
| FacR layout | — | 33 B packed struct → misaligned 4 B loads; SoA or 36 B padding is a few % and shrinks with spanlog growth pressure — bench, don't assume |
| hardware | — | the workload is int64/int32; an L40S/H100 is ~2–4× on int throughput. At fleet scale, $/prime decides, not ns/prime |

## Order of work when scaling reopens

1. Incremental first-hit in factorize (Wall C.1) — payable at 10¹⁵, verify with the
   existing gate, measure at 10¹⁵ and 10¹⁶.
2. Spanlog clamp + CUDA graphs (Wall A) — measure launch share at 10¹⁶ first.
3. Bucket sieve (Wall B) — with the window-coupling design resolved for shards.
4. Walk-bound reduction + device cofactor MR (Wall C.2) — crossover measurement.
5. Quota + fleet plan; only then micro-inventory.

Every step keeps the bar: gate green, diff_range at the target height, and the
falsifiable checks (level-S rung set) intact. The discipline is the product.

---

## CPU annex: what back-ports to `cloud-scan`, and what actually matters there

The reference is untouchable (cardinal rule); a second CPU scanner held to the same
diff bar is the sanctioned route (the CUDA port is precedent). Transferable lessons,
in value order: incremental first-hit (the CPU walk recomputes mulmod j0 per prime
per rung, rung_scan.cpp:535 — same fix as spec T1, inside the CPU's 40% walk share);
uint64 bitmask DP for r ≤ 63 (inside the 56% per-active-prime share); the level-S
confinement skip; the Jacobi/Legendre tables (the documented deliberately-unexploited
one). Non-transfers: sort pipeline, double-buffering, graphs. **Built and measured
(rung_scan2): 2.34× — 820 → 350 ns/prime at 10¹³ on a quiet box, each binary at
its own spanlog optimum (v2's moved to 20: the cheaper walk shifts the trade from
fixed-cost amortization back toward cache residency).** Composition: E1/E2 uarch
pass (division-free DP chain via 4 KB mod table; read-only branchless walk) took
862 → 379 under load; Barrett reciprocals (reopened by decision) added ~5% —
350/351 vs 368/371 alternated — by replacing the remaining ~15 per-call divq with
mulhi+correct. Remaining unbuilt: append-log accumulation, PGO. The GPU-side twin
of E1 (incremental first-hit) was BUILT AND KILLED at ~0.05 ns: per-thread latency
is hidden by occupancy there. Latency optimizations transfer GPU→CPU, not back.

**Economics: do not build it.** One L4 is ~14× better $/prime than the whole c4 box;
a 2× CPU scanner is still ~7× worse. Contingency only: GPU quota denied AND a 10¹⁶
deadline (N2_CPUS=200 headroom is the one asymmetry in CPU's favor — and even a
200-vCPU fleet at 400 ns/prime ≈ half an L4).

**The real CPU wall is verification, not scanning.** cloud-scan's emerging role is
the verification host, and CPython exact arithmetic is the pipeline's binding CPU
constraint: certificates grow ~40× per two decades — 585,677 at 10¹⁵ (hours on 22
cores), ~5M at 10¹⁶ (a day), 10¹⁸ infeasible. Fix order: (1) PyPy — the tools are
stdlib-only by design, so it is a drop-in; measure on a c15 chunk when the box
frees; (2) fleet-shard the chunks (embarrassingly parallel); (3) stratified
verification only as a NAMED policy decision — it weakens the bar and must never
happen by drift.
