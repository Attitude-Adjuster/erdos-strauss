# Design: shaving nanoseconds — the 10¹⁶⁺ architecture, paid for at 10¹⁵

**Status:** spec, awaiting the 10¹⁵ census completion (do not rebuild on the VM
under a running census).
**Companion:** `GPU_SCALE.md` (the ns ledger and wall analysis this implements).
**Baseline:** 9 ns/prime at 10¹⁵ and 10¹³, one L4, `--spanlog 24`, double-buffered.
**Target:** ≤ 7.5 ns/prime at 10¹⁵ **measured**, with every change architected so the
same code is what survives the 10¹⁶–10¹⁸ walls. At 10¹⁸, each ns saved ≈ 9 GPU-days
≈ $150; a durable 1.5 ns is ~2 fleet-weeks of value per full-height census.

---

## Non-negotiables (unchanged from the port, restated because they bind harder here)

1. `rung_scan.cpp` is read-only. All reuse via `ref_host`.
2. The bar per task: `make check` green on the box, `diff_range` clean at the
   measurement height with all emission paths, run-to-run byte-identical output
   (3× md5 test), `rung_scan.cpp` sha unchanged.
3. Determinism without allocation atomics: tallies may use order-independent
   atomics (sums, min); allocation stays count → scan → slice; compactions stable.
4. **Measure → decide → measure.** Every task opens by re-profiling (the bottleneck
   has moved after *every single task* so far) and closes with alternated A/B ×3 at
   30 windows of the target height plus one census-width confirmation. `nsys`
   api+kern summaries archived in the commit message.
5. **Kill criteria are honored.** Three hypotheses died this project (per-A
   factorization, class batching, sync placement); each died cleanly because the
   revert decision was pre-committed. Same rule here: below-threshold measurement →
   revert, record in RUNBOOK.md, move on. Negative results are deliverables.

## Memory budget

L4 = 23 GB. Current dual-slot footprint ≈ 5 GB at spanlog 24. Additions below are
itemized per task; nothing in this spec exceeds ~8 GB total. Spanlog growth (Task 5)
is the budget's real consumer — FacR at 33 B × npos dominates; a FacR diet
(SoA / 4-bit exponents) is Task 5's contingency, not a task of its own.

---

## Task 0 — operational hardening (zero ns, do first, 30 min)

`run_gpu.sh` runs `./rung_scan_cuda` from the build tree; `run8.sh` learned the
lesson years ago: **copy the binary into OUTDIR and execute the copy**, so a rebuild
mid-census cannot kill or corrupt a run (the linker rewrites in place). Port the
same three lines. Also: the sync must exclude `c15/` and future census outputs
(add `c1*/` to the exclude list) or a sync deletes a finished census from the VM.

**Verification:** launch a 4-shard toy census, rebuild mid-run, confirm completion.

---

## Task 1 — incremental first-hit in factorize (the audit's find)

**The claim.** `k_count`/`k_gen_*` recompute `j0 = (l − A0 mod l)·astep⁻¹ mod l`
per factor prime per rung per class — an emulated 64-bit modmul ~96× per window.
But `r → r+4` increments `A0 = (first+r)/4` by exactly 1, so

    j0(r+4) = (j0(r) − invastep_l) mod l      — subtract, conditional add.

**Design.**
- Per (slot, class): two `uint32_t` offset buffers of length `nfacp_w` (double
  buffered — within one kernel, items of the same prime must all read the
  *pre-update* value while one designated item writes the next-rung value; a single
  buffer is a read/write race across blocks). `j0 < l < 2³²` fits u32.
- `nfacp_w` is frozen per window at the *last* rung's `Amax` (the per-rung
  `l² ≤ Amax` cutoff moves by a handful of primes; the guard `l² > Amax(r)` runs
  per item, as today).
- Rung 3 initializes the buffer with the current modmul path (no new code path —
  the init IS the existing computation, writing its result down).
- `k_count` reads `j0[i]` from the current buffer, writes the shifted value to the
  next buffer (item `w == 0` only), swaps per rung. `k_gen_warp`/`k_gen_thread`
  read the same current buffer.
- Memory: ~1.07 M primes × 4 B × 2 buffers × 6 classes × 2 slots ≈ **103 MB** at
  10¹⁵ (~2.7 GB at 10¹⁸ — still fits; recorded in GPU_SCALE.md).
- Window advance re-initializes (windows are not consecutive per stream; the
  even/odd split makes the cross-window delta non-uniform — do NOT chase it).

**Expected:** 0.5–0.9 ns at 10¹⁵. **Kill criterion: < 0.3 ns at 10¹⁵ → revert.**

**OUTCOME (2026-08-09): KILLED.** Built, gate green (14 checks), measured 3.64 →
3.63 s at 30 windows of 10¹⁵ — ~0.05 ns/prime. Reverted per the pre-commitment.
The autopsy is the valuable part: the very same transformation, ported to the CPU
scanner (rung_scan2), contributed to a **2.27×** speedup — because there the modmul
sits on a serial dependency chain, while the GPU's thousands of in-flight threads
hide per-thread latency entirely. Latency optimizations transfer from GPU to CPU;
they do not transfer back. (Fourth confirmed kill; the discipline keeps paying.)

---

## Task 2 — `k_gen_warp` micro-pass (34% of kernel time; the honest core)

**Open with evidence, not a plan:** one `ncu` capture (memory vs compute bound,
achieved occupancy, L2 hit rate on the filter reads). The candidates below are
ordered guesses; the capture picks.

- 64 multiples per iteration (two ballots, one 64-bit mask write batch).
- `__ldg`/`__restrict__` on `filt` and the offset buffer (post-Task 1).
- Regime-A tiling: for small l, a warp's 32 consecutive multiples span few filter
  words — stage them once per iteration instead of 32 scattered loads.
- Launch-bound tuning (`__launch_bounds__`) if occupancy-capped.

**Expected:** 0.3–1.0 ns. Individually measured experiments; **each experiment's
kill criterion: < 0.1 ns or any determinism deviation → drop it.**

**OUTCOME (2026-08-09): capture ran, two experiments ran, both KILLED — negative.**
The capture itself was decisive: k_gen_warp at 90% memory SOL with DRAM near idle
(20 GB/s) = L2/LSU request-throughput wall; k_gen_thread 94% SOL at 114 GB/s =
genuinely DRAM-scattered (structural, Wall C's problem); k_reduce latency-bound at
low occupancy (fastdiv case confirmed, unbuilt); k_classify_tally register-capped
at 83% theoretical. The two pointed-at cheap remedies both LOST to the hardware's
defaults: `__ldg`/`__restrict__` on the gen kernels measured ~1% SLOWER (base
3.65–3.68 vs 3.68–3.73 s — the texture path adds latency and the probes have too
little reuse for L1 residency to pay it back), and `__launch_bounds__(256,6)` on
the fused kernel stacked another fraction on top (register spills in the DP beat
the occupancy gain). Both reverted. Kills #5 and #6.

**KILL #8 (2026-08-09): the sort-free scatter itself.** Built in full — k_invmap,
atomic slot claims in the gen kernels, per-position k_assemble; count/scan/sparse/
compact/sort/RLE/reduce and the last sync all deleted; gate green (14 checks), and
the determinism triple produced THE SAME output hash as the sort build — the
relaxed invariant held exactly. And it measured **11% slower** (3.72–3.78 →
4.13–4.20 s). The final refinement of the meta-lesson: not traffic VOLUME but
traffic PATTERN. The sort pipeline's extra passes were coalesced full-sector
transfers; the scatter's fewer bytes were random 56-byte-stride writes plus
per-position atomic RMW, and the memory system serves sectors, not bytes.
Coalesced-and-more beats random-and-less. Reverted.

**Standing verdict for the GPU at 10¹⁵, FINAL: 9 ns/prime, eight kills, zero
remaining candidates.** Every cheap experiment and both traffic-shape hypotheses
are now measured. 7 ns does not exist on this architecture at this height; the
next real gains are Wall C's structural rework at 10¹⁶ and raw SM count. Fastdiv was subsequently built too (Granlund–Montgomery exact division,
one mul+compare replacing the emulated div in k_reduce, plus literal-constant
stripping in k_base): gate green, and ~1% SLOWER — the trick trades ~80 ALU
instructions for two extra global loads per hit, and at k_reduce's occupancy the
ALU was free while the loads lengthen the memory critical path. Kill #7, and the
third GPU kill with the same autopsy. THE META-LESSON, now measured three ways
(T1, ldg/launch_bounds, fastdiv): on this saturated, latency-hiding machine, any
change that adds memory traffic to save arithmetic loses; only changes that REMOVE
memory traffic can win. Exactly two such changes exist: the sort-free scatter
(~0.5–0.8 ns; removes the sort/compact round trips; requires the allocation-atomics
invariant decision) and Wall C's structural rework (removes the scattered walks;
the 10¹⁶ program). 7 ns is reachable only through those two.

---

## Task 3 — walk-bound reduction with device cofactor resolution (Wall C.2)

**The claim.** Walking primes to √Amax is overkill: primes in `[npos, √Amax]` hit a
window < 1× each, yet each costs a thread. Cap the walk at `B_cut` and resolve
cofactors in the fused classify-tally kernel: after the append, `rem` may be
composite (at most two factors > B_cut, or a square) — device MR decides, device
Pollard splits (both exist, both deterministic: fixed bases, fixed c-sequence —
they are the `--check-factor` reference path).

- At 10¹⁵/spanlog 24, √Amax ≈ npos, so **this pays ~nothing at 10¹⁵ — its
  measurement height is 10¹⁶** (√Amax/npos ≈ 9.5× there). Do not judge it at 10¹⁵.
- `B_cut` is a flag (`--walk-bound`, default √Amax = today's behavior), so the
  crossover is swept empirically, per height. The per-A direct-path lesson (28 µs
  per position at starved occupancy) says the optimum is NOT `npos` — it is
  wherever MR cost × active positions crosses walked-prime cost, and MR runs at
  full width at rung 3 (2.1 M positions/class), so expect `B_cut` well above npos
  at low rungs. Consider a rung-dependent bound only if the sweep demands it.
- FacR semantics unchanged (multiset of (res, ex)); τ(q²) unchanged. The gate's
  three-way `--check-factor` covers it at every bound tested.

**Expected:** ~0 at 10¹⁵, ~1–3 ns at 10¹⁶, structural at 10¹⁸.
**Kill criterion: no measurable win at 10¹⁶ for any B_cut → keep the flag at
default, ship the code path disabled (it's also the escape hatch if bucket work
slips).**

---

## Task 4 — bucket sieve for the primality sieve (Wall B)

Oliveira e Silva buckets, per (slot, class): each sieving prime holds
(prime_idx, next_j) in the bucket of the window where its next hit lands; marking
consumes a window's bucket and re-files each prime forward. O(1) per hit instead of
O(π(√X)) per window.

- **Determinism:** marking writes a constant 0 (idempotent), so bucket processing
  order cannot change the sieve output. The output stays a pure function of the
  window.
- **Window coupling:** per-slot streams own the even/odd window subsequence — the
  bucket delta per prime is computed for a stride-2 window advance, so slots stay
  independent. Shard start = one full initialization scan (the current code path,
  again as initializer), amortized over the shard.
- Memory: 10 B/prime → ~0.5 GB at 10¹⁸, trivial at 10¹⁵–10¹⁶.
- **Only worth building when a ≥ 10¹⁶ census is actually scheduled** (sieve is ~8%
  at 10¹⁵ → measurable but small; ~26.7× more setup work by 10¹⁸).
  **Kill criterion at 10¹⁶: < 0.5 ns → shelve until 10¹⁷ planning.**

---

## Task 5 — window fixed cost: spanlog clamp + CUDA graphs (Wall A)

Gated on a measurement: at 10¹⁶, profile the launch+sync share per window. If ≥ 1 ns:
1. Lift the `--spanlog ≤ 27` clamp; audit every u32 position variable to 2³¹;
   re-sweep the optimum (the byte sieve map leaves L2 at ≥ 26 — expect the optimum
   to move, not vanish). FacR diet (SoA) if memory forces it.
2. Capture the per-rung kernel chain as a CUDA graph per (slot, class), instantiate
   once, update parameters per rung. ~10× cheaper replay than launch. The
   variable-`nact` grid sizes mean graph *update* (not re-instantiate) — verify
   update cost < launch savings before adopting.

**Kill criterion: launch+sync share < 0.5 ns at 10¹⁶ → defer to 10¹⁷ planning.**

---

## Sequencing and the meta-rule

    T0 (now, 30 min) → T1 (measure at 10¹⁵) → T2 (ncu first) →
    [10¹⁶ shakeout census: 29 h, ~$21 — this is where T3/T4/T5 get their numbers]
    → T3 → T5 → T4, re-profiling between every pair.

The 10¹⁶ shakeout is itself the first deliverable of the scale program: it
re-establishes correctness at the next height (diff_range + verify_cert there),
produces the profile that ranks T3/T4/T5, and its census output is publishable
science either way. Do not build T3–T5 on 10¹⁵ profiles: at 10¹⁵ their walls have
not arrived, and this project's record on optimizing ahead of measurement is 0-for-3.
