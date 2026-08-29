# DESIGN — `sieve/cuda`: the verification sieve on an NVIDIA L4

Status: **design, approved**. Ports `sieve/rung_scan3.cpp` to CUDA. Changes nothing about
the claim, the certificate families, the class tables, or the CPU scanners.

---

## 0. Why, and what the measurements say to port

From `SCALING_COVER.md` §1, single-thread budget over `[10¹², 1.5·10¹²)`:

| | time | share |
|---|---|---|
| marking (8.02·10⁹ marks) | 6.13 s | **90.9%** |
| stage D (23,182 prime survivors) | 0.53 s | 7.8% |
| composite survivors (112,386) | 0.09 s | 1.3% |

Marking dominates, and marking — a strided bit-clear over a cache-resident segment — is
what a GPU is best at. Stage D is Pollard rho plus a divisor DP on ~1.5·10⁻⁴ of
positions, which is what a GPU is worst at.

So: **marking, compaction and primality on the device; stage D on the host.**

## 1. The trap this design exists to avoid

If the device makes marking 20× faster, the budget becomes 0.31 s marking + 0.53 s stage
D, and **stage D goes from 7.8% of the work to 63% of it**. A port that moves only the
easy 91% lands at ~8×, not ~20×.

Two things prevent that, both inherited rather than invented:

- **The host tail is multi-threaded**, as it already is on CPU (OpenMP over lanes).
- **The host tail overlaps device work.** The census port's hardest-won result applies
  verbatim: *window w's CPU tail runs on a worker thread while window w+1's GPU ladder
  proceeds*, with results folded, flushed and merged strictly in window order after the
  join. That port measured 4.84 → 4.63 s from exactly this.

**Corollary that must be measured, not assumed:** the marking speedup and the tail cost
have to be measured *separately*, or a good marking kernel will be hidden behind an
unoverlapped tail and read as a bad port.

## 2. Two economics invert on the device

Both are predictions this port should test, not assumptions it should build on.

**The small-prime sieve should pay here.** It measured a 5% *loss* on CPU
(`2026-08-14-small-prime-sieve-design.md` §6): it removes only composite survivors, the
cheap ones, while adding ~10% more marking. On the device that trade reverses — marking
is the strong operation and a survivor costs an emulated 64-bit modexp, the weak one.
`--sieve P` already exists, defaults off, and is gated both ways; re-sweep it here.

**The pruned filter set should get LARGER.** The greedy balances marks against survivors
at a measured ratio of ~14,900:1 on CPU. On the device marks get much cheaper and
survivors relatively dearer, so the optimum moves toward *more* marking and fewer
survivors. `sieve/prune_filters.py` re-runs with GPU-measured constants; the resulting
table is published beside the CPU one rather than replacing it, since each is optimal for
its own machine.

## 3. Structure

One block per lane. The segment bitset lives in **shared memory** (32 KB at the measured
CPU optimum `--seg 262144`; the device optimum is its own sweep). Per-lane filter offsets
live in global memory, `4 × ncov` bytes per lane, written once per lane per shard.

```
  per (lane, segment):
    fill shared bitset               all positions alive
    mark covers          stage B     atomicAnd, one thread per filter
    mark small primes    stage B'    same, if --sieve P
    ballot + prefix sum              compact survivors
    SPRP → MR                        device
    stage D / E                      HOST, overlapped with the next window
```

**Why `atomicAnd` needs no coordination.** Marking only ever *clears* bits, so it is
idempotent and order-independent: two blocks clearing the same bit cannot disagree, and
no ordering is observable. This is the same argument the census port makes for its
composite marking writing a constant 0. It is the property that makes the whole kernel
lock-free.

**Determinism is structural**, exactly as in the census port: compaction is
count → exclusive prefix sum → write into your own slice, never atomics, so every write
lands at an index that is a pure function of the inputs. Atomics appear only where the
value is order-invariant (counters). Concurrency never crosses a window.

## 4. Layout

`census/cuda/` and `sieve/cuda/` — each product owns its GPU port, following the axis
established on 2026-08-16. The existing top-level `cuda/` moves to `census/cuda/`; it
reads as belonging to everything when it belongs to the census.

`sieve/cuda/ref_host.cpp` includes `sieve/rung_scan3.cpp` **exactly once** and exposes
narrow accessors, the pattern `census/cuda/ref_host.cpp` established — a second
`#include` of the same translation unit is an ODR violation, and that is a documented
trap in this repo.

## 5. Trust model — unchanged, and that is the point

Byte-identical `SUMMARY` and byte-identical **sorted** certificate set against
`rung_scan3`, and therefore transitively against the frozen `cover_scan`. Same bar the
census port meets, enforced the same way, by `sieve/cuda/tests/diff_cuda.sh`.

Emission *order* may differ (it already differs between `cover_scan` and `rung_scan3`);
the emitted **set** may not. `--sieve 0` for the diff, since the sieve stage changes
`mr` and `composite` by design.

The class-table digest check runs on the host at startup, unchanged: a binary whose
derivation drifted refuses to run.

## 6. Hardware, and its constraints

One NVIDIA L4, `g2-standard-12`, `sm_89`. From `RUNBOOK.md`:

- **`GPUS_ALL_REGIONS` is 1.** A GPU box for this project means `wm-trainer` is stopped.
- **L4 capacity in `us-central1` is not reliable** — a hard `STOCKOUT` killed two zones
  in minutes on 2026-08-08. The box is a disposable rsync mirror; recreate it in
  whichever zone has capacity rather than waiting. The error message names the zones
  that do.
- `nvcc` is not on the non-interactive PATH; the image ships no `g++` and no `make`.
  `tools/the sync driver` handles the first and one `apt-get install build-essential` the rest.

## 7. Expected result — a range, not a number

The census port measured ~30× against a same-box 12-thread CPU, but that workload is
dominated by per-prime factorization and this one by marking, so **that number must not
be carried across**. This project has paid twice for exactly that mistake.

Reasoning from structure instead: marking should go 10–20×, survivor tests 5–10× (the
modmul is emulated), and the host tail is bounded by overlap quality. Expect **8–15×**
against a 12-physical-core CPU box, which would put 10¹⁸ at roughly a day. Measure it;
the estimate is worth nothing beside a number.

## 8. Out of scope

- The production run. This spec covers the port and its gate.
- Stage D on the device. It is 7.8% of CPU work on ~1.5·10⁻⁴ of positions and is the
  wrong shape for the hardware; the host handles it, overlapped.
- Multi-GPU. `GPUS_ALL_REGIONS` is 1, so it is not reachable without a quota increase.
