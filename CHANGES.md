# Release history

This file tracks the **public distribution**. It exists so a downloaded copy can
identify itself: each release is a snapshot with no git history, so the version below is
the only thing that says which one you have.

## v3 — 2026-08-28

The verification of `[255255, 18446744065119614976)`.

- **The completed 10¹⁹ runs.** 460 shards, 2.085 × 10¹⁶ wheel positions,
  **156,377,459,709 primes**, every one certified, **survivors = 0**. The direct-search
  fallback was never reached. `docs/sieve/RUN_REPORT.md` is the operational record:
  scope, throughput, cost, what went wrong, and what the verification does and does not
  establish.
- **The shard records themselves** in `data/c19/` and `data/c64/`. Certificates were
  hashed rather than stored — the full stream is ~6 TB — so each shard's record carries
  its `SUMMARY`, the SHA-256 of its deterministic certificate stream, any `SURVIVOR`
  lines verbatim, and its provenance. `sieve/verify_shards.py` is the gate that checks
  the 460 records tile the interval and reconcile; `sieve/analyze_run.py` regenerates
  every number in the run report from them.
- **The re-derivation evidence** in `data/rederive/`: two shards re-run to identical
  certificate digests, 719,775 sampled certificates re-derived in exact rationals, and
  two windows — including the top 10¹⁵ below the ceiling — verified exhaustively.
  `sieve/verify_sample.py` is the verifier.
- **The CUDA port of the verification sieve** (`sieve/cuda/`), which produced the runs,
  alongside the census port that was already here.
- `tools/run19.sh`, the restartable shard driver, so the run itself is reproducible and
  not merely auditable.

The upper endpoint is `(2³² − 1)² − 2049`, not 2⁶⁴: the direct-search fallback needs
arithmetic headroom, so the last 8,589,936,640 integers below 2⁶⁴ are outside the
computation. Both scanners refuse to start above that bound rather than silently
declining to cover it.

## v2 — 2026-08-16

The verification sieve published alongside the census: `cover_scan.cpp` (the frozen
reference), `rung_scan3.cpp` (the optimized scanner, byte-identical to it),
`verify_covers.py` (the independent exact-arithmetic oracle), `prune_filters.py`, and
the certified class tables and filter sets in `tables/`.

## v1 — 2026-08-09

The classified census: the reference scanner, the optimized CPU scanner, the CUDA port,
the shard harness and verifiers, and the census output through 10¹⁵ — everything the
paper's data-availability section promises.
