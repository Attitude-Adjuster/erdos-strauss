# Verification sieve to 10^15 — 2026-08-18

`[255255, 10^15)`, 20 shards, `sieve/cuda/sieve_cuda` on one NVIDIA L4 with
`tables/filters/f_M2042040_mmax2000_gpu.txt`. **4.9 minutes wall.** 255,254 is the largest
`pmin` in the filter table; below it the range is settled by the existing censuses.

| | |
|---|---|
| positions | 1,130,242,306,420 |
| covered by certified covers | 1,130,198,943,629 (99.99616%) |
| reaching primality | 43,362,791 |
| composite | 35,520,569 |
| primes, all solved at stage D | 7,842,222 |
| stage E (`direct`) | 0 — never needed |
| **survivors** | **0** |

Every prime in `[255255, 10^15)` in one of the 2,308 hard classes mod 2,042,040 is
certified solvable; every other unit class is killed outright by the class table.
**This is not a new bound on Erdős–Straus** — 10^18 stands by other means — it is this
pipeline verified end to end at scale.

Checked, not asserted:

- shards tile `[255255, 10^15)` exactly, verified on the **boundaries** (`lo -> hi` chain),
  not merely by summing counters;
- `survivors = 0` in every shard;
- **all 7,842,222 certificates re-derived in exact rational arithmetic** by
  `sieve/verify_covers.py` — 0 bad — **on a different machine than produced them**;
- diffed against the independent CPU scanner `rung_scan3` **at 10^15**, over
  `[9.99e14, 10^15)`: byte-identical `SUMMARY`, identical certificate set.

`SUMMARY.txt` holds the twenty shard summaries. The certificate stream itself is 340 MB
and is not tracked; regenerate it with

    ./sieve/cuda/sieve_cuda 255255 1000000000000000 \
      --filters tables/filters/f_M2042040_mmax2000_gpu.txt --shard i/20

and check against `CERTIFICATES.sha256`.

**Emission-order caveat (2026-08-18):** these digests were produced by the lane-major
emitter (commit `b370b69`..`cbb08c8`). The driver now streams certificates round-major
(so a 10^19-scale shard does not hold its stream in RAM), which reorders lines: a
regeneration with a newer binary reproduces the same certificate SET -- compare with
`sort | sha256sum` -- but not these raw digests. The SUMMARY lines are order-free and
remain directly comparable.
