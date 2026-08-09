# Erdős–Straus certificate census

Code and data for a *classified* census of Erdős–Straus failures: for every prime of
the hard class below 10¹⁵, the complete ladder of failed rungs with a certificate
level attached to each. The conjecture itself is verified to 10¹⁸ by other means;
the product here is the **classification** — which local obstruction explains each
failure, and the statistics of the levels with height.

## The classification

For `4/p = 1/A + 1/B + 1/C` with `A = (p+r)/4`, rung `r` fails when no divisor of
`q² = (pA)²` lands in the class `−q (mod r)`. Each failed rung gets the finest
certificate that explains it:

| level | certificate |
|---|---|
| **J** | the Jacobi symbol `(·/r)` blocks the class |
| **RC** | some other real character mod `r` blocks it, Jacobi does not |
| **S** | no real character does, but `−q` lies outside the multiplicative support subgroup `H` |
| **R** | residual: no local certificate at all |

Below 10¹⁵ (hard class, 932,642,160,749 primes, 557,964,077,963 failed rungs):
**99.1773 % J**, 386,596 RC, 43,257 S, **0.8227 % R**; maximum search depth **155**
at p = 172,538,390,619,841. Every level-S failure occurs at rung 51 — and the released code proves this is forced: since `4A = p + r`, the
support subgroup always contains 4, which confines level S to an explicit finite set
of rungs (`51, 119, 123, 187, 195, 219, 255` below the scanner's cap), of which only
51 is reachable below rung 63. `python3 es_levels.py` derives the set from scratch.

## Layout

| path | what |
|---|---|
| `rung_scan.cpp` | **the reference scanner** — self-testing, unchanged since it produced the published census |
| `rung_scan2.cpp` | optimized CPU scanner, **2.34× the reference** (350 vs 820 ns/prime at 10¹³ on a quiet box, each at its own `--spanlog` optimum: 20 vs 22) — output-identical by `tests/diff2.sh`; use `BIN=rung_scan2 SPANLOG=20 ./run8.sh` for large CPU-only runs, the reference for trust anchoring |
| `Makefile`, `run8.sh` | build + shard harness (cores, sockets, machines) |
| `merge.py` | merge shards, validate tiling, reconcile histograms |
| `verify_cert.py` | exact-arithmetic re-derivation of every emitted certificate |
| `analyze_census.py` | level breakdown, residual-share fits, level-S localisation |
| `es_levels.py` | derives the admissible level-S rung set (the four-constraint) |
| `cuda/` | CUDA port (~9 ns/prime on one NVIDIA L4; ~90× one CPU core) |
| `c15/` | **the 10¹⁵ census**: `MERGED.txt`, `interesting.txt` (585,677 certificates, all re-verified in exact arithmetic — `verify.log`), `ANALYSIS.txt` |
| `c13/` | the 10¹³ census, same layout |
| `c_10^k/` | lower-height censuses, same format |
| `RUNBOOK.md`, `RUN_1e13.md`, `SCALING.md` | operations, reproduction, cost model |

## Reproduce

```sh
make && make check        # MUST print SELF-TEST PASSED — per machine, per flag set
./run8.sh 0 1000000000000 # 0 → 10^12 on 8 cores, ~25 min
python3 verify_cert.py census_out/interesting.txt
```

`make check` recomputes the entire *p* < 10⁵ census against an independent
Python/SymPy-derived reference: all four level counts, the full depth histogram, the
max depth, and the exact list of the 20 residual failures.

The CUDA scanner (`cuda/`, needs CUDA 12.x, `sm_89+`) produces byte-identical
statistics and an identical certificate set, enforced by `make -C cuda check`, which
diffs it against the CPU reference end-to-end and cross-checks every kernel against
an independent implementation (Miller–Rabin for the sieve, Pollard rho for the
factorizer, the reference's own `test_rung` for the classifier).

## Output format

One line per record, whitespace-separated:

```
SUMMARY lo=.. hi=.. primes=.. jac=.. rc=.. sup=.. res=.. escalated=.. pairpath=.. maxr=.. maxr_p=.. hist=r:n,r:n,...
REALCHAR p A r      # level-RC failure at rung r
SUPPORT  p A r      # level-S failure — must satisfy the four-constraint, verified
RESIDUAL p A r      # level-R failure (only with --emit-residual)
DEEP     p A r      # prime whose first hit is at rung r >= threshold
SAMPLE   p A r      # spot-check sample of first hits
```

Emission order is unspecified; `merge.py` sorts. The emitted *set* is deterministic
and bit-identical across machines, thread counts, window sizes, and the CPU/CUDA
implementations.

## Citing

Paper: *Local Obstructions and Slowly Growing Search Depth in the Erdős–Straus
Conjecture: A Classified Census to 10¹³* (working draft; see the data-availability
section for the correspondence between the paper's tables and these files).
