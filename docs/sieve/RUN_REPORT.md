# The 10¹⁹ verification runs — operational record

*Status: both runs complete and verified — run19 finished 250/250 on 2026-08-27 and
shut itself down. Every number below is generated from the 460 shard metas in the repo.*

Everything numeric here is generated from the shard metas by
`sieve/analyze_run.py`, and every claim is checked by `sieve/verify_shards.py`.
Neither reads a log or a note; both read the artifacts. If a number here
disagrees with the metas, the metas are right.

---

## 1. What was computed

Two runs of the verification sieve (`sieve/cuda/sieve_cuda`), together covering

    [255255, 18446744065119614976)

and establishing that **every prime in that range is Erdős–Straus solvable**, with a
certificate emitted and hashed for each one. Primes below 255,255 are settled by the
existing censuses — 255,254 is the largest `pmin` in the filter table, so the sieve
declines to start below it rather than pretend to cover it.

| | run19 | run64 |
|---|---|---|
| band | `[255255, 10¹⁹)` | `[10¹⁹, DIRECT_PMAX)` |
| shards | 250 | 210 = 3 boxes × 70 |
| boxes | `cloud-gpu` | `cloud-gpu-2`, `-3`, `-4` |
| binary commit | `4258bc9` | `115d127` |

The two runs used **different binaries but identical mathematical inputs** — same
filter table digest, same class-table digest, same wheel, same certificate bounds.
That is the distinction the gate enforces: `sieve/verify_shards.py` requires the
mathematical provenance to be uniform and merely *reports* the commit, because a run
may legitimately be resumed with a rebuilt binary but must never change what it is
computing.

### The upper limit is not 2⁶⁴, and the exact value matters

    DIRECT_PMAX = (2³² − 1)² − 2049 = 18446744065119614976

from `sieve/rung_scan3.cpp:42`. The `(2³² − 1)²` is inherited from the frozen
reference's `isqrt_u64`, whose `while ((s+1)*(s+1) <= n)` wraps once `s` reaches
2³² − 1; the 2049 is the headroom for `n ≤ p + 4ade ≤ p + 2048`. Both scanners
**refuse to start** above it rather than silently declining to solve, because a
silent skip would surface later as a SURVIVOR — which reads as a mathematical
result rather than a configuration error.

So the top **8,589,936,640** integers below 2⁶⁴ are *not* covered. The honest claim
is "verified to 1.8446744065×10¹⁹", never "the full 64-bit range".

---

## 2. Result

**survivors = 0** across both runs — every prime found was solved, none escaped.

<!--GENERATED:RESULT-->

| run | shards | lane positions | covered by a cover | to primality | composite | **primes** | stage D | stage E | **survivors** |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| run19 | 250 | 11,302,423,067,128,656 | 99.995101% | 553,752,228,769 | 467,767,414,950 | **85,984,813,819** | 85,984,813,819 | 0 | **0** |
| run64 | 210 | 9,546,867,496,374,255 | 99.995102% | 467,561,667,936 | 397,169,022,046 | **70,392,645,890** | 70,392,645,890 | 0 | **0** |
| **total** | 460 | 20,849,290,563,502,911 | 99.995101% | 1,021,313,896,705 | 864,936,436,996 | **156,377,459,709** | 156,377,459,709 | 0 | **0** |

That is **156,377,459,709 primes**, each with a certificate — one prime per 117,962,935 integers of the band, and 156,377,459,709 certificate lines (~6.26 TB of text, hashed and discarded).

---

## 3. Hardware and configuration

Both runs, every box, identical:

| | |
|---|---|
| machine | `g2-standard-12` — 12 vCPU (6 physical) Xeon @ 2.20 GHz, 47 GB, 38.5 MB L3 |
| GPU | 1 × NVIDIA L4, 23,034 MiB, `sm_89`, driver 580.173.02 |
| CUDA | 12.9, V12.9.41 |
| OS | Ubuntu 22.04.5 LTS, kernel 6.8.0-1065-gcp |
| disk | 150 GB pd-balanced |
| threads | 12 (host tail) |

Wheel and filter table, pinned by digest in every shard meta:

| | |
|---|---|
| wheel | M = 2,042,040, 2,308 hard lanes (spacing 884), `wheel_pmin` = 255,254 |
| covers | `mmax` = 10,000; uniform-rung certificates `r ≤ 63`, `u ≤ 64` |
| filter table | 3,745 filters, `f_M2042040_mmax2000_gpu.txt` |
| table sha256 | `6af5e7621b461872046c5026e4c8a40e81c46c0638a306ad8603838004fcceaf` |
| classes sha256 | `a77b2187201ff96ff3ecf822150f4152d841941deeb0cf616238bcbb0a648bbc` |

---

## 4. Timing, throughput, and cost against height

<!--GENERATED:TIMING-->

```
==============================================================================
run19  [255255, 10^19)
==============================================================================
band            [255255, 10000000000000000000)
                 = [2.553e+05, 1.000e+19)   width 9,999,999,999,999,744,745 integers
shards          250  x  39,999,999,999,998,979 integers each
boxes           1: c19

-- wheel and filter table
wheel           M=2,042,040  lanes=2308  spacing=884  wheel_pmin=255,254
covers          mmax=10,000  uniform-rung certs r<=63 u<=64
table sha256    6af5e7621b461872046c5026e4c8a40e81c46c0638a306ad8603838004fcceaf
classes sha256  a77b2187201ff96ff3ecf822150f4152d841941deeb0cf616238bcbb0a648bbc
binary commit   4258bc9

-- the sieve, stage by stage
positions       11,302,423,067,128,656   (1.130e+16)   = width * lanes/M, the wheel survivors
  covered       11,301,869,314,899,887   99.995101%   killed by a congruence cover (stage A/B)
  to primality  553,752,228,769   0.004899%
  Miller-Rabin  553,752,228,769 tested
    composite   467,767,414,950   84.472%
    PRIME       85,984,813,819   15.528%
      stage D   85,984,813,819  solved by rung certificate
      stage E   0  solved by direct search
      SURVIVOR  0  <-- must be 0
certificates    85,984,813,819 emitted and hashed (~3.44 TB at ~40 B/line, never stored)
prime density   1 per 116,299,606.4 integers of the band

-- timing
GPU time        207.89 GPU-hours  (8.662 GPU-days)
per shard       min 2659s   median 2999s   max 3414s   spread 1.28x
throughput      13,361,802,881,339 integers/s     7.484e-05 ns/integer
                114,891 primes/s        8,703.9 ns/prime
                15.102 Gpositions/s

-- cost (reconstructed from audit-log instance lifetimes, not an invoice)
instance hours  239.80 h billed vs 207.89 h computing = 86.7% duty cycle
at $1.00/h   $239.80
                $2.79 per 10^9 primes   $23.98 per 10^18 integers

-- cost against height (equal-width shards; this run measuring itself)
   bin shards  mid height  wall/shard  ns/integer   ns/prime   primes/shard   growth
     0     25    5.00e+17       2695s     0.00007     7415.7    363,443,809       --
     1     25    1.50e+18       2738s     0.00007     7795.1    351,264,658   1.016x
     2     25    2.50e+18       2798s     0.00007     8065.6    346,930,854   1.022x
     3     25    3.50e+18       2896s     0.00007     8414.4    344,175,067   1.035x
     4     25    4.50e+18       2966s     0.00007     8667.5    342,160,929   1.024x
     5     25    5.50e+18       3061s     0.00008     8988.8    340,562,123   1.032x
     6     25    6.50e+18       3093s     0.00008     9117.7    339,253,117   1.010x
     7     25    7.50e+18       3187s     0.00008     9424.2    338,133,236   1.030x
     8     25    8.50e+18       3229s     0.00008     9577.0    337,162,543   1.013x
     9     25    9.50e+18       3273s     0.00008     9731.4    336,306,217   1.014x
   fit: ns/integer grows 1.177x per decade of height (log-log least squares over 10 bins, 10^17.7..10^19.0)
```

```
==============================================================================
run64  [10^19, DIRECT_PMAX)
==============================================================================
band            [10000000000000000000, 18446744065119614976)
                 = [1.000e+19, 1.845e+19)   width 8,446,744,065,119,614,976 integers
shards          210  x  40,222,590,786,283,881 integers each
boxes           3: cloud-gpu-2, cloud-gpu-3, cloud-gpu-4

-- wheel and filter table
wheel           M=2,042,040  lanes=2308  spacing=884  wheel_pmin=255,254
covers          mmax=10,000  uniform-rung certs r<=63 u<=64
table sha256    6af5e7621b461872046c5026e4c8a40e81c46c0638a306ad8603838004fcceaf
classes sha256  a77b2187201ff96ff3ecf822150f4152d841941deeb0cf616238bcbb0a648bbc
binary commit   115d127

-- the sieve, stage by stage
positions       9,546,867,496,374,255   (9.547e+15)   = width * lanes/M, the wheel survivors
  covered       9,546,399,934,706,319   99.995102%   killed by a congruence cover (stage A/B)
  to primality  467,561,667,936   0.004898%
  Miller-Rabin  467,561,667,936 tested
    composite   397,169,022,046   84.945%
    PRIME       70,392,645,890   15.055%
      stage D   70,392,645,890  solved by rung certificate
      stage E   0  solved by direct search
      SURVIVOR  0  <-- must be 0
certificates    70,392,645,890 emitted and hashed (~2.82 TB at ~40 B/line, never stored)
prime density   1 per 119,994,694.9 integers of the band

-- timing
GPU time        203.33 GPU-hours  (8.472 GPU-days)
per shard       min 3078s   median 3502s   max 3723s   spread 1.21x
throughput      11,539,252,084,518 integers/s     8.666e-05 ns/integer
                96,165 primes/s        10,398.8 ns/prime
                13.042 Gpositions/s

-- cost (reconstructed from audit-log instance lifetimes, not an invoice)
instance hours  214.40 h billed vs 203.33 h computing = 94.8% duty cycle
at $1.00/h   $214.40
                $3.05 per 10^9 primes   $25.38 per 10^18 integers

-- cost against height (equal-width shards; this run measuring itself)
   bin shards  mid height  wall/shard  ns/integer   ns/prime   primes/shard   growth
     0     21    1.04e+19       3292s     0.00008     9756.4    337,464,471       --
     1     21    1.13e+19       3440s     0.00009    10212.9    336,865,501   1.045x
     2     21    1.21e+19       3337s     0.00008     9922.2    336,303,764   0.970x
     3     21    1.30e+19       3466s     0.00009    10322.7    335,791,148   1.039x
     4     21    1.38e+19       3558s     0.00009    10609.8    335,311,057   1.026x
     5     21    1.46e+19       3403s     0.00008    10161.1    334,858,752   0.956x
     6     21    1.55e+19       3530s     0.00009    10555.0    334,438,710   1.037x
     7     21    1.63e+19       3637s     0.00009    10888.3    334,041,634   1.030x
     8     21    1.72e+19       3661s     0.00009    10973.1    333,660,854   1.007x
     9     21    1.80e+19       3533s     0.00009    10599.4    333,294,866   0.965x
   fit: ns/integer grows 1.422x per decade of height (log-log least squares over 10 bins, 10^19.0..10^19.3)
```

Together: **411 GPU-hours** (17.1 GPU-days) of L4 time — 207.89 for run19, 203.33 for run64.

---

## 5. Calendar, and what went wrong

<!--GENERATED:TIMELINE-->

| when (UTC) | what |
|---|---|
| 2026-08-16 19:51 | `cloud-gpu` created (third attempt; two zones exhausted) |
| 2026-08-18 18:13 | final boot — the box then ran 9 days without a reboot |
| 2026-08-18 19:18 | run19 starts, `[255255, 10¹⁹)` × 250 |
| 2026-08-21 23:22 | quota raised to 4 GPUs; `cloud-gpu-2/3/4` created |
| 2026-08-21 23:28 | run64 starts on all three, `[10¹⁹, DIRECT_PMAX)` × 70 each |
| 2026-08-24 20:18 | `cloud-gpu-2` finishes 70/70, shuts itself down |
| 2026-08-24 22:48 | `cloud-gpu-3` finishes 70/70, shuts itself down |
| 2026-08-25 01:10 | `cloud-gpu-4` finishes 70/70, shuts itself down |
| 2026-08-26 16:51 | first attempt to restart the three boxes to pull metas — all zones exhausted |
| 2026-08-26 16:57 | GPU-less workaround attempted; `set-machine-type` refuses (G2 carries an explicit accelerator attachment) |
| 2026-08-27 16:19 | capacity returned; all three restarted, metas pulled and audited, boxes stopped |
| 2026-08-27 23:00 | run19 finishes 250/250, 0 FAILED |
| 2026-08-27 23:11 | `cloud-gpu` shuts itself down — 9 days, 250 shards, no operator |
| 2026-08-27 23:45 | restarted to pull `shard_249.meta`; audited; stopped 23:48 |

### Reconstructed cost

Priced from audit-log lifetimes at published on-demand rates — **a reconstruction, not an invoice**:

```
instance      created     ended             up     life   duty     vm $  disk $    total  machine
cloud-scan    08-07 20:09 08-09 16:14     2.1h    44.1h   4.7% $   2.04 $  1.09 $   3.13  c4-highcpu-24        CPU census box
cloud-gpu#1   08-07 22:13 08-08 13:22    14.1h    15.1h  93.1% $  14.10 $  0.31 $  14.41  g2-standard-12 + L4
cloud-gpu-b   08-08 12:39 08-09 16:13     9.0h    27.6h  32.6% $   8.99 $  0.57 $   9.55  g2-standard-12 + L4  CUDA dev
cloud-dev     08-13 15:43 08-22 03:28    73.7h   203.8h  36.2% $  22.11 $  1.26 $  23.37  c4-highcpu-8         dev box
cloud-gpu#2   08-16 19:51 08-27 23:47   239.8h   267.9h  89.5% $ 239.81 $  5.51 $ 245.31  g2-standard-12 + L4  run19, all 250 shards
cloud-gpu-2   08-21 23:21 08-27 16:29    69.1h   137.1h  50.4% $  69.11 $  2.82 $  71.93  g2-standard-12 + L4  run64 lower third
cloud-gpu-3   08-21 23:21 08-27 16:30    71.5h   137.1h  52.1% $  71.48 $  2.82 $  74.30  g2-standard-12 + L4  run64 middle third
cloud-gpu-4   08-21 23:21 08-27 16:46    73.8h   137.4h  53.7% $  73.84 $  2.82 $  76.67  g2-standard-12 + L4  run64 upper third
                                        553.1h                 $ 501.48 $ 17.19 $ 518.67  TOTAL

  run19 + run64 production:        454.2 instance-h   $ 468.21
  development / earlier boxes:      98.9 instance-h   $  50.46

  rejected API calls: 30 (not billed, and not counted above)
     26  ZONE_RESOURCE_POOL_EXHAUSTED
      4  QUOTA_EXCEEDED
```


---

## 6. What the certificates are, and why they are hashes

At ~40 bytes per line the two runs emitted on the order of **6 TB** of certificate
text against a 150 GB disk, so the stream is hashed and deleted, one shard at a time.
Each meta records the stream's sha256, the SUMMARY, every SURVIVOR line verbatim, and
the provenance. The scan is deterministic — byte-identical reruns are part of the
gate — so any shard is re-derivable and checkable at the cost of re-running it.

To re-derive one shard, with the table at the digest above:

    sieve/cuda/sieve_cuda LO HI --shard I/N --filters tables/filters/f_M2042040_mmax2000_gpu.txt --threads 12 \
      | grep -v '^SUMMARY ' | sha256sum

and compare against `cert_sha256` in `shard_I.meta`.

---

## 7. How this was verified

| check | what it rules out |
|---|---|
| `sieve/verify_shards.py --band` over each run | a missing, duplicated, or mis-tiled shard; a gap between boxes; non-uniform provenance |
| boundary tiling, not summed counters | the failure the first `--shard` gate missed: a floor-instead-of-ceil span drops `(HI−LO) mod N` integers, and at one lane position per 885 integers those almost never contain a prime, so every total still reconciles |
| `survivors = 0` and zero SURVIVOR lines | an unsolved prime, the one result that would matter |
| per-shard accounting identity | a prime lost between stages |
| distinct cert-stream digests | two shards having silently computed the same thing |
| metas pulled and audited on a different machine than produced them | a fault in the producing host |
| a shard re-run reproducing its `cert_sha256` byte for byte (§7 below) | a meta that commits to a stream nobody can reproduce, and any nondeterminism in the pipeline |
| `sieve/verify_sample.py` over a strided sample and both edges | a certificate that certifies nothing; a `p` that is composite, out of its band, or outside the hard class |

### Certificate-level re-derivation (2026-08-28)

The checks above establish that the shards tile the band and that each meta is a
faithful commitment to a stream. They say nothing about whether the certificates *in*
that stream are correct — and since the stream was hashed and deleted, the only way to
ask is to re-run a shard. Two were re-run, one per band, each on the box and with the
binary that originally produced it:

| | run19 shard 0 | run64 shard 69 |
|---|---|---|
| band | `[255255, 4.0000000000254234e16)` | `[1.5590940119293459e19, 1.5631162710079743e19)` |
| box / binary | `cloud-gpu` / `4258bc9` | `cloud-gpu-3` / `115d127` |
| certificates in the stream | 394,469,504 | 334,389,059 |
| `cert_sha256` vs the meta | **identical** | **identical** |
| `SUMMARY` vs the meta | **byte-identical** | **byte-identical** |
| wall, original → re-run | 2,659 s → 2,649 s | 3,548 s → 3,807 s |
| certificates verified in exact rationals | 389,224 | 330,551 |
| bad, composite, out-of-band, off-class | **0, 0, 0, 0** | **0, 0, 0, 0** |

The stream is ~16 GB and was again never stored: it was hashed on the fly and sampled
through a `tee` — every 1024th certificate, plus the first and last 2,000. **The edges
are not decoration.** A shard-boundary or offset-carry fault does not perturb a uniform
sample; it shows at the first or last certificate emitted. Both edges land inside their
band and verify (`s69`'s last certified prime is 15,631,162,686,827,084,329, some
2.3 × 10¹⁰ below its `hi`).

Verification ran on the laptop — a different machine, a different architecture, and a
different implementation from the one that produced the certificates:

    python3 sieve/verify_sample.py --lo LO --hi HI --certs N --stride 1024 \
        --sample sample.txt sample.txt head.txt tail.txt

`sieve/verify_sample.py` calls `verify_covers.py` for the exact-rational identity
`4/p = 1/A + 1/x + 1/y` and adds four checks that the identity alone cannot make:
**primality** of every certified `p` by an independent deterministic Miller–Rabin (the
scanner's own primality is a device kernel; this is unrelated code on unrelated
hardware), **range** containment in the shard's own band, **class** membership
`p ≡ 1 (mod 24)`, and the **count** against `certs/stride`, which is what makes the
sample a statement about the whole stream rather than about whatever landed in a file.

A sample says nothing about what it skipped, so two narrow windows were also scanned
fresh and verified **exhaustively** — every certificate, not every 1024th:

| | `w19` | `w64` |
|---|---|---|
| window | `[5e18, 5e18 + 10¹²)` | `[18445744065119614976, DIRECT_PMAX)`, the top **10¹⁵** |
| lane positions | 1,130,242,300 | 1,130,242,306,714 |
| covered by a cover | 99.995118% | 99.995102% |
| certificates | 8,413 | **8,284,620** |
| stage E (`direct`) | 0 | 0 |
| survivors | 0 | 0 |
| verified in exact rationals | **all 8,413** | **all 8,284,620** |
| bad, composite, out-of-band, off-class | 0, 0, 0, 0 | 0, 0, 0, 0 |

`w64` is the answer to the obvious objection about the top of the range. It sits
immediately below `DIRECT_PMAX`, where `p + 4ade` comes within 8.6 × 10⁹ of 2⁶⁴ and any
overflow in the covering arithmetic would show — and it behaves like everywhere else:
the same 99.995% coverage, no survivor, and 8.28 million certificates every one of which
satisfies `4/p = 1/A + 1/x + 1/y` in exact rational arithmetic, for a prime this tool
re-tested itself. The 10¹⁵ width also makes it a second, independent check that the
region `cloud-gpu-4` scanned is what it said it was: the window overlaps that box's
top shard, was produced on a different box, and agrees.

What this establishes, stated precisely: two shards' streams are **exactly** reproducible
byte for byte; a 1-in-1024 uniform sample of them plus both edges — 719,775
certificates — is correct in exact arithmetic; and two contiguous windows, one of them
the top 10¹⁵ of the whole verification, are correct **certificate for certificate**,
8,293,033 of them. It does not verify all 156,377,459,709; nothing short of re-running
the whole 411 GPU-hours and verifying every line would, and the pipeline is
deterministic, so that is a matter of cost, not of doubt. Earlier, at 10¹⁵,
`verify_covers.py` did verify **every** certificate emitted, on a different machine than
produced it, with 0 bad (`SCALING_COVER.md` §8.11).

The artifacts are in `data/rederive/`: the shard edges and sample digests in `s0/` and
`s69/`, and in `windows/` the complete `w19` certificate set (8,413 lines, small enough
to track), `w64`'s SUMMARY, both its edges, its digests, and the verifier's own output.

---

## 8. What this is and is not

It **is** a verification that every prime in `[255255, 1.8446744065×10¹⁹)` is
Erdős–Straus solvable, with a hashed, re-derivable certificate for each, produced by
a deterministic pipeline whose mathematical inputs are pinned by digest.

It is **not**:

- a proof of the conjecture;
- a claim about the full 64-bit range (see §1);
- a certificate archive — the certificates are hashes plus a reproduction recipe;
- a claim that stage E works (§2).
