# Re-derivation of two production shards (2026-08-28)

run19 and run64 hash their certificate stream and delete it — ~6.3 TB against a 150 GB
disk — so a shard meta commits to a stream it does not preserve. That is sound only if
the stream is reproducible, and it says nothing about whether the certificates in it are
correct. Two shards were re-run to settle both questions; `docs/sieve/RUN_REPORT.md` §7
has the result, and this directory has what was kept.

| | `s0/` | `s69/` |
|---|---|---|
| shard | run19 0/250 | run64 69/70 |
| band | `[255255, 40000000000254234)` | `[15590940119293459447, 15631162710079743316)` |
| meta | `data/c19/shard_0.meta` | `data/c64/cloud-gpu-3/shard_69.meta` |

Each directory holds:

    info.txt        what was run, on which binary, and the resulting cert_sha256
    summary.txt     the re-run's SUMMARY -- byte-identical to the meta's
    head.txt        the first 2,000 certificates emitted
    tail.txt        the last 2,000
    sample.sha256   digest of the strided sample (every 1024th certificate)

The **edges matter more than their size suggests**: a shard-boundary or offset-carry
fault does not perturb a uniform sample of a 16 GB stream, but it shows immediately at
the first or last line emitted.

The strided sample itself — 385,224 and 326,551 certificates, 35 MB — is not tracked.
It is a deterministic function of the shard, so re-running the shard regenerates it
exactly; `sample.sha256` is what lets you confirm you regenerated the same one. To
re-derive and re-verify:

    # on a box with an L4 and the binary at the commit named in info.txt
    sieve/cuda/sieve_cuda LO HI --shard I/N \
        --filters tables/filters/f_M2042040_mmax2000_gpu.txt --threads 12 \
      | tee >(grep -v '^SUMMARY ' | sha256sum) \
      | grep -v '^SUMMARY ' | awk 'NR % 1024 == 0' > sample.txt

    # anywhere, ideally not that machine
    python3 sieve/verify_sample.py --lo LO --hi HI --certs N --stride 1024 \
        --sample sample.txt sample.txt head.txt tail.txt

## windows/ — two narrow windows verified exhaustively

The sampling above is uniform, but a sample is still a sample. These two windows were
scanned fresh on `cloud-gpu-3` (binary `115d127`) and **every** certificate in them was
re-derived in exact rationals on the laptop:

| | `w19` | `w64` |
|---|---|---|
| window | `[5000000000000000000, 5000001000000000000)` | `[18445744065119614976, 18446744065119614976)` |
| width | 10¹² | 10¹⁵, ending exactly at `DIRECT_PMAX` |
| certificates | 8,413 | 8,284,620 |
| verified | all, 0 bad | all, 0 bad |

`w64` is the one that matters: it ends at `DIRECT_PMAX`, where `p + 4ade` comes within
8.6 × 10⁹ of 2⁶⁴. Coverage there is 99.995102%, `direct = 0`, `survivors = 0` — the top
of the range behaves like the rest of it.

    w19_certificates.txt   the complete set, all 8,413 (tracked in full)
    w64_summary.txt        the SUMMARY line
    w64_head.txt           the first 2,000 certificates
    w64_tail.txt           the last 2,000
    w64.sha256             digests of the full stdout and of the certificate-only stream
    w64_verify.txt         what sieve/verify_sample.py printed over all 8,284,620

`w64`'s 460 MB of certificates is not tracked; re-scan the window and check it against
`w64.sha256`. To reproduce either:

    sieve/cuda/sieve_cuda LO HI --filters tables/filters/f_M2042040_mmax2000_gpu.txt \
        --threads 12 > w.txt
    python3 sieve/verify_sample.py --lo LO --hi HI w.txt

## decades/ — one window per decade, 10⁶ to 10¹⁹

Fourteen windows anchored at each power of ten, scanned on `cloud-gpu-3` (binary
`115d127`) and verified in full — all 1,230 certificates, 0 bad. Below 10¹⁰ the window
is the whole decade; above it, 10¹⁰ wide, which yields ~100 stage-D certificates at
every height because their density is nearly flat (~1 per 1.2 × 10⁸ of width).

`TABLE.txt` has the summary; `d<k>.txt` the raw output; `decades_scan.txt` the run log.

Two things in it are worth knowing, and one of them is a trap:

- **Stage-D rungs stay small at every height** — max `r` over each window runs 11–47
  from 10⁶ to 10¹⁹ with no trend. The certificate search does not degrade with height.
- **These are NOT the census's search-depth records, and must never be quoted as
  such.** The census's max depth is 59 → 107 → 111 → 131 → 155 across 10⁶–10¹⁵
  (`data/c13/ANALYSIS.txt`, `data/c15/ANALYSIS.txt`), an exhaustive maximum over every
  hard-class prime below the bound. What the sieve emits is a different statistic on a
  different population: only primes that *survive the covers* ever get a rung
  certificate — about 1 in 20,000 of the hard class — and the rung is capped at
  `r ≤ 63` by the uniform-rung certificate bounds. A max over ~100 survivors is not a
  record over ~10¹¹ primes.
- At 10⁷ the whole decade emits **zero** certificates: every hard-class prime in
  `[10⁷, 10⁸)` is killed by a congruence cover. Covers are at their most effective on
  small primes, which is why the sieve's own coverage figure (99.995%) is a
  high-altitude number, not a universal one.

## binaries/ — the two production binaries

The one deliberate exception to this repo's rule that built scanners are never tracked.
These are not for reuse; they are provenance. 460 shard metas record hashes of streams
that *these exact binaries* produced, and the byte-for-byte reproduction in §7 of the
run report was done with them. Rebuilding from source may or may not reproduce the same
emission order under a different `nvcc`; keeping the binaries removes the question, and
they are 1.2 MB each.

    sieve_cuda_4258bc9    run19  (cloud-gpu)
    sieve_cuda_115d127    run64  (cloud-gpu-2/3/4)
    sha256.txt            digests, verified equal on the box and after transfer

Built for `sm_89` with CUDA 12.9 (V12.9.41) on Ubuntu 22.04.5, kernel 6.8.0-1066-gcp,
driver 580.173.02. They need an L4 (or another `sm_89` device) and a driver at least
that new; nothing else.
