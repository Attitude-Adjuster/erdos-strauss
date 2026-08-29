#!/usr/bin/env python3
"""verify_sample.py -- close the certificate-correctness gap for a hashed production run.

    python3 sieve/verify_sample.py --lo LO --hi HI --certs N --stride 1024 \
        sample.txt head.txt tail.txt

WHY THIS EXISTS. run19/run64 hash their certificate stream and delete it -- 6.3 TB
against a 150 GB disk -- so a shard meta commits to the stream without preserving it.
That is sound only if the stream is reproducible, and it says nothing about whether the
certificates in it are CORRECT. Re-running a shard settles the first (its cert_sha256
must reappear byte for byte); this tool settles the second on a deterministic sample of
that re-run, on a different machine than produced it.

WHAT IT ADDS OVER verify_covers.py, which it calls for the exact-rational identity:

  primality   every certified p is re-tested here with a deterministic Miller-Rabin
              (the first 12 primes as bases: exact below 3.3e24, so exact for anything
              this project can reach). The scanner's own primality is a device kernel;
              this is unrelated code on unrelated hardware.
  range       every p lies in [LO, HI) -- the shard actually certified its own band and
              not some neighbouring one. A shard-arithmetic fault that shifted a band
              would leave the identity check perfectly happy.
  class       every p is 1 mod 24. The reduction of the conjecture's hard case to
              p = 1 (mod 24) is derived in this project, not cited, so a certificate
              for anything else means the wheel is wrong.
  count       the sample's line count matches certs/stride to within one, which is what
              makes the sample a statement about the whole stream rather than about
              whatever happened to land in a file.

Any failure exits non-zero and names the line. Stdlib only.
"""
import sys, os, argparse

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from verify_covers import verify_output

_MR_BASES = (2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37)


def is_prime(n):
    """Deterministic for n < 3.317e24 with these bases (Sorenson-Webster)."""
    if n < 2: return False
    for p in _MR_BASES:
        if n % p == 0: return n == p
    d, s = n - 1, 0
    while d % 2 == 0: d //= 2; s += 1
    for a in _MR_BASES:
        x = pow(a, d, n)
        if x == 1 or x == n - 1: continue
        for _ in range(s - 1):
            x = x * x % n
            if x == n - 1: break
        else:
            return False
    return True


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('files', nargs='+')
    ap.add_argument('--lo', type=int, required=True)
    ap.add_argument('--hi', type=int, required=True)
    ap.add_argument('--certs', type=int, help='certs= from the shard meta')
    ap.add_argument('--stride', type=int, help='sampling stride used on the box')
    ap.add_argument('--sample', help='which of the files is the strided sample')
    args = ap.parse_args()

    bad = comp = out = cls = n = 0
    for path in args.files:
        for ln, line in enumerate(open(path), 1):
            t = line.split()
            if not t or t[0] not in ('RUNG', 'SOLVED'): continue
            n += 1
            p = int(t[1])
            if not is_prime(p):
                comp += 1; print(f"{path}:{ln}: NOT PRIME {p}")
            if not (args.lo <= p < args.hi):
                out += 1; print(f"{path}:{ln}: OUT OF BAND {p}")
            if p % 24 != 1:
                cls += 1; print(f"{path}:{ln}: NOT 1 mod 24 {p}")
    print(f"independent checks: {n} certificates; {comp} composite, {out} out of band, "
          f"{cls} outside the hard class")
    bad += comp + out + cls

    if args.certs and args.stride and args.sample:
        want = args.certs // args.stride
        got = sum(1 for line in open(args.sample) if line.split()[:1] in (['RUNG'], ['SOLVED']))
        ok = abs(got - want) <= 1
        print(f"sample count: {got} lines vs certs/stride = {args.certs}/{args.stride} "
              f"= {want}  {'-- OK' if ok else '-- MISMATCH'}")
        if not ok: bad += 1

    rc = verify_output(args.files)          # the exact-rational identity, unmodified
    return 1 if (bad or rc) else 0


if __name__ == '__main__':
    sys.exit(main())
