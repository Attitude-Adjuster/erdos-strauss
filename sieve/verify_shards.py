#!/usr/bin/env python3
"""Audit a completed sieve run from its shard metas. Exit 0 iff everything checks.

    python3 sieve/verify_shards.py data/c19
    python3 sieve/verify_shards.py --band data/c64/cloud-gpu-{2,3,4}

WHY THIS EXISTS, and what it checks that nothing else does: it compares shard
BOUNDARIES, not summed SUMMARY counters. The first --shard gate this project
wrote summed every counter and PASSED with a tail-dropping span deliberately
compiled in -- a floor-instead-of-ceil span drops only (HI-LO) mod N integers,
and at one wheel position per 885 integers a handful of dropped integers almost
always contain no prime, so every total still reconciled. Totals are not a
substitute for boundaries.

--band merges several directories into one contiguous range, which is how a run
split across boxes is checked: the seams BETWEEN boxes are exactly where a
hand-computed split goes wrong, and they are invisible to any per-box check.

Stdlib only.
"""
import sys, os, argparse
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from shard_meta import load_dir, load_run

# Provenance keys that must be identical across every shard of a run. binary_commit
# is deliberately NOT here: a run may legitimately be resumed with a rebuilt binary,
# and the mathematical inputs (table, class digest, wheel, certificate bounds) are
# what must not move. It is reported instead.
UNIFORM = ('table_sha256', 'classes_sha256', 'wheelM', 'lanes', 'mmax',
           'ucert_rmax', 'ucert_umax', 'wheel_pmin')


def check(ds, label, expect_lo=None, expect_hi=None, expect_n=None):
    ok = True
    def fail(msg):
        nonlocal ok
        print(f"  FAIL {msg}")
        ok = False
    def good(msg):
        print(f"  ok   {msg}")

    print(f"== {label}  ({len(ds)} shard metas)")
    if not ds:
        fail("no shard metas found")
        return False

    incomplete = [d.get('idx') for d in ds if not d['complete']]
    if incomplete:
        fail(f"metas without a SUMMARY line (shard truncated?): {incomplete}")
    else:
        good("every meta complete (SUMMARY present)")

    # completeness of the shard index set, per box
    for box in sorted({d.get('box', label) for d in ds}):
        sub = [d for d in ds if d.get('box', label) == box]
        n = sub[0].get('nshards')
        idxs = sorted(d['idx'] for d in sub)
        missing = sorted(set(range(n)) - set(idxs))
        dupes = len(idxs) != len(set(idxs))
        tag = f"{box}: " if len({d.get('box', label) for d in ds}) > 1 else ""
        if missing or dupes:
            fail(f"{tag}shard indices not exactly 0..{n-1} (missing {missing}, dupes {dupes})")
        else:
            good(f"{tag}shard indices 0..{n-1} complete, no duplicates")

    bad_rc = [(d.get('box'), d['idx'], d['rc']) for d in ds if d.get('rc') != 0]
    fail(f"shards with rc != 0: {bad_rc}") if bad_rc else good("every shard rc=0")

    # THE check: boundaries, in sorted-by-lo order, must tile the band exactly.
    gaps = [(a.get('hi'), b.get('lo')) for a, b in zip(ds, ds[1:]) if a.get('hi') != b.get('lo')]
    if gaps:
        fail(f"{len(gaps)} boundary gap(s)/overlap(s), first: hi={gaps[0][0]} then lo={gaps[0][1]} "
             f"(delta {gaps[0][1] - gaps[0][0]:+d})")
    else:
        good(f"boundaries tile exactly: [{ds[0]['lo']}, {ds[-1]['hi']})")
    if expect_lo is not None and ds[0]['lo'] != expect_lo:
        fail(f"band starts at {ds[0]['lo']}, expected {expect_lo}")
    if expect_hi is not None and ds[-1]['hi'] != expect_hi:
        fail(f"band ends at {ds[-1]['hi']}, expected {expect_hi}")
    if expect_n is not None and len(ds) != expect_n:
        fail(f"{len(ds)} shards, expected {expect_n}")

    # the mathematical result
    surv = sum(d.get('survivors', 0) for d in ds)
    lines = [l for d in ds for l in d['survivor_lines']]
    if surv or lines:
        fail(f"survivors={surv}, {len(lines)} SURVIVOR line(s) -- an unsolved prime")
        for l in lines[:20]:
            print("        ", l)
    else:
        good("survivors=0 across the band, no SURVIVOR lines")

    # per-shard accounting identity: every prime ends in exactly one terminal state
    bad = [d['idx'] for d in ds
           if d.get('primes') != d.get('rung', 0) + d.get('direct', 0) + d.get('survivors', 0)]
    if bad:
        fail(f"mr-composite != rung+direct+survivors in shards {bad[:10]}")
    else:
        good("accounting identity holds per shard: mr - composite = rung + direct + survivors")

    for k in UNIFORM:
        vals = {d.get(k) for d in ds}
        if len(vals) != 1:
            fail(f"{k} not uniform across shards: {sorted(map(str, vals))[:5]}")
    if ok:
        good(f"provenance uniform: table={ds[0]['table_sha256'][:8]}… "
             f"classes={ds[0]['classes_sha256'][:8]}… wheelM={ds[0]['wheelM']} "
             f"lanes={ds[0]['lanes']} mmax={ds[0]['mmax']} "
             f"ucert=(r<={ds[0]['ucert_rmax']},u<={ds[0]['ucert_umax']})")

    hashes = [d.get('cert_sha256') for d in ds]
    if len(set(hashes)) != len(hashes):
        fail("duplicate cert-stream sha256 -- two shards produced identical output")
    else:
        good(f"{len(hashes)} distinct cert-stream digests")

    commits = sorted({d.get('binary_commit') for d in ds})
    retries = sum(d.get('attempt', 1) - 1 for d in ds)
    print(f"  --   binary_commit={','.join(map(str, commits))}  retries={retries}")
    print(f"  ==> {'PASS' if ok else 'FAIL'}")
    return ok


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('dirs', nargs='+', help='directories holding shard_*.meta')
    ap.add_argument('--band', action='store_true',
                    help='merge all dirs into one contiguous band and check the seams between them')
    ap.add_argument('--expect-lo', type=int)
    ap.add_argument('--expect-hi', type=int)
    ap.add_argument('--expect-shards', type=int)
    a = ap.parse_args()

    if a.band:
        ok = check(load_run(a.dirs), ' + '.join(a.dirs),
                   a.expect_lo, a.expect_hi, a.expect_shards)
    else:
        ok = all([check(load_dir(d), d, a.expect_lo, a.expect_hi, a.expect_shards)
                  for d in a.dirs])
    return 0 if ok else 1


if __name__ == '__main__':
    sys.exit(main())
