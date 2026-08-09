#!/usr/bin/env python3
"""merge.py -- combine and validate the shard outputs of a rung_scan census.

    python3 merge.py OUTDIR

Reads OUTDIR/shard_*.out, checks that the shards tile the requested range with no
gap and no overlap, checks each shard's histogram against its own counters, sums
everything, and writes:

    OUTDIR/MERGED.txt       the merged summary and level table
    OUTDIR/interesting.txt  all REALCHAR / SUPPORT / DEEP / ESCALATE lines, sorted

Exits non-zero if any consistency check fails, so it is safe to use in a pipeline.
Scanner output is streamed per window and therefore unsorted; this script sorts.
"""
import sys
import os
import glob
from collections import Counter


def parse_summary(line):
    d = {}
    for kv in line.split()[1:]:
        if "=" not in kv:
            continue
        k, v = kv.split("=", 1)
        d[k] = v
    hist = Counter()
    for part in d.get("hist", "").split(","):
        if ":" in part:
            r, n = part.split(":")
            hist[int(r)] += int(n)
    return d, hist


def main(argv):
    if len(argv) < 2:
        print(__doc__)
        return 2
    outdir = argv[1]
    files = sorted(glob.glob(os.path.join(outdir, "shard_*.out")),
                   key=lambda f: int(f.rsplit("_", 1)[1].split(".")[0]))
    if not files:
        print(f"no shard_*.out files in {outdir}", file=sys.stderr)
        return 2

    errors, spans = [], []
    tot = Counter()
    hist = Counter()
    maxr, maxr_p = 0, 0
    interesting = []

    for f in files:
        summ = None
        with open(f) as fh:
            for line in fh:
                if line.startswith("SUMMARY "):
                    summ = line.strip()
                elif line[:8] in ("REALCHAR", "SUPPORT ", "DEEP ", "ESCALATE") or \
                        line.startswith(("REALCHAR", "SUPPORT", "DEEP", "ESCALATE")):
                    parts = line.split()
                    if len(parts) == 4:
                        interesting.append((int(parts[1]), int(parts[3]), parts[0], int(parts[2])))
        if summ is None:
            errors.append(f"{os.path.basename(f)}: no SUMMARY line (shard incomplete?)")
            continue

        d, h = parse_summary(summ)
        lo, hi = int(d["lo"]), int(d["hi"])
        spans.append((lo, hi, os.path.basename(f)))
        n = {k: int(d[k]) for k in ("primes", "jac", "rc", "sup", "res", "escalated", "pairpath")}

        # per-shard internal consistency: every prime either hit at some rung or escalated
        if sum(h.values()) + n["escalated"] != n["primes"]:
            errors.append(f"{os.path.basename(f)}: histogram {sum(h.values())} + escalated "
                          f"{n['escalated']} != primes {n['primes']}")
        if n["pairpath"]:
            errors.append(f"{os.path.basename(f)}: pair-DP path used {n['pairpath']}x "
                          f"(should be unreachable for r < 3p)")

        for k, v in n.items():
            tot[k] += v
        hist += h
        mr, mp = int(d["maxr"]), int(d["maxr_p"])
        if mr > maxr or (mr == maxr and mp and (not maxr_p or mp < maxr_p)):
            maxr, maxr_p = mr, mp

    # coverage: shards must tile [min, max) exactly
    spans.sort()
    for i in range(1, len(spans)):
        if spans[i][0] != spans[i - 1][1]:
            errors.append(f"coverage break between {spans[i-1][2]} (ends {spans[i-1][1]}) "
                          f"and {spans[i][2]} (starts {spans[i][0]})")

    failures = tot["jac"] + tot["rc"] + tot["sup"] + tot["res"]
    lines = []
    A = lines.append
    A(f"range              [{spans[0][0]}, {spans[-1][1]})")
    A(f"shards             {len(files)}")
    A(f"primes = 1 (24)    {tot['primes']:,}")
    A(f"failed rungs       {failures:,}")
    A("")
    A(f"  level J   (Jacobi)              {tot['jac']:>16,}   {100*tot['jac']/failures:7.4f}%"
      if failures else "  level J   0")
    A(f"  level RC  (other real char)     {tot['rc']:>16,}   {100*tot['rc']/failures:7.4f}%"
      if failures else "  level RC  0")
    A(f"  level S   (beyond real chars)   {tot['sup']:>16,}   {100*tot['sup']/failures:7.4f}%"
      if failures else "  level S   0")
    A(f"  level R   (residual miss)       {tot['res']:>16,}   {100*tot['res']/failures:7.4f}%"
      if failures else "  level R   0")
    A("")
    A(f"max depth          {maxr}  at p = {maxr_p:,}")
    A(f"escalations        {tot['escalated']}   (no hit within --rmax; must be 0)")
    A("")
    A("depth histogram:")
    for r in sorted(hist):
        A(f"  r = {r:>4}   {hist[r]:>16,}")

    report = "\n".join(lines)
    print(report)
    with open(os.path.join(outdir, "MERGED.txt"), "w") as fh:
        fh.write(report + "\n")

    interesting.sort()
    with open(os.path.join(outdir, "interesting.txt"), "w") as fh:
        for p, r, kind, a in interesting:
            fh.write(f"{kind} {p} {a} {r}\n")
    print(f"\nwrote {outdir}/MERGED.txt and {outdir}/interesting.txt "
          f"({len(interesting):,} non-J / deep lines)")

    if errors:
        print("\nCONSISTENCY ERRORS:", file=sys.stderr)
        for e in errors:
            print("  " + e, file=sys.stderr)
        return 1
    print("\nall consistency checks passed")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
