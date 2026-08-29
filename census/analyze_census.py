#!/usr/bin/env python3
"""analyze_census.py -- turn census output into the H1/H2/H3 evidence.

    python3 analyze_census.py DIR [DIR ...]

Each DIR is a run8.sh output directory (containing shard_*.out). Re-parses the
per-shard SUMMARY lines rather than MERGED.txt, so it is robust to formatting, and:

  * reports the certificate-level breakdown and residual share per run,
  * assembles the residual-share series across runs and fits share = C (log X)^-a,
  * does a leave-the-last-out holdout, since a fit that cannot predict the next
    decade is a curve, not a law,
  * checks the level-S localisation against the DERIVED admissible rung set
    (es_levels.py) -- a level-S failure at an inadmissible rung falsifies the
    four-constraint proposition, so it is flagged as such rather than as a curiosity,
  * reports depth-tail structure and whether the depth-107 record has fallen,
  * flags anything that should be zero and isn't.

Compares against the previously published (1 mod 24) series, and keeps the two
populations separate -- they are different measurements and their shares differ by
roughly a factor of two.
"""
import sys, os, glob, math
from collections import Counter

import es_levels

# residual share among failed rungs, published (1 mod 24) census
PUBLISHED_24 = {5: 20/854, 8: 4018/382612, 10: 181523/25603575, 11: 1302101/216444393}
# the single hard-class point computed by exact reference at 10^5
PUBLISHED_H = {5: 20/434}


def read_dir(d):
    tot, hist = Counter(), Counter()
    maxr, maxr_p, spans, hard = 0, 0, [], None
    for f in glob.glob(os.path.join(d, "shard_*.out")):
        summ = None
        with open(f) as fh:
            for line in fh:
                if line.startswith("SUMMARY "):
                    summ = line
        if summ is None:
            print(f"  ! {os.path.basename(f)}: incomplete (no SUMMARY)", file=sys.stderr)
            continue
        kv = dict(x.split("=", 1) for x in summ.split()[1:] if "=" in x)
        spans.append((int(kv["lo"]), int(kv["hi"])))
        for k in ("primes", "jac", "rc", "sup", "res", "escalated", "pairpath"):
            tot[k] += int(kv[k])
        for part in kv.get("hist", "").split(","):
            if ":" in part:
                r, n = part.split(":")
                hist[int(r)] += int(n)
        if int(kv["maxr"]) > maxr:
            maxr, maxr_p = int(kv["maxr"]), int(kv["maxr_p"])
    spans.sort()
    lo, hi = (spans[0][0], spans[-1][1]) if spans else (0, 0)
    # infer population from prime density: hard class is 6/192 of reduced residues
    if tot["primes"] and hi > 100:
        expected24 = hi / (8 * math.log(hi))
        hard = tot["primes"] < 0.5 * expected24
    return dict(lo=lo, hi=hi, tot=tot, hist=hist, maxr=maxr, maxr_p=maxr_p, hard=hard)


def levels_of(d):
    """Rung distribution of the SUPPORT / REALCHAR lines.

    interesting.txt is merge.py's sorted copy of exactly the lines already in
    shard_*.out, so reading BOTH double-counts every entry. Prefer the merged
    file when it exists and fall back to the shards only when it does not.
    """
    merged = os.path.join(d, "interesting.txt")
    files = [merged] if os.path.exists(merged) else glob.glob(os.path.join(d, "shard_*.out"))
    seen = Counter()
    for f in files:
        with open(f) as fh:
            for line in fh:
                p = line.split()
                if p and p[0] in ("SUPPORT", "REALCHAR") and len(p) == 4:
                    seen[(p[0], int(p[3]))] += 1
    return seen


def fit(points):                       # points: {log10(X): share}
    xs = [math.log(math.log(10 ** k)) for k in points]
    ys = [math.log(v) for v in points.values()]
    n = len(xs); xb = sum(xs)/n; yb = sum(ys)/n
    sxx = sum((x-xb)**2 for x in xs)
    if sxx == 0: return None
    b = sum((x-xb)*(y-yb) for x, y in zip(xs, ys))/sxx
    a = yb - b*xb
    ss = sum((y-(a+b*x))**2 for x, y in zip(xs, ys))
    tot = sum((y-yb)**2 for y in ys)
    return math.exp(a), -b, (1-ss/tot if tot else float('nan'))


def main(argv):
    if len(argv) < 2:
        print(__doc__); return 2
    runs = []
    for d in argv[1:]:
        r = read_dir(d)
        if not r["tot"]["primes"]:
            print(f"{d}: no complete shards", file=sys.stderr); continue
        r["dir"] = d
        r["levels"] = levels_of(d)
        runs.append(r)

    series = {}
    for r in sorted(runs, key=lambda x: x["hi"]):
        t = r["tot"]
        fails = t["jac"] + t["rc"] + t["sup"] + t["res"]
        k = round(math.log10(r["hi"]))
        pop = "hard class (6 squares mod 840)" if r["hard"] else "(1 mod 24)"
        print(f"\n=== {r['dir']}   [{r['lo']:,} , {r['hi']:,})   {pop} ===")
        print(f"  primes            {t['primes']:>16,}")
        print(f"  failed rungs      {fails:>16,}")
        for lab, key in (("J  (Jacobi)", "jac"), ("RC (other real chi)", "rc"),
                         ("S  (beyond real chi)", "sup"), ("R  (residual)", "res")):
            print(f"    level {lab:<22}{t[key]:>14,}   {100*t[key]/fails:8.4f}%")
        print(f"  max depth         {r['maxr']} at p = {r['maxr_p']:,}"
              + ("   <-- RECORD 107 BEATEN" if r["maxr"] > 107 else
                 "   (ties the standing record)" if r["maxr"] == 107 else ""))
        for k2, v in (("escalated", t["escalated"]), ("pairpath", t["pairpath"])):
            flag = "  <-- SHOULD BE ZERO, INVESTIGATE" if v else "  ok"
            print(f"  {k2:<17} {v}{flag}")
        # level-S localisation
        S = {rr: c for (kind, rr), c in r["levels"].items() if kind == "SUPPORT"}
        RC = {rr: c for (kind, rr), c in r["levels"].items() if kind == "REALCHAR"}
        if S:
            # The admissible set is DERIVED (es_levels.py), not tabulated: a rung can
            # carry level S only if some subgroup of (Z/r)* contains 4, omits -1, and
            # has (-1)H square in the quotient. A rung outside it is not "news" in the
            # interesting sense -- it falsifies the four-constraint proposition, or the
            # scanner is wrong. Either way it is the loudest thing in the report.
            allowed = set(es_levels.s_admissible_rungs(max(max(S), 3)))
            illegal = sorted(set(S) - allowed)
            note = ("   still ONLY r=51" if set(S) == {51} else
                    f"   <-- IMPOSSIBLE RUNGS {illegal}, this falsifies the "
                    f"four-constraint (admissible: {sorted(allowed)})" if illegal else
                    f"   new rung, but admissible (admissible: {sorted(allowed)})")
            print(f"  level-S by rung   {dict(sorted(S.items()))}{note}")
        if RC:
            print(f"  level-RC by rung  {dict(sorted(RC.items()))}")
        tail = {rr: n for rr, n in sorted(r["hist"].items()) if rr >= 71}
        if tail:
            print(f"  depth tail (>=71) {tail}")
        series[k] = (t["res"]/fails, r["hard"])

    # ---- H1 on DISJOINT decades
    #
    # Every run above is cumulative [0, X), so the runs are nested: the 10^13
    # census contains the whole 10^12 census. A holdout on the cumulative series
    # therefore predicts a number that is ~88% training data by weight, and will
    # look accurate even if the trend is wrong. Differencing consecutive runs
    # gives disjoint decades [X/10, X), where a holdout is a real one.
    cum = [r for r in sorted(runs, key=lambda x: x["hi"]) if r["lo"] == 0 and r["hard"]]
    if len(cum) >= 3:
        print("\n=== H1 on DISJOINT decades (cumulative runs differenced) ===")
        dec = {}
        for prev, cur in zip(cum, cum[1:]):
            tp, tc = prev["tot"], cur["tot"]
            fp = tp["jac"] + tp["rc"] + tp["sup"] + tp["res"]
            fc = tc["jac"] + tc["rc"] + tc["sup"] + tc["res"]
            df, dr = fc - fp, tc["res"] - tp["res"]
            if df <= 0 or dr <= 0:
                continue
            k = round(math.log10(cur["hi"]))
            dec[k] = dr / df
            print(f"  [10^{k-1}, 10^{k}) {100*dec[k]:8.4f}%   "
                  f"({dr:,} residual / {df:,} failed)")
        if len(dec) >= 3:
            ks = sorted(dec)
            g = fit({k: dec[k] for k in ks[:-1]})
            if g:
                pred = g[0]/math.log(10**ks[-1])**g[1]
                act = dec[ks[-1]]
                print(f"  honest holdout (fit on decades through 10^{ks[-2]}, all")
                print(f"  disjoint from the target): predicted {100*pred:.4f}%, "
                      f"actual {100*act:.4f}%, error {abs(pred-act)/act*100:.1f}%")

    # ---- H1 series
    for hard in (True, False):
        pts = {k: v for k, (v, h) in series.items() if h == hard}
        base = PUBLISHED_H if hard else PUBLISHED_24
        merged = dict(base); merged.update(pts)
        if len(merged) < 2: continue
        print(f"\n=== H1: residual share among failed rungs "
              f"[{'hard class' if hard else '(1 mod 24)'}] ===")
        for k in sorted(merged):
            tag = "  (new)" if k in pts else "  (published)"
            print(f"  X = 10^{k:<3} {100*merged[k]:8.4f}%{tag}")
        f = fit(merged)
        if f:
            C, a, r2 = f
            print(f"  fit: share ~ {C:.3f} / (log X)^{a:.3f}   R^2 = {r2:.5f}"
                  f"   -> {'consistent with H1' if a > 1 else 'does NOT vanish: a <= 1'}")
            if len(merged) >= 3:                      # holdout on the last point
                ks = sorted(merged); tr = {k: merged[k] for k in ks[:-1]}
                g = fit(tr)
                if g:
                    pred = g[0]/math.log(10**ks[-1])**g[1]
                    act = merged[ks[-1]]
                    print(f"  holdout (fit without 10^{ks[-1]}): predicted {100*pred:.4f}%,"
                          f" actual {100*act:.4f}%, error {abs(pred-act)/act*100:.1f}%")
            for k in (15, 18):
                print(f"  extrapolated at 10^{k}: {100*C/math.log(10**k)**a:.4f}%")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
