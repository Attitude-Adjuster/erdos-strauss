#!/usr/bin/env python3
"""Descriptive report for a production sieve run: timing, throughput, height scaling.

    python3 sieve/analyze_run.py data/c19
    python3 sieve/analyze_run.py --band data/c64/cloud-gpu-{2,3,4} --label run64
    python3 sieve/analyze_run.py data/c19 --csv shards.csv

This is the DESCRIPTIVE half; sieve/verify_shards.py is the pass/fail gate. Keeping
them apart mirrors census/analyze_census.py vs census/verify_cert.py.

The height-scaling table is the reason this exists. Shards are equal-WIDTH, so a
single production run measures cost against height directly, across every decade it
spans, with no separate benchmark and no extrapolation. This project has repeatedly
been burned by carrying a scaling law from one port to another (the census port's
1.17x/decade onto the sieve, the CPU's 1.25x onto the GPU); a run that reports its
own curve cannot be misquoted that way.

Stdlib only.
"""
import sys, os, argparse, statistics, math
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from shard_meta import load_dir, load_run

CSV_COLS = ('box', 'idx', 'lo', 'hi', 'width', 'wall', 'attempt', 'positions', 'covered',
            'mr', 'composite', 'primes', 'rung', 'direct', 'sieved', 'survivors', 'certs',
            'cert_sha256', 'binary_commit', 'finished')


def si(x, unit='', places=3):
    return f"{x:.{places}e}{unit}"


def report(ds, label, price_per_hour=None, instance_hours=None):
    tot = lambda k: sum(d.get(k, 0) for d in ds)
    lo, hi = ds[0]['lo'], ds[-1]['hi']
    width = hi - lo
    walls = [d['wall'] for d in ds]
    gpu_s = sum(walls)
    pos, cov = tot('positions'), tot('covered')
    primes, certs = tot('primes'), tot('certs')

    print(f"\n{'='*78}\n{label}\n{'='*78}")
    print(f"band            [{lo}, {hi})")
    print(f"                 = [{si(lo)}, {si(hi)})   width {width:,} integers")
    print(f"shards          {len(ds)}  x  {ds[0]['hi']-ds[0]['lo']:,} integers each")
    boxes = sorted({d.get('box') for d in ds if d.get('box')})
    if boxes:
        print(f"boxes           {len(boxes)}: {', '.join(boxes)}")

    print(f"\n-- wheel and filter table")
    print(f"wheel           M={ds[0]['wheelM']:,}  lanes={ds[0]['lanes']}  "
          f"spacing={ds[0]['wheelM']//ds[0]['lanes']}  wheel_pmin={ds[0]['wheel_pmin']:,}")
    print(f"covers          mmax={ds[0]['mmax']:,}  uniform-rung certs r<={ds[0]['ucert_rmax']} "
          f"u<={ds[0]['ucert_umax']}")
    print(f"table sha256    {ds[0]['table_sha256']}")
    print(f"classes sha256  {ds[0]['classes_sha256']}")
    print(f"binary commit   {','.join(sorted({str(d.get('binary_commit')) for d in ds}))}")

    print(f"\n-- the sieve, stage by stage")
    print(f"positions       {pos:,}   ({si(pos)})   = width * lanes/M, the wheel survivors")
    print(f"  covered       {cov:,}   {cov/pos*100:.6f}%   killed by a congruence cover (stage A/B)")
    print(f"  to primality  {pos-cov:,}   {(pos-cov)/pos*100:.6f}%")
    print(f"  Miller-Rabin  {tot('mr'):,} tested")
    print(f"    composite   {tot('composite'):,}   {tot('composite')/tot('mr')*100:.3f}%")
    print(f"    PRIME       {primes:,}   {primes/tot('mr')*100:.3f}%")
    print(f"      stage D   {tot('rung'):,}  solved by rung certificate")
    print(f"      stage E   {tot('direct'):,}  solved by direct search")
    print(f"      SURVIVOR  {tot('survivors'):,}  <-- must be 0")
    print(f"certificates    {certs:,} emitted and hashed "
          f"(~{certs*40/1e12:.2f} TB at ~40 B/line, never stored)")
    print(f"prime density   1 per {width/primes:,.1f} integers of the band")

    print(f"\n-- timing")
    print(f"GPU time        {gpu_s/3600:.2f} GPU-hours  ({gpu_s/86400:.3f} GPU-days)")
    print(f"per shard       min {min(walls)}s   median {statistics.median(walls):.0f}s   "
          f"max {max(walls)}s   spread {max(walls)/min(walls):.2f}x")
    print(f"throughput      {width/gpu_s:,.0f} integers/s     {si(gpu_s/width*1e9,' ns/integer')}")
    print(f"                {primes/gpu_s:,.0f} primes/s        {gpu_s/primes*1e9:,.1f} ns/prime")
    print(f"                {pos/gpu_s/1e9:.3f} Gpositions/s")
    if any(d.get('attempt', 1) > 1 for d in ds):
        print(f"retries         {sum(d.get('attempt',1)-1 for d in ds)}")

    if instance_hours:
        print(f"\n-- cost (reconstructed from audit-log instance lifetimes, not an invoice)")
        eff = gpu_s/3600/instance_hours*100
        print(f"instance hours  {instance_hours:.2f} h billed vs {gpu_s/3600:.2f} h computing "
              f"= {eff:.1f}% duty cycle")
        if price_per_hour:
            print(f"at ${price_per_hour:.2f}/h   ${instance_hours*price_per_hour:,.2f}")
            print(f"                ${instance_hours*price_per_hour/primes*1e9:,.2f} per 10^9 primes"
                  f"   ${instance_hours*price_per_hour/width*1e18:,.2f} per 10^18 integers")

    # ---- height scaling: equal-width shards, so wall IS the cost curve.
    # Bin by POSITION in the band, not by decade: the shards are equal-width, so a
    # band reaching 10^19 puts ~90% of its shards in the top decade and decade bins
    # collapse to one useful row. Position bins give the curve at even resolution;
    # the midpoint height of each bin is what it is plotted against.
    print(f"\n-- cost against height (equal-width shards; this run measuring itself)")
    print(f"   {'bin':>3} {'shards':>6} {'mid height':>11} {'wall/shard':>11} "
          f"{'ns/integer':>11} {'ns/prime':>10} {'primes/shard':>14} {'growth':>8}")
    NB = min(10, len(ds))
    bins = [ds[i*len(ds)//NB:(i+1)*len(ds)//NB] for i in range(NB)]
    prev, rows = None, []
    for i, b in enumerate(bins):
        if not b:
            continue
        w = sum(x['wall'] for x in b)
        wd = sum(x['hi'] - x['lo'] for x in b)
        p_ = sum(x['primes'] for x in b)
        mid = (b[0]['lo'] + b[-1]['hi']) // 2
        nsi = w / wd * 1e9
        rows.append((mid, nsi, w / p_ * 1e9))
        print(f"   {i:>3} {len(b):>6} {si(mid,'',2):>11} {w/len(b):>10.0f}s {nsi:>11.5f} "
              f"{w/p_*1e9:>10.1f} {p_/len(b):>14,.0f} {(f'{nsi/prev:.3f}x' if prev else '--'):>8}")
        prev = nsi
    # least-squares slope of log(ns/integer) against log10(height) -- the decade factor
    if len(rows) > 2:
        xs = [math.log10(m) for m, _, _ in rows]
        ys = [math.log(n) for _, n, _ in rows]
        n = len(xs)
        mx, my = sum(xs)/n, sum(ys)/n
        den = sum((x-mx)**2 for x in xs)
        if den > 0:
            slope = sum((x-mx)*(y-my) for x, y in zip(xs, ys))/den
            print(f"   fit: ns/integer grows {math.exp(slope):.3f}x per decade of height "
                  f"(log-log least squares over {n} bins, 10^{xs[0]:.1f}..10^{xs[-1]:.1f})")
    return ds


def write_csv(ds, path):
    import csv
    with open(path, 'w', newline='') as fh:
        w = csv.writer(fh)
        w.writerow(CSV_COLS)
        for d in ds:
            row = []
            for c in CSV_COLS:
                if c == 'width':
                    row.append(d['hi'] - d['lo'])
                else:
                    row.append(d.get(c, ''))
            w.writerow(row)
    print(f"\nwrote {path}  ({len(ds)} rows)")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('dirs', nargs='+')
    ap.add_argument('--band', action='store_true', help='merge dirs into one band')
    ap.add_argument('--label', default=None)
    ap.add_argument('--csv', default=None, help='write the per-shard table here')
    ap.add_argument('--instance-hours', type=float, default=None)
    ap.add_argument('--price', type=float, default=None, help='USD per instance-hour')
    a = ap.parse_args()

    ds = load_run(a.dirs) if a.band else load_dir(a.dirs[0])
    report(ds, a.label or ' + '.join(a.dirs), a.price, a.instance_hours)
    if a.csv:
        write_csv(ds, a.csv)


if __name__ == '__main__':
    sys.exit(main())
