#!/usr/bin/env python3
"""Parse the .meta files that tools/run19.sh writes, one per shard.

A shard meta is the ONLY durable artifact of a production sieve run: the raw
certificate stream is hashed and deleted (it would be ~3 TB), so the meta's
cert_sha256 plus the run's provenance is what makes a shard re-derivable.

Layout (see tools/run19.sh):

    # shard I/N  rc=R  wall=Ws  attempt=A  YYYY-MM-DD HH:MM:SS
    # binary_commit=... table_sha256=... threads=...
    certs=...
    cert_sha256=...
    SURVIVOR ...            (zero or more -- the alarm, verbatim)
    SUMMARY lo=... hi=... ...     (last; its presence marks the meta complete)

Stdlib only, deliberately: this runs on a bare VM and inside the container.
"""
import re, os, glob, datetime

HDR = re.compile(r'# shard (\d+)/(\d+)\s+rc=(-?\d+)\s+wall=(\d+)s\s+attempt=(\d+)\s+(.*)')


def parse_meta(path):
    """One .meta -> a flat dict. Ints stay ints; unknown keys are kept as strings."""
    d = {'path': path, 'survivor_lines': [], 'complete': False}
    with open(path) as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            m = HDR.match(line)
            if m:
                d.update(idx=int(m.group(1)), nshards=int(m.group(2)), rc=int(m.group(3)),
                         wall=int(m.group(4)), attempt=int(m.group(5)))
                try:
                    d['finished'] = datetime.datetime.strptime(m.group(6).strip(),
                                                               '%Y-%m-%d %H:%M:%S')
                except ValueError:
                    pass
            elif line.startswith('# '):
                for kv in line[2:].split():
                    k, _, v = kv.partition('=')
                    if k:
                        d[k] = int(v) if v.isdigit() else v
            elif line.startswith('SURVIVOR'):
                d['survivor_lines'].append(line)
            elif line.startswith('SUMMARY'):
                for kv in line.split()[1:]:
                    k, _, v = kv.partition('=')
                    d[k] = int(v) if v.isdigit() else v
                d['complete'] = True          # SUMMARY is written last
            elif '=' in line:
                k, _, v = line.partition('=')
                d[k] = int(v) if v.isdigit() else v
    # primes = Miller-Rabin tests that did not come back composite; equivalently
    # the three terminal states a prime can reach.
    if 'mr' in d and 'composite' in d:
        d['primes'] = d['mr'] - d['composite']
    return d


def load_dir(dirname):
    """All shard_*.meta in one directory, ordered by shard index."""
    paths = glob.glob(os.path.join(dirname, 'shard_*.meta'))
    ds = [parse_meta(p) for p in paths]
    ds.sort(key=lambda d: d.get('idx', -1))
    return ds


def load_run(dirs):
    """Several shard directories (one per box) merged into one band, ordered by lo."""
    ds = []
    for d in dirs:
        for s in load_dir(d):
            s['box'] = os.path.basename(os.path.normpath(d))
            ds.append(s)
    ds.sort(key=lambda s: s.get('lo', 0))
    return ds
