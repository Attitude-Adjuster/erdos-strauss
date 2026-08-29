#!/usr/bin/env python3
"""verify_covers.py -- independent exact-arithmetic trust anchor for cover_scan.

Everything cover_scan.cpp claims is re-derived here in unbounded Python integers,
with different code. One tool, five modes plus output verification:

  --identity            re-prove the boxed decomposition on an exhaustive small grid
                        plus random large tuples, in exact rational arithmetic
  --uniform [--verbose] search uniform rung certificates (r, u) -- a rung plus a fixed
                        witness whose conditions are forced by p's residue class alone
                        -- to r <= 63, u <= 64, and re-verify every one in exact
                        rational arithmetic
  --class-table [--wheel M] [--write | --check FILE]
                        regenerate the certified class table mod M (default 120120):
                        for M=120120, 220 surviving
                        classes plus a named killing certificate for each of the other
                        22,820 unit classes; write or byte-diff class_table_120120.txt
  --diff-filters FILE   byte-level check of `cover_scan --dump-filters` output:
                        every CLASS/FILTER line regenerated and compared
  --sweep N             EXHAUSTIVE coverage sweep of the unbounded family for p <= N
                        (reparameterized by (a,d,e,k), c=(a+e)/k -- see note below);
                        expected survivors: perfect squares + {288, 336, 4545}
  FILE [FILE...]        verify scanner output lines:
                          RUNG p A r u       exact witness: u | q^2, u ≡ -q (mod r),
                                             decomposition {A, (q+u)/r, q(q+u)/(ru)}
                          SOLVED p a c d k   exact identity instance
                          SURVIVOR p         counted and reported (exit 1 if any)

The boxed decomposition: m=4acd-1, e=ck-a>=1, p=mk-4a^2d  ==>  p+k=4ade and
    4/p = 1/(ade) + 1/(acdp) + 1/(cdep).
Proof: numerator over acdep is cp+e+a = (4acde-ck)+e+a = 4acde since ck=e+a.

Exhaustive-sweep note: any instance with p <= N has k >= 1, c >= 1, so
4ade = p+k <= p+a+e, giving (4a-1)(4e-1) <= 4N+1: the (a,d,e,k)-enumeration below
provably visits every instance with p <= N. The survivor list is therefore a
statement about the UNBOUNDED family -- but only below N. Do not extrapolate.
"""
import sys, hashlib, os
from fractions import Fraction
from math import gcd, isqrt
import random


# ---------------------------------------------------------------- shared pieces

# Artifacts live in tables/ at the repo root; resolve against this file's location so
# the tool works from any working directory.
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def table_path(M):
    return os.path.join(ROOT, "tables", f"class_table_{M}.txt")


def gen_covers(mmax):
    """(m, r, pmin, a, c, d) for all m=4acd-1 <= mmax, dedup (m,r) by min pmin."""
    best = {}
    nmax = (mmax + 1) // 4
    for a in range(1, nmax + 1):
        for d in range(1, nmax // a + 1):
            for c in range(1, nmax // (a * d) + 1):
                m = 4 * a * c * d - 1
                r = (-4 * a * a * d) % m
                pmin = m * (a // c + 1) - 4 * a * a * d
                key = (m, r)
                if key not in best or pmin < best[key][0]:
                    best[key] = (pmin, a, c, d)
    return sorted((m, r, p, a, c, d) for (m, r), (p, a, c, d) in best.items())


def gen_wheel_covers(M):
    """(m, res, pmin, a, c, d) for covers whose modulus DIVIDES M -- the only ones the
    wheel can use. Driven by the divisors of M rather than by enumerating every cover
    <= M, which at M = 2,042,040 would be ~4.4e7 triples. m = 3 (mod 4) is not a
    convenience filter: F1's modulus is 4N-1 always, so no other divisor can host a
    cover. Dedup and order match gen_covers exactly; --class-table asserts it."""
    divs, n = [1], M
    q = 2
    while q * q <= n:
        if n % q == 0:
            e = 0
            while n % q == 0: n //= q; e += 1
            divs = [d * q**i for d in divs for i in range(e + 1)]
        q += 1
    if n > 1: divs = [d * n**i for d in divs for i in range(2)]
    best = {}
    for m in sorted(divs):
        if m % 4 != 3: continue
        t = (m + 1) // 4
        for a in range(1, t + 1):
            if t % a: continue
            ta = t // a
            for c in range(1, ta + 1):
                if ta % c: continue
                d = ta // c
                res = (-4 * a * a * d) % m
                pmin = m * (a // c + 1) - 4 * a * a * d
                key = (m, res)
                if key not in best or pmin < best[key][0]: best[key] = (pmin, a, c, d)
    return sorted((m, r, p, a, c, d) for (m, r), (p, a, c, d) in best.items())


def uniform_certs(rmax=63, umax=64):
    """Uniform rung certificates: (L, res, r, u) such that EVERY p ≡ res (mod L)
    is solved by A=(p+r)/4, x=(q+u)/r, y=q(q+u)/(r u) with q = p*A.

    Both rung conditions -- u | q^2 and u ≡ -q (mod r) -- depend only on p mod L
    with L = 4*lcm(r,u): p = rho + L t gives A = A_rho + lcm(r,u) t, so q is fixed
    mod r and mod u.  gcd(u,r) = 1 is required for y's integrality:
    y = (q^2/u + q)/r and q^2/u ≡ -q (mod r) needs u invertible mod r.
    Deduped by (L, res) keeping the smallest (r, u) so the table is canonical."""
    out = {}
    for r in range(1, rmax + 1):
        for u in range(1, umax + 1):
            if gcd(u, r) != 1: continue
            L = 4 * (r * u // gcd(r, u))
            for rho in range(L):
                if (rho + r) % 4: continue           # A = (p+r)/4 must be integral
                ok = True
                for lift in (rho, rho + L):          # guard: the class really decides
                    A = (lift + r) // 4
                    q = lift * A
                    if (q * q) % u or (q + u) % r: ok = False; break
                if not ok: continue
                key = (L, rho)
                if key not in out or (r, u) < out[key]: out[key] = (r, u)
    return sorted((L, rho, r, u) for (L, rho), (r, u) in out.items())


def build_class_table(M, rmax=63, umax=64):
    """Every unit class mod M is killed by an F1 cover with m | M, or by an F2
    uniform certificate with L | M, or it survives.  The killer recorded is the
    first in canonical order (F1 by ascending (m, res), then F2 by ascending
    (L, res)) so the table is reproducible byte for byte."""
    f1 = [(m, res, a, c, d) for m, res, pmin, a, c, d in gen_wheel_covers(M)]
    f2 = [(L, res, r, u) for L, res, r, u in uniform_certs(rmax, umax) if M % L == 0]
    surv, kills = [], []
    for rho in range(1, M):
        if gcd(rho, M) != 1: continue
        killer = None
        for m, res, a, c, d in f1:
            if rho % m == res: killer = (rho, "F1", m, res, a, c, d); break
        if killer is None:
            for L, res, r, u in f2:
                if rho % L == res: killer = (rho, "F2", L, res, r, u); break
        if killer is None: surv.append(rho)
        else: kills.append(killer)
    return surv, kills


def fnv64(classes):
    h = 14695981039346656037
    for b in ",".join(map(str, classes)).encode():
        h = ((h ^ b) * 1099511628211) % (1 << 64)
    return h


# ---------------------------------------------------------------- modes

def mode_identity():
    random.seed(20260812)
    bad = n = 0
    for a in range(1, 13):
        for c in range(1, 13):
            for d in range(1, 13):
                for k in range(1, 9):
                    if c * k <= a: continue
                    n += 1
                    bad += _check_instance(a, c, d, k)
    for _ in range(20000):
        a, c, d, k = (random.randint(1, 5000) for _ in range(4))
        if c * k <= a: continue
        n += 1
        bad += _check_instance(a, c, d, k)
    print(f"identity: {n} instances, {bad} failures")
    return 1 if bad else 0


def _check_instance(a, c, d, k):
    m = 4 * a * c * d - 1
    e = c * k - a
    p = m * k - 4 * a * a * d
    if p <= 0: return 0
    ok = (p + k == 4 * a * d * e and
          Fraction(4, p) == Fraction(1, a * d * e) + Fraction(1, a * c * d * p)
                            + Fraction(1, c * d * e * p))
    if not ok: print(f"  FAIL identity a={a} c={c} d={d} k={k}")
    return 0 if ok else 1


def mode_diff_filters(path):
    """Regenerate and byte-compare a `cover_scan --dump-filters` dump."""
    M = mmax = None
    classes, filters = [], []
    for line in open(path):
        t = line.split()
        if line.startswith("# cover_scan filter table"):
            for tok in t:
                if tok.startswith("M="): M = int(tok[2:])
                if tok.startswith("mmax="): mmax = int(tok[5:])
        elif t and t[0] == "CLASS": classes.append(int(t[1]))
        elif t and t[0] == "FILTER":
            filters.append(tuple(int(x) for x in t[1:7]))
    assert M and mmax, "missing header"
    # The CLASS lines are the F1+F2 reduction over ALL unit classes -- the same
    # derivation as --class-table, NOT the old F1-only rho ≡ 1 (mod 24) wheel.
    surv, _ = build_class_table(M)
    ok = surv == classes
    print(f"CLASS lines: {'match' if ok else 'MISMATCH'} ({len(classes)} vs {len(surv)})")
    ref = [(m, r, p, a, c, d) for m, r, p, a, c, d in gen_covers(mmax)]
    got = [(m, r, p, a, c, d) for m, r, a, c, d, p in filters]   # dump order: m r a c d pmin
    ok2 = sorted(got) == ref
    print(f"FILTER lines: {'match' if ok2 else 'MISMATCH'} ({len(got)} vs {len(ref)})")
    if ok2:
        for m, r, pmin, a, c, d in ref[:1000]:      # spot-check semantic validity too
            assert m == 4*a*c*d - 1 and r == (-4*a*a*d) % m
    return 0 if ok and ok2 else 1


def mode_sweep(N):
    covered = bytearray(N + 1)
    a = 1
    while (4 * a - 1) * 3 <= 4 * N + 1:
        e = 1
        while (4 * a - 1) * (4 * e - 1) <= 4 * N + 1:
            s = a + e
            ks = [k for k in range(1, s + 1) if s % k == 0]
            d = 1
            while True:
                base = 4 * a * e * d                 # = 4ade = p + k
                if base - 1 > N and base - s > N: break
                for k in ks:
                    p = base - k
                    if 1 <= p <= N: covered[p] = 1
                d += 1
            e += 1
        a += 1
    surv = [n for n in range(2, N + 1) if not covered[n]]
    nonsq = [n for n in surv if isqrt(n) ** 2 != n]
    print(f"sweep to {N}: survivors={len(surv)}, non-squares={nonsq}")
    expect = [n for n in (288, 336, 4545) if n <= N]
    print("expected non-squares:", expect, "-- MATCH" if nonsq == expect else "-- DIFFERS")
    return 0 if nonsq == expect else 1


def verify_output(paths):
    bad = surv = nrung = nsolved = 0
    for path in paths:
        for ln, line in enumerate(open(path), 1):
            t = line.split()
            if not t: continue
            if t[0] == "RUNG":
                nrung += 1
                p, A, r, u = map(int, t[1:5])
                q = p * A
                x, rem = divmod(q + u, r)
                y, rem2 = divmod(q * (q + u), r * u)
                ok = (A == (p + r) // 4 and (q * q) % u == 0 and (q + u) % r == 0
                      and rem == 0 and rem2 == 0
                      and Fraction(4, p) == Fraction(1, A) + Fraction(1, x) + Fraction(1, y))
                if not ok: bad += 1; print(f"{path}:{ln}: BAD RUNG {p}")
            elif t[0] == "SOLVED":
                nsolved += 1
                p, a, c, d, k = map(int, t[1:6])
                m = 4 * a * c * d - 1
                e = c * k - a
                ok = (e >= 1 and p == m * k - 4 * a * a * d
                      and Fraction(4, p) == Fraction(1, a * d * e)
                          + Fraction(1, a * c * d * p) + Fraction(1, c * d * e * p))
                if not ok: bad += 1; print(f"{path}:{ln}: BAD SOLVED {p}")
            elif t[0] == "SURVIVOR":
                surv += 1
                print(f"{path}:{ln}: SURVIVOR {t[1]} (needs escalation)")
    print(f"verified: {nrung} RUNG, {nsolved} SOLVED; {bad} bad, {surv} survivors")
    return 1 if bad or surv else 0


def mode_class_table(M=120120, write=False, check=None):
    """Regenerate the certified class table and optionally write or diff it."""
    # The divisor-driven generator must equal the full enumeration restricted to m | M.
    # Cheap to assert at 120120, and it is the correctness argument for the fast path.
    if M == 120120:
        slow = [(m, r, p, a, c, d) for m, r, p, a, c, d in gen_covers(M) if M % m == 0]
        if slow != gen_wheel_covers(M):
            print("  FAIL gen_wheel_covers != gen_covers restricted to m | M"); return 1
        print(f"  ok  wheel covers: divisor == full ({len(slow)})")
    surv, kills = build_class_table(M)
    ncop = sum(1 for x in range(1, M) if gcd(x, M) == 1)
    s = ",".join(map(str, surv)).encode()
    digest = hashlib.sha256(s).hexdigest()
    print(f"M={M}: unit classes={ncop} survivors={len(surv)} kills={len(kills)} "
          f"fnv64={fnv64(surv):016x} sha256={digest}")
    if len(surv) + len(kills) != ncop:
        print("  FAIL survivors + kills != unit classes"); return 1

    f1 = [(m, res) for m, res, pmin, a, c, d in gen_wheel_covers(M)]
    f2 = {(L, res): (r, u) for L, res, r, u in uniform_certs() if M % L == 0}
    bad = 0
    for rho in surv:                                   # survivors must really survive
        for m, res in f1:
            if rho % m == res: bad += 1; print(f"  FAIL {rho} killed by F1 {m}")
        for L, res in f2:
            if rho % L == res: bad += 1; print(f"  FAIL {rho} killed by F2 {L}")
    for k in kills:                                    # kills must really kill
        rho, fam = k[0], k[1]
        if fam == "F1":
            _, _, m, res, a, c, d = k
            if not (m == 4*a*c*d - 1 and res == (-4*a*a*d) % m and M % m == 0
                    and rho % m == res):
                bad += 1; print(f"  FAIL bad F1 kill {rho}")
        else:
            _, _, L, res, r, u = k
            if not (M % L == 0 and rho % L == res and f2.get((L, res)) == (r, u)):
                bad += 1; print(f"  FAIL bad F2 kill {rho}")
    if bad: return 1
    print(f"class table: {len(surv)} survive, {len(kills)} killed, all witnesses check")

    # Free falsifiable check: the derivation is independent of Mordell, so it must
    # REPRODUCE Mordell -- every survivor lands in the six square classes mod 840.
    if M % 840 == 0:
        HARD840 = {1, 121, 169, 289, 361, 529}
        strays = [x for x in surv if x % 24 != 1 or x % 840 not in HARD840]
        if strays:
            print(f"  FAIL {len(strays)} survivors outside the hard class:", strays[:10])
            return 1
        print(f"  ok  all {len(surv)} survivors ≡ 1 (mod 24), reducing into "
              f"{sorted(HARD840)} (mod 840)")

    lines = [f"# certified class table mod {M}: every unit class is killed by a named",
             f"# certificate or survives.  F1: covers m=4acd-1 | M, kill rho ≡ -4a^2d (mod m).",
             f"# F2: uniform rung certificates (r,u), r <= 63 u <= 64, kill rho ≡ res (mod L).",
             f"# unit_classes={ncop} classes={len(surv)} kills={len(kills)} "
             f"fnv64={fnv64(surv):016x} sha256={digest}"]
    lines += [f"CLASS {x}" for x in surv]
    lines += [f"KILL {k[0]} F1 {k[2]} {k[3]} {k[4]} {k[5]} {k[6]}" if k[1] == "F1"
              else f"KILL {k[0]} F2 {k[2]} {k[3]} {k[4]} {k[5]}" for k in kills]
    body = "\n".join(lines) + "\n"
    if write:
        open(table_path(M), "w").write(body)
        print(f"wrote {table_path(M)}")
    if check:
        if open(check).read() == body:
            print(f"{check}: MATCH")
        else:
            print(f"{check}: MISMATCH"); return 1
    return 0


def mode_uniform(rmax=63, umax=64, per_cert=64, M=120120, verbose=False):
    """Search uniform rung certificates and re-verify them in exact rationals.

    Output is a summary by construction: at the default bounds the search finds
    thousands of certificates, and the ones that matter are recorded individually
    in class_table_120120.txt as KILL ... F2 lines.  --verbose lists them all."""
    certs = uniform_certs(rmax, umax)
    usable = [c for c in certs if M % c[0] == 0]
    print(f"uniform certificates: {len(certs)} (r <= {rmax}, u <= {umax}); "
          f"{len(usable)} usable at M={M}")
    if verbose:
        for L, res, r, u in certs:
            print(f"  UNIFORM L={L} res={res} r={r} u={u}")

    # The two that close the reduction on their own: p ≡ 3 (mod 4) -> rung 1, and
    # p ≡ 13 (mod 24) -> rung 3 with u=2 (this is 4/13 = 1/4 + 1/18 + 1/468).
    required = {(4, 3, 1, 1), (24, 13, 3, 2)}
    missing = required - {c for c in certs}
    if missing:
        print("  FAIL missing required certificates:", sorted(missing))
        return 1
    print("  ok  required (4,3,r=1,u=1) and (24,13,r=3,u=2) both present")

    bad = n = 0
    for L, res, r, u in certs:
        p = res if res >= 3 else res + L * ((3 - res + L - 1) // L)
        for _ in range(per_cert):
            A = (p + r) // 4
            q = p * A
            n += 1
            if (q + u) % r or (q * q) % u:
                bad += 1; print(f"  FAIL uniform r={r} u={u} at p={p}: conditions"); p += L; continue
            x, rem = divmod(q + u, r)
            y, rem2 = divmod(q * (q + u), r * u)
            if rem or rem2 or Fraction(4, p) != Fraction(1, A) + Fraction(1, x) + Fraction(1, y):
                bad += 1
                print(f"  FAIL uniform r={r} u={u} at p={p}")
            p += L
    print(f"uniform: {n} instances ({per_cert} per certificate), {bad} failures")
    return 1 if bad else 0


if __name__ == "__main__":
    args = sys.argv[1:]
    if not args:
        print(__doc__); sys.exit(1)
    if args[0] == "--identity":
        sys.exit(mode_identity())
    if args[0] == "--diff-filters":
        sys.exit(mode_diff_filters(args[1]))
    if args[0] == "--uniform":
        sys.exit(mode_uniform(verbose="--verbose" in args))
    if args[0] == "--class-table":
        M = int(args[args.index("--wheel") + 1]) if "--wheel" in args else 120120
        chk = None
        if "--check" in args: chk = args[args.index("--check") + 1]
        sys.exit(mode_class_table(M, write="--write" in args, check=chk))
    if args[0] == "--sweep":
        sys.exit(mode_sweep(int(args[1])))
    sys.exit(verify_output(args))
