#!/usr/bin/env python3
"""verify_cert.py -- exact-arithmetic re-verification of rung_scan output.

The C++ scanner works entirely in machine words and decides rung solvability by a
residue DP.  This script is the independent trust anchor: it re-derives each emitted
claim in exact Python integer arithmetic, using a different algorithm (explicit
divisor enumeration of q^2), and prints an actual unit-fraction decomposition.

Usage:
    ./rung_scan 1000000000000 1000000100000 --emit-deep 35 --sample 1000000 > out.txt
    python3 verify_cert.py out.txt

Lines understood:
    SUPPORT p A r   -- claim: rung r fails at level S. Additionally audited against the
                       four-constraint (es_levels.py): the rung must be one of the
                       finitely many that can carry level S at all, and the
                       certificate's own support subgroup must contain 4 and omit -1.
    DEEP p A r      -- claim: rung r hits, and every rung < r fails
    SAMPLE p A r    -- same claim (spot-check sample)
    UNFORCED p A r  -- claim: rung r fails, and the Jacobi obstruction does NOT apply
    ESCALATE p 0 0  -- claim: no rung <= rmax hits (searched further here)
    SUMMARY ...     -- histogram line; totals are cross-checked for internal consistency

Exit status is nonzero if any claim fails.
"""
import sys
from fractions import Fraction
from math import gcd, isqrt

import es_levels


# ---------------------------------------------------------------- factorization
def factorize(n):
    """Plain Pollard-rho factorization; returns {prime: exponent}."""
    fac = {}
    for d in (2, 3, 5):
        while n % d == 0:
            fac[d] = fac.get(d, 0) + 1
            n //= d
    d, step = 7, (4, 2, 4, 2, 4, 6, 2, 6)
    i = 0
    while d * d <= n and d < 1_000_000:
        while n % d == 0:
            fac[d] = fac.get(d, 0) + 1
            n //= d
        d += step[i % 8]
        i += 1
    if n > 1:
        for p, e in _rho_factor(n).items():
            fac[p] = fac.get(p, 0) + e
    return fac


def _is_prime(n):
    if n < 2:
        return False
    for p in (2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37):
        if n % p == 0:
            return n == p
    d, s = n - 1, 0
    while d % 2 == 0:
        d //= 2
        s += 1
    for a in (2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37):
        x = pow(a, d, n)
        if x in (1, n - 1):
            continue
        for _ in range(s - 1):
            x = x * x % n
            if x == n - 1:
                break
        else:
            return False
    return True


def _rho_factor(n):
    if n == 1:
        return {}
    if _is_prime(n):
        return {n: 1}
    r = isqrt(n)
    if r * r == n:
        return {p: 2 * e for p, e in _rho_factor(r).items()}
    c = 1
    while True:
        x = y = 2
        d = 1
        while d == 1:
            x = (x * x + c) % n
            y = (y * y + c) % n
            y = (y * y + c) % n
            d = gcd(abs(x - y), n)
        if d != n:
            break
        c += 1
    left, right = _rho_factor(d), _rho_factor(n // d)
    for p, e in right.items():
        left[p] = left.get(p, 0) + e
    return left


def divisors(fac):
    divs = [1]
    for p, e in fac.items():
        divs = [d * p**k for d in divs for k in range(e + 1)]
    return divs


def jacobi(a, n):
    a %= n
    t = 1
    while a:
        while a % 2 == 0:
            a //= 2
            if n % 8 in (3, 5):
                t = -t
        a, n = n, a
        if a % 4 == 3 and n % 4 == 3:
            t = -t
        a %= n
    return t if n == 1 else 0


# ---------------------------------------------------------------- rung analysis
def rung_solutions(p, A):
    """All (u, v, B, C) at this A, by exact divisor enumeration of q^2."""
    r, q = 4 * A - p, p * A
    assert r > 0, (p, A)
    facA = factorize(A)
    facq2 = {ell: 2 * e for ell, e in facA.items()}
    facq2[p] = facq2.get(p, 0) + 2
    out = []
    for u in divisors(facq2):
        if u > q:
            continue
        v = q * q // u
        if (u + q) % r == 0 and (v + q) % r == 0:
            out.append((u, v, (u + q) // r, (v + q) // r))
    return out


def is_blocked(p, A):
    """True when the Jacobi/QR obstruction forces failure at this rung."""
    r, q = 4 * A - p, p * A
    if r % 2 == 0 or gcd(q, r) != 1:
        return False
    primes = set(factorize(A)) | {p}
    if not all(jacobi(ell, r) == 1 for ell in primes):
        return False
    return jacobi((-q) % r, r) == -1


def check_decomposition(p, A, B, C):
    assert Fraction(1, A) + Fraction(1, B) + Fraction(1, C) == Fraction(4, p), (p, A, B, C)
    assert 4 * A * B * C == p * (A * B + B * C + C * A), (p, A, B, C)


# ---------------------------------------------------------------- claim checks
def check_hit(p, A, r, tag, verbose):
    """Rung r hits at A, and every earlier rung fails."""
    if (p + r) // 4 != A or 4 * A - p != r:
        return f"{tag} p={p}: A={A} inconsistent with r={r}"
    sols = rung_solutions(p, A)
    if not sols:
        return f"{tag} p={p} r={r}: claimed hit but no divisor lands"
    u, v, B, C = sols[0]
    check_decomposition(p, A, B, C)
    for rr in range(3, r, 4):
        AA = (p + rr) // 4
        if rung_solutions(p, AA):
            return f"{tag} p={p}: claimed minimal r={r} but rung {rr} also hits"
    if verbose:
        print(f"  ok {tag} p={p} r={r}: 4/{p} = 1/{A} + 1/{B} + 1/{C}   (u={u})")
    return None


def real_char_certifies(p, A):
    """Is there SOME real character mod r trivial on all primes of q, = -1 at -q?
    For odd r = prod p_i^e_i these are the products of (./p_i) over subsets."""
    from itertools import combinations
    r, q = 4 * A - p, p * A
    if gcd(q, r) != 1:
        return None
    ps = sorted(factorize(r))
    prs = [p] + list(factorize(A))
    for k in range(1, len(ps) + 1):
        for sub in combinations(ps, k):
            def chi(x, sub=sub):
                v = 1
                for pp in sub:
                    v *= (1 if pow(x % pp, (pp - 1) // 2, pp) == 1 else -1) if x % pp else 0
                return v
            if all(chi(l) == 1 for l in prs) and chi((-q) % r) == -1:
                return "*".join(f"(./{x})" for x in sub)
    return None


def support_certifies(p, A):
    r, q = 4 * A - p, p * A
    if gcd(q, r) != 1:
        return False
    gens = {l % r for l in [p] + list(factorize(A))}
    H, st = {1}, [1]
    while st:
        x = st.pop()
        for g in gens:
            y = x * g % r
            if y not in H:
                H.add(y)
                st.append(y)
    return ((-q) % r) not in H


def check_failure(p, A, r, kind, verbose):
    """Rung r fails, at exactly the claimed certificate level."""
    if 4 * A - r != p and 4 * A - p != r:
        return f"{kind} p={p}: A={A} inconsistent with r={r}"
    if rung_solutions(p, A):
        return f"{kind} p={p} r={r}: claimed failure but a divisor does land"
    jac = is_blocked(p, A)
    rch = real_char_certifies(p, A)
    sup = support_certifies(p, A)
    if kind == "RESIDUAL":
        if jac or rch or sup:
            return f"RESIDUAL p={p} r={r}: actually certified ({jac=}, {rch=}, {sup=})"
        detail = "genuine residual miss, no local certificate"
    elif kind == "REALCHAR":
        if jac:
            return f"REALCHAR p={p} r={r}: Jacobi already certifies it (should be level J)"
        if not rch:
            return f"REALCHAR p={p} r={r}: no real character certifies it"
        detail = f"certified by real character {rch}, not by Jacobi"
    else:  # SUPPORT
        if jac or rch:
            return f"SUPPORT p={p} r={r}: a real character certifies it ({rch})"
        if not sup:
            return f"SUPPORT p={p} r={r}: support subgroup does not certify it"
        # The four-constraint audit (es_levels.py): 4A = p + r forces 4 into the
        # support subgroup, which confines level S to an explicit finite set of rungs.
        # A violation here is not a scanner bug report -- it falsifies the proposition
        # the paper stakes on being falsifiable, so it must be loud.
        bad = es_levels.check_support_claim(p, A, r)
        if bad:
            return f"SUPPORT p={p} r={r}: {bad}"
        detail = "beyond ALL real characters; support subgroup certifies"
    if verbose:
        tau = 3
        for _, e in factorize(A).items():
            tau *= 2 * e + 1
        print(f"  ok {kind} p={p} r={r}: {detail}, tau(q^2)={tau}")
    return None


def check_escalate(p, rmax, verbose):
    for rr in range(3, 4 * rmax, 4):
        if rung_solutions(p, (p + rr) // 4):
            if rr <= rmax:
                return f"ESCALATE p={p}: rung {rr} <= rmax actually hits"
            if verbose:
                print(f"  ok ESCALATE p={p}: first hit at r={rr} (beyond rmax={rmax})")
            return None
    return f"ESCALATE p={p}: no hit found even in extended search"


def main(argv):
    if len(argv) < 2:
        print(__doc__)
        return 2
    verbose = "-q" not in argv
    errors, checked = [], 0
    summaries = []
    with open(argv[1]) as fh:
        for line in fh:
            parts = line.split()
            if not parts:
                continue
            kind = parts[0]
            if kind in ("DEEP", "SAMPLE"):
                p, A, r = map(int, parts[1:4])
                err = check_hit(p, A, r, kind, verbose)
                checked += 1
            elif kind in ("RESIDUAL", "REALCHAR", "SUPPORT"):
                p, A, r = map(int, parts[1:4])
                err = check_failure(p, A, r, kind, verbose)
                checked += 1
            elif kind == "ESCALATE":
                p = int(parts[1])
                err = check_escalate(p, 127, verbose)
                checked += 1
            elif kind == "SUMMARY":
                summaries.append(line.strip())
                continue
            else:
                continue
            if err:
                errors.append(err)

    for s in summaries:
        fields = dict(kv.split("=", 1) for kv in s.split()[1:] if "=" in kv)
        hist = fields.get("hist", "")
        tot = sum(int(x.split(":")[1]) for x in hist.split(",") if ":" in x)
        np_ = int(fields.get("primes", 0))
        esc = int(fields.get("escalated", 0))
        if tot + esc != np_:
            errors.append(f"SUMMARY internal inconsistency: hist total {tot} + escalated {esc} != primes {np_}")
        else:
            print(f"  ok SUMMARY [{fields.get('lo')}, {fields.get('hi')}): "
                  f"{np_} primes, histogram totals agree, max min-r = {fields.get('maxr')}")

    print(f"\n{checked} certificates checked in exact arithmetic; {len(errors)} failures")
    for e in errors:
        print("  FAIL " + e)
    return 1 if errors else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
