#!/usr/bin/env python3
"""prune_filters.py -- choose the certified filter set by unique-kill accounting.

Soundness is MONOTONE in the filter set: dropping a filter can only leave more
survivors, never manufacture a false certificate. Every survivor is then resolved
individually by stages C/D/E or reported. So the worst case of a bug here is a slower
run, not a wrong claim -- which is what makes it safe to be aggressive.

Cost of keeping filter i, per lane position:
    C_i = 1/step_i        bit-clears
        + 1/SEG           per-segment bookkeeping, paid whether it hits or not
        + setup/shard     (--device only) the per-(lane,filter) offset setup

Benefit: the survivor work its kills remove,
    B_i = (positions it kills) * [ ns_composite + phi * (ns_prime - ns_composite) ]
where phi is the prime fraction of the positions STILL ALIVE, measured exactly on the
domain by prime_masks() and identical for every candidate.

A composite survivor costs a base-2 SPRP and stops; a prime costs a 12-base MR, a
stage-D rung walk and an emitted certificate -- 28x more on the CPU and 260x more on the
CUDA port, where stage D is the only part that never moved to the GPU. So the split
matters, and phi used to be a GUESSED constant (0.171). It is now measured.

WHAT DOES NOT WORK, AND WAS TRIED (2026-08-17). The obvious stronger move is to credit
each filter with the specific PRIMES it kills, rather than its position kills times a
shared phi. The machinery is here and it is exact, so this looks free. It overfits, badly
and measurably:

                                    prime density of survivors
    filter set                  training domain (lo=1e12)   unseen (lo=2e12)
    published / shared phi              17.2%                    18.7%
    scored on per-filter primes          6.0%                    13.9%

The 6.0% is fitted, not real. Confirmed on the scanner itself over [1e12, 1.5e12): the
per-filter table left 50,060 surviving primes against the published table's 39,298, for
no change in wall (1.183 s vs 1.186 s on cloud-dev, 8 threads, mean of 3).

The reason is worth keeping. A filter's HIT SET is phase-structured -- that is exactly
why an anchored window predicts survivor density at all -- but WHICH of those hits are
prime is not a congruence property, so per-filter prime credit describes this stretch of
the number line and no other. The greedy then picks the best of 6,401 candidates, 4,533
times over, which is a winner's curse that compounds; and it bites hardest at the end,
where `remaining` is small, the per-filter prime counts are tiny, and the stopping
decision is actually made.

A shared phi cannot do this: being identical for every candidate it cannot bias the
choice between them, and it is an aggregate over millions of positions rather than the
handful any one filter touches. It moves only where the greedy STOPS.

Two families are candidates, and which of them survive is the question, not the
assumption:
  F1  Type II covers m = 4acd-1 with m <= mmax and m not dividing M
  F2  uniform rung certificates with L not dividing M -- the 5,417 left over after the
      wheel takes the 390 with L | M. Each kills a sub-progression INSIDE a lane, which
      is structurally what an F1 cover does in stage B. Measured: they add 1.32
      marks/pos and cut survivor density from 3.10e-4 to 2.23e-4.

The evaluation domain is fixed and published so the result is reproducible: ALL 220
lanes, 2^16 positions per lane, anchored at LO_REF.

ANCHORING AT LO_REF IS LOAD-BEARING. An earlier version indexed from J = 0, i.e. from
p = rho, on the theory that a filter's hit pattern depends only on (rho mod mod,
J mod step) and so a short window anywhere is as representative as any other. That is
true of MARKS per position -- which is just sum(npos/step), and was measured correct --
but it is false of the union's SURVIVOR DENSITY, which is only phase-independent over
lcm(steps), a number astronomically larger than 2^16. Anchoring at a tiny p aligns every
filter's phase against it and inflated survivor density 6x: 9.40e-4 against the 1.58e-4
this anchoring gives and the 1.48e-4 the scanner actually measures.

That error propagated into both cost constants and therefore into the chosen table.

USAGE
  ./prune_filters.py --self-check                 sanity-check the machinery, fast
  ./prune_filters.py --mmax 2000 --write tables/filters/  produce the published table
  ./prune_filters.py --mmax 2000 --device --write tables/filters/   the CUDA port's

--self-check verifies the prime mask POSITION BY POSITION against Miller-Rabin, because
a sieve has no witness test of its own and a mask that is merely a bit wrong changes the
chosen table without announcing itself.
"""
import sys, hashlib, math, os
from math import gcd, isqrt

# Artifacts live in tables/ at the repo root; resolve against this file.
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

M = 120120
LO_REF = 10**12      # scan floor the domain is anchored at; see the note above
NPOS = 1 << 16       # 65,536 positions per lane; see the note on the domain below
SEG = 262144                  # must match rung_scan3's default --seg

# Cost constants, in nanoseconds. MEASURED on cloud-dev (c4-highcpu-8) over
# [1e12, 1.5e12) with the 4,585-filter table, and cross-checked against a direct profile
# of stage D (tests/bench_staged.cpp):
#
#   marking     90.9% of runtime   8.02e9 marks          -> 0.76 ns per mark
#   stage D      7.8%              23,182 primes         -> 22.7 us per PRIME survivor
#   composites   1.3%              112,386               -> ~0.8 us per COMPOSITE survivor
#
# The survivor cost is NOT one number: a prime costs ~28x a composite, because it pays a
# 12-base MR and a stage-D rung walk (of which factorA is ~100% -- the divisor DP is 1%)
# and emits a certificate. Averaging the two is what made the small-prime sieve look like
# a winner when it is a 5% loss: that change can only remove COMPOSITES.
#
# Two earlier sets of constants were wrong. 1.2/400.0 were guesses, and the table they
# chose ran 5.7x slower than no pruning. 0.51/7600 came from a two-equation solve fed
# with survivor densities from the unanchored J=0 domain, which overstated them 6x.
NS_PER_MARK = 0.76
NS_PER_SURVIVOR_COMPOSITE = 800.0
NS_PER_SURVIVOR_PRIME = 22700.0
MEASURED = True

# These are no longer blended at a GUESSED prime fraction. prune() measures the live
# fraction on the domain instead, exactly, and updates it as coverage deepens. In the
# regime the pruner actually operates in that fraction turns out to be flat at ~0.186 --
# so this reproduces the published table byte for byte, and the old 0.171 was right to
# within 8%. It earns its place by being measured rather than assumed, and by tracking
# the collapse that does occur at much deeper coverage (see prime_masks).
DEVICE = False           # --device swaps in the L4 constants below

# --------------------------------------------------------------- device constants
#
# Fitted on one L4 (cloud-gpu, 2026-08-17). The DEVICE half comes from four filter sets
# at [1e12, 1.5e12), solving device_time = t0 + marks*c_m + survivors*c_s by least
# squares; all four rows reproduce within 2%:
#
#     mmax    marks      survivors   measured   model
#      500    3.71e9     1,047,744     0.155     0.155
#     1000    5.43e9       118,591     0.139     0.137
#     3000    8.93e9        41,071     0.169     0.172
#    10000    1.40e10       40,573     0.229     0.228
#
#   -> 0.010947 ns per mark, 39.83 ns per survivor ON THE DEVICE, 72.8 ms fixed.
#
# BUT THE DEVICE COST OF A SURVIVOR IS NOT ITS COST. Only marking, compaction and
# primality moved to the GPU; stages D and E stay on the HOST, and they are what a
# survivor that turns out to be PRIME actually costs. Measured in the same session, the
# 4,163-filter table over [1e12, 6e12): 1,242,946 survivors, 229,299 of them prime
# (18.4%), host tail 2.373 s wall on 12 threads -> 10.35 us of WALL per prime. Blended
# over survivors that is 1,904 ns, which swamps the 39.83 ns of device primality by 48x.
#
# THE SPEC'S PREDICTION IS CONFIRMED, and by a wide margin. Survivor:mark ratio is
# 177,600 on the device against 5,980 on the CPU -- THIRTY TIMES higher -- so the device
# wants a much LARGER filter set. The mechanism is not the one the spec gave (it argued
# from emulated modexp being dear); it is that marking got 69x cheaper while the part of
# a survivor that dominates never moved off the CPU at all.
#
# The model is additive while the implementation OVERLAPS tail and device, so wall is
# nearer max(sum device, sum tail) than their sum. That makes this additive form
# conservative in exactly the useful direction: it understates how nearly free extra
# marking is while the run is tail-bound.
# Re-measured 2026-08-18 after the 32-bit bitset (marking 2.59x) and the Montgomery/
# magic-inverse host tail (5-6x). Same method, same box, four filter sets, all rows
# within 3%. Both sides got cheaper by DIFFERENT factors, so the old table's balance is
# wrong twice over: survivor:mark fell 177,600 -> 106,500.
DEV_NS_PER_MARK = 0.004280
# A composite survivor costs one device primality test and nothing else -- it never
# reaches the host. A prime survivor costs that plus the whole of stage D, which did not
# move to the GPU. The 260x between them is the largest asymmetry in either cost model
# and is exactly what the old blended constant threw away.
DEV_NS_PER_SURVIVOR_COMPOSITE = 29.6
DEV_NS_PER_SURVIVOR_PRIME = 29.6 + 2316.0     # + host stage D, wall, 12 threads

# The device's THIRD cost term, which the CPU model does not have. sieve_cuda computes
# every (lane, filter) offset up front, once per shard: 7.03 ns per pair measured
# (12 threads, 2,308 lanes x 64,708 filters in 1.05 s). Per filter that is nlanes * 7.03
# ns of setup no matter how few positions it ever marks. It amortises with SHARD WIDTH,
# not with segment count, so it is divided by the positions one lane holds in a shard --
# which is what stops the larger set above from growing without bound.
DEV_NS_SETUP_PER_LANE_FILTER = 7.03
DEV_SHARD_POSITIONS = 2448532         # positions per lane in [1e12, 6e12), the measured shard


def gen_covers(mmax):
    """(m, res) for all m = 4acd-1 <= mmax, deduped, ascending."""
    best = set()
    nmax = (mmax + 1) // 4
    for a in range(1, nmax + 1):
        for d in range(1, nmax // a + 1):
            for c in range(1, nmax // (a * d) + 1):
                m = 4 * a * c * d - 1
                best.add((m, (-4 * a * a * d) % m))
    return sorted(best)


def gen_ucerts(rmax=63, umax=64):
    """(L, res) uniform rung certificates; see verify_covers.py for the derivation."""
    out = set()
    for r in range(1, rmax + 1):
        for u in range(1, umax + 1):
            if gcd(u, r) != 1:
                continue
            L = 4 * (r * u // gcd(r, u))
            for rho in range(L):
                if (rho + r) % 4:
                    continue
                if all(not ((((p * (p + r) // 4) ** 2) % u) or
                            (((p * (p + r) // 4) + u) % r))
                       for p in (rho, rho + L)):
                    out.add((L, rho))
    return sorted(out)


def hit_pattern(mod, res, rho, npos=NPOS):
    """(j0, step) for J in [0,npos) with first + M*J = res (mod mod), where `first` is
    the first position of lane rho at or above LO_REF -- exactly what sweep_lane
    computes. Returns None if the lane misses this progression entirely; `rhs % g` is
    the ONLY solvability condition, since per prime the minimum exponent goes wholly
    into g, so M/g and mod/g are automatically coprime."""
    first = LO_REF + ((rho + M - LO_REF % M) % M)
    g = gcd(M, mod)
    rhs = (res - first) % mod
    if rhs % g:
        return None
    step = mod // g
    if step == 1:
        return (0, 1)
    return ((rhs // g) * pow((M // g) % step, -1, step) % step, step)


_PAT = {}


def _stride_mask(step, npos):
    """Bit i set iff i ≡ 0 (mod step), for i < npos + step. Cached per step: the
    number of distinct steps is far smaller than the number of filters."""
    key = (step, npos)
    p = _PAT.get(key)
    if p is None:
        if step == 1:
            p = (1 << (npos + 1)) - 1
        else:
            unit = 1 << step                       # bit `step` set, bits below clear
            p = 0
            for _ in range((npos // step) + 2):
                p = (p << step) | 1
        _PAT[key] = p
    return p


def evaluate(filters, lanes, npos=NPOS):
    """For each filter, how many positions it kills that NO other filter kills.

    The per-position loop this replaces was 3.3e9 Python iterations. Instead the hit
    set of a filter on a lane is a periodic bit pattern, so the whole calculation is
    integer bitwise algebra that CPython runs in C:

        pass 1   once  |= hits ;  twice |= once_before & hits
        unique   = once & ~twice
        pass 2   uniq_i = popcount(hits_i & unique)

    Exact, not sampled, and stdlib-only (int.bit_count needs Python >= 3.10).
    Returns (uniq, marks, survivors, surviving_primes, positions)."""
    full = (1 << npos) - 1
    uniq = [0] * len(filters)
    marks = surv = surv_p = 0
    pmask = prime_masks(lanes, npos)
    for lane_index, rho in enumerate(lanes):
        hits = [0] * len(filters)
        once = twice = 0
        for i, (mod, res, _kind) in enumerate(filters):
            h = hit_pattern(mod, res, rho, npos)
            if h is None:
                continue
            j0, step = h
            m = (_stride_mask(step, npos) << j0) & full
            hits[i] = m
            marks += m.bit_count()
            twice |= once & m
            once |= m
        unique = once & ~twice & full
        surv += npos - once.bit_count()
        surv_p += (pmask[lane_index] & ~once).bit_count()
        for i, m in enumerate(hits):
            if m:
                uniq[i] += (m & unique).bit_count()
    return uniq, marks, surv, surv_p, len(lanes) * npos


# ------------------------------------------------------------------ primality

def _sieve_primes(n):
    """Primes <= n. n is ~10^6 here, and the bytearray slice assignment is the whole
    reason this is affordable in stdlib Python -- it runs at C speed."""
    s = bytearray(b"\x01") * (n + 1)
    s[0:2] = b"\x00\x00"
    for i in range(2, isqrt(n) + 1):
        if s[i]:
            s[i * i :: i] = b"\x00" * len(range(i * i, n + 1, i))
    return [i for i in range(2, n + 1) if s[i]]


def _prime_cache_path(nlanes, npos):
    return os.path.join(ROOT, ".primes",
                        f"pm_M{M}_lo{LO_REF}_n{npos}_L{nlanes}.bin")


def prime_masks(lanes, npos=NPOS, cache=True, verbose=False):
    """Bit J of mask[k] set iff position J of lane k is PRIME. Exact, not probable.

    WHY THIS EXISTS. The greedy's benefit term needs the prime fraction of the live
    positions, and that used to be a guessed constant. Two things make measuring it
    worth the trouble:

      - it calibrates the constant instead of assuming it (0.186 measured against 0.171
        assumed, in the regime the pruner stops in);
      - it TRACKS THE COLLAPSE that happens at much deeper coverage than a pruned table
        reaches. On the real scanner over [1e12, 6e12), raising mmax from 1000 to 3000
        cuts survivors 3.9x and surviving PRIMES 173x: past a certain depth the
        survivors are essentially the perfect squares, which covers cannot touch (a
        cover m = 4acd-1 kills p ≡ -4a^2d, and -4a^2d is a square mod m only when -d is
        a quadratic residue there). A pruned table stops far short of that, which is why
        the flat 0.171 survived this long -- but a generated set at mmax 3000 is deep
        inside it.

    It is ALSO what proved that per-filter prime credit overfits; see the module
    docstring. Diagnostics first, scoring input second.

    HOW. A position is p = first + M*J, so "q divides this position" is the congruence
    hit_pattern(q, 0, rho) already computes -- primality sieving is the same bit algebra
    as filter marking, over the primes up to isqrt(max position). Below that bound the
    test is EXACT, not probabilistic: every composite < hi has a factor <= isqrt(hi).

    Two things make it affordable, and both matter:
      - pow(M % q, -1, q) does not depend on the lane, so the modular inverse is hoisted
        out of the lane loop -- the same invariant, for the same reason, that took
        sieve/cuda's offset setup from 79% of its wall to 49% (3.60 s -> 1.05 s).
      - for q > npos the stride exceeds the window, so the filter hits at most one
        position per lane. _stride_mask would build and CACHE a q-bit integer for each
        such q -- about 5 GB of patterns to express one bit apiece -- so that path sets
        the single bit directly.
    """
    nlanes = len(lanes)
    nbytes = (npos + 7) // 8
    path = _prime_cache_path(nlanes, npos)
    if cache and os.path.exists(path):
        raw = open(path, "rb").read()
        if len(raw) == nlanes * nbytes:
            return [int.from_bytes(raw[k * nbytes:(k + 1) * nbytes], "little")
                    for k in range(nlanes)]

    hi = LO_REF + M * npos                 # strict upper bound on any position
    qmax = isqrt(hi)
    full = (1 << npos) - 1
    firsts = [LO_REF + ((rho + M - LO_REF % M) % M) for rho in lanes]
    comp = [0] * nlanes
    qs = _sieve_primes(qmax)
    if verbose:
        print(f"  prime mask: {nlanes} lanes x {npos} positions, sieving by "
              f"{len(qs)} primes <= {qmax} (exact below {hi})")
    for q in qs:
        if M % q == 0:
            continue                       # cannot divide a position coprime to M
        invq = pow(M % q, -1, q)           # lane-independent: hoisted out of the loop
        if q <= npos:
            pat = _stride_mask(q, npos)
            for k, f in enumerate(firsts):
                comp[k] |= (pat << ((-f) % q * invq % q)) & full
        else:
            for k, f in enumerate(firsts):
                j0 = (-f) % q * invq % q
                if j0 < npos:
                    comp[k] |= 1 << j0
    out = [full & ~c for c in comp]

    if cache:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "wb") as fh:
            for m in out:
                fh.write(m.to_bytes(nbytes, "little"))
    return out


def _is_prime(n):
    """Deterministic Miller-Rabin on the same 12 bases the scanners use. Only the
    self-check calls this; prime_masks is a sieve and needs no witness test."""
    if n < 2:
        return False
    for sp in (2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37):
        if n % sp == 0:
            return n == sp
    d, r = n - 1, 0
    while not d & 1:
        d >>= 1
        r += 1
    for a in (2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37):
        x = pow(a, d, n)
        if x in (1, n - 1):
            continue
        for _ in range(r - 1):
            x = x * x % n
            if x == n - 1:
                break
        else:
            return False
    return True


def _masks_for(filt, lanes, npos=NPOS):
    """Per-lane hit bitmasks for one filter, recomputed on demand. Storing them for
    all 14,158 candidates x 220 lanes would be ~25 GB; recomputing is two cached
    big-int ops."""
    mod, res, _kind = filt
    full = (1 << npos) - 1
    out = []
    for rho in lanes:
        h = hit_pattern(mod, res, rho, npos)
        out.append((_stride_mask(h[1], npos) << h[0]) & full if h else 0)
    return out


def filter_step(filt, lanes):
    """The stride this filter marks at. Lane-independent given g = gcd(M, mod), so the
    first lane it hits at all decides it."""
    for rho in lanes:
        h = hit_pattern(filt[0], filt[1], rho)
        if h:
            return h[1]
    return None                                    # hits no lane: pure overhead


def prune(filters, lanes, max_density=1.0, verbose=True):
    """FORWARD greedy weighted set cover, with lazy (CELF) re-evaluation.

    The obvious approach -- score every filter by its UNIQUE kills and drop the ones
    that do not pay -- is wrong on this input, and wrong in a way that looks right
    until you run it. With ~15 overlapping covers per position almost nothing is
    killed by exactly one filter, so every individual unique-kill count is ~0, the
    rule drops all 14,158 at once, and survivor density goes to 1. Marginal analysis
    cannot see a set that is collectively essential and individually redundant.

    So build up instead of tearing down: repeatedly take the filter with the best
    (survivor work removed) / cost ratio, and stop when the next one costs more in
    marking than the survivor tests it would save.

    Weighted by the live PRIME FRACTION of the survivors, measured on the domain rather
    than assumed -- see benefit_of, and the module docstring for why it is the aggregate
    fraction and emphatically not each filter's own prime kills. Position gains are
    monotone non-increasing as `remaining` shrinks and the fraction is clamped
    non-increasing, so a stale top-of-heap entry still only ever over-estimates and CELF
    stays exact rather than becoming approximate."""
    import heapq
    npos = NPOS
    pos = len(lanes) * npos
    remaining = [(1 << npos) - 1 for _ in lanes]
    rem_count = pos

    # The primes still alive, tracked alongside the positions still alive. remaining_p
    # is a subset of remaining by construction -- both are cleared by the same masks --
    # so a filter's prime kills are just (its hits) & remaining_p.
    pmask = prime_masks(lanes, npos, verbose=verbose)
    remaining_p = list(pmask)
    rem_primes = sum(r.bit_count() for r in remaining_p)
    phi_rose = [0]
    if verbose:
        print(f"  domain: {pos} positions, {rem_primes} prime "
              f"({rem_primes / pos:.4f} of them)")

    steps = [filter_step(f, lanes) for f in filters]
    cand = [i for i, s in enumerate(steps) if s is not None]

    def gain_of(i):
        """Positions killed that are still alive."""
        return sum((m & r).bit_count()
                   for m, r in zip(_masks_for(filters[i], lanes), remaining))

    # The prime fraction of what is still alive. This is the ONLY way primality enters
    # the score, and the restriction is not conservatism -- it is what the measurements
    # forced. See the note in prune()'s docstring: crediting each filter with the
    # specific primes IT kills overfits the domain badly, because primality is not a
    # congruence property and so per-filter prime credit does not transfer to any other
    # stretch of the number line. An aggregate over millions of positions does transfer,
    # and being identical for every candidate it cannot bias the choice between them --
    # it moves only where the greedy STOPS, which is the part the fixed 17.1% blend got
    # wrong.
    phi = [rem_primes / rem_count]

    def benefit_of(g):
        """Survivor work removed, in ns per lane-position -- the same units as cost_of."""
        return g * (NS_PER_SURVIVOR_COMPOSITE
                    + phi[0] * (NS_PER_SURVIVOR_PRIME
                                - NS_PER_SURVIVOR_COMPOSITE)) / pos

    def cost_of(i):
        c = (1.0 / steps[i] + 1.0 / SEG) * NS_PER_MARK
        if DEVICE:
            # Per-filter offset setup, amortised over the positions one lane holds in a
            # shard. This is the term that makes the device's optimal set smaller: a
            # filter that marks almost nothing still pays it in full.
            c += DEV_NS_SETUP_PER_LANE_FILTER / DEV_SHARD_POSITIONS
        return c

    if verbose:
        print(f"  scoring {len(cand)} candidates that hit at least one lane...")
    # Ordered by gain/COST, not by raw gain. Weighted set cover needs the ratio: with
    # raw-gain ordering the greedy picks expensive small-stride filters first and then
    # stops when the highest-GAIN filter fails its own cost test, even though far
    # cheaper filters would still pay for themselves. That bug produced a table 2.3x
    # slower than doing nothing.
    def score(i):
        return -benefit_of(gain_of(i)) / cost_of(i)

    heap = [(score(i), i, 0) for i in cand]
    heapq.heapify(heap)

    chosen, marks = [], 0
    stamp = 0
    while heap:
        neg, i, when = heapq.heappop(heap)
        if when != stamp:                          # stale: re-score and reinsert
            heapq.heappush(heap, (score(i), i, stamp))
            continue
        cost = cost_of(i)
        benefit = -neg * cost                      # heap key is benefit/cost
        if benefit == 0:
            break
        if benefit <= cost and rem_count / pos <= max_density:
            # Runtime says stop. The DENSITY CONSTRAINT can override it, and usually
            # does: every surviving position that turns out to be prime becomes an
            # emitted certificate, and the certificate stream is a published artifact
            # with a size budget. Optimising marks alone produced 4.2e-6 certificates
            # per integer -- 4.2e11 at 10^17, five orders past what is publishable --
            # so the constraint is what actually decides the table, not the cost model.
            break
        chosen.append(filters[i])
        masks = _masks_for(filters[i], lanes)
        for k, m in enumerate(masks):
            marks += m.bit_count()
            remaining[k] &= ~m
            remaining_p[k] &= ~m
        rem_count = sum(r.bit_count() for r in remaining)
        rem_primes = sum(r.bit_count() for r in remaining_p)
        # CELF is exact only while a stale heap key OVER-estimates, so the multiplier
        # must be non-increasing. Empirically it is -- covers cannot kill the square-rich
        # composite residue as fast as they kill primes, so the live prime fraction only
        # falls -- but clamping makes the bound hold by construction rather than by
        # observation, and `phi_rose` reports it if the empirical claim ever breaks.
        raw = rem_primes / rem_count if rem_count else 0.0
        if raw > phi[0]:
            phi_rose[0] += 1
        phi[0] = min(phi[0], raw)
        stamp += 1
        if verbose and len(chosen) % 200 == 0:
            print(f"    {len(chosen)} filters, {marks/pos:.2f} marks/pos, "
                  f"survivor density {rem_count/pos:.3e}, "
                  f"prime density {rem_primes/pos:.3e}, "
                  f"prime fraction of survivors {phi[0]:.4f}")
    if verbose and phi_rose[0]:
        print(f"  NOTE: the live prime fraction rose {phi_rose[0]}x and was clamped; "
              f"the CELF bound held but the model's monotonicity claim did not")
    return chosen, marks / pos, rem_count / pos, rem_primes / pos


def check_prime_mask(nlanes=3, npos=4096):
    """Verify the mask POSITION BY POSITION against Miller-Rabin, and check its density
    against the prime number theorem.

    The mask is a sieve, so it has no witness test of its own -- nothing inside it would
    notice an off-by-one in the hoisted inverse, a prime skipped because it divides M, or
    a sieve bound one short of isqrt. All three would produce a mask that is merely a bit
    wrong, which is exactly the kind of error that changes the chosen table and never
    announces itself. A small domain checked exhaustively catches all of them.
    """
    lanes = load_lanes()[:nlanes]
    masks = prime_masks(lanes, npos, cache=False)
    bad_p = bad_c = 0
    for k, rho in enumerate(lanes):
        first = LO_REF + ((rho + M - LO_REF % M) % M)
        m = masks[k]
        for j in range(npos):
            claimed = (m >> j) & 1
            actual = _is_prime(first + M * j)
            if claimed and not actual:
                bad_p += 1
            elif actual and not claimed:
                bad_c += 1
    n = nlanes * npos
    got = sum(m.bit_count() for m in masks)
    # Positions are coprime to M, so their prime density is M/phi(M) times the ambient
    # 1/ln p -- the wheel concentrates primes, and by a large factor at this modulus.
    phi = M
    q, mm = 2, M
    while q * q <= mm:
        if mm % q == 0:
            phi -= phi // q
            while mm % q == 0:
                mm //= q
        q += 1
    if mm > 1:
        phi -= phi // mm
    expect = n * (M / phi) / math.log(LO_REF + M * npos / 2)
    ratio = got / expect
    ok = not bad_p and not bad_c and 0.9 < ratio < 1.1
    print(f"  {'ok  ' if not bad_p and not bad_c else 'FAIL'} prime mask vs Miller-Rabin "
          f"on {n} positions: {bad_p} claimed-prime composites, "
          f"{bad_c} missed primes")
    print(f"  {'ok  ' if 0.9 < ratio < 1.1 else 'FAIL'} prime density {got}/{n} = "
          f"{got/n:.4f}, PNT expects {expect/n:.4f} (ratio {ratio:.3f})")
    print("self-check OK" if ok else "SELF-CHECK FAILED")
    return ok


def load_lanes(M=None):
    mm = M or globals()['M']
    path = os.path.join(ROOT, "tables", f"class_table_{mm}.txt")
    return [int(l.split()[1]) for l in open(path) if l.startswith("CLASS ")]


def build_candidates(mmax):
    """Only filters whose modulus does NOT divide M are candidates: those that do are
    already spent on the wheel, so at a larger M the candidate set shrinks."""
    f1 = [(m, r, "F1") for m, r in gen_covers(mmax) if M % m]
    f2 = [(L, r, "F2") for L, r in gen_ucerts() if M % L]
    return f1, f2


def write_table(path, filters, mmax, marks, dens, pdens, nlanes, max_density):
    body = [
        "# certified filter table for rung_scan3 -- pruned by unique-kill accounting.",
        "# Soundness is MONOTONE in this set: dropping a filter can only leave more",
        "# survivors, never manufacture a false certificate.",
        f"# M={M} mmax={mmax} seg={SEG} npos={NPOS} lanes={nlanes} "
        f"lo_ref={LO_REF} max_density={max_density}",
        f"# ns_per_mark={NS_PER_MARK} ns_per_survivor_composite={NS_PER_SURVIVOR_COMPOSITE} "
        f"ns_per_survivor_prime={NS_PER_SURVIVOR_PRIME} "
        f"measured={'yes' if MEASURED else 'NO-PLACEHOLDERS'}",
        f"# scored_on=positions_weighted_by_live_prime_fraction "
        f"device={'yes' if DEVICE else 'no'}",
        f"# filters={len(filters)} marks_per_pos={marks:.4f} survivor_density={dens:.6e} "
        f"prime_density={pdens:.6e}",
    ]
    body += [f"FILTER {mod} {res} {kind}" for mod, res, kind in filters]
    text = "\n".join(body) + "\n"
    digest = hashlib.sha256(text.encode()).hexdigest()
    text = text.replace("# filters=", f"# sha256={digest}\n# filters=", 1)
    open(path, "w").write(text)
    return digest


def main():
    args = sys.argv[1:]
    if "--self-check" in args:
        lanes = load_lanes()[:2]
        f1, f2 = build_candidates(200)
        uniq, marks, surv, surv_p, pos = evaluate(f1 + f2, lanes, npos=NPOS)
        print(f"self-check: {len(f1)} F1 + {len(f2)} F2 on {len(lanes)} lanes -> "
              f"{marks/pos:.2f} marks/pos, survivor density {surv/pos:.3e}, "
              f"prime density {surv_p/pos:.3e}")
        assert sum(uniq) <= pos, "unique kills cannot exceed positions"
        assert surv_p <= surv, "surviving primes cannot exceed surviving positions"
        return 0 if check_prime_mask() else 1
    global M, LO_REF, NS_PER_MARK, NS_PER_SURVIVOR_COMPOSITE, NS_PER_SURVIVOR_PRIME
    global DEVICE, DEV_SHARD_POSITIONS
    if "--device" in args:
        DEVICE = True
        NS_PER_MARK = DEV_NS_PER_MARK
        NS_PER_SURVIVOR_COMPOSITE = DEV_NS_PER_SURVIVOR_COMPOSITE
        NS_PER_SURVIVOR_PRIME = DEV_NS_PER_SURVIVOR_PRIME
        if "--shard-positions" in args:
            DEV_SHARD_POSITIONS = int(args[args.index("--shard-positions") + 1])
        print(f"device cost model: {NS_PER_MARK} ns/mark, "
              f"{NS_PER_SURVIVOR_COMPOSITE} ns/composite survivor, "
              f"{NS_PER_SURVIVOR_PRIME} ns/prime survivor, "
              f"{DEV_NS_SETUP_PER_LANE_FILTER} ns setup per (lane,filter) over "
              f"{DEV_SHARD_POSITIONS} positions/lane")
    if "--wheel" in args: M = int(args[args.index("--wheel") + 1])
    mmax = int(args[args.index("--mmax") + 1]) if "--mmax" in args else 2000
    max_density = (float(args[args.index("--max-density") + 1])
                   if "--max-density" in args else 1.0)
    outdir = args[args.index("--write") + 1] if "--write" in args else None
    lanes = load_lanes()
    f1, f2 = build_candidates(mmax)
    print(f"candidates: {len(f1)} F1 (mmax={mmax}) + {len(f2)} F2 = {len(f1)+len(f2)}")
    base_uniq, base_marks, base_surv, base_surv_p, pos = evaluate(f1 + f2, lanes)
    print(f"unpruned: {base_marks/pos:.2f} marks/pos, "
          f"survivor density {base_surv/pos:.3e}, prime density {base_surv_p/pos:.3e}")
    kept, marks, dens, pdens = prune(f1 + f2, lanes, max_density)
    nf1 = sum(1 for f in kept if f[2] == "F1")
    nf2 = sum(1 for f in kept if f[2] == "F2")
    print(f"pruned:   {len(kept)} filters ({nf1} F1 + {nf2} F2), "
          f"{marks:.2f} marks/pos, survivor density {dens:.3e}, "
          f"prime density {pdens:.3e}")
    if marks > 0:
        print(f"          {base_marks/pos/marks:.2f}x fewer marks, "
              f"{dens/(base_surv/pos):.2f}x the survivors, "
              f"{pdens/(base_surv_p/pos):.2f}x the surviving primes")
    if outdir:
        tag = ("gpu" if DEVICE else "runtime") if max_density >= 1.0 else f"d{max_density:g}"
        path = f"{outdir.rstrip('/')}/f_M{M}_mmax{mmax}_{tag}.txt"
        digest = write_table(path, kept, mmax, marks, dens, pdens, len(lanes),
                             max_density)
        print(f"wrote {path}  sha256={digest}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
