# The wheel: derived, never chosen

The verification sieve's stage A skips all but 2,308 of the 368,640 unit residue
classes mod `M = 2,042,040`. This document answers two questions a careful reader
should ask: **who decided which classes survive**, and **how can that decision be
checked from outside the project**.

The short answers: nobody decided, and four independent ways — the weakest of which
needs only pencil and paper.

## 1. Provenance rule: generate everything, hard-code nothing

No class list was copied from the literature. Terzi's published 198-class table
reportedly contains errors (Elsholtz–Tao note the residue classes were sound while
the checked-prime list was not), and transcription is exactly the kind of silent
trust this project is built to avoid. Instead, the surviving classes are the
*residue of a derivation*: every unit class mod `M` is tested against two families
of certificates, and a class survives only if **no certificate at all kills it**.
The killed classes are not discarded — every one of the 366,332 kills is recorded
in `tables/class_table_2042040.txt` with its witness, so the table is a complete,
finite proof object, not a summary.

The only human choices are **parameters**, and they are published in the table
header: the wheel modulus `M` and the F2 search bounds `(r ≤ 63, u ≤ 64)`. Change
either and you get a different (but equally certified) wheel; the lane count is a
parameter of the run, not a constant of the mathematics.

## 2. The two certificate families

**F1 — covering progressions.** For positive integers `a, c, d, k` with `ck > a`,
put `m = 4acd − 1` and `e = ck − a`; then every integer `p ≡ −4a²d (mod m)` with
`p ≥ p_min(a,c,d)` satisfies `p + k = 4ade` and

```
4/p = 1/(ade) + 1/(acdp) + 1/(cdep)
```

A class `ρ (mod M)` dies when some F1 cover with `m | M` has `ρ ≡ −4a²d (mod m)`:
every member of the class above `p_min` is then solved by one two-line identity.
The family is complete in a precise sense — reparameterized by `N = acd`,
`α = c²d`, it is exactly `{(N, α) : α | N²}` with `m = 4N − 1` — and its modulus is
**always ≡ 3 (mod 4)**. That last fact is a structural blindness: F1 can never
constrain a residue mod 8, which is why the classes `7, 13, 19 (mod 24)` survive
F1 alone, and why a second family is *necessary*, not decorative.

**F2 — uniform rung certificates.** A pair `(r, u)` with `gcd(r, u) = 1` such that
the rung-`r` decomposition with witness `u` —

```
A = (p+r)/4,   q = pA,   x = (q+u)/r,   y = q(q+u)/(r·u)
```

— works for **every** `p` in a residue class. Both rung conditions (`u | q²` and
`u ≡ −q (mod r)`) provably depend only on `p mod L` where `L = 4·lcm(r, u)`, so a
single exact-arithmetic check per class certifies the whole class. A class
`ρ (mod M)` dies when some F2 certificate with `L | M` has `ρ ≡ res (mod L)`.

Together the two families kill everything outside `p ≡ 1 (mod 24)`. This means the
classical reduction to the hard class is **derived here, not cited**: Mordell's
theorem is demoted from an input to a cross-check (§4, level 1).

**The soundness boundary.** Only certificates whose modulus divides `M` may touch
the wheel. An F2 certificate with `L ∤ M` kills a sub-progression *inside* a lane —
that is stage B's granularity, and applying it to the wheel would be a soundness
bug, not an optimization.

## 3. The derivation, and the table grammar

For each `ρ` coprime to `M`, in ascending order: try every F1 cover with `m | M`
(canonical order: ascending `(m, res)`), then every F2 certificate with `L | M`
(ascending `(L, res)`). Record the **first** killer, or survival. First-killer
canonical order makes the table reproducible **byte for byte** by any independent
implementation — there is no tie-breaking freedom to hide in.

```
# unit_classes=368640 classes=2308 kills=366332 fnv64=… sha256=…
CLASS 1                      # a surviving lane
KILL 19  F1 7 5 2 1 1        # KILL rho F1 m res a c d
KILL 211 F2 4 3 1 1          # KILL rho F2 L res r u
```

**Why `M = 2,042,040` and not bigger?** The next candidate, `38,798,760`, gives
2.97× fewer positions — and a ~250 MB kill table. This table's entire value is
being *a finite object a reader can check line by line*; at 250 MB it becomes a
thing you regenerate and hope. `2,042,040` (~14 MB, 1.62× fewer positions than the
older `120,120` wheel) is the largest wheel that keeps the proof object human-scale.
The larger wheel remains available via `--wheel` for anyone willing to regenerate
its table locally.

## 4. External verification, in increasing strength

**Level 0 — pencil and paper, no code.** Every `KILL` line is checkable by hand.

*F1 example*: `KILL 19 F1 7 5 2 1 1` claims class 19 dies by the cover
`(a,c,d) = (2,1,1)`. Check: `m = 4·2·1·1 − 1 = 7` ✓; `res = −4·2²·1 = −16 ≡ 5
(mod 7)` ✓; `19 ≡ 5 (mod 7)` ✓. Instantiate at `p = 19` itself: `k = (p+16)/7 = 5`,
`e = ck − a = 3`, and indeed `4/19 = 1/6 + 1/38 + 1/57` (common denominator 114:
`19 + 3 + 2 = 24 = 4·114/19`).

*F2 example*: `KILL 1621 F2 24 13 3 2` claims class 1621 dies because
`1621 ≡ 13 (mod 24)` and the certificate `(r,u) = (3,2)` solves everything
`≡ 13 (mod 24)`. Check `L = 4·lcm(3,2) = 24` ✓ and `1621 mod 24 = 13` ✓; then at
the smallest member `p = 13`: `A = (13+3)/4 = 4`, `q = 52`, `u | q²` is `2 | 2704`
✓, `(q+u) mod r = 54 mod 3 = 0` ✓, giving `x = 18`, `y = 468`, and
`4/13 = 1/4 + 1/18 + 1/468` (common denominator 468: `117 + 26 + 1 = 144 = 4·36`).
The `L`-periodicity argument in §2 is what promotes this single check to the whole
class. (The simplest F2 certificate, `KILL … F2 4 3 1 1`, is the classical
`p ≡ 3 (mod 4)` identity `4/p = 1/A + 1/(q+1) + 1/q(q+1)` — the textbook easy case
appears here as the first uniform certificate, as it should.)

**Level 1 — independent implementation.**

```sh
python3 sieve/verify_covers.py --class-table --wheel 2042040 \
        --check tables/class_table_2042040.txt
```

Stdlib-only Python re-derives the entire table from scratch in exact arithmetic and
must print `MATCH` (byte identity). Along the way it re-validates every kill
witness arithmetically, checks every survivor against every certificate (a
survivor that could have been killed is a failure), asserts
`survivors + kills = unit classes`, and — because the derivation nowhere assumes
Mordell — **requires every survivor to reduce into Mordell's six square classes
`{1, 121, 169, 289, 361, 529} (mod 840)`**. The derivation is independent of
Mordell, so it must *reproduce* Mordell; if it ever doesn't, either this code or a
century of number theory is wrong, and the check does not care which.

**Level 2 — the scanners refuse to disagree.** Both scanners re-derive the class
table at startup and compare it against a digest pinned per modulus in the source;
on mismatch they refuse to run. There is no code path that scans with an
unverified wheel at a pinned modulus.

**Level 3 — the frozen reference.** `cover_scan`, the sieve's frozen reference
implementation, computes the wheel by the slow, naive enumeration (every cover
with modulus `≤ M`, no divisor-driven shortcut) and independently reports
`lanes=2308` at `--wheel 2042040`. The fast generator in the optimized scanner is
additionally asserted equal to the naive one at `M = 120,120` on every `--verify`
run — that assertion is the entire correctness argument for the shortcut.

**Checksums.** The table header carries `sha256` and `fnv64` over the survivor
list, alongside the parameters `(M, rmax, umax)` that produced it. Quote all
three when citing a lane count: "2,308 lanes" is meaningless without the bounds,
because the lane count is a parameter, not a constant.
