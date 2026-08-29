# Makefile for rung_scan -- Erdos-Straus certificate census
#
#   make            build (native tuning, OpenMP on)
#   make check      run the self-test  <-- DO THIS on every machine before a real run
#   make portable   build without -march=native (heterogeneous clusters)
#   make clean

CXX      ?= g++
STD      := -std=c++17
WARN     := -Wall -Wextra
OPT      := -O3 -funroll-loops -fno-math-errno
NATIVE   ?= 1
OPENMP   ?= 1

ifeq ($(NATIVE),1)
  OPT += -march=native -mtune=native
endif
ifeq ($(OPENMP),1)
  OMPFLAG := -fopenmp
else
  OMPFLAG :=
endif

# -Icensus is LOAD-BEARING, not tidiness. cover_scan.cpp is FROZEN and contains
# `#include "rung_scan.cpp"`, which cannot resolve now that the two live in different
# directories -- and the file may not be edited to fix it. A quoted include searches the
# including file's directory first and then the -I list, so this makes it resolve with
# no source change. rung_scan3.cpp gets the same treatment so both build alike.
CXXFLAGS := $(STD) $(OPT) $(WARN) $(OMPFLAG) -Icensus
LDFLAGS  := $(OMPFLAG)

# Binaries are built at the repo root, where every test and runbook expects them.

all: rung_scan

# The optimized second scanner. NOT the reference: it earns trust via tests/diff2.sh
# (identical statistical SUMMARY + identical sorted certificate set vs rung_scan on
# every range tested) plus the same --verify self-test.
rung_scan2: census/rung_scan2.cpp
	$(CXX) $(CXXFLAGS) -o $@ $< $(LDFLAGS)

# The verification sieve's REFERENCE implementation: naive, simple, and frozen once
# the phase-1 plan lands. rung_scan3 must agree with it byte for byte (tests/diff3.sh).
# Depends on rung_scan.cpp because it #includes it (with main renamed) to reuse the
# frozen reference's factoring verbatim.
cover_scan: sieve/cover_scan.cpp census/rung_scan.cpp
	$(CXX) $(CXXFLAGS) -o $@ $< $(LDFLAGS)

# The OPTIMIZED verification scanner. NOT a reference: it earns trust via
# tests/diff3.sh (identical SUMMARY + identical sorted certificate set vs cover_scan
# on every range tested) plus the same --verify constants.
rung_scan3: sieve/rung_scan3.cpp census/rung_scan.cpp
	$(CXX) $(CXXFLAGS) -o $@ $< $(LDFLAGS)

rung_scan: census/rung_scan.cpp
	$(CXX) $(CXXFLAGS) -o $@ $< $(LDFLAGS)
	@echo
	@echo "Built. Now run 'make check' before trusting any output from this binary."

# Recomputes the whole p < 10^5 census and compares every statistic against the
# independent Python/SymPy reference. Guards against miscompilation and unsafe flags,
# not just logic errors -- so it must be run per machine, per compiler, per flag set.
check: rung_scan
	./rung_scan --verify

# The reference verification sieve's gate. Run on every machine and every flag set,
# for the same reason `make check` is: it guards against miscompilation and unsafe
# optimization flags, not just logic. The --mmax 200 is deliberate -- it leaves enough
# survivors to exercise stages C/D/E, so the gate can actually fail on the survivor
# path. At the default mmax the same range emits nothing and proves much less.
check-cover: cover_scan
	./tests/cover_smoke.sh 1000000000000 1000200000000 --mmax 200

# The differential gate: the optimized scanner against the frozen reference.
# --wheel is passed explicitly: cover_scan defaults to 120120 and rung_scan3 now
# defaults to 2042040, so leaving it implicit would compare two different wheels.
# The 2042040 row is slow (~33 s) because the FROZEN reference still enumerates every
# cover <= M; that is the price of not touching it, and it is worth paying.
check-diff3: cover_scan rung_scan3
	./tests/diff3.sh 1000000000000 1000200000000 --wheel 120120 --mmax 200
	./tests/diff3.sh 1000000000000 1000010000000 --wheel 120120 --mmax 2000
	./tests/diff3.sh 1000000000000 1000200000000 --wheel 2042040 --mmax 200

# Stage E, which no scan has ever reached: `direct=0` across 1.13e12 positions to 10^15,
# because stage D resolves everything that survives the covers. Calling it directly is the
# only way to exercise it -- and the rows either side of 2^63 are the ones that matter,
# because the old guard declined SILENTLY there and an unsolved prime is then reported as
# a survivor, which reads as a mathematical result rather than the numeric limit it is.
# Certificates go to stdout; verify_covers.py re-derives them in exact rationals.
# The fast stage-D arithmetic (Montgomery + magic-inverse trial division) against the
# frozen factorA: canonical factorizations compared on 32k adversarial values -- hard
# semiprimes both sides of 2^63, prime squares, smooth numbers, the TD_BOUND^2 and
# DIRECT_NMAX boundaries -- plus factor-primality and product reconstruction, which
# are independent of the frozen comparison. ~4 s.
check-factor: rung_scan3
	./rung_scan3 --check-factor 2000

check-direct: rung_scan3
	./rung_scan3 --check-direct 9223372036854770000     20 > /tmp/d_below.txt
	./rung_scan3 --check-direct 9223372036854775808     20 > /tmp/d_at63.txt
	./rung_scan3 --check-direct 9999999999999990000     20 > /tmp/d_e19.txt
	./rung_scan3 --check-direct 18446744065119000000    20 > /tmp/d_ceil.txt
	python3 sieve/verify_covers.py /tmp/d_below.txt /tmp/d_at63.txt /tmp/d_e19.txt /tmp/d_ceil.txt

portable:
	$(MAKE) NATIVE=0

serial:
	$(MAKE) OPENMP=0

clean:
	rm -f rung_scan rung_scan2 cover_scan rung_scan3

.PHONY: check-factor check-direct all check check-cover check-diff3 portable serial clean
