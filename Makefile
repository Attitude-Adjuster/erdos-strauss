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

CXXFLAGS := $(STD) $(OPT) $(WARN) $(OMPFLAG)
LDFLAGS  := $(OMPFLAG)

all: rung_scan

# The optimized second scanner. NOT the reference: it earns trust via tests/diff2.sh
# (identical statistical SUMMARY + identical sorted certificate set vs rung_scan on
# every range tested) plus the same --verify self-test.
rung_scan2: rung_scan2.cpp
	$(CXX) $(CXXFLAGS) -o $@ $< $(LDFLAGS)

rung_scan: rung_scan.cpp
	$(CXX) $(CXXFLAGS) -o $@ $< $(LDFLAGS)
	@echo
	@echo "Built. Now run 'make check' before trusting any output from this binary."

# Recomputes the whole p < 10^5 census and compares every statistic against the
# independent Python/SymPy reference. Guards against miscompilation and unsafe flags,
# not just logic errors -- so it must be run per machine, per compiler, per flag set.
check: rung_scan
	./rung_scan --verify

portable:
	$(MAKE) NATIVE=0

serial:
	$(MAKE) OPENMP=0

clean:
	rm -f rung_scan rung_scan2

.PHONY: all check portable serial clean
