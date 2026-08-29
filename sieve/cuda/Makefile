# The verification sieve on CUDA. Earns trust exactly as the census port does:
# byte-identical SUMMARY and an identical sorted certificate set vs the CPU scanner
# (tests/diff_cuda.sh), plus the CPU scanner's own --verify running inside this binary.
NVCC    ?= nvcc
ARCH    := -gencode arch=compute_89,code=sm_89
CXXSTD  := -std=c++17
OPT     := -O3
# -I../../census is LOAD-BEARING: sieve/rung_scan3.cpp contains
# `#include "rung_scan.cpp"`, and the frozen census/rung_scan.cpp lives two levels up.
# Same flag the root Makefile carries for the same reason. Remove it and nothing builds.
NVFLAGS := $(CXXSTD) $(OPT) $(ARCH) -Xcompiler -fopenmp -I../../census -I../..

SRC := sieve_cuda.cu mark.cu survivors.cu scan.cu ref_host.cpp
HDR := ref_host.h common.cuh mark.cuh survivors.cuh scan.cuh

sieve_cuda: $(SRC) $(HDR)
	$(NVCC) $(NVFLAGS) -o $@ $(SRC)

# The CPU scanner builds at the repo root; build it here rather than assuming it
# exists, since the sync driver deliberately excludes binaries and a stale one would silently
# pass a diff against old behaviour.
../../rung_scan3:
	$(MAKE) -C ../.. rung_scan3

check: sieve_cuda ../../rung_scan3
	./sieve_cuda --verify
	./sieve_cuda --check-mark --wheel 120120  --mmax 200
	./sieve_cuda --check-mark --wheel 120120  --mmax 200   --sieve 1000
	./sieve_cuda --check-mark --wheel 2042040 --mmax 10000
	./sieve_cuda --check-mark --wheel 2042040 --mmax 10000 --sieve 1000
	./sieve_cuda --check-compact --wheel 120120  --mmax 200
	./sieve_cuda --check-compact --wheel 2042040 --mmax 10000 --sieve 1000
	./sieve_cuda --check-primality 1000000000000 2000000
	./tests/diff_cuda.sh 1000000000000 1000200000000 --wheel 120120 --mmax 200
	./tests/diff_cuda.sh 1000000000000 1000200000000 --wheel 2042040 --mmax 200
# The two rows above fit in ONE segment, so neither can fail on the offset carry --
# off[i] = j - n is invisible until a lane runs a second segment. --seg 4096 over a
# 5e10 range gives six segments per lane and 250k certificates for 8 s. A gate that
# cannot fail on the path you care about is decoration.
	./tests/diff_cuda.sh 1000000000000 1050000000000 --wheel 2042040 --mmax 200 --seg 4096
# The device-tuned filter table is a published artifact and the default the port should
# be run with, so it is gated like one -- and at --seg 4096, because a pruned table's
# whole point is a different marks/survivors balance and the carry path is where a
# filter-set change would show up as a wrong answer rather than a slow one.
	./tests/diff_cuda.sh 1000000000000 1050000000000 --seg 4096 \
	  --filters ../../tables/filters/f_M2042040_mmax2000_gpu.txt
# --shard is how a production run is driven and had no coverage at all until 2026-08-18.
# Three shapes: one that divides evenly, one that does not (a floor-vs-ceil span shows up
# only in the remainder), and one with more shards than the range can fill, which is the
# empty-shard path the tiling check has to tolerate.
	./tests/shard_cuda.sh 1000000000000 1000200000000 8  --wheel 2042040 --mmax 200 --sieve 0
	./tests/shard_cuda.sh 1000000000000 1000200000000 7  --wheel 2042040 --mmax 200 --sieve 0
	./tests/shard_cuda.sh 1000000000000 1000010000000 64 --wheel 2042040 --mmax 200 --sieve 0
# Every other diff row sits at 10^12. This one sits at the top of the range the port has
# actually been run over, because agreeing with the CPU scanner at one height is not the
# same claim as agreeing with it at the height you intend to run at.
	./tests/diff_cuda.sh 999000000000000 1000000000000000 \
	  --filters ../../tables/filters/f_M2042040_mmax2000_gpu.txt

clean:
	rm -f sieve_cuda

.PHONY: check clean ../../rung_scan3
