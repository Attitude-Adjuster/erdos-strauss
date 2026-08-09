NVCC   ?= nvcc
ARCH   := -gencode arch=compute_89,code=sm_89 -gencode arch=compute_90,code=sm_90
CXXSTD := -std=c++17
OPT    := -O3
NVFLAGS := $(CXXSTD) $(OPT) $(ARCH) -Xcompiler -fopenmp

SRC := rung_scan_cuda.cu tables.cu factorize.cu classify.cu sieve.cu census.cu gate.cu ref_host.cpp
HDR := common.cuh tables.cuh factorize.cuh classify.cuh sieve.cuh census.cuh ref_host.h

rung_scan_cuda: $(SRC) $(HDR)
	$(NVCC) $(NVFLAGS) -o $@ $(SRC)

# `check` needs a freshly built ../rung_scan (the CPU reference binary) for
# tests/diff_range.sh to diff against. Build it here rather than assuming it
# already exists -- the binary is deliberately not distributed, and a
# stale one would silently pass a diff against old behavior.
../rung_scan:
	$(MAKE) -C ..

check: rung_scan_cuda ../rung_scan
	./rung_scan_cuda --verify
	./rung_scan_cuda --check-tables
	./tests/diff_range.sh 0 100000000 --hard840
	./tests/diff_range.sh 0 100000000
	# small, fast factorization diff against factorA -- the full-scale gate is
	# `make gate`, which takes minutes because factorA runs on millions of A.
	./rung_scan_cuda --gate 1000000000000 1000100000000 --hard840 --spanlog 14 --reps 1
	./rung_scan_cuda --gate 1000000000000 1000100000000 --spanlog 14 --reps 1
	# classifier vs the reference's test_rung, every rung r = 3 mod 4 with r <= 63,
	# in both fast (S confined to r=51) and strict (closure everywhere) mode.
	./rung_scan_cuda --check-classify 9999000000000 10000000000000 --hard840 --spanlog 16
	./rung_scan_cuda --check-classify 9999000000000 10000000000000 --spanlog 16
	# level S is ~0.13 events per 1e9 of height, so no affordable window contains one.
	# Replay the census's own 1,309 level-S certificates instead.
	./rung_scan_cuda --check-s tests/level_s_certificates.txt
	# the two factorization paths vs each other AND vs the reference's factorA
	./rung_scan_cuda --check-factor 1000000000000000 1000010000000000 --hard840
	./rung_scan_cuda --check-factor 0 100000000 --hard840 --spanlog 14
	# device sieve vs the reference's sieve AND vs Miller-Rabin at every position
	./rung_scan_cuda --check-sieve 9999000000000 10000000000000 --hard840
	./rung_scan_cuda --check-sieve 9999000000000 10000000000000
	# Task 5: the census itself. Low ranges first, then a high-offset run with every
	# emission path live (RESIDUAL/DEEP/SAMPLE), which the low ranges never reach.
	./tests/diff_range.sh 0 100000000 --hard840 --emit-residual --emit-deep 35
	./tests/diff_range.sh 9999000000000 9999200000000 --hard840 \
	    --emit-residual --emit-deep 35 --sample 100000
	# A deep ladder at a large window: enough primes stay active past r=63 that the
	# GPU/CPU handoff boundary is actually exercised. This is the case that caught the
	# classifier being handed r=67 with zero-filled tables -- keep it in the gate.
	./tests/diff_range.sh 0 10000000000 --hard840 --spanlog 24
	# the CPU handoff point is claimed to be a performance knob, not a correctness
	# one -- assert it instead of trusting it.
	./rung_scan_cuda --check-threshold 9999000000000 9999100000000 --hard840 --spanlog 16

# The GO/NO-GO measurement itself: one full window at spanlog 22, all 6 classes.
gate: rung_scan_cuda
	./rung_scan_cuda --gate 9999000000000 10000000000000 --hard840

clean:
	rm -f rung_scan_cuda

.PHONY: check gate clean ../rung_scan
