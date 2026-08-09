// facr_append.cuh -- the one step of factorization that two TUs must agree on.
//
// After the strided walk has divided out every prime <= sqrt(Amax), whatever remains
// of A is prime by construction (rung_scan.cpp:544 makes the same append). Appending
// it is the last step of building a FacR, and it is needed in two places: k_finish in
// factorize.cu (the standalone path the gate and the check harnesses test) and the
// fused classify-tally kernel in classify.cu (the census path, which appends in
// registers instead of writing the FacR back just to re-read it). One shared inline
// is what keeps those two paths incapable of drifting apart.
#pragma once
#include <cstdio>
#include "factorize.cuh"

__device__ __forceinline__ void facr_append_cofactor(uint64_t rem, uint32_t r, FacR &f) {
  if (rem > 1) {
    if (f.k >= FACR_MAX) { printf("FATAL: FacR overflow on cofactor\n"); __trap(); }
    f.res[f.k] = (uint8_t)(rem % r);
    f.ex[f.k]  = 1;
    ++f.k;
  }
}
