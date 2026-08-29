// common.cuh -- small shared utilities for the CUDA port. No kernels here; this
// is glue (error checking, table-layout constants) used by tables.cu today and by
// the classifier kernels in later tasks.
#pragma once
#include <cstdio>
#include <cstdlib>
#include <cuda_runtime.h>

// Every CUDA runtime call in this port is wrapped in this so a device-side or
// allocation failure aborts loudly with file:line instead of silently producing
// wrong statistics -- the whole point of RUNBOOK.md's verification discipline.
#define CUDA_CHECK(expr) do {                                              \
    cudaError_t _cuda_check_err = (expr);                                  \
    if (_cuda_check_err != cudaSuccess) {                                  \
      fprintf(stderr, "CUDA error %s:%d: %s\n", __FILE__, __LINE__,        \
              cudaGetErrorString(_cuda_check_err));                        \
      abort();                                                             \
    }                                                                      \
  } while (0)
