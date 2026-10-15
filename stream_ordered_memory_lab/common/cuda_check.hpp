#pragma once

#include <cuda_runtime.h>

#include <cstdlib>
#include <iostream>

inline void cuda_check(cudaError_t status, const char* expression,
                       const char* file, int line) {
  if (status == cudaSuccess) {
    return;
  }
  std::cerr << "CUDA_ERROR expression=\"" << expression << "\" code="
            << static_cast<int>(status) << " name=" << cudaGetErrorName(status)
            << " message=\"" << cudaGetErrorString(status) << "\" at=" << file
            << ':' << line << '\n';
  std::exit(EXIT_FAILURE);
}

#define CUDA_CHECK(expression) \
  cuda_check((expression), #expression, __FILE__, __LINE__)

#define CUDA_KERNEL_CHECK() CUDA_CHECK(cudaGetLastError())

inline bool memory_pools_supported(int device = 0) {
  int supported = 0;
  CUDA_CHECK(cudaDeviceGetAttribute(
      &supported, cudaDevAttrMemoryPoolsSupported, device));
  return supported != 0;
}

