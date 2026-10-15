#pragma once

#include "common/cuda_check.hpp"

#include <cuda_runtime.h>

#include <cstdint>
#include <iomanip>
#include <iostream>
#include <string>

struct PoolStats {
  std::uint64_t used_current = 0;
  std::uint64_t used_high = 0;
  std::uint64_t reserved_current = 0;
  std::uint64_t reserved_high = 0;
  std::size_t device_free = 0;
  std::size_t device_total = 0;
};

inline PoolStats read_pool_stats(cudaMemPool_t pool) {
  PoolStats stats;
  CUDA_CHECK(cudaMemPoolGetAttribute(
      pool, cudaMemPoolAttrUsedMemCurrent, &stats.used_current));
  CUDA_CHECK(cudaMemPoolGetAttribute(pool, cudaMemPoolAttrUsedMemHigh,
                                     &stats.used_high));
  CUDA_CHECK(cudaMemPoolGetAttribute(
      pool, cudaMemPoolAttrReservedMemCurrent, &stats.reserved_current));
  CUDA_CHECK(cudaMemPoolGetAttribute(pool, cudaMemPoolAttrReservedMemHigh,
                                     &stats.reserved_high));
  CUDA_CHECK(cudaMemGetInfo(&stats.device_free, &stats.device_total));
  return stats;
}

inline double mib(std::uint64_t bytes) {
  return static_cast<double>(bytes) / (1024.0 * 1024.0);
}

inline void print_pool_stats(const std::string& phase, cudaMemPool_t pool) {
  const PoolStats stats = read_pool_stats(pool);
  std::cout << std::fixed << std::setprecision(2)
            << "POOL_STATS phase=" << phase
            << " used_current_mib=" << mib(stats.used_current)
            << " used_high_mib=" << mib(stats.used_high)
            << " reserved_current_mib=" << mib(stats.reserved_current)
            << " reserved_high_mib=" << mib(stats.reserved_high)
            << " device_free_mib=" << mib(stats.device_free) << '\n';
}

