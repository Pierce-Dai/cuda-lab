#pragma once

#include <cuda_runtime.h>

#include <cstddef>
#include <cstdint>
#include <vector>

__host__ __device__ inline std::uint32_t initial_value(std::size_t index) {
  return static_cast<std::uint32_t>((index % 251U) + 1U);
}

__global__ void initialize_values(std::uint32_t* values, std::size_t count) {
  const std::size_t index = blockIdx.x * blockDim.x + threadIdx.x;
  if (index < count) {
    values[index] = initial_value(index);
  }
}

__global__ void add_to_values(std::uint32_t* values, std::size_t count,
                              std::uint32_t increment) {
  const std::size_t index = blockIdx.x * blockDim.x + threadIdx.x;
  if (index < count) {
    values[index] += increment;
  }
}

__global__ void verify_and_add(std::uint32_t* values, std::size_t count,
                               std::uint32_t increment, unsigned int* errors) {
  const std::size_t index = blockIdx.x * blockDim.x + threadIdx.x;
  if (index < count) {
    if (values[index] != initial_value(index)) {
      atomicAdd(errors, 1U);
    }
    values[index] += increment;
  }
}

inline dim3 grid_for(std::size_t count, unsigned int block_size = 256U) {
  return dim3(static_cast<unsigned int>((count + block_size - 1U) / block_size));
}

inline bool verify_host_values(const std::vector<std::uint32_t>& values,
                               std::uint32_t increment,
                               std::size_t* first_bad_index = nullptr) {
  for (std::size_t i = 0; i < values.size(); ++i) {
    if (values[i] != initial_value(i) + increment) {
      if (first_bad_index != nullptr) {
        *first_bad_index = i;
      }
      return false;
    }
  }
  return true;
}

inline std::uint64_t host_checksum(
    const std::vector<std::uint32_t>& values) {
  std::uint64_t checksum = 0;
  for (const auto value : values) {
    checksum += value;
  }
  return checksum;
}

