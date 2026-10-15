#include "common/cuda_check.hpp"
#include "common/workload.cuh"

#include <cuda_runtime.h>

#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <iostream>
#include <string>
#include <vector>

namespace {

std::size_t parse_elements(int argc, char** argv) {
  std::size_t elements = 1U << 20U;
  for (int i = 1; i < argc; ++i) {
    const std::string argument = argv[i];
    if (argument == "--elements" && i + 1 < argc) {
      elements = std::stoull(argv[++i]);
    } else if (argument == "--help") {
      std::cout << "Usage: lab01_lifetime [--elements N]\n";
      std::exit(EXIT_SUCCESS);
    } else {
      std::cerr << "Unknown argument: " << argument << '\n';
      std::exit(EXIT_FAILURE);
    }
  }
  return elements;
}

void same_stream_case(std::size_t count) {
  constexpr unsigned int block_size = 256U;
  constexpr std::uint32_t increment = 7U;
  const std::size_t bytes = count * sizeof(std::uint32_t);
  std::vector<std::uint32_t> host(count);

  cudaStream_t stream = nullptr;
  CUDA_CHECK(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));

  std::uint32_t* device = nullptr;
  CUDA_CHECK(cudaMallocAsync(reinterpret_cast<void**>(&device), bytes, stream));
  std::cout << "ALLOC case=same_stream ptr=" << static_cast<void*>(device)
            << " bytes=" << bytes << '\n';

  initialize_values<<<grid_for(count, block_size), block_size, 0, stream>>>(
      device, count);
  CUDA_KERNEL_CHECK();
  add_to_values<<<grid_for(count, block_size), block_size, 0, stream>>>(
      device, count, increment);
  CUDA_KERNEL_CHECK();
  CUDA_CHECK(cudaMemcpyAsync(host.data(), device, bytes,
                             cudaMemcpyDeviceToHost, stream));
  CUDA_CHECK(cudaFreeAsync(device, stream));
  CUDA_CHECK(cudaStreamSynchronize(stream));

  std::size_t bad_index = 0;
  const bool ok = verify_host_values(host, increment, &bad_index);
  std::cout << "RESULT case=same_stream status=" << (ok ? "PASS" : "FAIL")
            << " checksum=" << host_checksum(host);
  if (!ok) {
    std::cout << " first_bad_index=" << bad_index;
  }
  std::cout << '\n';
  CUDA_CHECK(cudaStreamDestroy(stream));

  if (!ok) {
    std::exit(EXIT_FAILURE);
  }
}

void cross_stream_case(std::size_t count) {
  constexpr unsigned int block_size = 256U;
  constexpr std::uint32_t increment = 11U;
  const std::size_t bytes = count * sizeof(std::uint32_t);
  std::vector<std::uint32_t> host(count);

  cudaStream_t owner_stream = nullptr;
  cudaStream_t consumer_stream = nullptr;
  cudaEvent_t allocation_ready = nullptr;
  cudaEvent_t work_done = nullptr;
  CUDA_CHECK(cudaStreamCreateWithFlags(&owner_stream, cudaStreamNonBlocking));
  CUDA_CHECK(
      cudaStreamCreateWithFlags(&consumer_stream, cudaStreamNonBlocking));
  CUDA_CHECK(cudaEventCreateWithFlags(&allocation_ready, cudaEventDisableTiming));
  CUDA_CHECK(cudaEventCreateWithFlags(&work_done, cudaEventDisableTiming));

  std::uint32_t* device = nullptr;
  CUDA_CHECK(
      cudaMallocAsync(reinterpret_cast<void**>(&device), bytes, owner_stream));
  std::cout << "ALLOC case=cross_stream ptr=" << static_cast<void*>(device)
            << " bytes=" << bytes << '\n';

  initialize_values<<<grid_for(count, block_size), block_size, 0, owner_stream>>>(
      device, count);
  CUDA_KERNEL_CHECK();
  CUDA_CHECK(cudaEventRecord(allocation_ready, owner_stream));

  CUDA_CHECK(cudaStreamWaitEvent(consumer_stream, allocation_ready, 0));
  add_to_values<<<grid_for(count, block_size), block_size, 0, consumer_stream>>>(
      device, count, increment);
  CUDA_KERNEL_CHECK();
  CUDA_CHECK(cudaMemcpyAsync(host.data(), device, bytes,
                             cudaMemcpyDeviceToHost, consumer_stream));
  CUDA_CHECK(cudaEventRecord(work_done, consumer_stream));

  CUDA_CHECK(cudaStreamWaitEvent(owner_stream, work_done, 0));
  CUDA_CHECK(cudaFreeAsync(device, owner_stream));
  CUDA_CHECK(cudaStreamSynchronize(owner_stream));

  std::size_t bad_index = 0;
  const bool ok = verify_host_values(host, increment, &bad_index);
  std::cout << "DEPENDENCY allocation_ready=event consumer_done=event\n";
  std::cout << "RESULT case=cross_stream status=" << (ok ? "PASS" : "FAIL")
            << " checksum=" << host_checksum(host);
  if (!ok) {
    std::cout << " first_bad_index=" << bad_index;
  }
  std::cout << '\n';

  CUDA_CHECK(cudaEventDestroy(work_done));
  CUDA_CHECK(cudaEventDestroy(allocation_ready));
  CUDA_CHECK(cudaStreamDestroy(consumer_stream));
  CUDA_CHECK(cudaStreamDestroy(owner_stream));

  if (!ok) {
    std::exit(EXIT_FAILURE);
  }
}

}  // namespace

int main(int argc, char** argv) {
  const std::size_t elements = parse_elements(argc, argv);
  CUDA_CHECK(cudaSetDevice(0));
  if (!memory_pools_supported()) {
    std::cout << "SKIP reason=memory_pools_not_supported\n";
    return 0;
  }

  cudaDeviceProp properties{};
  CUDA_CHECK(cudaGetDeviceProperties(&properties, 0));
  std::cout << "LAB name=stream_ordered_lifetime gpu=\"" << properties.name
            << "\" elements=" << elements << '\n';
  same_stream_case(elements);
  cross_stream_case(elements);
  std::cout << "LAB_RESULT status=PASS\n";
  return 0;
}

