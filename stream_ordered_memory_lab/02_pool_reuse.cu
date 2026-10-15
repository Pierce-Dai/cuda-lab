#include "common/cuda_check.hpp"
#include "common/pool_stats.hpp"
#include "common/workload.cuh"

#include <cuda_runtime.h>

#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <iostream>
#include <limits>
#include <string>

namespace {

struct Options {
  std::size_t allocation_mib = 32;
  std::size_t release_threshold_mib = 64;
  int follow_events = 1;
  int opportunistic = 1;
  int internal_dependencies = 1;
};

Options parse_options(int argc, char** argv) {
  Options options;
  for (int i = 1; i < argc; ++i) {
    const std::string argument = argv[i];
    auto next_value = [&]() -> std::string {
      if (i + 1 >= argc) {
        std::cerr << "Missing value after " << argument << '\n';
        std::exit(EXIT_FAILURE);
      }
      return argv[++i];
    };
    if (argument == "--allocation-mib") {
      options.allocation_mib = std::stoull(next_value());
    } else if (argument == "--release-threshold-mib") {
      options.release_threshold_mib = std::stoull(next_value());
    } else if (argument == "--follow-events") {
      options.follow_events = std::stoi(next_value());
    } else if (argument == "--opportunistic") {
      options.opportunistic = std::stoi(next_value());
    } else if (argument == "--internal-dependencies") {
      options.internal_dependencies = std::stoi(next_value());
    } else if (argument == "--help") {
      std::cout
          << "Usage: lab02_pool_reuse [--allocation-mib N] "
             "[--release-threshold-mib N] [--follow-events 0|1] "
             "[--opportunistic 0|1] [--internal-dependencies 0|1]\n";
      std::exit(EXIT_SUCCESS);
    } else {
      std::cerr << "Unknown argument: " << argument << '\n';
      std::exit(EXIT_FAILURE);
    }
  }
  return options;
}

cudaMemPool_t create_pool(const Options& options) {
  cudaMemPoolProps properties{};
  properties.allocType = cudaMemAllocationTypePinned;
  properties.handleTypes = cudaMemHandleTypeNone;
  properties.location.type = cudaMemLocationTypeDevice;
  properties.location.id = 0;

  cudaMemPool_t pool = nullptr;
  CUDA_CHECK(cudaMemPoolCreate(&pool, &properties));

  std::uint64_t threshold =
      options.release_threshold_mib * 1024ULL * 1024ULL;
  int follow_events = options.follow_events;
  int opportunistic = options.opportunistic;
  int internal_dependencies = options.internal_dependencies;
  CUDA_CHECK(cudaMemPoolSetAttribute(
      pool, cudaMemPoolAttrReleaseThreshold, &threshold));
  CUDA_CHECK(cudaMemPoolSetAttribute(
      pool, cudaMemPoolReuseFollowEventDependencies, &follow_events));
  CUDA_CHECK(cudaMemPoolSetAttribute(
      pool, cudaMemPoolReuseAllowOpportunistic, &opportunistic));
  CUDA_CHECK(cudaMemPoolSetAttribute(
      pool, cudaMemPoolReuseAllowInternalDependencies,
      &internal_dependencies));
  return pool;
}

}  // namespace

int main(int argc, char** argv) {
  const Options options = parse_options(argc, argv);
  CUDA_CHECK(cudaSetDevice(0));
  if (!memory_pools_supported()) {
    std::cout << "SKIP reason=memory_pools_not_supported\n";
    return 0;
  }

  const std::size_t bytes = options.allocation_mib * 1024ULL * 1024ULL;
  const std::size_t elements = bytes / sizeof(std::uint32_t);
  constexpr unsigned int block_size = 256U;

  std::cout << "LAB name=pool_reuse allocation_mib=" << options.allocation_mib
            << " release_threshold_mib=" << options.release_threshold_mib
            << " follow_events=" << options.follow_events
            << " opportunistic=" << options.opportunistic
            << " internal_dependencies=" << options.internal_dependencies
            << '\n';

  cudaMemPool_t pool = create_pool(options);
  cudaStream_t stream = nullptr;
  CUDA_CHECK(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));
  print_pool_stats("created", pool);

  std::uint32_t* first = nullptr;
  CUDA_CHECK(cudaMallocFromPoolAsync(reinterpret_cast<void**>(&first), bytes,
                                     pool, stream));
  initialize_values<<<grid_for(elements, block_size), block_size, 0, stream>>>(
      first, elements);
  CUDA_KERNEL_CHECK();
  CUDA_CHECK(cudaStreamSynchronize(stream));
  std::cout << "ADDRESS generation=1 ptr=" << static_cast<void*>(first) << '\n';
  print_pool_stats("after_first_allocation", pool);

  CUDA_CHECK(cudaFreeAsync(first, stream));
  CUDA_CHECK(cudaStreamSynchronize(stream));
  print_pool_stats("after_first_free", pool);

  std::uint32_t* second = nullptr;
  CUDA_CHECK(cudaMallocFromPoolAsync(reinterpret_cast<void**>(&second), bytes,
                                     pool, stream));
  initialize_values<<<grid_for(elements, block_size), block_size, 0, stream>>>(
      second, elements);
  CUDA_KERNEL_CHECK();
  CUDA_CHECK(cudaStreamSynchronize(stream));
  std::cout << "ADDRESS generation=2 ptr=" << static_cast<void*>(second)
            << " reused=" << (second == first ? "yes" : "no") << '\n';
  print_pool_stats("after_second_allocation", pool);

  CUDA_CHECK(cudaFreeAsync(second, stream));
  CUDA_CHECK(cudaStreamSynchronize(stream));
  print_pool_stats("after_second_free", pool);

  CUDA_CHECK(cudaMemPoolTrimTo(pool, 0));
  print_pool_stats("after_trim_to_zero", pool);

  CUDA_CHECK(cudaStreamDestroy(stream));
  CUDA_CHECK(cudaMemPoolDestroy(pool));
  std::cout << "LAB_RESULT status=PASS note=address_reuse_is_observation_not_api_guarantee\n";
  return 0;
}
