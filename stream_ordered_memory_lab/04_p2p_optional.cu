#include "common/cuda_check.hpp"

#include <cuda_runtime.h>

#include <cstdlib>
#include <iostream>

int main() {
  int device_count = 0;
  CUDA_CHECK(cudaGetDeviceCount(&device_count));
  std::cout << "LAB name=p2p_pool_access device_count=" << device_count << '\n';
  if (device_count < 2) {
    std::cout << "SKIP reason=requires_at_least_two_cuda_devices\n";
    return 0;
  }

  int can_access = 0;
  CUDA_CHECK(cudaDeviceCanAccessPeer(&can_access, 1, 0));
  if (!can_access) {
    std::cout << "SKIP reason=device_1_cannot_access_device_0\n";
    return 0;
  }

  CUDA_CHECK(cudaSetDevice(0));
  cudaMemPoolProps properties{};
  properties.allocType = cudaMemAllocationTypePinned;
  properties.handleTypes = cudaMemHandleTypeNone;
  properties.location.type = cudaMemLocationTypeDevice;
  properties.location.id = 0;
  cudaMemPool_t pool = nullptr;
  CUDA_CHECK(cudaMemPoolCreate(&pool, &properties));

  cudaMemAccessDesc access{};
  access.location.type = cudaMemLocationTypeDevice;
  access.location.id = 1;
  access.flags = cudaMemAccessFlagsProtReadWrite;
  CUDA_CHECK(cudaMemPoolSetAccess(pool, &access, 1));

  cudaMemAccessFlags flags = cudaMemAccessFlagsProtNone;
  CUDA_CHECK(cudaMemPoolGetAccess(&flags, pool, &access.location));
  std::cout << "ACCESS owner_device=0 peer_device=1 flags="
            << static_cast<int>(flags) << '\n';
  CUDA_CHECK(cudaMemPoolDestroy(pool));

  const bool ok = flags == cudaMemAccessFlagsProtReadWrite;
  std::cout << "LAB_RESULT status=" << (ok ? "PASS" : "FAIL") << '\n';
  return ok ? EXIT_SUCCESS : EXIT_FAILURE;
}

