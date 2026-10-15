#include <cuda.h>
#include <cuda_runtime.h>

#include <cstdlib>
#include <iostream>
#include <vector>

/*
 * CUDA VMM（Virtual Memory Management）基础示例
 *
 * 目标：用 VMM 分配一块 GPU 显存，把它传给普通 CUDA kernel 使用，
 * 最后把结果复制回 CPU 校验。
 *
 * 本例只讲单进程、单 GPU 下的 VMM 核心，不涉及跨进程 handle、
 * Fabric Memory、IMEX Channel 或 Multicast。
 *
 * cudaMalloc() 把物理显存、虚拟地址、映射和访问权限封装在一次调用中。
 * VMM 则把它们拆成可以单独控制的步骤：
 *
 *   cuMemAddressReserve()  预留虚拟地址
 *   cuMemCreate()          分配物理显存，得到 allocation handle
 *   cuMemMap()             把物理显存映射到虚拟地址
 *   cuMemSetAccess()       授予指定 GPU 访问权限
 */

// 检查 CUDA Runtime API 调用。这个宏只在 main() 中使用。
#define CUDA_CHECK(call)                                                        \
    do {                                                                        \
        const cudaError_t error__ = (call);                                     \
        if (error__ != cudaSuccess) {                                           \
            std::cerr << "Runtime API 失败：" << __FILE__ << ':'               \
                      << __LINE__ << "，" << cudaGetErrorString(error__)        \
                      << '\n';                                                  \
            return EXIT_FAILURE;                                                \
        }                                                                       \
    } while (0)

// 检查 CUDA Driver API 调用。VMM 的核心接口属于 Driver API。
#define CU_CHECK(call)                                                          \
    do {                                                                        \
        const CUresult error__ = (call);                                        \
        if (error__ != CUDA_SUCCESS) {                                          \
            const char* name__ = nullptr;                                       \
            const char* message__ = nullptr;                                    \
            cuGetErrorName(error__, &name__);                                   \
            cuGetErrorString(error__, &message__);                              \
            std::cerr << "Driver API 失败：" << __FILE__ << ':'                \
                      << __LINE__ << "，"                                      \
                      << (name__ ? name__ : "未知错误") << " - "              \
                      << (message__ ? message__ : "无错误描述") << '\n';       \
            return EXIT_FAILURE;                                                \
        }                                                                       \
    } while (0)

// VMM 映射完成后，虚拟地址可以像 cudaMalloc 返回的指针一样传给 kernel。
__global__ void fill_sequence(int* data, std::size_t count) {
    const std::size_t index =
        static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (index < count) {
        data[index] = static_cast<int>(index * 2 + 1);
    }
}

int main() {
    constexpr int device_ordinal = 0;
    constexpr std::size_t element_count = 1ULL << 20;
    constexpr std::size_t requested_bytes = element_count * sizeof(int);

    /*
     * 本例混合使用两套 API：
     *   Driver API 负责 VMM；
     *   Runtime API 负责 <<< >>> kernel 和 cudaMemcpy。
     *
     * 先让 Runtime API 初始化 device 0 的 primary context，之后两套 API
     * 就在同一个当前 context 中工作。
     */
    CUDA_CHECK(cudaSetDevice(device_ordinal));
    CUDA_CHECK(cudaFree(nullptr));

    CUdevice device;
    CU_CHECK(cuDeviceGet(&device, device_ordinal));

    // 使用 VMM 前先查询当前 GPU 和驱动是否支持它。
    int vmm_supported = 0;
    CU_CHECK(cuDeviceGetAttribute(
        &vmm_supported,
        CU_DEVICE_ATTRIBUTE_VIRTUAL_MEMORY_MANAGEMENT_SUPPORTED,
        device));
    if (!vmm_supported) {
        std::cerr << "当前 GPU 或驱动不支持 CUDA VMM。\n";
        return EXIT_FAILURE;
    }

    /*
     * 描述物理内存：创建位于 device 0 上的固定设备内存。
     * requestedHandleTypes 保持为 0，因为本例不导出给其他进程。
     */
    CUmemAllocationProp properties{};
    properties.type = CU_MEM_ALLOCATION_TYPE_PINNED;
    properties.location.type = CU_MEM_LOCATION_TYPE_DEVICE;
    properties.location.id = device;

    /*
     * VMM 分配大小必须是设备分配粒度的整数倍。
     * 因此先查询最小粒度，再把请求大小向上对齐。
     */
    std::size_t granularity = 0;
    CU_CHECK(cuMemGetAllocationGranularity(
        &granularity, &properties, CU_MEM_ALLOC_GRANULARITY_MINIMUM));
    const std::size_t allocation_bytes =
        ((requested_bytes + granularity - 1) / granularity) * granularity;

    /*
     * 第一步：预留连续虚拟地址。
     *
     * 这一步只占虚拟地址空间，没有分配同等大小的物理显存，
     * 所以 virtual_address 暂时不能被 kernel 访问。
     * alignment=0、address=0 表示让 CUDA 选择默认对齐和地址。
     */
    CUdeviceptr virtual_address = 0;
    CU_CHECK(cuMemAddressReserve(
        &virtual_address, allocation_bytes, 0, 0, 0));

    /*
     * 第二步：分配物理显存。
     *
     * allocation_handle 表示物理内存资源，但它不是 GPU 指针，
     * 不能直接传给 kernel 或 cudaMemcpy()。
     */
    CUmemGenericAllocationHandle allocation_handle;
    CU_CHECK(cuMemCreate(
        &allocation_handle, allocation_bytes, &properties, 0));

    /*
     * 第三步：把物理显存映射到预留的虚拟地址。
     * offset 当前必须为 0。映射建立后还需要显式设置访问权限。
     */
    CU_CHECK(cuMemMap(
        virtual_address, allocation_bytes, 0, allocation_handle, 0));

    /*
     * 第四步：允许 device 0 读写这段地址。
     *
     * cuMemMap() 只负责建立映射，不会自动授予访问权限；遗漏这里可能
     * 导致 kernel 报 illegal memory access。多 GPU 程序可以传入多个
     * CUmemAccessDesc，只授权真正需要访问这块内存的 GPU。
     */
    CUmemAccessDesc access{};
    access.location.type = CU_MEM_LOCATION_TYPE_DEVICE;
    access.location.id = device;
    access.flags = CU_MEM_ACCESS_FLAGS_PROT_READWRITE;
    CU_CHECK(cuMemSetAccess(
        virtual_address, allocation_bytes, &access, 1));

    // 至此 virtual_address 才真正成为 device 0 可读写的 GPU 地址。
    int* device_data = reinterpret_cast<int*>(virtual_address);
    constexpr int threads_per_block = 256;
    const int block_count = static_cast<int>(
        (element_count + threads_per_block - 1) / threads_per_block);

    fill_sequence<<<block_count, threads_per_block>>>(
        device_data, element_count);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    // 映射后的 VMM 地址也能直接交给 Runtime API 的 cudaMemcpy()。
    std::vector<int> host_data(element_count);
    CUDA_CHECK(cudaMemcpy(
        host_data.data(), device_data, requested_bytes,
        cudaMemcpyDeviceToHost));

    bool correct = true;
    for (std::size_t index = 0; index < element_count; ++index) {
        const int expected = static_cast<int>(index * 2 + 1);
        if (host_data[index] != expected) {
            std::cerr << "校验失败：index=" << index
                      << "，期望值=" << expected
                      << "，实际值=" << host_data[index] << '\n';
            correct = false;
            break;
        }
    }

    /*
     * 三类资源分别释放：
     *   1. cuMemUnmap()       删除虚拟地址到物理显存的映射；
     *   2. cuMemRelease()     释放物理内存 handle；
     *   3. cuMemAddressFree() 释放预留的虚拟地址范围。
     */
    CU_CHECK(cuMemUnmap(virtual_address, allocation_bytes));
    CU_CHECK(cuMemRelease(allocation_handle));
    CU_CHECK(cuMemAddressFree(virtual_address, allocation_bytes));

    if (!correct) {
        return EXIT_FAILURE;
    }

    std::cout << "VMM 示例运行成功。\n"
              << "请求字节数：" << requested_bytes << '\n'
              << "实际映射字节数：" << allocation_bytes << '\n'
              << "设备最小分配粒度：" << granularity << '\n';
    return EXIT_SUCCESS;
}
