#include <cuda.h>
#include <iostream>

int main() {
    CUresult result = cuInit(0);
    // 检查 result

    CUdevice device;
    result = cuDeviceGet(&device, 0);
    // 检查 result

    int deviceSupportsVmm = 0;
    result = cuDeviceGetAttribute(
        &deviceSupportsVmm,
        CU_DEVICE_ATTRIBUTE_VIRTUAL_MEMORY_MANAGEMENT_SUPPORTED,
        device
    );
    // 检查 result

    std::cout << "VMM supported: " << deviceSupportsVmm << '\n';
    return 0;
}