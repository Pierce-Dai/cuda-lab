#include <cuda.h>

#include <cstdlib>
#include <iostream>

#if CUDA_VERSION < 12040
#error "此程序需要 CUDA Toolkit 12.4 或更新版本"
#endif

#define CU_CHECK(call)                                                        \
    do {                                                                      \
        const CUresult error = (call);                                        \
        if (error != CUDA_SUCCESS) {                                          \
            const char* name = nullptr;                                       \
            const char* description = nullptr;                                \
            cuGetErrorName(error, &name);                                     \
            cuGetErrorString(error, &description);                            \
            std::cerr << (name ? name : "未知 CUDA 错误") << ": "           \
                      << (description ? description : "无错误说明") << '\n'; \
            return EXIT_FAILURE;                                              \
        }                                                                     \
    } while (0)

int main(int argc, char** argv) {
    const int ordinal = argc > 1 ? std::atoi(argv[1]) : 0;

    CU_CHECK(cuInit(0));

    int device_count = 0;
    CU_CHECK(cuDeviceGetCount(&device_count));
    if (ordinal < 0 || ordinal >= device_count) {
        std::cerr << "GPU 编号无效，可用范围为 0 到 " << device_count - 1 << '\n';
        return EXIT_FAILURE;
    }

    CUdevice device;
    CU_CHECK(cuDeviceGet(&device, ordinal));

    char device_name[256]{};
    CU_CHECK(cuDeviceGetName(device_name, sizeof(device_name), device));

    int fabric_supported = 0;
    CU_CHECK(cuDeviceGetAttribute(
        &fabric_supported,
        CU_DEVICE_ATTRIBUTE_HANDLE_TYPE_FABRIC_SUPPORTED,
        device));

    std::cout << "GPU " << ordinal << ": " << device_name << '\n'
              << "Fabric Memory Handle 支持: "
              << (fabric_supported ? "是" : "否") << '\n';
    return fabric_supported ? EXIT_SUCCESS : 2;
}