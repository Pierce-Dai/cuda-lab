#include <cuda_runtime.h>
#include <iostream>
#include <vector>
#include <cmath>
#include <chrono>
#include <cstdio>
#include <iomanip>

#define CHECK_CUDA(call) do { \
    cudaError_t err = (call); \
    if (err != cudaSuccess) { \
        std::cerr << "CUDA error at " << __FILE__ << ":" << __LINE__ \
                  << ", call: " << #call \
                  << ", message: " << cudaGetErrorString(err) << std::endl; \
        exit(EXIT_FAILURE); \
    } \
} while(0)

__global__ void vectorAdd(const float* A, const float* B, float* C, int N) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < N) {
        C[i] = A[i] + B[i];
    }
}

void printCudaVersion(const char* name, int version) {
    std::cout << name << ": "
              << version / 1000 << "."
              << (version % 1000) / 10
              << " (" << version << ")"
              << std::endl;
}

void runVectorAdd(int N) {
    size_t bytes = N * sizeof(float);
    std::vector<float> h_A(N), h_B(N), h_C(N), h_D(N);
    float *d_A = nullptr;
    float *d_B = nullptr;
    float *d_C = nullptr;

    for(int i = 0; i < N; i++) {
        h_A[i] = i * 1.0f;
        h_B[i] = i * 2.0f;
    }

    auto cpu_start = std::chrono::high_resolution_clock::now();
    for(int i = 0; i < N; i++) {
        h_C[i] = h_A[i] + h_B[i];
    }
    auto cpu_end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double, std::milli> cpu_duration = cpu_end - cpu_start;

    CHECK_CUDA(cudaMalloc((void**)&d_A, bytes));
    CHECK_CUDA(cudaMalloc((void**)&d_B, bytes));
    CHECK_CUDA(cudaMalloc((void**)&d_C, bytes));

    cudaEvent_t start, stop;
    CHECK_CUDA(cudaEventCreate(&start));
    CHECK_CUDA(cudaEventCreate(&stop));

    float h2d_ms = 0.0f;
    CHECK_CUDA(cudaEventRecord(start));
    CHECK_CUDA(cudaMemcpy(d_A, h_A.data(), bytes, cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_B, h_B.data(), bytes, cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaEventRecord(stop));
    CHECK_CUDA(cudaEventSynchronize(stop));
    CHECK_CUDA(cudaEventElapsedTime(&h2d_ms, start, stop));

    int blockSize = 256;
    int numBlocks = (N + blockSize - 1) / blockSize;

    float kernel_ms = 0.0f;
    CHECK_CUDA(cudaEventRecord(start));
    vectorAdd<<<numBlocks, blockSize>>>(d_A, d_B, d_C, N);
    CHECK_CUDA(cudaGetLastError());
    CHECK_CUDA(cudaEventRecord(stop));
    CHECK_CUDA(cudaEventSynchronize(stop));
    CHECK_CUDA(cudaEventElapsedTime(&kernel_ms, start, stop));

    float d2h_ms = 0.0f;
    CHECK_CUDA(cudaEventRecord(start));
    CHECK_CUDA(cudaMemcpy(h_D.data(), d_C, bytes, cudaMemcpyDeviceToHost));
    CHECK_CUDA(cudaEventRecord(stop));
    CHECK_CUDA(cudaEventSynchronize(stop));
    CHECK_CUDA(cudaEventElapsedTime(&d2h_ms, start, stop));

    bool correct = true;
    for(int i = 0; i < N; i++) {
        if (std::fabs(h_D[i] - h_C[i]) > 1e-5) {
            std::cerr << "Error at index " << i << ": "
                      << h_D[i] << " != " << h_C[i] << std::endl;
            correct = false;
            break;
        }
    }

    float total_gpu_ms = h2d_ms + kernel_ms + d2h_ms;

    std::cout << std::setw(12) << N
              << std::setw(14) << bytes
              << std::setw(14) << cpu_duration.count()
              << std::setw(12) << h2d_ms
              << std::setw(14) << kernel_ms
              << std::setw(12) << d2h_ms
              << std::setw(14) << total_gpu_ms
              << std::setw(10) << (correct ? "PASS" : "FAIL")
              << std::endl;

    CHECK_CUDA(cudaFree(d_A));
    CHECK_CUDA(cudaFree(d_B));
    CHECK_CUDA(cudaFree(d_C));
    CHECK_CUDA(cudaEventDestroy(start));
    CHECK_CUDA(cudaEventDestroy(stop));
}

int main() {
    int devCount = -1;
    int devID = -1;
    CHECK_CUDA(cudaGetDeviceCount(&devCount));
    if (devCount == 0) {
        std::cout << "No CUDA device found." << std::endl;
        return EXIT_FAILURE;
    }
    std::cout << "device count: " << devCount << std::endl;
    CHECK_CUDA(cudaGetDevice(&devID));
    if (devID < 0) {
        std::cout << "get device id fail!" << std::endl;
        return EXIT_FAILURE;
    }
    std::cout << "current device: " << devID << std::endl;

    cudaDeviceProp prop;
    CHECK_CUDA(cudaGetDeviceProperties(&prop, devID));
    std::cout << "current device name: " << prop.name << std::endl;
    std::printf("device uuid: ");
    const size_t uuidLen = sizeof(prop.uuid.bytes) / sizeof(prop.uuid.bytes[0]);
    for(size_t i = 0; i < uuidLen; ++i) {
        std::printf("%02x ",static_cast<unsigned int>(
            static_cast<unsigned char>(prop.uuid.bytes[i])));
    }
    std::printf("\n");
    std::cout << "total global memory: " << prop.totalGlobalMem << std::endl;
    std::cout << "multiProcessor count: " << prop.multiProcessorCount << std::endl;
    std::cout << "local L1CacheSupported: " << prop.localL1CacheSupported << std::endl;
    std::cout << "global L1CacheSupported: " << prop.globalL1CacheSupported << std::endl;
    std::cout << "L2 Cache size: " << prop.l2CacheSize << std::endl;
    std::cout << "max threads per block: " << prop.maxThreadsPerBlock << std::endl;
    std::cout << "max threads per SM: " << prop.maxThreadsPerMultiProcessor << std::endl;
    std::cout << "Compute Capability: " << prop.major << "." << prop.minor << std::endl;

    int runtimeVersion = 0;
    int driverVersion  = 0;
    CHECK_CUDA(cudaRuntimeGetVersion(&runtimeVersion));
    CHECK_CUDA(cudaDriverGetVersion(&driverVersion));
    printCudaVersion("CUDA runtime version", runtimeVersion);
    printCudaVersion("CUDA driver version", driverVersion);

    int testSizes[] = {
        10000,
        100000,
        1000000,
        10000000
    };

    std::cout << std::endl;
    std::cout << std::setw(12) << "N"
              << std::setw(14) << "Bytes"
              << std::setw(14) << "CPU(ms)"
              << std::setw(12) << "H2D(ms)"
              << std::setw(14) << "Kernel(ms)"
              << std::setw(12) << "D2H(ms)"
              << std::setw(14) << "Total(ms)"
              << std::setw(10) << "Result"
              << std::endl;

    for (int N : testSizes) {
        runVectorAdd(N);
    }

    return 0;
}
