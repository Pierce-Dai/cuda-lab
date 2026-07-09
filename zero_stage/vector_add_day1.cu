#include <cuda_runtime.h>
#include <iostream>
#include <vector>
#include <cmath>
#include <chrono>

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

int main() {
    int N = 1 << 20;
    std::vector<float> h_A(N), h_B(N), h_C(N), h_D(N);
    float *d_A, *d_B, *d_C;

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
    std::cout << "CPU time elapsed: " << cpu_duration.count() << " ms" << std::endl;

    size_t bytes = N * sizeof(float);

    cudaEvent_t start, stop;
    CHECK_CUDA(cudaEventCreate(&start));
    CHECK_CUDA(cudaEventCreate(&stop));
    CHECK_CUDA(cudaEventRecord(start));

    CHECK_CUDA(cudaMalloc((void**)&d_A, bytes));
    CHECK_CUDA(cudaMalloc((void**)&d_B, bytes));
    CHECK_CUDA(cudaMalloc((void**)&d_C, bytes));

    CHECK_CUDA(cudaMemcpy(d_A, h_A.data(), bytes, cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_B, h_B.data(), bytes, cudaMemcpyHostToDevice));

    int blockSize = 256;
    int numBlocks = (N + blockSize - 1) / blockSize;

    vectorAdd<<<numBlocks, blockSize>>>(d_A, d_B, d_C, N);
    CHECK_CUDA(cudaGetLastError());
    CHECK_CUDA(cudaMemcpy(h_D.data(), d_C, bytes, cudaMemcpyDeviceToHost));
    CHECK_CUDA(cudaEventRecord(stop));
    CHECK_CUDA(cudaEventSynchronize(stop));
    float milliseconds = 0;
    CHECK_CUDA(cudaEventElapsedTime(&milliseconds, start, stop));
    std::cout << "GPU Time elapsed: " << milliseconds << " ms" << std::endl;

    // Verify the result
    for(int i = 0; i < N; i++) {
        if (fabs(h_D[i] - h_C[i]) > 1e-5) {
            std::cerr << "Error at index " << i << ": " << h_D[i] << " != " << h_C[i] << std::endl;
            return -1;
        }
    }

    // Free device memory
    CHECK_CUDA(cudaFree(d_A));
    CHECK_CUDA(cudaFree(d_B));
    CHECK_CUDA(cudaFree(d_C));
    CHECK_CUDA(cudaEventDestroy(start));
    CHECK_CUDA(cudaEventDestroy(stop));

    return 0;
}