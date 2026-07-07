#include <cuda_runtime.h>
#include <iostream>
#include <vector>

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
    std::vector<float> h_A(N), h_B(N), h_C(N);
    std::vector<float> d_A(N), d_B(N), d_C(N);

    for(int i = 0; i < N; i++) {
        h_A[i] = i * 1.0f;
        h_B[i] = i * 2.0f;
        h_C[i] = h_A[i] + h_B[i];
    }
    size_t bytes = N *sizeof(float);
    for(int i = 0; i < N; i++) {
        if (h_C[i] != h_A[i] + h_B[i]) {
            std::cerr << "Error at index " << i << ": " << h_C[i] << " != " << h_A[i] + h_B[i] << std::endl;
            return -1;
        }
    }
}