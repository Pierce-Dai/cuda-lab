// This program computes the sum of two matrix of length N
// By: Nick from CoffeeBeforeArch

#include <algorithm>
#include <cassert>
#include <iostream>
#include <vector>
#include "../utils.h"

// CUDA kernel for vector addition
// __global__ means this is called from the CPU, and runs on the GPU
__global__ void vectorAdd(const int *__restrict a, const int *__restrict b,
                          int *__restrict c, int rows, int cols) {
    // Calculate global thread ID
    //按照二维矩阵的方式计算线程ID
    // Map CUDA x to matrix columns and CUDA y to matrix rows.
    // When rows are visualized from top to bottom, (0, 0) is the top-left.
    int col = blockIdx.x * blockDim.x + threadIdx.x; //意味着视原点在左上角
    int row = blockIdx.y * blockDim.y + threadIdx.y;

    // Boundary check
    if (col < cols && row < rows) {
        size_t index = static_cast<size_t>(row) * cols + col; //行主序
        c[index] = a[index] + b[index];
    }
}

// Check vector add result
void verify_result(std::vector<int> &a, std::vector<int> &b,
                    std::vector<int> &c) {
    for (size_t i = 0; i < a.size(); i++) {
        assert(c[i] == a[i] + b[i]);
    }
}

int main() {
    // Array size of 2^16 (65536 elements)
    constexpr size_t rows = size_t{1000};
    constexpr size_t cols = size_t{1500};
    constexpr size_t N    = rows * cols;
    constexpr size_t bytes = sizeof(int) * N;

    // Vectors for holding the host-side (CPU-side) data
    std::vector<int> a;
    a.reserve(N);
    std::vector<int> b;
    b.reserve(N);
    std::vector<int> c(N);

    // Initialize random numbers in each array
    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < cols; j++) {
            a.push_back(rand() % 100);
            b.push_back(rand() % 100);
        }
    }

    // Allocate memory on the device
    int *d_a, *d_b, *d_c;
    CHECK_CUDA(cudaMalloc(&d_a, bytes));
    CHECK_CUDA(cudaMalloc(&d_b, bytes));
    CHECK_CUDA(cudaMalloc(&d_c, bytes));

    // Copy data from the host to the device (CPU -> GPU)
    CHECK_CUDA(cudaMemcpy(d_a, a.data(), bytes, cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_b, b.data(), bytes, cudaMemcpyHostToDevice));

    // Threads per CTA (1024)
    // int NUM_THREADS = 1 << 10;
    // int NUM_THREADS = 128;
    // int NUM_THREADS = 256;
    // int NUM_THREADS = 512;
    dim3 threadPerBlock (32, 8);
    // dim3 threadPerBlock (16, 16);


    // CTAs per Grid
    // We need to launch at LEAST as many threads as we have elements
    // This equation pads an extra CTA to the grid if N cannot evenly be divided
    // by NUM_THREADS (e.g. N = 1025, NUM_THREADS = 1024)
    // int NUM_BLOCKS = (N + NUM_THREADS - 1) / NUM_THREADS;
    dim3 blocksPerGrid (
        (cols + threadPerBlock.x - 1) / threadPerBlock.x,
        (rows + threadPerBlock.y - 1) / threadPerBlock.y
    );


    // Launch the kernel on the GPU
    // Kernel calls are asynchronous (the CPU program continues execution after
    // call, but no necessarily before the kernel finishes)
    cudaEvent_t start,stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start);
    vectorAdd<<<blocksPerGrid, threadPerBlock>>>(d_a, d_b, d_c, rows, cols);
    cudaGetLastError();
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    float time = 0.0f;
    cudaEventElapsedTime(&time, start, stop);
    std::cout << "time: " << time << std::endl;

    // Copy sum vector from device to host
    // cudaMemcpy is a synchronous operation, and waits for the prior kernel
    // launch to complete (both go to the default stream in this case).
    // Therefore, this cudaMemcpy acts as both a memcpy and synchronization
    // barrier.
    cudaMemcpy(c.data(), d_c, bytes, cudaMemcpyDeviceToHost);

    // Check result for errors
    verify_result(a, b, c);

    // Free memory on device
    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_c);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    std::cout << "COMPLETED SUCCESSFULLY\n";

    return 0;
}
