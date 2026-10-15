// 原仓库 https://github.com/CoffeeBeforeArch/cuda_programming
// 中的文件在baseline的基础做了优化，包括：
/*
 * 目录                  主要区别                          解决的问题
 * baseline              朴素矩阵乘法                      建立基准实现
 * noAliasing/tmp        使用局部变量累加                  减少对全局内存 C 的访问
 * noAliasing/restrict   给指针添加 __restrict__           告诉编译器三个矩阵不重叠
 * tiled                 使用共享内存分块                  复用矩阵数据，显著减少全局内存读取
 * rectangular           支持 M, N, K 不相等               处理矩形矩阵
 * nonMultiple           尺寸不是 32 的倍数                处理不完整的边缘 tile
 *
 * 本代码只做了baseline，其他优化后续再实现
 */

#include <iostream>
#include <vector>
#include <cassert>
#include <iomanip>

__global__ void matrixMultiply(int *a, int *b, int *c, int N) {
    int col = blockDim.x * blockIdx.x + threadIdx.x;
    int row = blockDim.y * blockIdx.y + threadIdx.y;

    
    if (col < N && row < N) {
        c [row * N + col] = 0;
        for (int i = 0; i < N; ++i) {
            c [row * N + col] += a[row * N + i] * b[i * N + col];
        }
    }
}

void verify(std::vector<int> &a, std::vector<int> &b, std::vector<int> &c, int N) {
    for (int i = 0; i < N; ++i) {
        for (int j = 0; j < N; ++j) {
            int tmp = 0;
            for (int k = 0; k < N; ++k) {
                tmp += a[i * N + k] * b[k * N + j];
            }
            assert(tmp == c[i * N + j]);
        }
    }
}

void printMatrix(const std::vector<int> &matrix, int N, const char *name) {
    std::cout << name << ":\n";
    for (int row = 0; row < N; ++row) {
        for (int col = 0; col < N; ++col) {
            std::cout << std::setw(6) << matrix[row * N + col];
        }
        std::cout << '\n';
    }
    std::cout << '\n';
}

int main() {
    int N = 10;

    size_t bytes = sizeof(int) * N * N;

    //主机内存数据
    std::vector<int> a;
    a.resize(N * N);
    std::vector<int> b;
    b.resize(N * N);
    std::vector<int> c;
    c.assign(N * N, 0);

    for (int i = 0; i < N; ++i) {
        int tmp = 0;
        for (int j = 0; j < N; ++j) {
            a[i * N + j] = tmp++;
            b[i * N + j] = tmp++; 
        }
    }

    printMatrix(a, N, "Matrix A before calculation");
    printMatrix(b, N, "Matrix B before calculation");

    int *d_a, *d_b, *d_c;
    cudaMalloc(&d_a, bytes);
    cudaMalloc(&d_b, bytes);
    cudaMalloc(&d_c, bytes);

    cudaMemcpy(d_a, a.data(), bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, b.data(), bytes, cudaMemcpyHostToDevice);
    cudaGetLastError();

    dim3 threadsPerBlock(16, 4);

    dim3 blocks(
        (N + threadsPerBlock.x - 1) / threadsPerBlock.x,
        (N + threadsPerBlock.y - 1) / threadsPerBlock.y
    );

    matrixMultiply<<<blocks, threadsPerBlock>>>(d_a, d_b, d_c, N);
    cudaDeviceSynchronize();

    // for (int i = 0; i < N; ++i) {
    //     for (int j = 0; j < N; ++j) {
    //         for (int k = 0; k < N; ++k) {
    //             c[i * k + j] += a[i * N + k] * b[k * N + j];
    //         }
    //     } 
    // }
    cudaMemcpy(c.data(), d_c, bytes, cudaMemcpyDeviceToHost);
    printMatrix(c, N, "Matrix C after calculation");
    verify(a, b, c, N);

    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_c);

    return 0;
}