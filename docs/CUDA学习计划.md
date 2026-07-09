# CUDA 系统学习计划

> 目标：从 CUDA 基础语法、并行编程模型、内存优化、经典并行模式、矩阵乘法优化、性能分析、异步流水线，到最终完成一个可展示的 CUDA 项目。

---

## 目录

- [一、学习总目标](#一学习总目标)
- [二、总体学习路线](#二总体学习路线)
- [三、推荐 GitHub 项目总览](#三推荐-github-项目总览)
- [四、阶段 0：环境、编译、运行、计时](#四阶段-0环境编译运行计时)
- [五、阶段 1：CUDA 编程模型](#五阶段-1cuda-编程模型)
- [六、阶段 2：内存模型、Coalescing、Shared Memory](#六阶段-2内存模型coalescingshared-memory)
- [七、阶段 3：并行模式：Reduction、Histogram、Scan](#七阶段-3并行模式reductionhistogramscan)
- [八、阶段 4：SGEMM 矩阵乘法深度优化](#八阶段-4sgemm-矩阵乘法深度优化)
- [九、阶段 5：Profiler 与性能分析](#九阶段-5profiler-与性能分析)
- [十、阶段 6：Stream、异步、Pipeline、工程化](#十阶段-6stream异步pipeline工程化)
- [十一、阶段 7：综合项目](#十一阶段-7综合项目)
- [十二、推荐执行顺序](#十二推荐执行顺序)
- [十三、个人仓库结构建议](#十三个人仓库结构建议)
- [十四、每个阶段的 README 模板](#十四每个阶段的-readme-模板)
- [十五、最终验收标准](#十五最终验收标准)

---

# 一、学习总目标

完成这套学习计划之后，你应该具备以下能力：

1. **能独立写 CUDA kernel**
   
   - 理解 `threadIdx`
   - 理解 `blockIdx`
   - 理解 `blockDim`
   - 理解 `gridDim`
   - 能正确设计一维、二维、三维 kernel indexing

2. **理解 CUDA 性能瓶颈**
   
   - global memory 访问
   - memory coalescing
   - shared memory
   - register
   - occupancy
   - memory-bound / compute-bound
   - warp divergence
   - atomic contention

3. **能实现 CUDA 经典并行模式**
   
   - map
   - reduce
   - scan
   - histogram
   - stencil
   - matrix transpose
   - convolution

4. **能从零优化 SGEMM**
   
   - naive GEMM
   - coalesced memory access
   - shared memory tiling
   - register blocking
   - vectorized load
   - 与 cuBLAS 对比

5. **会使用性能分析工具**
   
   - CUDA events
   - Nsight Compute
   - Nsight Systems

6. **知道什么时候该自己写 kernel，什么时候该用库**
   
   - Thrust
   - CUB
   - cuBLAS
   - CUTLASS

7. **完成一个可展示项目**
   
   - mini-CUDA-BLAS
   - CUDA 图像处理库
   - Heat Diffusion / Stencil Solver
   - mini-attention / mini-FlashAttention

---

# 二、总体学习路线

| 阶段   | 主题                 | 时间    | 最终可见成果                                     |
| ---- | ------------------ | -----:| ------------------------------------------ |
| 阶段 0 | 环境、编译、运行、计时        | 1–2 天 | 跑通 CUDA，能计时，能检查错误                          |
| 阶段 1 | CUDA 编程模型          | 1 周   | 会写一维、二维 kernel                             |
| 阶段 2 | 内存模型、shared memory | 2 周   | 能优化 transpose / stencil                    |
| 阶段 3 | 并行模式               | 2 周   | 会写 reduction、histogram、scan                |
| 阶段 4 | SGEMM              | 2–3 周 | 从 naive GEMM 优化到 tiled / register blocking |
| 阶段 5 | 性能分析               | 1 周   | 会用 profiler 定位瓶颈                           |
| 阶段 6 | stream 与工程化        | 1–2 周 | 会写异步 pipeline                              |
| 阶段 7 | 综合项目               | 2–4 周 | 完成一个可展示 CUDA 项目                            |

---

# 三、推荐 GitHub 项目总览

## 1. NVIDIA CUDA Samples

GitHub：

```text
https://github.com/NVIDIA/cuda-samples
```

用途：

- 环境检查
- 官方基础例子
- vector add
- deviceQuery
- bandwidthTest
- matrixMul
- transpose
- nbody
- stream
- async copy

适合阶段：

- 阶段 0
- 阶段 2
- 阶段 5
- 阶段 6

---

## 2. CoffeeBeforeArch CUDA Programming

GitHub：

```text
https://github.com/CoffeeBeforeArch/cuda_programming
```

用途：

- CUDA 入门
- vector addition
- matrix multiplication
- sum reduction
- histogram
- convolution

适合阶段：

- 阶段 1
- 阶段 2
- 阶段 3
- 阶段 7

---

## 3. eunomia-bpf/basic-cuda-tutorial

GitHub：

```text
https://github.com/eunomia-bpf/basic-cuda-tutorial
```

用途：

- CUDA 基础教程
- 有中文文档
- vector add
- GPU architecture
- programming methods
- convolution
- attention
- profiling

适合阶段：

- 阶段 1
- 阶段 2
- 阶段 5
- 阶段 7

---

## 4. GPU-Puzzles

GitHub：

```text
https://github.com/srush/GPU-Puzzles
```

用途：

- 训练 GPU 并行思维
- indexing
- map
- zip
- reduce
- scan
- matrix multiply

适合阶段：

- 阶段 1
- 阶段 3

---

## 5. NVIDIA CCCL

GitHub：

```text
https://github.com/NVIDIA/cccl
```

用途：

- Thrust
- CUB
- libcudacxx
- 高性能并行 primitives

适合阶段：

- 阶段 3
- 阶段 7

使用方式：

- 先自己写 reduction / scan / histogram
- 再用 CUB / Thrust 实现同样功能
- 最后做性能对比

---

## 6. siboehm/SGEMM_CUDA

GitHub：

```text
https://github.com/siboehm/SGEMM_CUDA
```

用途：

- 从零优化 CUDA SGEMM
- naive GEMM
- coalescing
- shared memory
- block tiling
- register tiling
- vectorized memory access
- warptiling

适合阶段：

- 阶段 4
- 阶段 5
- 阶段 7

---

## 7. NVIDIA CUTLASS

GitHub：

```text
https://github.com/NVIDIA/cutlass
```

用途：

- 工业级 GEMM
- tensor core
- tile hierarchy
- pipeline
- 高性能矩阵计算

适合阶段：

- 阶段 4 后期
- 阶段 7

注意：

> CUTLASS 不适合一开始就读。建议完成自己的 SGEMM 之后再看。

---

## 8. LeetCUDA

GitHub：

```text
https://github.com/xlite-dev/LeetCUDA
```

用途：

- 200+ CUDA kernels
- elementwise
- reduce
- layer norm
- RMS norm
- online softmax
- SGEMM
- HGEMM
- FlashAttention

适合阶段：

- 阶段 3 后期
- 阶段 4
- 阶段 7，尤其是 AI Infra 方向

---

# 四、阶段 0：环境、编译、运行、计时

## 时间

1–2 天。

## 阶段目标

你要做到：

1. 知道自己的 GPU 型号。
2. 知道自己的 CUDA 版本。
3. 知道自己的 GPU compute capability。
4. 能编译 `.cu` 文件。
5. 能运行 CUDA kernel。
6. 会写 CUDA error check。
7. 会用 CUDA event 计时。
8. 能区分：
   - H2D：Host to Device
   - Kernel time
   - D2H：Device to Host
   - Total GPU time

---

## 看得见的效果

你能运行：

```text
deviceQuery
bandwidthTest
vectorAdd
```

并能输出类似：

```text
GPU: RTX xxxx
CUDA Version: xx.x
Array size: 10000000

CPU time: xxx ms
H2D time: xxx ms
Kernel time: xxx ms
D2H time: xxx ms
Total GPU time: xxx ms

Result: PASS
```

---

## 推荐项目

### NVIDIA CUDA Samples

GitHub：

```text
https://github.com/NVIDIA/cuda-samples
```

重点看：

```text
Samples/1_Utilities/deviceQuery
Samples/1_Utilities/bandwidthTest
Samples/0_Introduction/vectorAdd
```

---

## 怎么利用项目

不要只跑官方代码，要这样做：

1. clone 项目。
2. 跑通 `deviceQuery`。
3. 跑通 `bandwidthTest`。
4. 跑通 `vectorAdd`。
5. 自己新建一个 `00_env_check/vector_add.cu`。
6. 不看官方代码，自己重写 vector add。
7. 加上：
   - CPU baseline
   - CUDA error check
   - CUDA event 计时
   - correctness check
8. 改变数组大小，观察性能变化。

---

## 自己的阶段项目

目录：

```text
cuda-labs/
  00_env_check/
    vector_add.cu
    README.md
```

---

## README 需要记录

```text
GPU 型号：
CUDA 版本：
N：
CPU time：
H2D time：
Kernel time：
D2H time：
Total GPU time：
结果是否正确：
```

---

## 阶段验收标准

完成后你应该能回答：

1. CUDA 程序如何编译？
2. host code 和 device code 有什么区别？
3. kernel launch 是什么？
4. 为什么小数据量 GPU 不一定比 CPU 快？
5. CUDA event 统计的是哪部分时间？

---

# 五、阶段 1：CUDA 编程模型

## 时间

1 周。

## 阶段目标

你要掌握：

1. `threadIdx`
2. `blockIdx`
3. `blockDim`
4. `gridDim`
5. 一维 indexing
6. 二维 indexing
7. kernel launch 配置
8. 边界检查
9. 每个 thread 负责一个元素的并行思想

---

## 看得见的效果

你能自己写出类似代码：

```cpp
dim3 block(16, 16);
dim3 grid((width + block.x - 1) / block.x,
          (height + block.y - 1) / block.y);
```

并能解释：

```text
每个 thread 处理图像中的哪个像素？
为什么需要边界检查？
block size 改变后性能为什么可能变化？
```

---

## 推荐项目 1：CoffeeBeforeArch CUDA Programming

GitHub：

```text
https://github.com/CoffeeBeforeArch/cuda_programming
```

重点看：

```text
01_vector_addition
02_matrix_mul
```

---

## 怎么利用 CoffeeBeforeArch

1. 先跑通项目原代码。
2. 自己重写 vector add。
3. 把一维 vector add 改成二维 matrix add。
4. 尝试不同 block size：
   - 128 threads/block
   - 256 threads/block
   - 512 threads/block
5. 记录性能差异。
6. 阅读 matrix multiplication 的 naive 写法，先不急着优化。

---

## 推荐项目 2：basic-cuda-tutorial

GitHub：

```text
https://github.com/eunomia-bpf/basic-cuda-tutorial
```

重点看：

```text
01-vector-addition.cu
01-vector-addition.zh.md
03-gpu-programming-methods.cu
03-gpu-programming-methods.zh.md
04-gpu-architecture.zh.md
```

---

## 怎么利用 basic-cuda-tutorial

1. 先读 `.zh.md` 中文文档。
2. 再跑 `.cu` 示例。
3. 复制核心思想到你自己的代码中，但不要照抄。
4. 每个例子后写一句总结：

```text
这个例子教会了我 CUDA 的哪个机制？
```

---

## 阶段项目：图像基础处理

自己实现：

```text
cuda-labs/
  01_cuda_programming_model/
    grayscale.cu
    invert.cu
    brightness.cu
    README.md
```

功能：

1. RGB 转灰度。
2. 图片反色。
3. 亮度调整。
4. CPU / GPU 结果对比。

---

## 阶段验收标准

你应该能做到：

1. 正确处理二维图像索引。
2. 正确处理边界。
3. 能解释为什么 `width * height` 个像素适合 GPU 并行。
4. 能比较不同 block size 的性能：
   - `8x8`
   - `16x16`
   - `32x8`
5. 能独立写出简单 CUDA kernel。

---

# 六、阶段 2：内存模型、Coalescing、Shared Memory

## 时间

2 周。

## 阶段目标

你要掌握：

1. global memory
2. memory coalescing
3. shared memory
4. shared memory tile
5. register 的基本概念
6. memory bandwidth
7. bank conflict 的基本概念
8. memory-bound kernel

---

## 看得见的效果

你能写出两个版本：

```text
慢版本：naive global memory
快版本：coalesced + shared memory
```

并能解释：

```text
为什么快？
快在哪里？
瓶颈从哪里转移到了哪里？
```

---

## 推荐项目 1：NVIDIA CUDA Samples

GitHub：

```text
https://github.com/NVIDIA/cuda-samples
```

重点看：

```text
Samples/6_Performance/transpose
Samples/0_Introduction/matrixMul
```

---

## 怎么利用 transpose sample

正确使用方式：

1. 不看官方代码，自己先写 naive transpose。
2. 测时间。
3. 计算 bandwidth。
4. 再看官方 transpose sample。
5. 学习 shared memory tile 的写法。
6. 自己重新实现 tiled transpose。
7. 对比：
   - naive transpose
   - coalesced transpose
   - shared memory transpose

---

## 推荐项目 2：CoffeeBeforeArch Convolution

GitHub：

```text
https://github.com/CoffeeBeforeArch/cuda_programming
```

重点看：

```text
05_convolution
```

---

## 阶段项目：Matrix Transpose + Blur

目录：

```text
cuda-labs/
  02_memory_shared/
    transpose_naive.cu
    transpose_tiled.cu
    blur_naive.cu
    blur_shared.cu
    README.md
```

---

## README 表格

```text
Transpose:

version              time(ms)    bandwidth(GB/s)
naive
tiled shared memory


Blur:

version              time(ms)    speedup
global memory
shared memory tile
```

---

## 阶段验收标准

你应该能回答：

1. 什么是 coalesced access？
2. 为什么矩阵转置 naive 版本会慢？
3. shared memory 为什么能提升局部访问？
4. 为什么 shared memory 版本也可能不快？
5. 什么情况下 kernel 是 memory-bound？
6. 如何计算有效内存带宽？

---

# 七、阶段 3：并行模式：Reduction、Histogram、Scan

## 时间

2 周。

## 阶段目标

掌握 CUDA 里最常见的并行模式：

1. map
2. reduce
3. scan
4. histogram
5. scatter
6. gather
7. atomic
8. warp-level thinking
9. block-level reduction

---

## 看得见的效果

你能实现：

```text
sum
max
argmax
dot product
histogram
prefix sum 简化版
```

并能解释：

```text
atomic 为什么会慢？
shared memory reduction 为什么更快？
一个 block 内 reduce 之后，block 之间如何合并？
什么时候应该用 CUB？
```

---

## 推荐项目 1：CoffeeBeforeArch Reduction / Histogram

GitHub：

```text
https://github.com/CoffeeBeforeArch/cuda_programming
```

重点看：

```text
03_sum_reduction
04_histogram
```

---

## 怎么利用 CoffeeBeforeArch

### 学 reduction

1. 先写 naive GPU sum。
2. 再看项目中的 reduction。
3. 自己实现 shared memory reduction。
4. 加上多 block 的二阶段 reduction。
5. 和 CPU 结果比较。
6. 记录不同 N 下的时间。

### 学 histogram

1. 先写 global atomic 版本。
2. 再写 shared memory private histogram。
3. 最后合并 partial histograms。
4. 测试不同数据分布：
   - 均匀分布
   - 全部集中到一个 bin
   - 随机分布

---

## 推荐项目 2：GPU-Puzzles

GitHub：

```text
https://github.com/srush/GPU-Puzzles
```

---

## 怎么利用 GPU-Puzzles

这个项目不适合当主线工程，但适合训练并行思维。

建议：

1. 学完 thread/block/grid 后开始做。
2. 每天做 1–2 题。
3. 不要急着看答案。
4. 每道题写一句总结：

```text
这一题训练的是：索引 / map / zip / reduce / scan / matmul
```

---

## 推荐项目 3：NVIDIA CCCL

GitHub：

```text
https://github.com/NVIDIA/cccl
```

用途：

- Thrust
- CUB
- 高性能并行 primitives

---

## 怎么利用 CCCL

正确顺序：

1. 先自己写 reduction。
2. 再用 CUB 做同样的事。
3. 对比性能。
4. 反推 CUB 为什么快。

例如：

```text
自己写 reduce_sum
CUB DeviceReduce::Sum
对比时间
分析差距
```

---

## 阶段项目：Reduction + Histogram Benchmark

目录：

```text
cuda-labs/
  03_parallel_patterns/
    reduce_naive.cu
    reduce_shared.cu
    reduce_warp.cu
    histogram_global_atomic.cu
    histogram_shared_atomic.cu
    README.md
```

---

## README 表格

```text
Reduction:

version          N          time(ms)      bandwidth(GB/s)
CPU
naive GPU
shared memory
warp optimized
CUB


Histogram:

version                  image size     time(ms)
global atomic
shared memory partial
```

---

## 阶段验收标准

你应该能做到：

1. 写出 block-level reduction。
2. 解释 atomic contention。
3. 理解 shared memory reduction 的基本模式。
4. 知道什么时候应该直接用 CUB。
5. 看懂别人写的 reduce / histogram kernel。

---

# 八、阶段 4：SGEMM 矩阵乘法深度优化

## 时间

2–3 周。

这是 CUDA 学习的核心阶段之一。

---

## 阶段目标

你要从零写矩阵乘法，并逐步优化：

1. naive GEMM
2. global memory coalescing
3. shared memory tiled GEMM
4. register blocking
5. vectorized load
6. block tiling
7. warp tiling
8. 与 cuBLAS 对比

---

## 看得见的效果

你最终能输出：

```text
version                     GFLOPS       relative to cuBLAS
naive
coalesced
shared memory tiled
register blocking
vectorized load
warp tiling
cuBLAS
```

---

## 推荐项目 1：siboehm/SGEMM_CUDA

GitHub：

```text
https://github.com/siboehm/SGEMM_CUDA
```

---

## 怎么利用 SGEMM_CUDA

这个项目非常好，但不要一开始就看答案。

正确使用方式：

### 第一步：自己先写 2 个版本

```text
sgemm_v0_naive.cu
sgemm_v1_shared_memory.cu
```

### 第二步：再看 siboehm 的前 3 个版本

重点理解：

```text
1. Naive
2. GMEM Coalescing
3. SMEM Caching
```

### 第三步：自己复现

不要复制代码，自己复现：

```text
sgemm_v2_coalesced.cu
sgemm_v3_tiled.cu
```

### 第四步：继续看后续优化

继续学习：

```text
1D Blocktiling
2D Blocktiling
Avoid Bank Conflicts
Vectorized Mem Access
Double Buffering
Warptiling
```

### 第五步：写优化日志

每优化一个版本，写：

```text
我改了什么：
为什么理论上会更快：
实际是否更快：
Nsight Compute 看到什么变化：
下一步瓶颈是什么：
```

---

## 推荐项目 2：NVIDIA CUTLASS

GitHub：

```text
https://github.com/NVIDIA/cutlass
```

---

## 怎么利用 CUTLASS

CUTLASS 不适合新手一开始读。你应该在完成自己的 SGEMM 后这样用：

1. 先读 Efficient GEMM 相关文档。
2. 看 CUTLASS 怎么分层：
   - thread
   - warp
   - threadblock
   - grid
3. 不要求完全看懂模板。
4. 重点理解：
   - tile 层级
   - memory hierarchy
   - pipeline
   - tensor core usage

---

## 阶段项目：My SGEMM Lab

目录：

```text
cuda-labs/
  04_sgemm/
    sgemm_v0_naive.cu
    sgemm_v1_coalesced.cu
    sgemm_v2_shared_memory.cu
    sgemm_v3_register_blocking.cu
    sgemm_v4_vectorized.cu
    sgemm_cublas.cu
    README.md
```

---

## README 表格

```text
M=N=K=1024:

version                  time(ms)      GFLOPS      % cuBLAS
naive
coalesced
shared memory
register blocking
vectorized
cuBLAS


M=N=K=4096:

version                  time(ms)      GFLOPS      % cuBLAS
naive
coalesced
shared memory
register blocking
vectorized
cuBLAS
```

---

## 阶段验收标准

你应该能回答：

1. naive GEMM 为什么慢？
2. shared memory tiling 为什么有效？
3. register blocking 是在解决什么问题？
4. vectorized load 为什么可能更快？
5. 为什么你的 GEMM 还是比 cuBLAS 慢？
6. CUTLASS 的 tile 层级大概是什么？
7. 什么是 arithmetic intensity？

---

# 九、阶段 5：Profiler 与性能分析

## 时间

1 周。

---

## 阶段目标

你要学会：

1. 用 CUDA event 做基本计时。
2. 用 Nsight Compute 看 kernel 指标。
3. 区分 memory-bound 和 compute-bound。
4. 看 occupancy。
5. 看 memory throughput。
6. 看 warp divergence。
7. 看 shared memory bank conflict。
8. 用数据解释性能问题。

---

## 看得见的效果

你能拿一个 kernel 写出类似分析：

```text
这个 kernel 当前瓶颈是 global memory load。
原因是 memory throughput 接近上限，但 compute utilization 不高。
下一步应该优化 memory access pattern。
```

---

## 推荐项目

继续使用：

```text
https://github.com/siboehm/SGEMM_CUDA
https://github.com/NVIDIA/cuda-samples
```

---

## 怎么利用项目

选三个 kernel 分析：

```text
transpose_naive
reduce_shared
sgemm_v2_shared_memory
```

每个 kernel 写一页 profiling notes。

---

## Profiling Notes 模板

```text
Kernel 名称：
输入规模：
运行时间：
occupancy：
global memory throughput：
shared memory usage：
主要瓶颈：
下一步优化：
```

---

## 阶段项目

目录：

```text
cuda-labs/
  05_profiling/
    reports/
      transpose_naive.md
      reduction.md
      sgemm.md
    README.md
```

---

## 阶段验收标准

你应该能做到：

1. 不再只说“这个版本快/慢”。
2. 能说清楚“为什么快/慢”。
3. 能用 profiler 数据支持判断。
4. 能根据 profiler 结果提出下一步优化方案。

---

# 十、阶段 6：Stream、异步、Pipeline、工程化

## 时间

1–2 周。

---

## 阶段目标

你要掌握：

1. CUDA stream
2. async memory copy
3. pinned memory
4. H2D / kernel / D2H overlap
5. 多 kernel pipeline
6. batch processing
7. Nsight Systems timeline 的基本阅读

---

## 看得见的效果

你能写出一个 pipeline：

```text
batch 0: H2D -> kernel -> D2H
batch 1:      H2D -> kernel -> D2H
batch 2:           H2D -> kernel -> D2H
```

并能观察到 copy 和 compute 是否 overlap。

---

## 推荐项目：NVIDIA CUDA Samples

GitHub：

```text
https://github.com/NVIDIA/cuda-samples
```

重点找：

```text
simpleStreams
async copy
concurrent kernels
stream
```

---

## 怎么利用项目

你应该：

1. 跑通官方 stream 示例。
2. 自己写一个批处理数组加法。
3. 再改成图像批处理。
4. 比较：
   - pageable memory
   - pinned memory
   - single stream
   - multi stream

---

## 阶段项目：Image Pipeline

目录：

```text
cuda-labs/
  06_stream_pipeline/
    pipeline_single_stream.cu
    pipeline_multi_stream.cu
    pipeline_pinned_memory.cu
    README.md
```

功能：

```text
读取 N 张图像
H2D
grayscale kernel
blur kernel
sobel kernel
D2H
保存结果
```

---

## README 表格

```text
version                  images/sec      total time(ms)
single stream
multi stream
pinned memory
multi stream + pinned
```

---

## 阶段验收标准

你应该能回答：

1. stream 解决什么问题？
2. pinned memory 为什么有用？
3. 什么情况下多 stream 没有提升？
4. 如何判断 copy 和 compute 是否 overlap？
5. Nsight Systems 和 Nsight Compute 分别看什么？

---

# 十一、阶段 7：综合项目

## 时间

2–4 周。

这个阶段你要选择一个方向，不要再写零散 demo。

---

# 方向 A：mini-CUDA-BLAS

## 适合方向

```text
HPC
CUDA 优化
AI Infra
底层算子
```

---

## 项目目标

实现一个小型 BLAS：

```text
vector add
dot product
reduction
matrix transpose
SGEMM
batched GEMM 简化版
```

---

## 可见成果

```text
CPU baseline
CUDA kernel
CUB/cuBLAS 对比
benchmark 表格
profiling 报告
```

---

## 推荐参考项目

```text
https://github.com/CoffeeBeforeArch/cuda_programming
https://github.com/siboehm/SGEMM_CUDA
https://github.com/NVIDIA/cccl
https://github.com/NVIDIA/cutlass
```

---

## 怎么利用项目

- CoffeeBeforeArch：参考基础 kernel 结构。
- SGEMM_CUDA：参考 GEMM 优化路线。
- CCCL：对比 CUB/Thrust 性能。
- CUTLASS：理解工业级 GEMM 设计。

---

## 项目目录建议

```text
cuda-labs/
  07_final_project/
    mini_cuda_blas/
      include/
      src/
      benchmarks/
      tests/
      README.md
```

---

# 方向 B：CUDA 图像处理库

## 适合方向

```text
计算机视觉
工程应用
图像处理
```

---

## 项目目标

实现：

```text
RGB to grayscale
Gaussian blur
Sobel edge detection
histogram
histogram equalization
resize
convolution
```

---

## 可见成果

输入：

```text
image.png
```

输出：

```text
gray.png
blur.png
edge.png
equalized.png
```

性能报告：

```text
CPU time
GPU time
speedup
```

---

## 推荐参考项目

```text
https://github.com/CoffeeBeforeArch/cuda_programming
https://github.com/eunomia-bpf/basic-cuda-tutorial
https://github.com/NVIDIA/cuda-samples
```

---

## 怎么利用项目

- CoffeeBeforeArch：重点看 convolution。
- basic-cuda-tutorial：看 CNN convolution 和 profiling/tracing。
- CUDA Samples：参考图像、纹理、性能示例。

---

## 项目目录建议

```text
cuda-labs/
  07_final_project/
    cuda_image_processing/
      include/
      src/
      images/
      output/
      benchmarks/
      README.md
```

---

# 方向 C：Heat Diffusion / Stencil Solver

## 适合方向

```text
科学计算
HPC
数值计算
```

---

## 项目目标

实现二维热扩散：

```cpp
new_grid[i][j] = 0.25f * (
    old_grid[i-1][j] +
    old_grid[i+1][j] +
    old_grid[i][j-1] +
    old_grid[i][j+1]
);
```

---

## 可见成果

1. 生成热力图。
2. 输出每秒迭代次数。
3. 对比：
   - naive global memory
   - shared memory tile
   - multi-step kernel
   - stream pipeline

---

## 推荐参考项目

```text
https://github.com/NVIDIA/cuda-samples
https://github.com/eunomia-bpf/basic-cuda-tutorial
```

---

## 项目目录建议

```text
cuda-labs/
  07_final_project/
    heat_diffusion_solver/
      src/
      visualization/
      benchmarks/
      README.md
```

---

# 方向 D：mini-attention / mini-FlashAttention

## 适合方向

```text
大模型推理
AI Infra
CUDA kernel
Tensor Core
```

---

## 前置条件

建议你至少完成：

```text
reduction
softmax
SGEMM
shared memory
profiling
```

再做 attention。

---

## 项目目标

按顺序做：

```text
naive attention: QK^T -> softmax -> V
tiled attention
online softmax
shared memory optimized attention
PyTorch correctness check
latency benchmark
```

---

## 推荐项目：LeetCUDA

GitHub：

```text
https://github.com/xlite-dev/LeetCUDA
```

---

## 怎么利用 LeetCUDA

不要从 FlashAttention 开始。按这个顺序：

1. Easy
   
   - elementwise
   - relu
   - gelu
   - histogram

2. Medium
   
   - mat_trans
   - reduce
   - layer norm
   - RMS norm
   - online softmax

3. Hard
   
   - sgemv
   - sgemm

4. Hard+
   
   - hgemm
   - flash-attention

---

## 项目目录建议

```text
cuda-labs/
  07_final_project/
    mini_attention/
      attention_cpu.py
      attention_naive.cu
      attention_tiled.cu
      online_softmax.cu
      benchmark.py
      README.md
```

---

## README 记录

```text
B：
H：
N：
D：
PyTorch result diff：
CUDA result diff：
latency：
memory usage：
```

---

# 十二、推荐执行顺序

如果你现在没有明确方向，建议按下面顺序走。

---

## 第 1 轮：基础能力

使用项目：

```text
https://github.com/NVIDIA/cuda-samples
https://github.com/CoffeeBeforeArch/cuda_programming
https://github.com/eunomia-bpf/basic-cuda-tutorial
```

完成：

```text
vector add
matrix add
image grayscale
convolution
```

---

## 第 2 轮：性能能力

使用项目：

```text
https://github.com/NVIDIA/cuda-samples
https://github.com/CoffeeBeforeArch/cuda_programming
https://github.com/NVIDIA/cccl
```

完成：

```text
transpose naive vs shared memory
reduction naive vs shared memory vs CUB
histogram global atomic vs shared memory
```

---

## 第 3 轮：核心算子能力

使用项目：

```text
https://github.com/siboehm/SGEMM_CUDA
https://github.com/NVIDIA/cutlass
```

完成：

```text
SGEMM naive
SGEMM tiled
SGEMM register blocking
SGEMM vs cuBLAS
```

---

## 第 4 轮：综合项目

根据方向选择：

| 方向       | 项目                    |
| -------- | --------------------- |
| AI Infra | mini-attention        |
| HPC      | mini-CUDA-BLAS        |
| 视觉应用     | CUDA 图像处理库            |
| 科学计算     | Heat Diffusion Solver |

---

# 十三、个人仓库结构建议

建议你从第一天就建自己的仓库，而不是只 clone 别人的项目。

```text
cuda-labs/
  00_env_check/
  01_cuda_programming_model/
  02_memory_shared/
  03_parallel_patterns/
  04_sgemm/
  05_profiling/
  06_stream_pipeline/
  07_final_project/
  README.md
```

---

## 推荐初始化命令

```bash
mkdir cuda-labs
cd cuda-labs

mkdir 00_env_check
mkdir 01_cuda_programming_model
mkdir 02_memory_shared
mkdir 03_parallel_patterns
mkdir 04_sgemm
mkdir 05_profiling
mkdir 06_stream_pipeline
mkdir 07_final_project

touch README.md
```

---

# 十四、每个阶段的 README 模板

每个阶段都建议写一个 README，模板如下：

```markdown
# 阶段名称

## Goal

这一阶段我要学什么？

## Kernels

我实现了哪些 kernel？

## Correctness

CPU/GPU 结果是否一致？

## Benchmark

| version | input size | time(ms) | bandwidth / GFLOPS | speedup |
|---|---:|---:|---:|---:|
| CPU | | | | |
| GPU naive | | | | |
| GPU optimized | | | | |

## Profiling

使用的工具：

- CUDA events
- Nsight Compute
- Nsight Systems

观察到的瓶颈：

```text
填写你的分析
```

## Optimization Log

| version | change | expected improvement | actual result |
| ------- | ------ | -------------------- | ------------- |
| v0      |        |                      |               |
| v1      |        |                      |               |

## What I Learned

这一阶段我真正理解了什么？

## Next Step

下一步准备优化什么？

```
---

# 十五、最终验收标准

当你走完整条路线之后，你应该能做到以下事情。

---

## 基础能力

你应该能：

1. 编译 `.cu` 文件。
2. 写 CUDA error check。
3. 用 CUDA event 计时。
4. 写一维 kernel。
5. 写二维 kernel。
6. 正确处理边界。
7. 解释 block / grid / thread 的关系。

---

## 性能能力

你应该能解释：

1. 什么是 memory coalescing？
2. 什么是 shared memory？
3. 什么是 bank conflict？
4. 什么是 occupancy？
5. 什么是 memory-bound？
6. 什么是 compute-bound？
7. atomic 为什么会慢？
8. shared memory 为什么能提升性能？

---

## 算子能力

你应该能实现：

1. vector add
2. matrix add
3. image grayscale
4. matrix transpose
5. convolution / blur
6. reduction
7. histogram
8. scan 简化版
9. SGEMM
10. stream pipeline

---

## 工具能力

你应该能使用：

1. CUDA events
2. Nsight Compute
3. Nsight Systems
4. CUB
5. Thrust
6. cuBLAS

---

## 项目能力

你最终应该至少完成一个可展示项目：

```text
mini-CUDA-BLAS
CUDA 图像处理库
Heat Diffusion Solver
mini-attention
```

一个合格的最终项目应该包含：

```text
代码
README
CPU baseline
CUDA 实现
correctness check
benchmark
profiling notes
optimization log
```

---

# 一句话总结

学习 CUDA 不要靠“看完文档”，而要靠：

```text
跑通官方例子
自己重写 kernel
加入 CPU baseline
做 correctness check
做 benchmark
用 profiler 找瓶颈
逐步优化
写总结
最后做完整项目
```

推荐主线：

```text
NVIDIA/cuda-samples
  -> CoffeeBeforeArch/cuda_programming
  -> basic-cuda-tutorial
  -> GPU-Puzzles
  -> NVIDIA/cccl
  -> siboehm/SGEMM_CUDA
  -> NVIDIA/cutlass
  -> LeetCUDA
```

第一天只做一件事：

```text
自己写出带 CPU baseline、CUDA event 计时、error check 的 vector add。
```