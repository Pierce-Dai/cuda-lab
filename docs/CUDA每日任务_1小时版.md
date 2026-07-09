# CUDA 每日学习任务（每天 1 小时版）

> 周期：13 周 / 91 天。  
> 节奏：每天只追一个明确目标，完成一个可见产物。  
> 默认最终项目：先在第 73 天选择方向；如果没有特别偏好，建议做 `mini-CUDA-BLAS`，它最能串起 vector add、reduction、transpose、SGEMM、benchmark、profiling。

## 每天固定 1 小时结构

| 时间 | 做什么 |
|---:|---|
| 10 分钟 | 复盘昨天的代码、README、遗留问题 |
| 15 分钟 | 学当天唯一知识点，最多只看一小段文档或参考代码 |
| 25 分钟 | 在 VS Code 里写/改一个 `.cu` 文件 |
| 5 分钟 | 跑 correctness check / benchmark |
| 5 分钟 | 写学习日志：今天学了什么、卡在哪里、明天做什么 |

建议每天结束都更新当前阶段的 `README.md`，至少写三行：

```text
今天完成：
测试结果：
明天继续：
```

---

## 第 1 周：环境、编译、运行、计时

| 天数 | 今日目标 | 实操任务 | 验收标准 |
|---:|---|---|---|
| Day 1 | 搭好 CUDA 学习仓库 | 创建 `cuda-labs/` 和 8 个阶段目录；记录 `nvcc --version`、GPU 型号 | 能在 VS Code 里打开仓库，README 记录环境信息 |
| Day 2 | 跑通官方样例 | 跑 `deviceQuery`、`bandwidthTest`、`vectorAdd` | 三个样例能运行，并记录关键输出 |
| Day 3 | 自己写 vector add | 新建 `00_env_check/vector_add.cu`，实现 CPU baseline + GPU kernel | CPU/GPU 结果一致 |
| Day 4 | 加 CUDA 错误检查 | 写 `CUDA_CHECK` 宏；给 malloc/memcpy/kernel launch 加检查 | 故意制造错误时能打印错误信息 |
| Day 5 | 加 CUDA event 计时 | 分别统计 H2D、kernel、D2H、total GPU time | 输出完整计时表 |
| Day 6 | 做输入规模实验 | 测试不同 N，例如 `1e4/1e6/1e7/1e8` | 能解释小数据量 GPU 为什么不一定更快 |
| Day 7 | 阶段复盘 | 整理 `00_env_check/README.md` | 能回答：如何编译 `.cu`、什么是 kernel launch、CUDA event 统计什么 |

---

## 第 2 周：CUDA 编程模型

| 天数 | 今日目标 | 实操任务 | 验收标准 |
|---:|---|---|---|
| Day 8 | 掌握一维 indexing | 改写 vector add，打印/验证 `threadIdx/blockIdx/blockDim/gridDim` | 能写出 `idx = blockIdx.x * blockDim.x + threadIdx.x` |
| Day 9 | 写二维 matrix add | 新建 `01_cuda_programming_model/matrix_add.cu` | 正确处理 `row/col` 和边界 |
| Day 10 | 理解 block size | 对 matrix add 测 `8x8`、`16x16`、`32x8` | README 记录性能差异 |
| Day 11 | 写 grayscale kernel | 实现 RGB 转灰度，先可用数组模拟图片 | CPU/GPU 输出一致 |
| Day 12 | 写 invert kernel | 实现图像反色 | 能复用二维 indexing |
| Day 13 | 写 brightness kernel | 实现亮度调整，并处理 clamp 到 `[0,255]` | 不出现越界和溢出结果 |
| Day 14 | 阶段复盘 | 整理 `01_cuda_programming_model/README.md` | 能解释每个 thread 处理哪个像素，以及为什么要边界检查 |

---

## 第 3 周：Global Memory 与 Coalescing

| 天数 | 今日目标 | 实操任务 | 验收标准 |
|---:|---|---|---|
| Day 15 | 理解 global memory 访问 | 写一个连续读写 copy kernel 和 stride copy kernel | 能观察 stride 访问变慢 |
| Day 16 | 写 naive transpose | 新建 `02_memory_shared/transpose_naive.cu` | CPU/GPU 转置结果一致 |
| Day 17 | 给 transpose 加 benchmark | 统计时间并计算有效 bandwidth | README 有 `time(ms)` 和 `GB/s` |
| Day 18 | 阅读官方 transpose 思路 | 只看 CUDA Samples transpose 的 tile 思路，不抄代码 | 写下 coalescing 为什么重要 |
| Day 19 | 实现 tiled transpose | 新建 `transpose_tiled.cu`，使用 shared memory tile | 结果正确 |
| Day 20 | 对比 naive vs tiled | 测多个矩阵尺寸 | README 有对比表和 speedup |
| Day 21 | 周复盘 | 回答 coalesced access、memory-bound、bandwidth 计算 | 能说清 naive transpose 慢在哪里 |

---

## 第 4 周：Shared Memory、Blur、Stencil

| 天数 | 今日目标 | 实操任务 | 验收标准 |
|---:|---|---|---|
| Day 22 | 写 naive blur | 新建 `blur_naive.cu`，实现 3x3 blur | CPU/GPU 结果一致 |
| Day 23 | shared memory tile 入门 | 给 blur 加 shared memory 版本骨架 | 能正确加载 tile 和 halo |
| Day 24 | 完成 shared blur | 新建 `blur_shared.cu` | 输出正确，README 有 speedup |
| Day 25 | 理解 bank conflict | 给 transpose 尝试 padding，例如 `TILE_DIM+1` | 记录 padding 前后差异 |
| Day 26 | 写 naive stencil | 实现二维 5-point stencil / heat diffusion 一步更新 | 结果正确 |
| Day 27 | shared stencil | 尝试把 stencil 改成 shared memory tile | 能解释 shared 版本什么时候不一定更快 |
| Day 28 | 阶段复盘 | 整理 `02_memory_shared/README.md` | 能回答 shared memory、bank conflict、memory-bound |

---

## 第 5 周：Reduction 基础

| 天数 | 今日目标 | 实操任务 | 验收标准 |
|---:|---|---|---|
| Day 29 | 理解 reduce 问题 | 写 CPU sum/max/dot baseline | 有可复用 correctness check |
| Day 30 | 写 naive GPU sum | 新建 `03_parallel_patterns/reduce_naive.cu` | 结果正确，但能指出低效点 |
| Day 31 | 写 block-level reduction | 使用 shared memory 做 block 内求和 | 单 block / 多 block 局部结果正确 |
| Day 32 | 二阶段 reduction | 第一次 kernel 输出 partial sums，第二次合并 | 任意大 N 结果正确 |
| Day 33 | 扩展到 max/argmax | 实现 max 或 argmax 简化版 | CPU/GPU 一致 |
| Day 34 | 实现 dot product | 先 elementwise multiply，再 reduction | 输出正确并有计时 |
| Day 35 | 周复盘 | 整理 reduction 表格 | 能解释 block 内 reduce 后 block 之间怎么合并 |

---

## 第 6 周：Histogram、Scan、CUB 对比

| 天数 | 今日目标 | 实操任务 | 验收标准 |
|---:|---|---|---|
| Day 36 | 写 global atomic histogram | 新建 `histogram_global_atomic.cu` | 结果正确 |
| Day 37 | 测 atomic contention | 测均匀分布、随机分布、集中到一个 bin | 能解释 atomic 为什么会慢 |
| Day 38 | 写 shared private histogram | 新建 `histogram_shared_atomic.cu` | 正确合并 partial histograms |
| Day 39 | histogram 对比 | 对比 global atomic vs shared partial | README 有表格 |
| Day 40 | 写单 block scan | 实现 prefix sum 简化版 | 小数组结果正确 |
| Day 41 | 用 CUB/Thrust 对比 | 用 CUB DeviceReduce 或 Thrust reduce 做同类任务 | 能说出什么时候该用库 |
| Day 42 | 阶段复盘 | 整理 `03_parallel_patterns/README.md` | 能回答 reduction、histogram、scan、atomic、CUB |

---

## 第 7 周：SGEMM 起步

| 天数 | 今日目标 | 实操任务 | 验收标准 |
|---:|---|---|---|
| Day 43 | 建 GEMM benchmark 框架 | 新建 `04_sgemm/`，写矩阵初始化、CPU GEMM、误差检查 | 有可复用测试框架 |
| Day 44 | 写 naive GPU GEMM | 新建 `sgemm_v0_naive.cu` | 小矩阵和 CPU 一致 |
| Day 45 | 完善 naive benchmark | 测 `M=N=K=256/512/1024` | 输出 time 和 GFLOPS |
| Day 46 | 理解 coalescing 版本 | 调整线程映射，写 `sgemm_v1_coalesced.cu` | 结果正确 |
| Day 47 | 对比 v0/v1 | 测多种尺寸 | README 有 GFLOPS 表 |
| Day 48 | shared memory tiled GEMM | 写 `sgemm_v2_shared_memory.cu` 初版 | 结果正确 |
| Day 49 | 周复盘 | 写 SGEMM v0-v2 优化日志 | 能解释 naive GEMM 慢在哪里 |

---

## 第 8 周：SGEMM 深入优化

| 天数 | 今日目标 | 实操任务 | 验收标准 |
|---:|---|---|---|
| Day 50 | tile size 调参 | 对 `16x16`、`32x32` 等 tile 做 benchmark | 找到当前机器上较好的 tile |
| Day 51 | register blocking 入门 | 写 `sgemm_v3_register_blocking.cu` 简化版 | 结果正确 |
| Day 52 | vectorized load 入门 | 尝试 `float4` 或向量化加载思路 | 能记录是否真的变快 |
| Day 53 | 加 cuBLAS baseline | 写 `sgemm_cublas.cu` 或单独 benchmark | 有 cuBLAS 对比数据 |
| Day 54 | 大尺寸 benchmark | 测 `1024` 和 `4096` 尺寸 | README 有 `% cuBLAS` |
| Day 55 | 阅读 SGEMM_CUDA 前 3 版 | 对照自己的 v0-v2，不复制代码 | 写下 3 条可改进点 |
| Day 56 | 复现一个改进点 | 把一个读到的优化迁移到自己的实现 | 有 before/after 数据 |

---

## 第 9 周：SGEMM 收束

| 天数 | 今日目标 | 实操任务 | 验收标准 |
|---:|---|---|---|
| Day 57 | 看 register/occupancy | 编译时打开寄存器信息，例如 `ptxas -v` | README 记录 register 使用 |
| Day 58 | 处理 bank conflict/padding | 在 shared tiled GEMM 中尝试 padding | 有性能对比 |
| Day 59 | 理解 block tiling | 阅读 1D/2D block tiling 思路 | 写出自己的 tile 层级图或说明 |
| Day 60 | 理解 warp tiling | 只做概念学习，不强求实现 | 能解释为什么 warp tiling 更复杂 |
| Day 61 | 写完整优化日志 | 给 v0-v4 每版写：改了什么、预期、实际 | README 有 optimization log |
| Day 62 | 清理 SGEMM 代码 | 统一命名、参数、计时输出 | 所有版本可独立编译运行 |
| Day 63 | 阶段复盘 | 整理 `04_sgemm/README.md` | 能回答 shared tiling、register blocking、vectorized load、cuBLAS 差距 |

---

## 第 10 周：Profiler 与性能分析

| 天数 | 今日目标 | 实操任务 | 验收标准 |
|---:|---|---|---|
| Day 64 | 熟悉 profiler 命令 | 确认 Nsight Compute / Nsight Systems 可用 | 记录常用命令 |
| Day 65 | profile transpose | 分析 `transpose_naive` 或 `transpose_tiled` | 写 `05_profiling/reports/transpose.md` |
| Day 66 | profile reduction | 分析 `reduce_shared` | 写 `reduction.md` |
| Day 67 | profile SGEMM | 分析 `sgemm_v2_shared_memory` | 写 `sgemm.md` |
| Day 68 | 归纳瓶颈 | 对三个 kernel 判断 memory-bound / compute-bound | 每个判断都有数据依据 |
| Day 69 | 根据 profiler 改一个点 | 选择一个最小优化，例如 block size、padding、访问模式 | 有 before/after |
| Day 70 | 阶段复盘 | 整理 `05_profiling/README.md` | 不再只说“快/慢”，能说“为什么快/慢” |

---

## 第 11 周：Stream、异步、Pipeline

| 天数 | 今日目标 | 实操任务 | 验收标准 |
|---:|---|---|---|
| Day 71 | 理解 stream | 跑通或阅读 `simpleStreams` 思路 | 能解释 stream 解决什么 |
| Day 72 | single stream batch | 写 `pipeline_single_stream.cu`，批处理 vector add 或图像任务 | 结果正确 |
| Day 73 | pinned memory | 加 pageable vs pinned memory 对比 | README 有 H2D/D2H 时间 |
| Day 74 | multi stream | 写 `pipeline_multi_stream.cu` | 能运行多个 batch |
| Day 75 | 组合图像 pipeline | 串起 grayscale -> blur -> sobel 简化流程 | 输出正确 |
| Day 76 | Nsight Systems timeline | 观察 copy/kernel 是否 overlap | README 放 timeline 结论 |
| Day 77 | 阶段复盘 | 整理 `06_stream_pipeline/README.md` | 能回答 pinned memory、多 stream、overlap 判断 |

---

## 第 12 周：综合项目 MVP

| 天数 | 今日目标 | 实操任务 | 验收标准 |
|---:|---|---|---|
| Day 78 | 选择最终项目方向 | 从 mini-CUDA-BLAS、图像处理库、Heat Diffusion、mini-attention 中选一个 | README 写清项目目标和范围 |
| Day 79 | 设计项目结构 | 创建 `07_final_project/<project>/include/src/tests/benchmarks` | 目录清晰 |
| Day 80 | 定义 API 和测试入口 | 写最小 API、命令行参数、测试入口 | 能编译空项目 |
| Day 81 | 接入第一个 kernel | 迁移一个已完成 kernel，例如 vector add / grayscale / stencil | correctness check 通过 |
| Day 82 | 接入第二个 kernel | 迁移 reduction / blur / transpose 等 | correctness check 通过 |
| Day 83 | 接入 benchmark | 统一输出 `time(ms)`、speedup、GFLOPS 或 bandwidth | 有 benchmark 表 |
| Day 84 | MVP README | 写项目介绍、如何编译、如何运行、当前结果 | 别人能按 README 跑起来 |

---

## 第 13 周：综合项目完善与展示

| 天数 | 今日目标 | 实操任务 | 验收标准 |
|---:|---|---|---|
| Day 85 | 加 CPU baseline | 给项目核心功能补 CPU baseline | CPU/GPU 结果可比 |
| Day 86 | 加库对比 | 根据方向接入 CUB、Thrust、cuBLAS 或 PyTorch 对比 | 有第三方库 baseline |
| Day 87 | profile 核心 kernel | 用 Nsight 分析项目里最重要的 kernel | 写 profiling notes |
| Day 88 | 做一次优化 | 基于 profiler 做一个最小优化 | 有 before/after 数据 |
| Day 89 | 整理测试和 benchmark | 统一测试命令，补充多个输入规模 | README 有完整表格 |
| Day 90 | 准备展示材料 | 写项目亮点、限制、下一步 | README 像一个可展示项目 |
| Day 91 | 总复盘 | 回顾 13 周成果，列出下一轮强化方向 | 能清楚说明自己掌握了哪些 CUDA 能力 |

---

## 每日和我协作的固定提示词

每天开始时发：

```text
开始今天的 CUDA 学习。今天是 Day X，每天 1 小时。
请按 CUDA每日任务_1小时版.md 带我完成今天目标：
1. 先说明今天要学什么
2. 给我最小代码任务
3. 我写完后帮我 review / debug
4. 最后帮我更新 README 和学习日志
```

每天结束时发：

```text
收尾今天的学习。请总结：
1. 今天完成了什么
2. 代码是否达到验收标准
3. 我需要补的概念
4. 明天 Day X+1 应该做什么
```

