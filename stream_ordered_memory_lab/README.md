# CUDA Stream-Ordered Memory Lab

这组 lab 只研究一个问题：**一块 GPU 内存的“可用时间、复用方式和可见范围”分别由谁决定？**

同一个 workload 始终是连续数组的初始化、加法和校验。实验只改变内存管理路径，因此输出可以直接用于技术分享，不会被复杂算法抢走注意力。

## 环境要求

- Linux 或 WSL2；
- CUDA Toolkit 11.2+（本项目在 CUDA 12.8 上验证）；
- 支持 stream-ordered memory allocator 的 GPU/驱动；
- Lab 03 还要求支持 `cudaMemHandleTypePosixFileDescriptor`；
- Lab 04 需要至少两张支持 P2P 的 CUDA GPU，本机只有一张卡时会正常输出 `SKIP`。

构建并运行：

```bash
cd /home/pierce_dai/workspace/cuda-lab/stream_ordered_memory_lab
chmod +x run_all.sh
./run_all.sh
```

也可以单独构建：

```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build --parallel
./build/lab01_lifetime
./build/lab02_pool_reuse --allocation-mib 64 --release-threshold-mib 0
./build/lab03_pool_ipc
./build/lab04_p2p_optional
```

## Lab 01：地址不是对象，stream 顺序才定义生命周期

展示两条正确路径：

1. 同一个 stream 中依次执行 `cudaMallocAsync → kernel → memcpy → cudaFreeAsync`；
2. 分配/释放在 owner stream，使用在 consumer stream，通过两个 event 建立依赖。

跨 stream 的核心不是“拿到了指针就能用”，而是 consumer 必须发生在 allocation 之后，free 必须发生在最后一次 use 之后。错误写法如下（项目不会执行它）：

```cpp
cudaMallocAsync(&p, bytes, stream_a);
kernel<<<grid, block, 0, stream_b>>>(p);  // 没有等待 allocation，未定义行为
cudaFreeAsync(p, stream_a);              // 也可能排在 stream_b 的使用之前
```

观察输出中的 `ALLOC`、`DEPENDENCY` 和 `RESULT`。两次虚拟地址可能相同，也可能不同；地址相同不是 API 保证。

## Lab 02：memory pool 把“free”拆成逻辑释放与物理保留

程序创建显式 pool，并记录：

- `used_current/high`：仍被应用持有的分配；
- `reserved_current/high`：pool 从系统保留的物理内存；
- `cudaMemGetInfo`：设备层面的剩余显存；
- 两次同尺寸分配的地址，以及是否观察到地址复用；
- `cudaMemPoolTrimTo(0)` 前后的变化。

默认开启三项复用策略。可以逐项关闭，讨论 CUDA 为了复用内存是否允许跟随显式 event、利用机会式同步点，或插入内部依赖：

```bash
./build/lab02_pool_reuse --follow-events 0 --opportunistic 0 --internal-dependencies 0
```

关键结论：`cudaFreeAsync` 结束的是应用层分配生命周期，但 pool 可以继续保留 backing memory；`trim` 才显式要求归还未使用的保留内存。实际归还量和地址复用都是运行时观察结果，不应写成绝对保证。

## Lab 03：跨进程共享的不是裸指针，而是一组能力与顺序

父进程在 `fork` 之后才初始化 CUDA，创建可导出的 pool，并完成：

1. `cudaMemPoolExportToShareableHandle` 导出 pool 的 POSIX FD；
2. `SCM_RIGHTS` 把 FD 传给子进程；
3. `cudaMemPoolExportPointer` 导出 allocation metadata；
4. 导出 interprocess event，表示初始化 kernel 已完成。

子进程完成：

1. `cudaMemPoolImportFromShareableHandle` 导入 pool；
2. `cudaMemPoolImportPointer` 得到本进程可用的 device pointer；
3. 等待 IPC event 后再读取/修改数据；
4. 先 `cudaFreeAsync` 释放 imported mapping，再通知父进程。

父进程收到结果后验证修改，最后才释放 original allocation。输出同时打印两个进程看到的地址；它们相同或不同都合法，程序不会依赖该现象。

这个 lab 最适合向下追问：FD 经 `sendmsg` 进入另一进程后，Runtime/Driver/KMD 分别要维护什么对象？VA 映射、页表、访问权限和 backing allocation 如何关联？IPC event 又为什么不能被“已经拿到 pointer”替代？

限制：CUDA 文档指出，使用 IPC 导入/导出的 memory pool 当前不能把物理内存块释放回 OS，因此这类 pool 的 release threshold/trim 行为与普通 pool 不同，不应用 Lab 02 的结论直接类推。

## Lab 04：P2P 能力门控

该程序不伪造双卡环境。它查询设备数和 peer access；满足条件时，用 `cudaMemPoolSetAccess` 给 device 1 授予对 device 0 pool 的读写权限，并用 `cudaMemPoolGetAccess` 回读验证。单卡机器输出 `SKIP` 是预期结果。

## 现场演示建议

建议依次只展示四组日志：

1. Lab 01 的两个 event 依赖；
2. Lab 02 中 `used=0` 但 `reserved>0`；
3. Lab 02 trim 后的 reserved 变化；
4. Lab 03 中“IPC event ready → child import/use/free → parent free”的顺序。

这些日志可以形成从 Runtime API 到 KMD 的讨论入口，但不能单凭日志断言公司产品的底层实现。对标时只记录：API 是否存在、输入边界、返回错误、可观察语义、性能曲线和系统能力要求，不记录或猜测敏感实现。

