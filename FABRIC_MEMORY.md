# CUDA Fabric Memory：原理、边界与完整示例

> 本文讨论 CUDA VMM（Virtual Memory Management）中的 **Fabric Handle 共享**，不是一种新的 `cudaMalloc` 分配器，也不是 NVLink SHARP Multicast。文中 API 以 CUDA 12.4+ 为前提；示例在 Linux 上编写，需有受支持的 GPU、NVLink fabric 和 IMEX 环境才能实际运行。最后核对日期：2026-09-17。

## 1. 先用一句话理解

Fabric Memory 是**能够通过受支持的 NVLink/NVSwitch fabric 被其他 GPU 映射访问的物理内存**。在 CUDA VMM 中，创建内存时请求 `CU_MEM_HANDLE_TYPE_FABRIC`，便可将该分配导出为 `CUmemFabricHandle`，交给另一进程——在支持的系统上甚至可以是另一节点的进程——导入并映射。映射成功并设置权限后，GPU kernel 对远端内存执行普通 load/store。[CUDA VMM 官方指南](https://docs.nvidia.com/cuda/cuda-programming-guide/04-special-topics/virtual-memory-management.html)

这不意味着 GPU 上另外出现了一个叫“Fabric Memory”的独立内存池。示例中物理分配仍指定某块 GPU 的设备内存；“Fabric”强调它的**跨 fabric 可达性和可跨 OS 实例传递的共享句柄**。

```text
节点 A / 进程 A                                  节点 B / 进程 B
GPU A 物理显存                                   GPU B kernel
      │                                              │ 普通 load/store
      ├─ cuMemCreate(requestedHandleTypes=FABRIC)   ▼
      ├─ cuMemExportToShareableHandle()       进程 B 的虚拟地址
      │             │                                ▲
      │             └── 控制面传送 Fabric Handle ────┤
      │                  （例如 TCP、MPI）          cuMemImportFromShareableHandle()
      │                                        + cuMemAddressReserve()
      │                                        + cuMemMap()
      │                                        + cuMemSetAccess()
      └──────────── GPU 数据访问经 NVLink fabric ─────┘

            双方访问同一 IMEX Channel：控制谁有资格导入句柄
```

这里有两条完全不同的路径：

- **控制面**：应用把 Fabric Handle 从导出进程送到导入进程。TCP/MPI 承载的是句柄、大小和同步消息，不是这个示例中的数组数据。
- **数据面**：导入方 GPU 通过自己的虚拟地址直接访问被映射的物理内存。路径和速度取决于系统 NVLink/NVSwitch 拓扑；IMEX Channel 不负责搬运这些字节。

## 2. 各组件分别干什么

| 组件 | 职责 | 容易混淆的点 |
| --- | --- | --- |
| `CUmemGenericAllocationHandle` | 当前进程内代表 VMM 物理分配的句柄 | 不是 GPU 指针，不能传给 kernel，也不能直接跨进程复制数值 |
| `CUmemFabricHandle` | 从可共享的分配导出的、可交给另一进程的 Fabric 句柄 | 只是共享凭据，不包含整块显存的数据 |
| IMEX daemon / Channel | 为 Fabric Handle 的导出、导入提供系统服务和按用户/通道隔离 | 不是 NVLink 数据传输引擎，也不替代应用的句柄传送机制 |
| `CUdeviceptr` | **本进程**预留并完成映射、授权后的虚拟地址 | 原进程的数值地址不能直接拿到另一进程解引用；双方要分别预留/映射 |
| NVLink/NVSwitch fabric | 让相关 GPU 的物理内存可被远端 GPU 访问 | 不是只要有 Fabric Handle 就自动拥有跨节点 NVLink 拓扑 |

普通 VMM 可以完全不涉及 Fabric：单 GPU 内预留/映射内存时不需要 IMEX；同一 Linux 节点内跨进程共享还可用 POSIX FD 句柄。Fabric Handle 的特殊价值在于：**同一套 VMM 映射 API 可以延伸到支持的跨节点 NVLink 系统**。[CUDA VMM 官方指南：句柄类型与共享流程](https://docs.nvidia.com/cuda/cuda-programming-guide/04-special-topics/virtual-memory-management.html)

## 3. 它解决什么问题，不能解决什么问题

### 能做的

1. **跨进程共享同一物理分配**。导出/导入句柄后，各进程分别预留地址、映射并设置本地 GPU 权限；不是先把整块数据复制到对方显存。
2. **受支持的跨节点直接寻址**。若 GPU 属于同一个可用 NVLink fabric、系统软件和 IMEX 已配置，导入方可以把远端物理显存映射进自己的虚拟地址空间。
3. **按分配/映射范围控制可访问性**。VMM 让应用选择共享哪块内存，而不必让一张 GPU 自动映射 peer 的全部既有与未来 `cudaMalloc` 分配。
4. **保留 VMM 的虚拟地址能力**。可以先预留大范围虚拟地址，再按粒度映射物理内存；同一套 `cuMemMap`、`cuMemSetAccess`、`cuMemUnmap` 生命周期仍适用。

### 不会自动带来的

- **不会让远端显存等价于本地 HBM**。远端访问仍受互连带宽、延迟、跳数、争用和访问模式约束；单个 kernel 不会仅因换成 Fabric Handle 就自动提速。
- **不会在不具备相关硬件拓扑的两台普通服务器之间创造 GPU 直连**。TCP 能传句柄，不代表目标 GPU 能映射源 GPU 的物理内存。
- **不会自动完成跨 GPU 的生产者/消费者同步**。必须先让写入完成并满足所需的内存可见性与顺序要求，再通知读取方；不能把控制面消息当成可替代 CUDA 内存模型的万能栅栏。
- **不会自动广播数据给多张 GPU**。普通 Fabric VMM Unicast 映射与 Multicast 对象/NVLink SHARP 是不同功能。
- **不会替你管理安全的控制面**。IMEX 限制导入资格，但应用仍要安全传送句柄并处理身份、端口和生命周期。

## 4. 使用前提与限制

1. **CUDA Toolkit 12.4 或更新版本**提供 `CU_MEM_HANDLE_TYPE_FABRIC`；还要有相应驱动及支持的平台。VMM 还要求 UVA。编译通过只说明头文件/链接器接受这些 API，**不说明运行平台支持 Fabric**。[CUDA VMM 官方指南](https://docs.nvidia.com/cuda/cuda-programming-guide/04-special-topics/virtual-memory-management.html)
2. 运行时分别查询 `CU_DEVICE_ATTRIBUTE_VIRTUAL_MEMORY_MANAGEMENT_SUPPORTED` 与 `CU_DEVICE_ATTRIBUTE_HANDLE_TYPE_FABRIC_SUPPORTED`。属性为 0 就不要继续尝试 Fabric 分配。两端都应检查。[CUDA VMM 官方指南：能力查询](https://docs.nvidia.com/cuda/cuda-programming-guide/04-special-topics/virtual-memory-management.html)
3. 跨节点直接访问要求相关 GPU 位于**受支持、可互达的 NVLink fabric**；普通 PCIe P2P 或仅有网络连接不等同于这个条件。具体可达性和拓扑由平台管理员验证。
4. Fabric Handle 共享需要 NVIDIA IMEX 相关系统服务与 Channel 配置。驱动应暴露 `nvidia-caps-imex-channels` 设备，导出和导入进程所属用户须有**同一 IMEX Channel** 的访问权限；多用户隔离宜为不同用户配置不同 Channel。管理员配置文件位置及权限，不要在应用里随意 `mknod` 或放宽 `/dev` 权限。[CUDA VMM 官方指南：IMEX](https://docs.nvidia.com/cuda/cuda-programming-guide/04-special-topics/virtual-memory-management.html)、[Driver API：导入权限](https://docs.nvidia.com/cuda/cuda-driver-api/group__CUDA__VA.html)
5. 在 `cuMemCreate` 前设置 `CUmemAllocationProp::requestedHandleTypes = CU_MEM_HANDLE_TYPE_FABRIC`；大小按 `cuMemGetAllocationGranularity` 向上取整。导入方需要本进程自己的地址预留与映射，且 `cuMemMap` 后还要调用 `cuMemSetAccess`。[Driver API：VMM](https://docs.nvidia.com/cuda/cuda-driver-api/group__CUDA__VA.html)
6. CUDA pointer 是进程局部的。跨进程只传**可共享的 handle**，不能把 `CUdeviceptr`、`cudaMalloc` 指针或 `CUmemGenericAllocationHandle` 的整数值发过去当作可用地址。[CUDA IPC 官方指南](https://docs.nvidia.com/cuda/cuda-programming-guide/04-special-topics/inter-process-communication.html)
7. 此处的远端映射是**寻址模型**。若应用需要逐操作的完成状态/故障反馈，可研究 Compute Fabric Transport 的 endpoint 模型；普通 Fabric VMM load/store 不提供同样的逐操作完成报告。[CUDA Compute Fabric Transport](https://docs.nvidia.com/cuda/cuda-programming-guide/04-special-topics/compute-fabric-transport.html)

典型 Linux 前置检查（只读；Channel 创建与授权由管理员负责）：

```bash
nvcc --version
nvidia-smi -L
grep nvidia-caps-imex-channels /proc/devices
ls -l /dev/nvidia-caps-imex-channels/
```

这些命令只确认部分软件配置。是否拥有合适 NVLink fabric、两端是否同 Channel，仍须结合平台文档与运行时能力检查；不能只凭 `nvidia-smi -L` 判断。常见消费级单卡或普通 WSL 环境可以**编译**示例，但通常无法完成 Fabric/IMEX 运行验证。

## 5. API 使用顺序

```text
导出端                                导入端
查询 VMM/Fabric 能力                  查询 VMM/Fabric 能力
设置 CUmemAllocationProp             等待控制面收到大小、Fabric Handle
查询 granularity，向上对齐           cuMemImportFromShareableHandle()
cuMemCreate()                         cuMemAddressReserve()
cuMemAddressReserve()                 cuMemMap()
cuMemMap()                            cuMemSetAccess()
cuMemSetAccess()                      kernel 访问本地映射地址
kernel 写入并完成                      kernel 完成后回执
cuMemExportToShareableHandle()        cuMemUnmap()
发送大小、Fabric Handle               cuMemRelease()
等待导入端回执                        cuMemAddressFree()
cuMemUnmap() / cuMemRelease()
cuMemAddressFree()
```

`cuMemMap` 只建立“虚拟地址 → 物理分配”的关联；**没有 `cuMemSetAccess`，kernel 访问仍会失败**。导出方也必须在导入方结束访问后才释放仍被使用的资源。[CUDA VMM 官方指南：映射、权限与释放](https://docs.nvidia.com/cuda/cuda-programming-guide/04-special-topics/virtual-memory-management.html)

## 6. 完整可编译示例：两个进程、两个 GPU

完整源码：[VMM/fabric_memory_ipc.cu](VMM/fabric_memory_ipc.cu)。代码包含：能力查询、对齐分配、Fabric Handle 导出/导入、TCP 完整发送/接收、两侧独立映射与权限设置、GPU 写入与远端 GPU 校验、完成回执，以及异常路径的资源清理。**它不依赖 MPI**，但仍要求上节所述的平台条件。

### 6.1 编译

在每个参与节点的 Linux/WSL CUDA 环境中，使用兼容的 Toolkit 编译同一源码。示例限定两端均为 Linux x86-64、相同 CUDA ABI；因此原样传输 `CUmemFabricHandle` 和一个 `uint64_t` 大小。生产系统若混用 ABI/架构，应另行定义并验证控制面协议。

```bash
cd /home/pierce_dai/workspace/cuda-lab
nvcc -std=c++17 -O2 VMM/fabric_memory_ipc.cu -lcuda -o fabric_memory_ipc
```

### 6.2 运行

必须先由管理员配置 NVLink fabric、IMEX daemon 与双方可访问的同一 Channel。服务器进程运行于源 GPU 所在节点；客户端运行于目标 GPU 所在节点。以下地址与 GPU 编号仅为示意：

```bash
# 节点 A：在本节点 CUDA device 0 上创建物理内存，监听 TCP 50051。
./fabric_memory_ipc server 50051 0

# 节点 B：本节点 CUDA device 0 导入并访问节点 A 的内存。
./fabric_memory_ipc client <节点A的可达IP> 50051 0
```

同一节点做两 GPU 测试时，服务器可用 device 0，客户端改用 device 1，并把地址设为 `127.0.0.1`。**单节点测试能验证部分流程，但不能据此宣称跨节点 Fabric 已验证。**

成功时导入端会打印“GPU 直接读取远端映射并校验成功”，导出端会收到确认消息后再释放资源。控制面 TCP 连接仅用于演示，不带 TLS/认证；只应在可信实验网络中使用。真实系统应采用受保护的控制面，并设置超时、故障清理和监控。

### 6.3 这个 kernel 实际做了什么

- 导出方 `fill_kernel`：每个 thread 写一个 `data[i] = i + 7`，使用 256 threads/block，有边界检查；连续写入是 coalesced 的，演示里为跨 GPU 可见性加入 `__threadfence_system()`。它是教学用法，不是高性能写入实现。
- 导入方 `verify_kernel`：每个 thread 从**远端映射地址**读一个元素并校验；若错误，则原子设置保存在**本地显存**的标志。读取连续且可合并，但瓶颈可能是远端延迟/互连带宽；此示例不需要 shared memory。
- 导出端先等待写 kernel 完成再传句柄；导入端等读 kernel 完成再回执。这个顺序是资源生命周期与正确性的一部分。生产并发读写还需更细的跨 GPU 同步设计。

## 7. 与相似功能的区别

| 方式 | 解决的主要问题 | 共享范围 / 关键区别 | 何时优先考虑 |
| --- | --- | --- | --- |
| `cudaMalloc` + `cudaDeviceEnablePeerAccess` | 一进程内让一张 GPU 访问另一张 GPU 的分配 | 按 peer 设备开启；对 peer 的既有与后续分配整体生效，可能增加分配/映射开销 | 同机少量 GPU，全部或大部分分配都需要互访，追求简单 |
| 普通 VMM Unicast | 精确选择物理分配、虚拟地址和可访问 GPU | 单进程也可用；并非必须 Fabric/IMEX | 动态映射、只共享部分内存、自己实现分配器 |
| CUDA 传统 IPC | 不同进程共享 GPU 分配 | 本机进程间交换 IPC 句柄；Linux 支持；不是跨节点 Fabric 方案 | 单机进程隔离且现有应用使用传统 CUDA IPC |
| VMM + POSIX FD handle | 不同 Linux 进程共享 VMM 分配 | 同一 OS/节点；传 FD 要用合适的 OS IPC（例如 Unix socket 的 `SCM_RIGHTS`），不能直接发送 FD 整数值 | 单机跨进程、需要 VMM 的细粒度控制，但无需 Fabric |
| **VMM + Fabric Handle + IMEX** | 在支持的平台上跨进程，包含跨节点共享同一物理分配 | 句柄跨 OS 实例；IMEX 授权；导入方仍需映射和设置访问权 | 受支持的 NVLink fabric 集群，需远端 GPU 直接寻址 |
| Unified Memory（`cudaMallocManaged`） | 统一指针与自动迁移/放置策略 | 重点是托管内存模型，不等同于导出 Fabric Handle | 希望简化数据放置而非自行控制远端映射 |
| NCCL / NVSHMEM | 高层多 GPU 通信与同步 | 提供 collective、put/get 等通信抽象，可能内部使用 VMM | 常规分布式训练、科学计算、通信模式已有库支持 |
| Multicast / NVLink SHARP | 多 GPU 副本上的广播、归约等协作 | 多个物理副本与 `multimem` 指令；不同于单份内存的普通 Unicast 映射 | 特定 NVSwitch 硬件及低层通信库开发 |
| Compute Fabric Transport | 针对 endpoint 发起异步 fabric 操作 | 资源/endpoint 模型，有逐操作完成状态；区别于映射地址后普通 load/store | 需要更明确的 fabric 错误反馈或实现底层通信库 |

P2P 是“GPU 之间直接访问/传输”的能力范畴，并不是 Fabric VMM 的对立面。Fabric VMM 可以实现某些 P2P **直接寻址**场景；差异主要在配置粒度、进程/节点边界和平台支持。`cudaDeviceEnablePeerAccess` 的“全部既有及后续分配”行为见 [CUDA 多 GPU 编程指南](https://docs.nvidia.com/cuda/cuda-programming-guide/03-advanced/multi-gpu-systems.html)；传统 IPC、VMM、Fabric 的边界见 [CUDA IPC 指南](https://docs.nvidia.com/cuda/cuda-programming-guide/04-special-topics/inter-process-communication.html) 与 [CUDA VMM 指南](https://docs.nvidia.com/cuda/cuda-programming-guide/04-special-topics/virtual-memory-management.html)。

## 8. 使用场景与选型

- **跨节点 NVLink 系统中共享大型只读数据**：一次创建并让多个进程映射；先评估远端读取延迟、热点争用、复制一份到本地显存是否反而更快。
- **低层通信库或自定义调度器**：需要自己决定哪些分配暴露给哪些 GPU、何时映射/撤销、如何维护生命周期。
- **跨进程 GPU 数据结构**：生产者写入共享缓冲区，消费者 GPU 直接读取；需要明确写入完成、读写权限、同步协议和故障恢复。
- **仅单机两卡互访**：通常先试普通 P2P；若映射全部分配代价高或需细粒度授权，再试 VMM。若没有跨 OS 实例需求，不应为了“Fabric”而增加 IMEX 运维复杂度。
- **普通以太网/InfiniBand 集群**：不要把 Fabric Handle 当作通用 RDMA 方案。优先考虑 NCCL、NVSHMEM、CUDA-aware MPI 或平台提供的通信机制；Fabric VMM 的直接映射前提仍是受支持的 NVLink fabric。

性能测试要把**初始化/导出/导入/映射耗时**与**稳定状态下的远端 kernel 访问耗时**分开记录；比较同一问题下的本地 HBM、普通 P2P、Fabric 远端映射和显式复制方案。记录 GPU 型号、互连拓扑、数据大小、访问模式、并发数及正确性，不应只看一次端到端时间。

## 9. 常见失败及定位

| 现象 | 优先检查 |
| --- | --- |
| 编译报 `CU_MEM_HANDLE_TYPE_FABRIC` 不存在 | Toolkit 是否为 12.4+；编译器是否指向预期 CUDA 安装 |
| 能编译，但能力查询返回 0 | GPU/驱动/平台不支持；升级 Toolkit 不能替代硬件支持 |
| `cuMemCreate` 或导出失败 | 分配属性是否请求 FABRIC、粒度是否对齐、IMEX daemon/Channel 是否配置 |
| 导入报 `CUDA_ERROR_NOT_PERMITTED` | 两端用户是否被授权访问同一 IMEX Channel；不要直接放宽设备文件权限 |
| 导入或映射报 `NOT_SUPPORTED` | 两端平台、NVLink fabric 可达性、驱动与设备支持情况 |
| kernel 报非法地址 | 是否每个进程分别 `cuMemAddressReserve`/`cuMemMap`，并且在映射后调用 `cuMemSetAccess` |
| 数据偶发错误 | 写入是否完成、跨 GPU 可见性/同步是否正确、读写是否并发竞争、导出端是否过早释放 |
| 速度不及复制方案 | 小数据/随机访问的远端延迟、互连带宽和争用；比较完整工作负载，不仅比较单个 API |

## 10. 参考资料

- [NVIDIA CUDA Programming Guide — Virtual Memory Management](https://docs.nvidia.com/cuda/cuda-programming-guide/04-special-topics/virtual-memory-management.html)
- [NVIDIA CUDA Driver API — Virtual Memory Management](https://docs.nvidia.com/cuda/cuda-driver-api/group__CUDA__VA.html)
- [NVIDIA CUDA Programming Guide — Interprocess Communication](https://docs.nvidia.com/cuda/cuda-programming-guide/04-special-topics/inter-process-communication.html)
- [NVIDIA CUDA Programming Guide — Programming Systems with Multiple GPUs](https://docs.nvidia.com/cuda/cuda-programming-guide/03-advanced/multi-gpu-systems.html)
- [NVIDIA CUDA Programming Guide — Compute Fabric Transport](https://docs.nvidia.com/cuda/cuda-programming-guide/04-special-topics/compute-fabric-transport.html)
