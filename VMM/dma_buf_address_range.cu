/*
 * cuMemGetHandleForAddressRange：把 GPU 地址范围导出为 DMA-BUF FD
 * =====================================================================
 *
 * 一、先理解它解决什么问题
 *
 * CUDA 指针只告诉 CUDA：“我要访问这段 GPU 虚拟地址。”
 * Linux 中另一个支持 DMA-BUF 的驱动/设备，并不能仅凭这个指针找到、
 * 导入和使用对应的显存。这个 API 为指定地址范围生成 DMA-BUF 文件
 * 描述符（FD），让兼容的外部子系统通过自己的导入接口引用这段内存。
 *
 * 类比：
 *   CUdeviceptr                  = CUDA 地址空间中的“门牌号”
 *   CUmemGenericAllocationHandle = CUDA 管理物理分配的“资源凭证”
 *   DMA-BUF FD                   = Linux 子系统可以接收的“共享凭证”
 *
 * 三者不是同一种东西，不能互相强制转换！导出不是复制显存内容，
 * 也不是分配第二份显存，更不是启动一次 DMA 传输。
 * 典型用途是支持 DMA-BUF 的 GPUDirect RDMA 等互操作路径；
 * 能否使用还取决于 GPU、驱动、内核、外部设备及其驱动。
 *
 * 本文件完整演示 CUDA 生产者侧的生命周期：
 *   初始化 -> 查询能力 -> VMM 分配/映射 -> 写入测试数据 -> 导出 FD
 *          -> 检查 FD 和原 CUDA 数据 -> close -> 解除映射/释放资源
 *
 * 本例不包含外部消费者，不宣称验证了 RDMA 或跨设备读写。
 * 本次按你的要求单独学习这个 API，不代表学习计划中的阶段已经验收。
 *
 * 二、函数原型和每个参数
 *
 * CUresult cuMemGetHandleForAddressRange(
 *     void* handle,
 *     CUdeviceptr dptr,
 *     size_t size,
 *     CUmemRangeHandleType handleType,
 *     unsigned long long flags);
 *
 * 1. handle：输出位置，不是输入的分配 handle。
 *    当 handleType 为 CU_MEM_RANGE_HANDLE_TYPE_DMA_BUF_FD 时，CUDA
 *    将它解释成 int*。正确写法是 int fd = -1; 然后传 &fd。
 *    不能传 fd 本身、&dptr 或 CUmemGenericAllocationHandle*。
 *    只有返回 CUDA_SUCCESS 后，才能把输出当成有效 FD。
 *
 * 2. dptr：要导出的 CUDA 虚拟地址范围的起点，不是 &dptr。
 *    本例来自 cuMemAddressReserve，并通过 cuMemMap 完整映射。
 *    官方也支持 cuMemAlloc 的设备分配路径；不能因此推断任意
 *    managed memory、内存池分配或任意 CPU 指针都适用。
 *    地址必须按 host page size（主机页大小）对齐。
 *
 * 3. size：从 dptr 开始要导出的字节数，不是元素数量。
 *    必须为主机页大小的整数倍，而且整个范围都必须有真实内存映射。
 *    可以导出符合条件的子范围，不一定要导出整个 allocation。
 *    如果范围跨多个物理分配，它们必须在同一设备上且分配属性相同。
 *    本例只用一个物理分配，避免掩盖这些约束。
 *
 * 4. handleType：决定输出句柄的类型和大小。
 *    本例固定为 CU_MEM_RANGE_HANDLE_TYPE_DMA_BUF_FD，输出是 int。
 *    不要混用另一个枚举中的 CU_MEM_HANDLE_TYPE_POSIX_FILE_DESCRIPTOR
 *    或 CU_MEM_HANDLE_TYPE_FABRIC，它们属于另一套导出接口。
 *
 * 5. flags：本例传 0，采用默认映射选择，不要求特殊路径。
 *    CU_MEM_RANGE_FLAG_DMA_BUF_MAPPING_TYPE_PCIE 要求通过 PCIe BAR1
 *    映射；平台不支持就报错。它不是“自动更快”开关。
 *
 * 返回值：CUDA_SUCCESS 表示导出成功；常见错误包括
 * CUDA_ERROR_INVALID_VALUE（地址、大小、类型或 flags 等不符合要求）
 * 和 CUDA_ERROR_NOT_SUPPORTED（平台/内存路径不支持）。
 * 初始化、context 等调用也必须检查错误，不能只检查最后一次导出。
 *
 * 三、与相似 API 的区别
 *
 * cuMemGetHandleForAddressRange：
 *   输入 CUDA 地址 + 长度，输出 DMA-BUF FD，面向兼容外部子系统。
 * cuMemExportToShareableHandle：
 *   输入 VMM allocation handle，配合 cuMemImportFromShareableHandle
 *   等实现相应类型的 CUDA 共享；创建时要请求对应的可导出句柄类型。
 * cuMemRetainAllocationHandle：
 *   从已有映射取得 CUDA allocation handle 引用，不是 DMA-BUF FD。
 * cuIpcGetMemHandle：
 *   CUDA IPC 的另一套机制，返回的也不是本例的 DMA-BUF FD。
 *
 * 本例的 requestedHandleTypes 保持为 0，因为不走
 * cuMemExportToShareableHandle 路径。虽然某些接口都返回 int FD，
 * 但 FD 背后的内核对象/协议可能不同，不能把本例 FD 随意交给
 * cuMemImportFromShareableHandle 或 CUDA external-memory 导入接口。
 *
 * 四、按顺序看流程
 *
 * [1] cuInit / cuDeviceGet：选择 GPU，查询 VMM 和 DMA-BUF 能力。
 *     VMM_SUPPORTED=1 不代表 DMA_BUF_SUPPORTED=1。
 *     WSL 可以正常运行 CUDA，但某些 Linux DMA-BUF 路径仍不支持；
 *     以当前 WSL 中的能力查询和真实调用为准，不靠系统名称猜测。
 *
 * [2] 保留并设置 primary context。这里仅用 Driver API，不混用 Runtime。
 *
 * [3] 查询 VMM 最小分配粒度 G 和主机页大小 P。
 *     VMM 的大小/映射约束与本 API 的主机页对齐约束是两回事。
 *     本例验证二者的兼容性，然后把实际分配大小向上对齐：
 *       alignment = max(G, P)        // 验证它是二者的公倍数
 *       allocation_bytes = ceil(logical_bytes / alignment) * alignment
 *     导出的是实际分配范围，逻辑数据仅占前 logical_bytes 字节。
 *     填充区也先清零，避免将未初始化数据共享给外部消费者。
 *
 * [4] cuMemAddressReserve：只预留地址，没有分配物理显存。
 *     cuMemCreate：创建物理分配，得到 allocation handle，不是指针。
 *     cuMemMap：建立“虚拟地址 -> 物理分配”的映射。
 *     cuMemSetAccess：授权本 GPU 读写；映射本身不会自动授予权限。
 *
 * [5] memset 清零并 H2D 写入测试数据，cuCtxSynchronize 确认 CUDA
 *     写入完成。本 API 不是同步屏障，也不自动建立跨设备执行顺序。
 *
 * [6] cuMemGetHandleForAddressRange(&fd, address, allocation_bytes,
 *                                 CU_MEM_RANGE_HANDLE_TYPE_DMA_BUF_FD, 0);
 *     得到 FD 后用 fcntl 查询有效性，并设置 FD_CLOEXEC 防止 exec 泄漏。
 *     fcntl 只验证文件描述符，不证明外部设备能导入/访问 GPU 数据。
 *
 * [7] 真实外部消费者应接在这里。概念流程（不是通用可执行代码）：
 *       producer: 等 CUDA 写完 -> 导出/传递 FD
 *       consumer: 用对应子系统的 DMA-BUF 导入接口注册这段范围
 *                 -> 按该子系统协议读写 -> 等待外部操作完成
 *                 -> 注销导入资源 -> 通知 producer 可以回收
 *
 *     例如支持该路径的 RDMA 软件栈可能使用 ibv_reg_dmabuf_mr：
 *       pd      = 已建立的 RDMA protection domain
 *       offset  = 0（相对这个 DMA-BUF 的起点，不是原 CUDA allocation）
 *       length  = 不超过导出范围的注册长度
 *       iova    = 按 verbs 文档约束设置的注册地址，不等于通用物理地址
 *       fd      = 本例导出的 FD
 *       access  = 根据实际读写需求选择的访问权限
 *     随后还需要队列、完成事件和 ibv_dereg_mr 等外部生命周期管理。
 *     本例不引入这些依赖，也不把“拿到了 FD”当作 RDMA 成功。
 *
 *     跨进程必须通过 Unix domain socket 的 SCM_RIGHTS 等方式传 FD；
 *     把整数 fd 写入文件/消息并发送，不会为对方创建有效描述符。
 *     不要假设 read(fd) 就能读取显存，或任意平台都支持 mmap(fd)。
 *     新版 CUDA 文档提供 DMA_BUF_MMAP_SUPPORTED 能力查询；只有在
 *     对应版本和平台明确支持时才讨论 mmap，本 CUDA 12.8 示例不使用它。
 *     外部设备写回后，CUDA 读取前还须满足该互操作路径要求的完成通知、
 *     可见性和内存排序；cuCtxSynchronize 不能替代外部设备的完成协议。
 *
 * [8] 本例只从原 CUDA 地址 D2H 校验，因此验证的是：
 *     “原 CUDA 数据正确 + 导出调用成功 + FD 有效”，不是外部读写。
 *
 * [9] 所有使用者停止访问后，先注销外部导入资源并关闭相关 FD，
 *     再 cuMemUnmap -> cuMemRelease -> cuMemAddressFree -> 释放 context。
 *     close(fd) 只关闭当前进程的这个描述符，不替代 VMM 的三步回收。
 *     地址范围的底层物理分配一旦更换，必须重新取得句柄，并更新消费者；
 *     不要假设旧 FD 会自动跟随相同虚拟地址的新映射。
 *
 * 五、编译运行（在仓库根目录的 WSL 终端）
 *
 *   nvcc -std=c++17 -Xcompiler=-Wall,-Wextra \
 *       VMM/dma_buf_address_range.cu -lcuda -o /tmp/dma_buf_address_range
 *   /tmp/dma_buf_address_range
 *   echo $?
 *
 * 退出码：0 = 本例校验和清理成功；77 = 能力查询明确不支持；
 *         1 = CUDA/系统调用/数据校验/清理失败，查看具体错误。
 * 77 是明确的 SKIP，不是成功；不会悄悄换成另一种 FD 或 IPC 路径。
 * 如果工具环境报驱动不匹配/访问受限，请在自己的 WSL 终端复测，
 * 不能据此直接认定你的 WSL CUDA 环境不可用。
 *
 * 本例是 API 生命周期实验，不是计算性能 benchmark：没有 kernel，
 * 所以 CPU 计算 baseline、kernel time 和 block/grid 配置不适用。
 * 有 H2D/D2H 正确性检查，但不报告未测量的传输时间或性能结论。
 * 做实验请记录 GPU、驱动/Toolkit、逻辑/实际大小、能力值及校验结果；
 * 若后续测性能，应分别测 H2D、导出调用 CPU 耗时、D2H 和端到端时间，
 * 不要把 FD 导出耗时解释成 GPU 数据传输时间。
 *
 * 六、亲手练习
 * 1. 改变 logical_bytes，观察逻辑大小、分配大小和对齐的关系。
 * 2. 在支持导出的平台上分配至少两个 alignment 单位，尝试导出后半段；
 *    先推导 address + offset、size、映射范围必须满足的条件。
 * 3. 用自己的话解释：为什么 fd、address、allocation_handle 不能互换？
 *    总结至少三个错误：传错输出参数、对齐错误、过早释放/缺少同步。
 *
 * 依据：本机 CUDA 12.8 cuda.h 的 API 契约，以及 NVIDIA 官方文档：
 * https://docs.nvidia.com/cuda/cuda-driver-api/group__CUDA__MEM.html
 * https://docs.nvidia.com/cuda/cuda-driver-api/group__CUDA__VA.html
 * 在线文档可能更新；较新版本的 host allocation / mmap 扩展不在本例范围。
 */

#include <cuda.h>

#include <algorithm>
#include <array>
#include <cerrno>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <limits>

#include <fcntl.h>
#include <unistd.h>

static bool check_cuda(CUresult result, const char* expression, int line) {
    if (result == CUDA_SUCCESS) {
        return true;
    }
    const char* name = nullptr;
    const char* message = nullptr;
    cuGetErrorName(result, &name);
    cuGetErrorString(result, &message);
    std::cerr << "CUDA error at line " << line << ": " << expression
              << " -> " << (name ? name : "unknown")
              << " (" << static_cast<int>(result) << "), "
              << (message ? message : "no description") << '\n';
    return false;
}

#define CU_OK(call) check_cuda((call), #call, __LINE__)

int main() {
    constexpr int skipped = 77;
    constexpr std::size_t logical_bytes = 4096 + 13;
    CUdevice device = 0;
    CUcontext context = nullptr;
    CUdeviceptr address = 0;
    CUmemGenericAllocationHandle allocation_handle = 0;
    std::size_t allocation_bytes = 0;
    int fd = -1;
    bool context_retained = false;
    bool address_reserved = false;
    bool allocation_created = false;
    bool mapped = false;

    // 将主体放进 lambda，使任一步失败后仍统一执行末尾的资源清理。
    const auto run = [&]() -> int {
        if (!CU_OK(cuInit(0)) || !CU_OK(cuDeviceGet(&device, 0))) {
            return EXIT_FAILURE;
        }
        char name[256]{};
        int driver_version = 0;
        int vmm_supported = 0;
        int dma_buf_supported = 0;
        if (!CU_OK(cuDeviceGetName(name, sizeof(name), device)) ||
            !CU_OK(cuDriverGetVersion(&driver_version)) ||
            !CU_OK(cuDeviceGetAttribute(
                &vmm_supported,
                CU_DEVICE_ATTRIBUTE_VIRTUAL_MEMORY_MANAGEMENT_SUPPORTED,
                device)) ||
            !CU_OK(cuDeviceGetAttribute(
                &dma_buf_supported, CU_DEVICE_ATTRIBUTE_DMA_BUF_SUPPORTED,
                device))) {
            return EXIT_FAILURE;
        }
        std::cout << "GPU: " << name << "\nDriver CUDA API version: "
                  << driver_version << "\nBuild CUDA_VERSION: " << CUDA_VERSION
                  << "\nVMM_SUPPORTED=" << vmm_supported
                  << ", DMA_BUF_SUPPORTED=" << dma_buf_supported << std::endl;
        if (!vmm_supported || !dma_buf_supported) {
            std::cerr << "SKIP: this device/driver environment does not support "
                         "the required VMM + DMA-BUF path.\n";
            return skipped;
        }

        if (!CU_OK(cuDevicePrimaryCtxRetain(&context, device))) {
            return EXIT_FAILURE;
        }
        context_retained = true;
        if (!CU_OK(cuCtxSetCurrent(context))) {
            return EXIT_FAILURE;
        }

        CUmemAllocationProp properties{};
        properties.type = CU_MEM_ALLOCATION_TYPE_PINNED;
        properties.location.type = CU_MEM_LOCATION_TYPE_DEVICE;
        properties.location.id = device;
        std::size_t granularity = 0;
        if (!CU_OK(cuMemGetAllocationGranularity(
                &granularity, &properties, CU_MEM_ALLOC_GRANULARITY_MINIMUM))) {
            return EXIT_FAILURE;
        }
        errno = 0;
        const long page_size_result = sysconf(_SC_PAGESIZE);
        if (page_size_result <= 0) {
            std::cerr << "sysconf(_SC_PAGESIZE) failed: "
                      << (errno ? std::strerror(errno) : "invalid page size")
                      << '\n';
            return EXIT_FAILURE;
        }
        const auto page_size = static_cast<std::size_t>(page_size_result);
        const std::size_t alignment = std::max(granularity, page_size);
        if (granularity == 0 || alignment % granularity != 0 ||
            alignment % page_size != 0 ||
            (alignment & (alignment - 1)) != 0 ||
            logical_bytes > std::numeric_limits<std::size_t>::max() -
                                (alignment - 1)) {
            std::cerr << "Unsupported alignment or allocation size overflow.\n";
            return EXIT_FAILURE;
        }
        allocation_bytes =
            ((logical_bytes + alignment - 1) / alignment) * alignment;
        std::cout << "Host page size: " << page_size
                  << "\nVMM granularity: " << granularity
                  << "\nLogical bytes: " << logical_bytes
                  << "\nAllocated/exported bytes: " << allocation_bytes << '\n';

        if (!CU_OK(cuMemAddressReserve(
                &address, allocation_bytes, alignment, 0, 0))) {
            return EXIT_FAILURE;
        }
        address_reserved = true;
        if (!CU_OK(cuMemCreate(
                &allocation_handle, allocation_bytes, &properties, 0))) {
            return EXIT_FAILURE;
        }
        allocation_created = true;
        if (!CU_OK(cuMemMap(
                address, allocation_bytes, 0, allocation_handle, 0))) {
            return EXIT_FAILURE;
        }
        mapped = true;
        CUmemAccessDesc access{};
        access.location = properties.location;
        access.flags = CU_MEM_ACCESS_FLAGS_PROT_READWRITE;
        if (!CU_OK(cuMemSetAccess(address, allocation_bytes, &access, 1))) {
            return EXIT_FAILURE;
        }

        std::array<unsigned char, logical_bytes> input{};
        std::array<unsigned char, logical_bytes> output{};
        for (std::size_t i = 0; i < input.size(); ++i) {
            input[i] = static_cast<unsigned char>((i * 17 + 3) % 256);
        }
        if (!CU_OK(cuMemsetD8(address, 0, allocation_bytes)) ||
            !CU_OK(cuMemcpyHtoD(address, input.data(), input.size())) ||
            !CU_OK(cuCtxSynchronize())) {
            return EXIT_FAILURE;
        }

        if (!CU_OK(cuMemGetHandleForAddressRange(
                &fd, address, allocation_bytes,
                CU_MEM_RANGE_HANDLE_TYPE_DMA_BUF_FD, 0))) {
            return EXIT_FAILURE;
        }
        const int fd_flags = fcntl(fd, F_GETFD);
        if (fd_flags == -1) {
            std::perror("fcntl(F_GETFD)");
            return EXIT_FAILURE;
        }
        if (fcntl(fd, F_SETFD, fd_flags | FD_CLOEXEC) == -1) {
            std::perror("fcntl(F_SETFD)");
            return EXIT_FAILURE;
        }
        std::cout << "Exported DMA-BUF FD: " << fd << '\n';

        // 外部消费者的导入/操作/等待/注销应插在这里，本例不模拟其成功。
        if (!CU_OK(cuMemcpyDtoH(output.data(), address, output.size())) ||
            !CU_OK(cuCtxSynchronize())) {
            return EXIT_FAILURE;
        }
        for (std::size_t i = 0; i < input.size(); ++i) {
            if (output[i] != input[i]) {
                std::cerr << "FAIL: byte " << i << ", expected "
                          << static_cast<unsigned int>(input[i]) << ", got "
                          << static_cast<unsigned int>(output[i]) << '\n';
                return EXIT_FAILURE;
            }
        }
        std::cout << "CUDA round-trip data: PASS (" << logical_bytes
                  << " bytes)\nDMA-BUF export and FD validity: PASS\n"
                     "External DMA-BUF consumer access: NOT TESTED\n";
        return EXIT_SUCCESS;
    };

    int status = run();
    // 错误路径也尝试同步并清理；清理错误必须影响最终退出码。
    if (mapped && !CU_OK(cuCtxSynchronize())) {
        status = EXIT_FAILURE;
    }
    if (fd >= 0 && close(fd) == -1) {
        // Linux 下不要因 EINTR 盲目重试 close，以免误关复用后的 FD。
        std::perror("close(DMA-BUF FD)");
        status = EXIT_FAILURE;
    }
    if (mapped && !CU_OK(cuMemUnmap(address, allocation_bytes))) {
        status = EXIT_FAILURE;
    }
    if (allocation_created && !CU_OK(cuMemRelease(allocation_handle))) {
        status = EXIT_FAILURE;
    }
    if (address_reserved && !CU_OK(cuMemAddressFree(address, allocation_bytes))) {
        status = EXIT_FAILURE;
    }
    if (context_retained && !CU_OK(cuDevicePrimaryCtxRelease(device))) {
        status = EXIT_FAILURE;
    }
    if (status == EXIT_SUCCESS) {
        std::cout << "Cleanup: PASS\n";
    }
    return status;
}
