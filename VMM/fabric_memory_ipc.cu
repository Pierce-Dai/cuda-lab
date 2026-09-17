// CUDA 12.4+ / Linux：用 Fabric Handle 在两个进程之间共享一块 GPU 显存。
// 这是教学示例；必须在支持 Fabric Handle、NVLink fabric 和 IMEX 的平台运行。
#include <cuda.h>
#include <cuda_runtime.h>

#include <arpa/inet.h>
#include <netdb.h>
#include <sys/socket.h>
#include <unistd.h>

#include <cerrno>
#include <cstdint>
#include <cstring>
#include <cstdlib>
#include <iostream>
#include <stdexcept>
#include <string>

#if CUDA_VERSION < 12040
#error "此示例需要 CUDA Toolkit 12.4 或更新版本"
#endif

namespace {

constexpr int kElements = 1024;
constexpr std::size_t kDataBytes = kElements * sizeof(int);

// 每个线程负责连续数组中的一个元素；越界线程不写入。
__global__ void fill_kernel(int* data, int count) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < count) {
        data[i] = i + 7;
        // 写入要对系统中的其他 GPU 可见，然后再通过主机控制面通知对端。
        __threadfence_system();
    }
}

// 对端 GPU 直接读取远端映射；错误标志保存在对端本地显存。
__global__ void verify_kernel(const int* data, int count, int* mismatch) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < count && data[i] != i + 7) {
        atomicExch(mismatch, 1);
    }
}

void check_cu(CUresult result, const char* operation) {
    if (result == CUDA_SUCCESS) return;
    const char* name = nullptr;
    const char* description = nullptr;
    cuGetErrorName(result, &name);
    cuGetErrorString(result, &description);
    throw std::runtime_error(std::string(operation) + ": " +
                             (name ? name : "未知 CUDA Driver 错误") + " / " +
                             (description ? description : "无说明"));
}

void check_cuda(cudaError_t result, const char* operation) {
    if (result != cudaSuccess) {
        throw std::runtime_error(std::string(operation) + ": " +
                                 cudaGetErrorString(result));
    }
}

void check_sys(bool ok, const char* operation) {
    if (!ok) {
        throw std::runtime_error(std::string(operation) + ": " +
                                 std::strerror(errno));
    }
}

// 文件描述符由本对象独占；异常退出时也会关闭 socket。
struct Socket {
    explicit Socket(int value = -1) : fd(value) {}
    ~Socket() { if (fd >= 0) close(fd); }
    Socket(const Socket&) = delete;
    Socket& operator=(const Socket&) = delete;
    int fd;
};

// VMM 的三个资源各自独立，退出时按“解除映射 -> 释放句柄 -> 释放地址”清理。
struct VmmRegion {
    CUdeviceptr address = 0;
    CUmemGenericAllocationHandle allocation = 0;
    std::size_t bytes = 0;
    bool has_allocation = false;
    bool is_mapped = false;

    ~VmmRegion() {
        if (is_mapped) cuMemUnmap(address, bytes);
        if (has_allocation) cuMemRelease(allocation);
        if (address != 0) cuMemAddressFree(address, bytes);
    }
};

struct DeviceFlag {
    int* ptr = nullptr;
    ~DeviceFlag() { if (ptr) cudaFree(ptr); }
};

// send()/recv() 可能只处理一部分字节；循环直到完整传输。
void send_all(int fd, const void* source, std::size_t length) {
    const auto* data = static_cast<const unsigned char*>(source);
    while (length != 0) {
        const ssize_t sent = send(fd, data, length, MSG_NOSIGNAL);
        if (sent < 0 && errno == EINTR) continue;
        check_sys(sent > 0, "send");
        data += sent;
        length -= static_cast<std::size_t>(sent);
    }
}

void recv_all(int fd, void* destination, std::size_t length) {
    auto* data = static_cast<unsigned char*>(destination);
    while (length != 0) {
        const ssize_t received = recv(fd, data, length, 0);
        if (received < 0 && errno == EINTR) continue;
        if (received == 0) throw std::runtime_error("连接提前关闭");
        check_sys(received > 0, "recv");
        data += received;
        length -= static_cast<std::size_t>(received);
    }
}

int parse_number(const char* text, int min, int max, const char* name) {
    char* end = nullptr;
    errno = 0;
    const long value = std::strtol(text, &end, 10);
    if (errno != 0 || *text == '\0' || *end != '\0' ||
        value < min || value > max) {
        throw std::runtime_error(std::string(name) + " 超出有效范围");
    }
    return static_cast<int>(value);
}

void select_device(int ordinal) {
    int count = 0;
    check_cuda(cudaGetDeviceCount(&count), "cudaGetDeviceCount");
    if (ordinal >= count) throw std::runtime_error("指定的 GPU 编号不存在");
    check_cuda(cudaSetDevice(ordinal), "cudaSetDevice");
    check_cuda(cudaFree(nullptr), "初始化 CUDA primary context");
    check_cu(cuInit(0), "cuInit");

    CUdevice device;
    check_cu(cuDeviceGet(&device, ordinal), "cuDeviceGet");
    int vmm = 0;
    int fabric = 0;
    check_cu(cuDeviceGetAttribute(
                 &vmm, CU_DEVICE_ATTRIBUTE_VIRTUAL_MEMORY_MANAGEMENT_SUPPORTED,
                 device), "查询 VMM 支持");
    check_cu(cuDeviceGetAttribute(
                 &fabric, CU_DEVICE_ATTRIBUTE_HANDLE_TYPE_FABRIC_SUPPORTED,
                 device), "查询 Fabric Handle 支持");
    if (!vmm || !fabric) {
        throw std::runtime_error("当前 GPU/驱动不支持 VMM Fabric Handle");
    }
}

void reserve_map_access(VmmRegion& region, int ordinal) {
    // 两个进程必须各自预留地址。不要跨进程传 CUdeviceptr 数值。
    check_cu(cuMemAddressReserve(&region.address, region.bytes, 0, 0, 0),
             "cuMemAddressReserve");
    check_cu(cuMemMap(region.address, region.bytes, 0, region.allocation, 0),
             "cuMemMap");
    region.is_mapped = true;

    // cuMemMap 本身不授予读写权；授权当前进程所选的 GPU。
    CUmemAccessDesc access{};
    access.location.type = CU_MEM_LOCATION_TYPE_DEVICE;
    access.location.id = ordinal;
    access.flags = CU_MEM_ACCESS_FLAGS_PROT_READWRITE;
    check_cu(cuMemSetAccess(region.address, region.bytes, &access, 1),
             "cuMemSetAccess");
}

int open_listener(int port) {
    const int fd = socket(AF_INET, SOCK_STREAM, 0);
    check_sys(fd >= 0, "socket");
    Socket guard(fd);
    const int yes = 1;
    check_sys(setsockopt(fd, SOL_SOCKET, SO_REUSEADDR, &yes, sizeof(yes)) == 0,
              "setsockopt");
    sockaddr_in address{};
    address.sin_family = AF_INET;
    address.sin_addr.s_addr = htonl(INADDR_ANY);
    address.sin_port = htons(static_cast<std::uint16_t>(port));
    check_sys(bind(fd, reinterpret_cast<sockaddr*>(&address), sizeof(address)) == 0,
              "bind");
    check_sys(listen(fd, 1) == 0, "listen");
    guard.fd = -1;
    return fd;
}

int connect_to(const char* host, int port) {
    addrinfo hints{};
    hints.ai_family = AF_INET;
    hints.ai_socktype = SOCK_STREAM;
    addrinfo* results = nullptr;
    const std::string port_text = std::to_string(port);
    const int error = getaddrinfo(host, port_text.c_str(), &hints, &results);
    if (error != 0) throw std::runtime_error(gai_strerror(error));
    int connected = -1;
    for (addrinfo* item = results; item != nullptr; item = item->ai_next) {
        const int fd = socket(item->ai_family, item->ai_socktype, item->ai_protocol);
        if (fd < 0) continue;
        if (connect(fd, item->ai_addr, item->ai_addrlen) == 0) {
            connected = fd;
            break;
        }
        close(fd);
    }
    freeaddrinfo(results);
    if (connected < 0) throw std::runtime_error("无法连接到导出进程");
    return connected;
}

void run_server(int port, int ordinal) {
    select_device(ordinal);
    VmmRegion region;
    CUmemAllocationProp properties{};
    properties.type = CU_MEM_ALLOCATION_TYPE_PINNED;
    properties.location.type = CU_MEM_LOCATION_TYPE_DEVICE;
    properties.location.id = ordinal;
    properties.requestedHandleTypes = CU_MEM_HANDLE_TYPE_FABRIC;

    std::size_t granularity = 0;
    check_cu(cuMemGetAllocationGranularity(
                 &granularity, &properties, CU_MEM_ALLOC_GRANULARITY_MINIMUM),
             "cuMemGetAllocationGranularity");
    region.bytes = ((kDataBytes + granularity - 1) / granularity) * granularity;
    check_cu(cuMemCreate(&region.allocation, region.bytes, &properties, 0),
             "cuMemCreate：检查 IMEX daemon/channel 配置");
    region.has_allocation = true;
    reserve_map_access(region, ordinal);

    constexpr int threads = 256;
    fill_kernel<<<(kElements + threads - 1) / threads, threads>>>(
        reinterpret_cast<int*>(region.address), kElements);
    check_cuda(cudaGetLastError(), "启动 fill_kernel");
    check_cuda(cudaDeviceSynchronize(), "等待写入完成");

    CUmemFabricHandle shareable{};
    check_cu(cuMemExportToShareableHandle(
                 &shareable, region.allocation, CU_MEM_HANDLE_TYPE_FABRIC, 0),
             "cuMemExportToShareableHandle");

    Socket listener(open_listener(port));
    std::cout << "等待导入进程连接，端口 " << port << std::endl;
    Socket peer(accept(listener.fd, nullptr, nullptr));
    check_sys(peer.fd >= 0, "accept");

    // 控制面只传大小与 handle；数据本身仍留在 GPU 显存中。
    // 此示例限定两端均为 Linux x86-64、相同 CUDA ABI，因此直接传 uint64_t。
    const std::uint64_t bytes = static_cast<std::uint64_t>(region.bytes);
    send_all(peer.fd, &bytes, sizeof(bytes));
    send_all(peer.fd, &shareable, sizeof(shareable));

    // 等对端 GPU 完成访问再释放物理内存，避免悬空映射。
    unsigned char result = 0;
    recv_all(peer.fd, &result, sizeof(result));
    if (result != 1) throw std::runtime_error("导入进程校验失败");
    std::cout << "导出端：收到校验成功回执。" << std::endl;
}

void run_client(const char* host, int port, int ordinal) {
    select_device(ordinal);
    Socket peer(connect_to(host, port));
    std::uint64_t bytes = 0;
    CUmemFabricHandle shareable{};
    recv_all(peer.fd, &bytes, sizeof(bytes));
    recv_all(peer.fd, &shareable, sizeof(shareable));
    if (bytes < kDataBytes || bytes > static_cast<std::uint64_t>(SIZE_MAX)) {
        throw std::runtime_error("收到无效的映射大小");
    }

    VmmRegion region;
    region.bytes = static_cast<std::size_t>(bytes);
    check_cu(cuMemImportFromShareableHandle(
                 &region.allocation, &shareable, CU_MEM_HANDLE_TYPE_FABRIC),
             "cuMemImportFromShareableHandle：检查双方 IMEX Channel 权限");
    region.has_allocation = true;
    reserve_map_access(region, ordinal);

    DeviceFlag flag;
    check_cuda(cudaMalloc(&flag.ptr, sizeof(int)), "cudaMalloc 校验标志");
    check_cuda(cudaMemset(flag.ptr, 0, sizeof(int)), "cudaMemset 校验标志");
    constexpr int threads = 256;
    verify_kernel<<<(kElements + threads - 1) / threads, threads>>>(
        reinterpret_cast<const int*>(region.address), kElements, flag.ptr);
    check_cuda(cudaGetLastError(), "启动 verify_kernel");
    check_cuda(cudaDeviceSynchronize(), "等待远端读取完成");

    int mismatch = 0;
    check_cuda(cudaMemcpy(&mismatch, flag.ptr, sizeof(int), cudaMemcpyDeviceToHost),
               "复制校验标志");
    const unsigned char result = mismatch == 0 ? 1 : 0;
    send_all(peer.fd, &result, sizeof(result));
    if (mismatch != 0) throw std::runtime_error("远端数据校验失败");
    std::cout << "导入端：GPU 直接读取远端映射并校验成功。" << std::endl;
}

}  // namespace

int main(int argc, char** argv) {
    try {
        if (argc == 4 && std::strcmp(argv[1], "server") == 0) {
            run_server(parse_number(argv[2], 1, 65535, "端口"),
                       parse_number(argv[3], 0, 1024, "GPU 编号"));
        } else if (argc == 5 && std::strcmp(argv[1], "client") == 0) {
            run_client(argv[2], parse_number(argv[3], 1, 65535, "端口"),
                       parse_number(argv[4], 0, 1024, "GPU 编号"));
        } else {
            std::cerr << "用法：\n"
                      << "  " << argv[0] << " server <端口> <本地GPU编号>\n"
                      << "  " << argv[0]
                      << " client <服务器地址> <端口> <本地GPU编号>\n";
            return EXIT_FAILURE;
        }
        return EXIT_SUCCESS;
    } catch (const std::exception& error) {
        std::cerr << "失败：" << error.what() << std::endl;
        return EXIT_FAILURE;
    }
}
