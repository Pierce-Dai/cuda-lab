#include "common/cuda_check.hpp"
#include "common/workload.cuh"

#include <cuda_runtime.h>

#include <sys/socket.h>
#include <sys/types.h>
#include <sys/wait.h>
#include <unistd.h>

#include <cerrno>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <iostream>
#include <string>
#include <vector>

namespace {

struct ExportMessage {
  cudaMemPoolPtrExportData pointer_data{};
  cudaIpcEventHandle_t ready_event{};
  std::uint64_t elements = 0;
};

struct ChildResult {
  int status = 1;
  unsigned int device_errors = 0;
  std::uint64_t checksum = 0;
  std::uintptr_t imported_address = 0;
};

bool send_all(int socket, const void* buffer, std::size_t size) {
  const auto* bytes = static_cast<const char*>(buffer);
  std::size_t sent = 0;
  while (sent < size) {
    const ssize_t result = send(socket, bytes + sent, size - sent, 0);
    if (result < 0 && errno == EINTR) {
      continue;
    }
    if (result <= 0) {
      return false;
    }
    sent += static_cast<std::size_t>(result);
  }
  return true;
}

bool receive_all(int socket, void* buffer, std::size_t size) {
  auto* bytes = static_cast<char*>(buffer);
  std::size_t received = 0;
  while (received < size) {
    const ssize_t result = recv(socket, bytes + received, size - received, 0);
    if (result < 0 && errno == EINTR) {
      continue;
    }
    if (result <= 0) {
      return false;
    }
    received += static_cast<std::size_t>(result);
  }
  return true;
}

bool send_fd_with_message(int socket, int fd, const ExportMessage& message) {
  iovec io{};
  io.iov_base = const_cast<ExportMessage*>(&message);
  io.iov_len = sizeof(message);

  char control[CMSG_SPACE(sizeof(int))]{};
  msghdr header{};
  header.msg_iov = &io;
  header.msg_iovlen = 1;
  header.msg_control = control;
  header.msg_controllen = sizeof(control);

  cmsghdr* cmsg = CMSG_FIRSTHDR(&header);
  cmsg->cmsg_level = SOL_SOCKET;
  cmsg->cmsg_type = SCM_RIGHTS;
  cmsg->cmsg_len = CMSG_LEN(sizeof(int));
  std::memcpy(CMSG_DATA(cmsg), &fd, sizeof(fd));

  const ssize_t result = sendmsg(socket, &header, 0);
  return result == static_cast<ssize_t>(sizeof(message));
}

bool receive_fd_with_message(int socket, int* fd, ExportMessage* message) {
  iovec io{};
  io.iov_base = message;
  io.iov_len = sizeof(*message);

  char control[CMSG_SPACE(sizeof(int))]{};
  msghdr header{};
  header.msg_iov = &io;
  header.msg_iovlen = 1;
  header.msg_control = control;
  header.msg_controllen = sizeof(control);

  ssize_t result = -1;
  do {
    result = recvmsg(socket, &header, 0);
  } while (result < 0 && errno == EINTR);
  if (result != static_cast<ssize_t>(sizeof(*message))) {
    return false;
  }

  const cmsghdr* cmsg = CMSG_FIRSTHDR(&header);
  if (cmsg == nullptr || cmsg->cmsg_level != SOL_SOCKET ||
      cmsg->cmsg_type != SCM_RIGHTS || cmsg->cmsg_len != CMSG_LEN(sizeof(int))) {
    return false;
  }
  std::memcpy(fd, CMSG_DATA(cmsg), sizeof(*fd));
  return true;
}

cudaMemPool_t create_exportable_pool() {
  cudaMemPoolProps properties{};
  properties.allocType = cudaMemAllocationTypePinned;
  properties.handleTypes = cudaMemHandleTypePosixFileDescriptor;
  properties.location.type = cudaMemLocationTypeDevice;
  properties.location.id = 0;

  cudaMemPool_t pool = nullptr;
  CUDA_CHECK(cudaMemPoolCreate(&pool, &properties));
  return pool;
}

int child_process(int socket) {
  unsigned char control = 0;
  if (!receive_all(socket, &control, sizeof(control))) {
    std::cerr << "CHILD_ERROR receive_control_failed\n";
    return 2;
  }
  if (control == 0) {
    return 0;
  }

  ExportMessage message{};
  int pool_fd = -1;
  if (!receive_fd_with_message(socket, &pool_fd, &message)) {
    std::cerr << "CHILD_ERROR receive_export_message_failed\n";
    return 3;
  }

  CUDA_CHECK(cudaSetDevice(0));
  cudaMemPool_t imported_pool = nullptr;
  CUDA_CHECK(cudaMemPoolImportFromShareableHandle(
      &imported_pool, &pool_fd, cudaMemHandleTypePosixFileDescriptor, 0));
  close(pool_fd);

  std::uint32_t* imported = nullptr;
  CUDA_CHECK(cudaMemPoolImportPointer(reinterpret_cast<void**>(&imported),
                                      imported_pool, &message.pointer_data));

  cudaEvent_t ready_event = nullptr;
  CUDA_CHECK(cudaIpcOpenEventHandle(&ready_event, message.ready_event));
  cudaStream_t stream = nullptr;
  CUDA_CHECK(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));
  CUDA_CHECK(cudaStreamWaitEvent(stream, ready_event, 0));

  unsigned int* device_errors = nullptr;
  CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&device_errors),
                        sizeof(unsigned int)));
  CUDA_CHECK(cudaMemsetAsync(device_errors, 0, sizeof(unsigned int), stream));

  constexpr unsigned int block_size = 256U;
  constexpr std::uint32_t increment = 1U;
  verify_and_add<<<grid_for(message.elements, block_size), block_size, 0,
                   stream>>>(imported, message.elements, increment,
                             device_errors);
  CUDA_KERNEL_CHECK();

  ChildResult child_result{};
  child_result.imported_address = reinterpret_cast<std::uintptr_t>(imported);
  CUDA_CHECK(cudaMemcpyAsync(&child_result.device_errors, device_errors,
                             sizeof(unsigned int), cudaMemcpyDeviceToHost,
                             stream));
  std::vector<std::uint32_t> host(message.elements);
  CUDA_CHECK(cudaMemcpyAsync(host.data(), imported,
                             host.size() * sizeof(std::uint32_t),
                             cudaMemcpyDeviceToHost, stream));
  CUDA_CHECK(cudaStreamSynchronize(stream));
  child_result.checksum = host_checksum(host);
  child_result.status =
      child_result.device_errors == 0 && verify_host_values(host, increment)
          ? 0
          : 1;

  CUDA_CHECK(cudaFree(device_errors));
  CUDA_CHECK(cudaFreeAsync(imported, stream));
  CUDA_CHECK(cudaStreamSynchronize(stream));
  CUDA_CHECK(cudaEventDestroy(ready_event));
  CUDA_CHECK(cudaStreamDestroy(stream));
  CUDA_CHECK(cudaMemPoolDestroy(imported_pool));

  if (!send_all(socket, &child_result, sizeof(child_result))) {
    std::cerr << "CHILD_ERROR send_result_failed\n";
    return 4;
  }
  return child_result.status;
}

std::size_t parse_elements(int argc, char** argv) {
  std::size_t elements = 1U << 20U;
  for (int i = 1; i < argc; ++i) {
    const std::string argument = argv[i];
    if (argument == "--elements" && i + 1 < argc) {
      elements = std::stoull(argv[++i]);
    } else if (argument == "--help") {
      std::cout << "Usage: lab03_pool_ipc [--elements N]\n";
      std::exit(EXIT_SUCCESS);
    } else {
      std::cerr << "Unknown argument: " << argument << '\n';
      std::exit(EXIT_FAILURE);
    }
  }
  return elements;
}

}  // namespace

int main(int argc, char** argv) {
  const std::size_t elements = parse_elements(argc, argv);
  int sockets[2] = {-1, -1};
  if (socketpair(AF_UNIX, SOCK_SEQPACKET, 0, sockets) != 0) {
    std::cerr << "HOST_ERROR socketpair_failed errno=" << errno << '\n';
    return EXIT_FAILURE;
  }

  // CUDA is intentionally not initialized until after fork. Each process creates
  // its own runtime state and communicates only through exported CUDA handles.
  const pid_t child = fork();
  if (child < 0) {
    std::cerr << "HOST_ERROR fork_failed errno=" << errno << '\n';
    return EXIT_FAILURE;
  }
  if (child == 0) {
    close(sockets[0]);
    const int result = child_process(sockets[1]);
    close(sockets[1]);
    return result;
  }

  close(sockets[1]);
  CUDA_CHECK(cudaSetDevice(0));
  int supported_handle_types = 0;
  CUDA_CHECK(cudaDeviceGetAttribute(
      &supported_handle_types, cudaDevAttrMemoryPoolSupportedHandleTypes, 0));
  if ((supported_handle_types & cudaMemHandleTypePosixFileDescriptor) == 0) {
    const unsigned char control = 0;
    if (!send_all(sockets[0], &control, sizeof(control))) {
      std::cerr << "HOST_ERROR send_skip_control_failed\n";
    }
    std::cout << "SKIP reason=posix_fd_mempool_handle_not_supported mask="
              << supported_handle_types << '\n';
    close(sockets[0]);
    int child_status = 0;
    waitpid(child, &child_status, 0);
    return 0;
  }

  const unsigned char control = 1;
  if (!send_all(sockets[0], &control, sizeof(control))) {
    std::cerr << "HOST_ERROR send_run_control_failed\n";
    return EXIT_FAILURE;
  }

  cudaMemPool_t pool = create_exportable_pool();
  cudaStream_t stream = nullptr;
  CUDA_CHECK(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));
  cudaEvent_t ready_event = nullptr;
  CUDA_CHECK(cudaEventCreateWithFlags(
      &ready_event, cudaEventDisableTiming | cudaEventInterprocess));

  const std::size_t bytes = elements * sizeof(std::uint32_t);
  std::uint32_t* exported = nullptr;
  CUDA_CHECK(cudaMallocFromPoolAsync(reinterpret_cast<void**>(&exported), bytes,
                                     pool, stream));
  initialize_values<<<grid_for(elements), 256, 0, stream>>>(exported, elements);
  CUDA_KERNEL_CHECK();
  CUDA_CHECK(cudaEventRecord(ready_event, stream));

  ExportMessage message{};
  message.elements = elements;
  CUDA_CHECK(cudaMemPoolExportPointer(&message.pointer_data, exported));
  CUDA_CHECK(cudaIpcGetEventHandle(&message.ready_event, ready_event));
  int exported_pool_fd = -1;
  CUDA_CHECK(cudaMemPoolExportToShareableHandle(
      &exported_pool_fd, pool, cudaMemHandleTypePosixFileDescriptor, 0));

  std::cout << "LAB name=pool_ipc elements=" << elements
            << " parent_pid=" << getpid() << " child_pid=" << child << '\n';
  std::cout << "PARENT export_ptr=" << static_cast<void*>(exported)
            << " pool_handle=posix_fd ready_dependency=ipc_event\n";
  if (!send_fd_with_message(sockets[0], exported_pool_fd, message)) {
    std::cerr << "HOST_ERROR send_export_message_failed\n";
    close(exported_pool_fd);
    return EXIT_FAILURE;
  }
  close(exported_pool_fd);

  ChildResult child_result{};
  if (!receive_all(sockets[0], &child_result, sizeof(child_result))) {
    std::cerr << "HOST_ERROR receive_child_result_failed\n";
    close(sockets[0]);
    int child_status = 0;
    waitpid(child, &child_status, 0);
    return EXIT_FAILURE;
  }
  std::cout << "CHILD import_ptr=reinterpreted:0x" << std::hex
            << child_result.imported_address << std::dec
            << " device_errors=" << child_result.device_errors
            << " checksum=" << child_result.checksum << '\n';
  std::cout << "ORDER child_import_free=complete parent_original_free=now_safe\n";

  std::vector<std::uint32_t> host(elements);
  CUDA_CHECK(cudaMemcpyAsync(host.data(), exported, bytes,
                             cudaMemcpyDeviceToHost, stream));
  CUDA_CHECK(cudaStreamSynchronize(stream));
  const bool parent_verified = verify_host_values(host, 1U);

  CUDA_CHECK(cudaFreeAsync(exported, stream));
  CUDA_CHECK(cudaStreamSynchronize(stream));
  CUDA_CHECK(cudaEventDestroy(ready_event));
  CUDA_CHECK(cudaStreamDestroy(stream));
  CUDA_CHECK(cudaMemPoolDestroy(pool));
  close(sockets[0]);

  int wait_status = 0;
  waitpid(child, &wait_status, 0);
  const bool child_ok = WIFEXITED(wait_status) && WEXITSTATUS(wait_status) == 0 &&
                        child_result.status == 0;
  const bool ok = child_ok && parent_verified;
  std::cout << "RESULT child=" << (child_ok ? "PASS" : "FAIL")
            << " parent_post_ipc_verify="
            << (parent_verified ? "PASS" : "FAIL") << '\n';
  std::cout << "LAB_RESULT status=" << (ok ? "PASS" : "FAIL") << '\n';
  return ok ? EXIT_SUCCESS : EXIT_FAILURE;
}
