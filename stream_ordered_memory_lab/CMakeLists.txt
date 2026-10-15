cmake_minimum_required(VERSION 3.24)

project(stream_ordered_memory_lab LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
  set(CMAKE_CUDA_ARCHITECTURES native)
endif()

function(add_cuda_lab target source)
  add_executable(${target} ${source})
  target_include_directories(${target} PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
  target_compile_options(${target} PRIVATE
    $<$<COMPILE_LANGUAGE:CXX>:-Wall;-Wextra;-Wpedantic>
    $<$<COMPILE_LANGUAGE:CUDA>:--expt-relaxed-constexpr;-lineinfo>
  )
endfunction()

add_cuda_lab(lab01_lifetime 01_lifetime.cu)
add_cuda_lab(lab02_pool_reuse 02_pool_reuse.cu)
add_cuda_lab(lab03_pool_ipc 03_pool_ipc.cu)
add_cuda_lab(lab04_p2p_optional 04_p2p_optional.cu)

