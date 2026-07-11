# Stage 0: Environment, Build, Run, Timing

## Goal

This stage verifies that CUDA can compile and run, and records a basic benchmark for vector addition.

## Environment

- GPU: NVIDIA GeForce RTX 2070 with Max-Q Design
- Device count: 1
- Current device: 0
- Compute capability: 7.5
- Global memory: 8589606912 bytes
- SM count: 36
- L2 cache size: 4194304 bytes
- Max threads per block: 1024
- Max threads per SM: 1024
- CUDA runtime version: 12.8
- CUDA driver version: 13.0

Note: CUDA commands should be run in the WSL environment for this project. The Codex tool environment may report a driver/runtime mismatch even when WSL works correctly.

## Build

```bash
nvcc -O2 vector_add.cu -o vector_add
```

## Run

```bash
./vector_add
```

## Official CUDA Samples

The following official samples were run in WSL.

### deviceQuery

Path:

```bash
~/workspace/cuda-samples/cpp/1_Utilities/build/deviceQuery
```

Key output:

```text
Detected 1 CUDA Capable device(s)

Device 0: "NVIDIA GeForce RTX 2070 with Max-Q Design"
  CUDA Driver Version / Runtime Version          13.0 / 12.8
  CUDA Capability Major/Minor version number:    7.5
  Total amount of global memory:                 8192 MBytes (8589606912 bytes)
  (036) Multiprocessors, (064) CUDA Cores/MP:    2304 CUDA Cores
  GPU Max Clock rate:                            1125 MHz (1.12 GHz)
  Memory Clock rate:                             5501 Mhz
  Memory Bus Width:                              256-bit
  L2 Cache Size:                                 4194304 bytes
  Warp size:                                     32
  Maximum number of threads per multiprocessor:  1024
  Maximum number of threads per block:           1024
  Concurrent copy and kernel execution:          Yes with 2 copy engine(s)
  Device supports Unified Addressing (UVA):      Yes
  Device supports Managed Memory:                Yes

Result = PASS
```

### Official vectorAdd

Path:

```bash
~/workspace/cuda-samples/cpp/0_Introduction/vectorAdd/build
```

Key output:

```text
[Vector addition of 50000 elements]
CUDA kernel launch with 196 blocks of 256 threads
Test PASSED
Done
```

### bandwidthTest

`bandwidthTest` is not present in the current official CUDA samples tree being used here, so it is recorded as not applicable for this environment instead of blocking Stage 0.

## Benchmark Result

```text
           N         Bytes       CPU(ms)     H2D(ms)    Kernel(ms)     D2H(ms)     Total(ms)    Result
       10000         40000        0.1215     1.37718       2.64858    0.144224       4.16998      PASS
      100000        400000        0.7222    0.916672      0.374528     0.41264       1.70384      PASS
     1000000       4000000       9.41631     1.32029       0.33184     1.00557        2.6577      PASS
    10000000      40000000       106.071     22.1249      0.632352     11.1685       33.9258      PASS
```

## Notes

- The GPU kernel uses one thread per output element.
- Grid size is computed as `(N + blockSize - 1) / blockSize`.
- The boundary check `if (i < N)` is required because `N` may not be divisible by the block size.
- Global memory access is coalesced because adjacent threads read and write adjacent `float` elements.
- Shared memory is not useful for this kernel because each input element is read once and there is no data reuse.
- This vector add kernel is memory-bound; for small `N`, CPU can be faster because H2D, D2H, and kernel launch overhead dominate total GPU time.

## Review Questions

### How to compile a CUDA program?

Use `nvcc` to compile `.cu` files. For this stage:

```bash
nvcc -O2 vector_add.cu -o vector_add
```

### What is the difference between host code and device code?

Host code runs on the CPU and controls allocation, data transfer, kernel launch, timing, and result checking. Device code runs on the GPU, such as the `__global__` `vectorAdd` kernel.

### What is a kernel launch?

A kernel launch starts GPU execution from host code using the `<<<grid, block>>>` syntax. In this stage:

```cpp
vectorAdd<<<numBlocks, blockSize>>>(d_A, d_B, d_C, N);
```

`blockSize = 256`, and `numBlocks = (N + blockSize - 1) / blockSize`.

### Why can CPU be faster for small input sizes?

For small `N`, the GPU does very little arithmetic, but still pays fixed costs: H2D copy, kernel launch overhead, synchronization, and D2H copy. CPU only runs a simple loop over local memory, so it can win when the GPU overhead is larger than the useful parallel work.

## Stage 0 Checklist

- [x] Know GPU model
- [x] Know CUDA version
- [x] Know compute capability
- [x] Compile a `.cu` file
- [x] Run a CUDA kernel in WSL
- [x] Use CUDA error checking
- [x] Use CUDA event timing
- [x] Record CPU, H2D, kernel, D2H, total GPU time
- [x] Check correctness
- [x] Test multiple input sizes
- [x] Record official `deviceQuery`
- [x] Record official `vectorAdd`
- [x] Record `bandwidthTest` status: not present in current CUDA samples tree

## Stage Status

Stage 0 is complete for this learning repo. The remaining CUDA samples difference is documented as an environment/sample-version issue, not a missing CUDA concept.
