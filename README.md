# CUDA Divisible Numbers

##  Overview

Here is an overview of the project:

1. **Argument Parsing & Setup**  
   - Reads two integers from the command line: the divisor X and the upper bound Y.  
   - Verifies that both values are valid (e.g., X > 1, Y ≥ 2).

2. **Host Memory Allocation & Initialization**  
   - Allocates a host array of size Y + 1 (type `char`) and sets all entries to 0.  
   - This array will eventually hold flags (0 or 1) indicating whether each index is divisible by X.

3. **Device Memory Allocation & Data Transfer**  
   - Allocates a device (GPU) array of identical size.  
   - Copies the zero‐initialized host array into device memory using `cudaMemcpy`.  
   - By transferring a blank array first, the kernel can safely mark only those positions that satisfy the divisibility condition.

4. **Kernel Launch Configuration**  
   - Defines a fixed block size of 256 threads.  
   - Calculates the required number of blocks so that `blocks × 256 ≥ Y + 1`.  
   - This ensures every index from 0 through Y (including 2…Y) is potentially handled by exactly one thread.

5. **CUDA Kernel Execution**  
   - Each thread computes its global index `idx = blockIdx.x × blockDim.x + threadIdx.x`.  
   - If `idx` is between 2 and Y (inclusive) and `idx % X == 0`, it writes `1` into `deviceArray[idx]`.  
   - Otherwise, the thread does nothing (no branching or further checks).

6. **Copying Results Back & Host-Side Aggregation**  
   - After kernel completion, the device array is copied back to the host.  
   - A simple for-loop on the CPU scans through indices 2…Y and counts how many entries are flagged `1`.  
   - Prints the total count and also echoes the number of blocks and threads per block used in the kernel launch.

7. **Cleanup**  
   - Frees device memory (`cudaFree`) and host memory (`free`) before exiting.  
   - Any error codes from CUDA calls (e.g., `cudaMalloc`, `cudaMemcpy`) trigger an immediate program abort to avoid silent failures.


## Tech Stack

- **Language:**  
  - C++ (CUDA C)

- **GPU Framework:**  
  - NVIDIA CUDA (compute capability ≥ 3.0)

- **Compiler & Build Tools:**  
  - `nvcc` (CUDA Toolkit 12.x or later)  
  - Linux (tested on Ubuntu 22.04)

- **Runtime Libraries:**  
  - CUDA Runtime API (`cudaMalloc`, `cudaMemcpy`, `cudaFree`)  
  - Standard C I/O (`printf`)

- **Performance Analysis:**  
  - Host-side wall-clock timing (e.g., `std::chrono` or `clock()`)  
  - CUDA events (`cudaEvent_t`) for measuring kernel execution (see report for specifics)


## Key Takeaways

- **Host vs. Device Speedup (Y = 10⁷, X = 7):**  
  - **CPU-only (serial):** ≈ 0.120 s (baseline)  
  - **GPU (256 threads/block, 39 062 blocks):** ≈ 0.005 s → Speedup ≈ 24×  
  - *Insight:* Massive parallelism on thousands of GPU threads makes the trivial modulus-check task extremely fast compared to serial.

- **Kernel Launch Overhead (Y ≈ 10⁵):**  
  - For smaller ranges, data-transfer (`cudaMemcpy`) + kernel-launch latency dominate.  
  - Example: Y = 10⁵ → GPU time ~ 0.002 s vs. CPU ~ 0.001 s → Speedup < 1 (overhead outweighs compute).

- **Memory Transfer Cost:**  
  - Copying a (Y + 1)-byte array back and forth is significant when Y < 10⁶.  
  - As Y grows (≥ 10⁷), compute time dominates, making GPU usage favorable.

- **Occupancy & Thread Utilization:**  
  - 256 threads/block yields high occupancy on modern GPUs (96+ SMs).  
  - Each thread performs a single `%` operation and one memory write—compute-light but hides memory latency effectively at high occupancy.

- **General Lessons:**  
  1. **Choose Y Large Enough:** GPU overhead pays off only with millions of independent tasks.  
  2. **Memory Coalescing:** Writing to `A[idx]` is coalesced (consecutive indices), maximizing global-memory throughput.  
  3. **Divide-and-Conquer Approach:** Even a simple “divisible by X” check benefits from mapping each index to its own thread, demonstrating CUDA’s suitability for embarrassingly parallel tasks.


