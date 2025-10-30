// Microbenchmark for comparing different LDS layouts
// This will cause compiler to generate ds_read2_b64 vs ds_read2st64_b64
#include <hip/hip_runtime.h>
#include <hip/hip_bf16.h>
#include <stdio.h>
#include <chrono>

#define CHECK_HIP(call) \
    do { \
        hipError_t err = call; \
        if (err != hipSuccess) { \
            printf("HIP error at %s:%d: %s\n", __FILE__, __LINE__, hipGetErrorString(err)); \
            exit(1); \
        } \
    } while(0)

// Use short vector type instead of bfloat16
using s16x4 = short __attribute__((__vector_size__(4 * sizeof(short))));

// Layout 1: Compact layout (BLOCK_K = 16, forces consecutive reads)
// Compiler should generate ds_read2_b64 with small offsets
template<int WARP_M, int WARP_K, int MMA_M, int MMA_K>
__global__ void benchmark_compact_layout(float *output, int iterations) {
    constexpr int BLOCK_K = 16;  // Small K dimension forces compact layout
    __shared__ __hip_bfloat16 A_smem[WARP_M * BLOCK_K];
    
    int tid = threadIdx.x;
    int lane_id = tid % 64;
    
    // Initialize
    for (int i = tid; i < WARP_M * BLOCK_K; i += blockDim.x) {
        A_smem[i] = __float2bfloat16((float)i);
    }
    __syncthreads();
    
    float sum = 0.0f;
    
    // Access pattern that should generate ds_read2_b64
    for (int iter = 0; iter < iterations; iter++) {
        for (int mma_id_m = 0; mma_id_m < WARP_M / MMA_M; mma_id_m++) {
            for (int mma_id_k = 0; mma_id_k < BLOCK_K / MMA_K; mma_id_k++) {
                int row = mma_id_m * MMA_M + (lane_id % 16);
                int col = mma_id_k * MMA_K + (lane_id / 16) * 4;
                
                // Consecutive access - should use ds_read2_b64
                s16x4 data = reinterpret_cast<s16x4*>(&A_smem[row * BLOCK_K + col])[0];
                __hip_bfloat16* ptr = reinterpret_cast<__hip_bfloat16*>(&data);
                sum += __bfloat162float(ptr[0]) + __bfloat162float(ptr[1]);
                sum += __bfloat162float(ptr[2]) + __bfloat162float(ptr[3]);
            }
        }
    }
    
    if (tid == 0) output[0] = sum;
}

// Layout 2: Strided layout (BLOCK_K = 64, may generate ds_read2st64_b64)
// With proper swizzle pattern
template<int WARP_M, int WARP_K, int MMA_M, int MMA_K>
__global__ void benchmark_strided_layout(float *output, int iterations) {
    constexpr int BLOCK_K = 64;  // Larger K with stride
    __shared__ __hip_bfloat16 A_smem[WARP_M * BLOCK_K];
    
    int tid = threadIdx.x;
    int lane_id = tid % 64;
    
    for (int i = tid; i < WARP_M * BLOCK_K; i += blockDim.x) {
        A_smem[i] = __float2bfloat16((float)i);
    }
    __syncthreads();
    
    float sum = 0.0f;
    
    // Access pattern with larger stride
    for (int iter = 0; iter < iterations; iter++) {
        for (int mma_id_m = 0; mma_id_m < WARP_M / MMA_M; mma_id_m++) {
            for (int mma_id_k = 0; mma_id_k < WARP_K / MMA_K; mma_id_k++) {
                int row = mma_id_m * MMA_M + (lane_id % 16);
                int col = mma_id_k * MMA_K + (lane_id / 16) * 4;
                
                // Access with stride - may use ds_read2st64_b64
                s16x4 data = reinterpret_cast<s16x4*>(&A_smem[row * BLOCK_K + col])[0];
                __hip_bfloat16* ptr = reinterpret_cast<__hip_bfloat16*>(&data);
                sum += __bfloat162float(ptr[0]) + __bfloat162float(ptr[1]);
                sum += __bfloat162float(ptr[2]) + __bfloat162float(ptr[3]);
            }
        }
    }
    
    if (tid == 0) output[0] = sum;
}

// Layout 3: Transpose layout (row-major to col-major, definitely needs strided access)
template<int WARP_M, int WARP_K, int MMA_M, int MMA_K>
__global__ void benchmark_transpose_layout(float *output, int iterations) {
    constexpr int BLOCK_K = 64;
    __shared__ __hip_bfloat16 A_smem[WARP_M * BLOCK_K];  // stored col-major
    
    int tid = threadIdx.x;
    int lane_id = tid % 64;
    
    for (int i = tid; i < WARP_M * BLOCK_K; i += blockDim.x) {
        A_smem[i] = __float2bfloat16((float)i);
    }
    __syncthreads();
    
    float sum = 0.0f;
    
    // Transposed access - definitely needs strided reads
    for (int iter = 0; iter < iterations; iter++) {
        for (int mma_id_m = 0; mma_id_m < WARP_M / MMA_M; mma_id_m++) {
            for (int mma_id_k = 0; mma_id_k < WARP_K / MMA_K; mma_id_k++) {
                int row = mma_id_m * MMA_M + (lane_id % 16);
                int col = mma_id_k * MMA_K + (lane_id / 16) * 4;
                
                // Transpose access: read column-wise from row-major storage
                __hip_bfloat16 vals[4];
                vals[0] = A_smem[col * WARP_M + row];
                vals[1] = A_smem[(col+1) * WARP_M + row];
                vals[2] = A_smem[(col+2) * WARP_M + row];
                vals[3] = A_smem[(col+3) * WARP_M + row];
                
                sum += __bfloat162float(vals[0]) + __bfloat162float(vals[1]);
                sum += __bfloat162float(vals[2]) + __bfloat162float(vals[3]);
            }
        }
    }
    
    if (tid == 0) output[0] = sum;
}

template<typename KernelFunc>
double benchmark_kernel(KernelFunc kernel, const char* name, 
                        int block_size, int iterations, int warmup_runs, int bench_runs) {
    float *d_output;
    CHECK_HIP(hipMalloc(&d_output, sizeof(float)));
    
    // Warmup
    for (int i = 0; i < warmup_runs; i++) {
        kernel<<<1, block_size>>>(d_output, iterations);
    }
    CHECK_HIP(hipDeviceSynchronize());
    
    // Benchmark
    auto start = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < bench_runs; i++) {
        kernel<<<1, block_size>>>(d_output, iterations);
    }
    CHECK_HIP(hipDeviceSynchronize());
    auto end = std::chrono::high_resolution_clock::now();
    
    double elapsed_ms = std::chrono::duration<double, std::milli>(end - start).count();
    double avg_time_us = (elapsed_ms * 1000.0) / bench_runs;
    
    printf("%-55s: %8.3f us\n", name, avg_time_us);
    
    CHECK_HIP(hipFree(d_output));
    return avg_time_us;
}

int main() {
    hipDeviceProp_t prop;
    CHECK_HIP(hipGetDeviceProperties(&prop, 0));
    
    printf("==================================================================\n");
    printf("LDS Layout Microbenchmark - Instruction Generation Comparison\n");
    printf("==================================================================\n");
    printf("Device: %s\n", prop.name);
    printf("LDS per block: %zu KB\n", prop.sharedMemPerBlock / 1024);
    printf("==================================================================\n\n");
    
    constexpr int WARP_M = 64;
    constexpr int WARP_K_SMALL = 16;
    constexpr int WARP_K_LARGE = 64;
    constexpr int MMA_M = 16;
    constexpr int MMA_K = 16;
    constexpr int BLOCK_SIZE = 256;
    constexpr int ITERATIONS = 5000;
    constexpr int WARMUP = 10;
    constexpr int RUNS = 100;
    
    printf("Configuration:\n");
    printf("  WARP_M=%d, MMA_M=%d, MMA_K=%d\n", WARP_M, MMA_M, MMA_K);
    printf("  Block size: %d threads\n", BLOCK_SIZE);
    printf("  Iterations: %d\n", ITERATIONS);
    printf("  Benchmark runs: %d\n\n", RUNS);
    
    printf("==================================================================\n");
    printf("Test 1: Compact Layout (BLOCK_K=16)\n");
    printf("Expected: Consecutive LDS reads, may use ds_read2_b64\n");
    printf("==================================================================\n");
    double time_compact = benchmark_kernel(
        benchmark_compact_layout<WARP_M, WARP_K_SMALL, MMA_M, MMA_K>,
        "Compact layout (K=16, consecutive access)",
        BLOCK_SIZE, ITERATIONS, WARMUP, RUNS
    );
    
    printf("\n==================================================================\n");
    printf("Test 2: Strided Layout (BLOCK_K=64)\n");
    printf("Expected: Strided LDS reads, may use ds_read2st64_b64\n");
    printf("==================================================================\n");
    double time_strided = benchmark_kernel(
        benchmark_strided_layout<WARP_M, WARP_K_LARGE, MMA_M, MMA_K>,
        "Strided layout (K=64, with stride)",
        BLOCK_SIZE, ITERATIONS, WARMUP, RUNS
    );
    
    printf("\n==================================================================\n");
    printf("Test 3: Transpose Access\n");
    printf("Expected: Strided/scattered reads, definitely needs special handling\n");
    printf("==================================================================\n");
    double time_transpose = benchmark_kernel(
        benchmark_transpose_layout<WARP_M, WARP_K_LARGE, MMA_M, MMA_K>,
        "Transpose access (cross-stride reads)",
        BLOCK_SIZE, ITERATIONS, WARMUP, RUNS
    );
    
    printf("\n==================================================================\n");
    printf("Summary:\n");
    printf("==================================================================\n");
    printf("Compact layout:       %8.3f us (baseline)\n", time_compact);
    printf("Strided layout:       %8.3f us (%.2f%% %s)\n", 
           time_strided,
           fabs((time_strided - time_compact) / time_compact * 100),
           time_strided > time_compact ? "slower" : "faster");
    printf("Transpose access:     %8.3f us (%.2f%% %s)\n",
           time_transpose,
           fabs((time_transpose - time_compact) / time_compact * 100),
           time_transpose > time_compact ? "slower" : "faster");
    
    printf("\n==================================================================\n");
    printf("Next steps:\n");
    printf("  1. Check generated assembly with: --show-asm flag\n");
    printf("  2. Use rocprof for detailed profiling:\n");
    printf("     rocprof --stats ./lds_layout_benchmark\n");
    printf("  3. Look for LDS bank conflicts and instruction mix\n");
    printf("==================================================================\n");
    
    return 0;
}

