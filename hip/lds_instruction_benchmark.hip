// Microbenchmark for comparing ds_read2_b64 vs ds_read2st64_b64 performance
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

// Kernel using ds_read2st64_b64 (stride = 64 * 8 = 512 bytes)
__global__ void benchmark_ds_read2st64_b64(float *output, int iterations) {
    __shared__ float smem[2048];  // 8KB LDS
    
    int tid = threadIdx.x;
    int lane_id = tid % 64;
    
    // Initialize shared memory
    for (int i = tid; i < 2048; i += blockDim.x) {
        smem[i] = (float)i;
    }
    __syncthreads();
    
    // Base address calculation
    int base_addr = (lane_id % 16) * 2;  // Each thread has different base
    
    double sum = 0.0;
    
    // Inline assembly to use ds_read2st64_b64
    // This instruction reads from addr and addr + 64*8*offset1
    for (int iter = 0; iter < iterations; iter++) {
        long long v01, v23;
        
        // Read with stride 64 (64 * 8 bytes = 512 bytes apart)
        asm volatile(
            "ds_read2st64_b64 %0, %2 offset1:4"
            : "=v"(v01), "=v"(v23)
            : "v"(base_addr * 4)  // byte address
            : "memory"
        );
        
        float *vptr = (float*)&v01;
        sum += vptr[0] + vptr[1];
        vptr = (float*)&v23;
        sum += vptr[0] + vptr[1];
        
        // Add more reads to stress the LDS
        asm volatile(
            "ds_read2st64_b64 %0, %2 offset0:8 offset1:12"
            : "=v"(v01), "=v"(v23)
            : "v"(base_addr * 4)
            : "memory"
        );
        vptr = (float*)&v01;
        sum += vptr[0] + vptr[1];
        vptr = (float*)&v23;
        sum += vptr[0] + vptr[1];
    }
    
    // Prevent optimization
    if (tid == 0) {
        output[0] = (float)sum;
    }
}

// Kernel using ds_read2_b64 (standard offset, 8 bytes per offset)
__global__ void benchmark_ds_read2_b64(float *output, int iterations) {
    __shared__ float smem[2048];
    
    int tid = threadIdx.x;
    int lane_id = tid % 64;
    
    // Initialize shared memory
    for (int i = tid; i < 2048; i += blockDim.x) {
        smem[i] = (float)i;
    }
    __syncthreads();
    
    int base_addr = (lane_id % 16) * 2;
    
    double sum = 0.0;
    
    // Inline assembly to use ds_read2_b64
    // This instruction reads from addr + offset0*8 and addr + offset1*8
    for (int iter = 0; iter < iterations; iter++) {
        long long v01, v23;
        
        // Read with simple offsets (in units of 8 bytes)
        asm volatile(
            "ds_read2_b64 %0, %2 offset1:4"
            : "=v"(v01), "=v"(v23)
            : "v"(base_addr * 4)  // byte address
            : "memory"
        );
        
        float *vptr = (float*)&v01;
        sum += vptr[0] + vptr[1];
        vptr = (float*)&v23;
        sum += vptr[0] + vptr[1];
        
        asm volatile(
            "ds_read2_b64 %0, %2 offset0:8 offset1:12"
            : "=v"(v01), "=v"(v23)
            : "v"(base_addr * 4)
            : "memory"
        );
        vptr = (float*)&v01;
        sum += vptr[0] + vptr[1];
        vptr = (float*)&v23;
        sum += vptr[0] + vptr[1];
    }
    
    if (tid == 0) {
        output[0] = (float)sum;
    }
}

// More intensive version with computation to hide latency
__global__ void benchmark_ds_read2st64_b64_with_compute(float *output, int iterations) {
    __shared__ float smem[2048];
    
    int tid = threadIdx.x;
    int lane_id = tid % 64;
    
    for (int i = tid; i < 2048; i += blockDim.x) {
        smem[i] = (float)i;
    }
    __syncthreads();
    
    int base_addr = (lane_id % 16) * 2;
    double sum = 0.0;
    
    for (int iter = 0; iter < iterations; iter++) {
        long long v01, v23, w01, w23;
        
        // Issue multiple reads
        asm volatile(
            "ds_read2st64_b64 %0, %2 offset1:4"
            : "=v"(v01), "=v"(v23)
            : "v"(base_addr * 4)
            : "memory"
        );
        
        asm volatile(
            "ds_read2st64_b64 %0, %2 offset0:8 offset1:12"
            : "=v"(w01), "=v"(w23)
            : "v"((base_addr + 32) * 4)
            : "memory"
        );
        
        // Some computation to hide latency (like MFMA would)
        float *vptr = (float*)&v01;
        float *wptr = (float*)&w01;
        sum += vptr[0] * wptr[0] + vptr[1] * wptr[1];
        vptr = (float*)&v23;
        wptr = (float*)&w23;
        sum += vptr[0] * wptr[0] + vptr[1] * wptr[1];
        
        vptr = (float*)&v01;
        wptr = (float*)&w01;
        sum += vptr[0] + vptr[1] + wptr[0] + wptr[1];
        vptr = (float*)&v23;
        wptr = (float*)&w23;
        sum += vptr[0] + vptr[1] + wptr[0] + wptr[1];
    }
    
    if (tid == 0) {
        output[0] = (float)sum;
    }
}

__global__ void benchmark_ds_read2_b64_with_compute(float *output, int iterations) {
    __shared__ float smem[2048];
    
    int tid = threadIdx.x;
    int lane_id = tid % 64;
    
    for (int i = tid; i < 2048; i += blockDim.x) {
        smem[i] = (float)i;
    }
    __syncthreads();
    
    int base_addr = (lane_id % 16) * 2;
    double sum = 0.0;
    
    for (int iter = 0; iter < iterations; iter++) {
        long long v01, v23, w01, w23;
        
        asm volatile(
            "ds_read2_b64 %0, %2 offset1:4"
            : "=v"(v01), "=v"(v23)
            : "v"(base_addr * 4)
            : "memory"
        );
        
        asm volatile(
            "ds_read2_b64 %0, %2 offset0:8 offset1:12"
            : "=v"(w01), "=v"(w23)
            : "v"((base_addr + 32) * 4)
            : "memory"
        );
        
        float *vptr = (float*)&v01;
        float *wptr = (float*)&w01;
        sum += vptr[0] * wptr[0] + vptr[1] * wptr[1];
        vptr = (float*)&v23;
        wptr = (float*)&w23;
        sum += vptr[0] * wptr[0] + vptr[1] * wptr[1];
        
        vptr = (float*)&v01;
        wptr = (float*)&w01;
        sum += vptr[0] + vptr[1] + wptr[0] + wptr[1];
        vptr = (float*)&v23;
        wptr = (float*)&w23;
        sum += vptr[0] + vptr[1] + wptr[0] + wptr[1];
    }
    
    if (tid == 0) {
        output[0] = (float)sum;
    }
}

double benchmark_kernel(void (*kernel)(float*, int), const char* name, 
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
    
    printf("%-50s: %8.3f us (avg over %d runs)\n", name, avg_time_us, bench_runs);
    
    CHECK_HIP(hipFree(d_output));
    return avg_time_us;
}

int main() {
    // Get device properties
    hipDeviceProp_t prop;
    CHECK_HIP(hipGetDeviceProperties(&prop, 0));
    printf("==========================================================\n");
    printf("Device: %s\n", prop.name);
    printf("LDS per CU: %zu KB\n", prop.sharedMemPerBlock / 1024);
    printf("==========================================================\n\n");
    
    const int BLOCK_SIZE = 256;  // 4 wavefronts
    const int ITERATIONS = 10000;
    const int WARMUP_RUNS = 10;
    const int BENCH_RUNS = 100;
    
    printf("Configuration:\n");
    printf("  Block size: %d threads (%d wavefronts)\n", BLOCK_SIZE, BLOCK_SIZE / 64);
    printf("  Iterations per kernel: %d\n", ITERATIONS);
    printf("  Warmup runs: %d\n", WARMUP_RUNS);
    printf("  Benchmark runs: %d\n\n", BENCH_RUNS);
    
    printf("==========================================================\n");
    printf("Test 1: Pure LDS read throughput (latency bound)\n");
    printf("==========================================================\n");
    
    double time_st64 = benchmark_kernel(
        benchmark_ds_read2st64_b64,
        "ds_read2st64_b64 (stride=64)",
        BLOCK_SIZE, ITERATIONS, WARMUP_RUNS, BENCH_RUNS
    );
    
    double time_normal = benchmark_kernel(
        benchmark_ds_read2_b64,
        "ds_read2_b64 (standard)",
        BLOCK_SIZE, ITERATIONS, WARMUP_RUNS, BENCH_RUNS
    );
    
    printf("\nSpeedup: %.2fx (%s is faster)\n", 
           time_st64 / time_normal,
           time_st64 > time_normal ? "ds_read2_b64" : "ds_read2st64_b64");
    
    printf("\n==========================================================\n");
    printf("Test 2: LDS read with computation (throughput bound)\n");
    printf("==========================================================\n");
    
    double time_st64_compute = benchmark_kernel(
        benchmark_ds_read2st64_b64_with_compute,
        "ds_read2st64_b64 + compute",
        BLOCK_SIZE, ITERATIONS, WARMUP_RUNS, BENCH_RUNS
    );
    
    double time_normal_compute = benchmark_kernel(
        benchmark_ds_read2_b64_with_compute,
        "ds_read2_b64 + compute",
        BLOCK_SIZE, ITERATIONS, WARMUP_RUNS, BENCH_RUNS
    );
    
    printf("\nSpeedup: %.2fx (%s is faster)\n",
           time_st64_compute / time_normal_compute,
           time_st64_compute > time_normal_compute ? "ds_read2_b64" : "ds_read2st64_b64");
    
    printf("\n==========================================================\n");
    printf("Summary:\n");
    printf("==========================================================\n");
    printf("Pure LDS read:\n");
    printf("  ds_read2st64_b64: %.3f us\n", time_st64);
    printf("  ds_read2_b64:     %.3f us\n", time_normal);
    printf("  Difference:       %.3f us (%.2f%%)\n", 
           time_st64 - time_normal,
           (time_st64 - time_normal) / time_normal * 100);
    
    printf("\nWith computation:\n");
    printf("  ds_read2st64_b64: %.3f us\n", time_st64_compute);
    printf("  ds_read2_b64:     %.3f us\n", time_normal_compute);
    printf("  Difference:       %.3f us (%.2f%%)\n",
           time_st64_compute - time_normal_compute,
           (time_st64_compute - time_normal_compute) / time_normal_compute * 100);
    
    printf("\n==========================================================\n");
    
    return 0;
}

