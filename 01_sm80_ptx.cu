// #include <stdint.h>
// #include <stdio.h>
#include <cuda_fp16.h>
#include <cuda_bf16.h>
#include <cuda_pipeline.h>
#include "00_rtc.cu"



/* simple kernel, how to use nvrtc and call it in python*/
__global__ __launch_bounds__(256) void add_kernel(const float *A, const float *B, float *C, int size) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size / 4){
        float4 a = reinterpret_cast<const float4*>(A)[idx];
        float4 b = reinterpret_cast<const float4*>(B)[idx];
        float4 c = reinterpret_cast<float4*>(C)[idx];
        c.x = a.x + b.x;
        c.y = a.y + b.y;
        c.z = a.z + b.z;
        c.w = a.w + b.w;
        reinterpret_cast<float4*>(C)[idx] = c;
    }
}


// template<int elts>
constexpr int elts = 16 * 16;
constexpr bool ASYNC_USE_PTX = false;

// https://zhuanlan.zhihu.com/p/1887108012579197523
__global__ __launch_bounds__(32) void async_cp_kernel(half *ptr) {
    __shared__ half smem[elts];
    for (int i = 0; i < elts; i += blockDim.x) {
        smem[i] =  __float2half(0.0);
    }
    __syncthreads();
    
    print_mem(ptr);
    // half *src = ptr + threadIdx.x / 2 * 16 + threadIdx.x % 2 * 8;
    half *src = ptr + threadIdx.x * 8;
    half *dst = smem + threadIdx.x * 8;
    // 每个 thread 拷贝自己的 n 个过来, 最多 16 个Byte 单条命令?
/*
 * https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#data-movement-and-conversion-instructions-non-bulk-copy
cp.async.ca.shared{::cta}.global{.level::cache_hint}{.level::prefetch_size}
                         [dst], [src], cp-size{, src-size}{, cache-policy} ;
cp.async.cg.shared{::cta}.global{.level::cache_hint}{.level::prefetch_size}
                         [dst], [src], 16{, src-size}{, cache-policy} ;
cp.async.ca.shared{::cta}.global{.level::cache_hint}{.level::prefetch_size}
                         [dst], [src], cp-size{, ignore-src}{, cache-policy} ;
cp.async.cg.shared{::cta}.global{.level::cache_hint}{.level::prefetch_size}
                         [dst], [src], 16{, ignore-src}{, cache-policy} ;

.level::cache_hint =     { .L2::cache_hint }
.level::prefetch_size =  { .L2::64B, .L2::128B, .L2::256B }
cp-size =                { 4, 8, 16 }
*/
    if constexpr (ASYNC_USE_PTX){
        int addr = __cvta_generic_to_shared(dst);

        asm("cp.async.cg.shared.global [%0], [%1], 16;\n"
            : 
            : "r"(addr), "l"(src)
        );
        asm volatile("cp.async.commit_group;\n" ::);
        asm volatile("cp.async.wait_group 0;\n" ::);
    } else { 
        __pipeline_memcpy_async(dst, src, 16);
        __pipeline_commit();
        __pipeline_wait_prior(0); 
    }
    __syncthreads();

    /// print smem
    print_mem(smem);
}






/*
传入 ld_matrix的顺序, 所以传入一个是 8 个元素...
m8n8 x4? means for that?
0      0
1     16
2     32
3     48
4     64
5     80
6     96
7    112
8    128
9    144
10    160
11    176
12    192
13    208
14    224
15    240
16      8
17     24
18     40
19     56
20     72
21     88
22    104
23    120
24    136
25    152
26    168
27    184
28    200
29    216
30    232
31    248
*/

__global__ void ld_matrix_kernel(half *d_ptr) {
    constexpr int elts = 4*8*8;
    __shared__ half smem[elts];

    /// init and print smem
    int tid = threadIdx.x;
    if(tid == 0) {
        for(int i = 0; i < elts; i ++) {
            smem[i] = __float2half(i);
        }
        print_mem(smem);
    }

    /// ldmatrix
    uint32_t regs[4];
    half *ptr = smem + tid % 16 * 16 + tid / 16 * 8;
    printf("%d %6.lf\n", threadIdx.x, __half2float(ptr[0]));
    uint32_t addr = __cvta_generic_to_shared(ptr);
    asm("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];\n"
        : "=r"(regs[0]), "=r"(regs[1]), "=r"(regs[2]), "=r"(regs[3])
        : "r"(addr)
    );

    /// print thread regs
    if(tid == 0) {
        printf("\non device:\n");

        for(int i = 0; i < 4; i ++) {
            half *data = reinterpret_cast<half*>(&regs[i]);
            printf("%6.lf, %6.lf\n", __half2float(data[0]), __half2float(data[1]));
        }
    }
    if(tid < 32){
        int row = tid / 4;
        int col = tid % 4;
        reinterpret_cast<uint32_t*>(d_ptr)[row       * 8 + col] = regs[0];
        reinterpret_cast<uint32_t*>(d_ptr)[(row + 8) * 8 + col] = regs[1];
        reinterpret_cast<uint32_t*>(d_ptr)[row       * 8 + col + 4] = regs[2];
        reinterpret_cast<uint32_t*>(d_ptr)[(row + 8) * 8 + col + 4] = regs[3]; 
    }
    __syncthreads();
    print_mem(d_ptr);
}



