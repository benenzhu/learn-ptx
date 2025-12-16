#include <cuda_fp16.h>
#include <cuda_bf16.h>
#include <cuda_pipeline.h>
#include "00_rtc.cu"




// template<int elts>
constexpr int elts = 16 * 16;
constexpr bool ASYNC_USE_PTX = false;

// https://zhuanlan.zhihu.com/p/1887108012579197523
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