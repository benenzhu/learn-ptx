#ifdef __CUDACC_RTC__
using int8_t = signed char;
using uint8_t = unsigned char;
using int16_t = signed short;
using uint16_t = unsigned short;
using int32_t = signed int;
using uint32_t = unsigned int;
using int64_t = signed long long;
using uint64_t = unsigned long long;
using cuuint64_t = unsigned long long;

namespace std {
template <class T, T v>
struct integral_constant {
  static constexpr T value = v;
  using value_type = T;
  using type = integral_constant;  // using injected-class-name

  __device__ constexpr operator value_type() const noexcept { return value; }

  __device__ constexpr value_type operator()() const noexcept { return value; }  // since c++14
};

using false_type = integral_constant<bool, false>;
using true_type = integral_constant<bool, true>;

template <class T, class U>
struct is_same : false_type {};

template <class T>
struct is_same<T, T> : true_type {};

template <class T, class U>
inline constexpr bool is_same_v = is_same<T, U>::value;
} 
#define CU_TENSOR_MAP_NUM_QWORDS 16

struct CUtensorMap_st {
#if defined(__cplusplus) && (__cplusplus >= 201103L)
    alignas(64)
#elif __STDC_VERSION__ >= 201112L
    _Alignas(64)
#endif
        cuuint64_t opaque[CU_TENSOR_MAP_NUM_QWORDS];
};

using CUtensorMap = CUtensorMap_st;

#else
#include <cuda.h>
#include <string>
#endif
#include <cuda_fp16.h>
#include <cuda_pipeline.h>
#include <cuda/barrier>
#include <cuda/ptx>
// typedef struct CUtensorMap_st {
//     #if defined(__cplusplus) && (__cplusplus >= 201103L)
//         alignas(64)
//     #elif __STDC_VERSION__ >= 201112L
//         _Alignas(64)
//     #endif
//         cuuint64_t opaque[CU_TENSOR_MAP_NUM_QWORDS];
// } CUtensorMap;



__device__ void print_mem(half *ptr, int row=16, int col=16){
    // __syncthreads();
    if(threadIdx.x == 0) {
    
        // printf("gmem data:\n");

        for(int i = 0; i < row; i ++) {
            if(i % 8 == 0 && i != 0) {
                printf("\n");
            }
            for(int j = 0; j < col; j++) {
                if(j % 8 == 0 && j != 0) {
                    printf("  ");
                }
                printf("%6.1lf ",  __half2float(ptr[i*col+j]));
            }
            printf("\n");
        }
    }
}
// nvcc -o mma mma.cu -O2 -arch=sm_80 -std=c++17 -lcublas

// a: row major
// b: col major
// c: row major
// template <int elts, bool print>
constexpr int elts = 16 * 16;
constexpr bool print = true;
__global__ void mma_ptx_kernel(half *c_ptr, half *a_ptr, half *b_ptr, half *d_ptr) {
    int tid = threadIdx.x;

    // ldgsts
    __shared__ half smem_a[elts];
    __shared__ half smem_b[elts];

    half *src = a_ptr + tid / 2 * 16 + tid % 2 * 8;
    half *dst = smem_a + tid * 8;
    uint32_t addr = __cvta_generic_to_shared(dst);
    asm("cp.async.cg.shared.global [%0], [%1], 16;\n"
        :
        :"r"(addr), "l"(src)
    );
    asm("cp.async.commit_group;\n"::);

    src = b_ptr + tid / 2 * 16 + tid % 2 * 8;
    dst = smem_b + tid * 8;
    addr = __cvta_generic_to_shared(dst);
    asm("cp.async.cg.shared.global [%0], [%1], 16;\n"
        :
        :"r"(addr), "l"(src)
    );
    asm("cp.async.commit_group;\n"::);

    asm("cp.async.wait_group 0;\n"::);
    __syncthreads();

    if(tid == 0 && print) {
        printf("\nsmem_a:\n");
        print_mem(smem_a);


        printf("\nsmem_b:\n");
        print_mem(smem_b);

    }

    // ldmatrix
    uint32_t a_regs[4];
    uint32_t b_regs[4];
    src = smem_a + tid % 16 * 16 + tid / 16 * 8;
    /*	mul.wide.s32 	%rd57, %r97, 2;
        add.s64 	%rd59, %rd27, %rd57;
        cvt.u32.u64 	%r83, %rd59;*/

    addr = __cvta_generic_to_shared(src);
    asm("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];\n"
        :"=r"(a_regs[0]), "=r"(a_regs[1]), "=r"(a_regs[2]), "=r"(a_regs[3])
        :"r"(addr)
    );

    src = smem_b + tid % 16 * 16 + tid / 16 * 8;
    addr = __cvta_generic_to_shared(src);
    asm("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];\n"
        :"=r"(b_regs[0]), "=r"(b_regs[1]), "=r"(b_regs[2]), "=r"(b_regs[3])
        :"r"(addr)
    );

    if(tid == 0 && print) {
        printf("\na_regs:\n");

        for(int i = 0; i < 4; i ++) {
            half *data = reinterpret_cast<half*>(&a_regs[i]);
            printf("%6.2lf, %6.2lf\n", __half2float(data[0]), __half2float(data[1]));
        }

        printf("\nb_regs:\n");
        for(int i = 0; i < 4; i ++) {
            half *data = reinterpret_cast<half*>(&b_regs[i]);
            printf("%6.2lf, %6.2lf\n", __half2float(data[0]), __half2float(data[1]));
        }
    }

    // mma
    float accum[8] = {0};

    asm volatile(
        "mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 "
        "{%0, %1, %2, %3}, "
        "{%4, %5, %6, %7}, "
        "{%8, %9}, "
        "{%10, %11, %12, %13};\n"
        : "=f"(accum[0]),"=f"(accum[1]),"=f"(accum[4]),"=f"(accum[5])
        : "r"(a_regs[0]),"r"(a_regs[1]),"r"(a_regs[2]),"r"(a_regs[3]),
          "r"(b_regs[0]),"r"(b_regs[2]),
          "f"(accum[0]),"f"(accum[1]),"f"(accum[4]),"f"(accum[5])
    );

    asm volatile(
        "mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 "
        "{%0, %1, %2, %3}, "
        "{%4, %5, %6, %7}, "
        "{%8, %9}, "
        "{%10, %11, %12, %13};\n"
        : "=f"(accum[2]),"=f"(accum[3]),"=f"(accum[6]),"=f"(accum[7])
        : "r"(a_regs[0]),"r"(a_regs[1]),"r"(a_regs[2]),"r"(a_regs[3]),
          "r"(b_regs[1]),"r"(b_regs[3]),
          "f"(accum[2]),"f"(accum[3]),"f"(accum[6]),"f"(accum[7])
    );
    
    if(tid == 0 && print) {
        printf("\naccum:\n");
        for(int i = 0; i < 8; i ++) {
            printf("%.2lf ", accum[i]);
        }
        printf("\n");
    }

    // sts
    __shared__ half smem_c[elts];
    int row = tid / 4;
    int col = tid % 4 * 2;
    for(int i = 0; i < 2; i ++) {
        for(int j = 0; j < 2; j ++) {
            smem_c[(i*8+row)*16+(j*8+col)+0] = __float2half(accum[i*4+j*2+0]);
            smem_c[(i*8+row)*16+(j*8+col)+1] = __float2half(accum[i*4+j*2+1]);
        }
    }
    __syncthreads();

    // stg
    src = smem_c + tid / 2 * 16 + tid % 2 * 8;
    dst = c_ptr + tid / 2 * 16 + tid % 2 * 8;
    for(int i = 0; i < 8; i ++) {
        dst[i] = src[i];
    }
    if(tid == 0){
        printf("\nc_ptr:\n");
        print_mem(c_ptr);
        printf("\nd_ptr:\n");
        print_mem(d_ptr);
    }
}






using barrier = cuda::barrier<cuda::thread_scope_block>;
namespace cde = cuda::device::experimental;

// nvcc -o tma tma.cu -O2 -arch=sm_90a -std=c++17 && ./tma

__launch_bounds__(32*4)
__global__ void tma_1d_kernel(half* ptr, int elts)
{
  // Shared memory buffer. The destination shared memory buffer of
  // a bulk operations should be 16 byte aligned.
 extern __shared__ __align__(16) half smem[];
  
  ////////////////// global mem -> shared mem //////////////////
  // 1. a) Initialize shared memory barrier with the number of threads participating in the barrier.
  //    b) Make initialized barrier visible in async proxy.
  #pragma nv_diag_suppress static_var_with_dynamic_init
  __shared__ barrier bar;
  if (threadIdx.x == 0) { 
/*  mov.u64 	%rd9, _ZZ13tma_1d_kernelP6__halfiE3bar;
	cvt.u32.u64 	%r10, %rd9;
    mbarrier.init.shared.b64 [%r10], %r11;*/
    init(&bar, blockDim.x);                      // a)
/*  fence.proxy.async.shared::cta; */
    cde::fence_proxy_async_shared_cta();         // b)
  }
/* bar.sync 	0; */
  __syncthreads();

  // 2. Initiate TMA transfer to copy global to shared memory.
  if (threadIdx.x == 0) {
    // 3a. cuda::memcpy_async arrives on the barrier and communicates
    //     how many bytes are expected to come in (the transaction count)
/*
 * cp.async.bulk.shared::cluster.global.mbarrier::complete_tx::bytes [%r12], [%rd10], %r16, [%r15]
 * mbarrier.expect_tx.relaxed.cta.shared::cta.b64 [%r15], %r16;
*/
    cuda::memcpy_async(
        smem, 
        ptr,
        cuda::aligned_size_t<16>(sizeof(half)*elts),
        bar
    );
  }
  // 3b. All threads arrive on the barrier
/*
  mbarrier.arrive.shared::cta.b64                             %rd13,  [%r17], %r18; 
*/
  barrier::arrival_token token = bar.arrive();
  
  // 3c. Wait for the data to have arrived.
/* ??? wait + sleep? .............. 这么长的代码?
 * 	mov.u64 %rd14, %globaltimer;
 * 	mbarrier.try_wait.shared.b64 p, [%r17], %rd13;	
 *  selp.b32 %r20, 1, 0, p;
 * 	
*/
  bar.wait(std::move(token));

  // 4. Compute saxpy and write back to shared memory
  for (int i = threadIdx.x; i < elts; i += blockDim.x) {
  /*add.f16 %rs2,%rs3,%rs1*/
    smem[i] = __hadd(smem[i], __float2half(1.0));
  }
  
  ////////////////// shared mem -> global mem //////////////////
  // 5. Wait for shared memory writes to be visible to TMA engine.
// 同上
  cde::fence_proxy_async_shared_cta();   // b)
  __syncthreads();
  // After syncthreads, writes by all threads are visible to TMA engine.

  if(threadIdx.x == 0) {
    printf("\ndata on device: %d\n", elts);
    print_mem(smem, 32, 32);
    // for(int i = 0; i < elts; i ++) {
    //     printf("%.2lf ", __half2float(smem[i]));
    // }
  }

  // 6. Initiate TMA transfer to copy shared memory to global memory
  if (threadIdx.x == 0) {
/*	cp.async.bulk.global.shared::cta.bulk_group [%rd36], [%r62], %r63; */
    cde::cp_async_bulk_shared_to_global(
            ptr, smem, sizeof(half)*elts);
    // 7. Wait for TMA transfer to have finished reading shared memory.
    // Create a "bulk async-group" out of the previous bulk copy operation.
/*  cp.async.bulk.commit_group;*/
    cde::cp_async_bulk_commit_group();
    // Wait for the group to have completed reading from shared memory.
/*  cp.async.bulk.wait_group.read 0 */
    cde::cp_async_bulk_wait_group_read<0>();
  }
}


#define WIDTH 32
#define HEIGHT 32
#define W 16
#define H 16

extern "C"
__global__ void tma_2d_kernel(const __grid_constant__ CUtensorMap tensor_map) {
    // The destination shared memory buffer of a bulk tensor operation should be
    // 128 byte aligned.
    __shared__ __align__(128) half smem[W][H];

    // The top-left corner of the tile is indicated by the indices x and y。
    int x, y;
    if(blockIdx.x == 0) {
        x = 0, y = 0;
    } else {
        x = 16, y = 16;
    }

    // Initialize shared memory barrier with the number of threads participating in the barrier.
    #pragma nv_diag_suppress static_var_with_dynamic_init
    __shared__ barrier bar;

    if (threadIdx.x == 0) {
        // Initialize barrier. All `blockDim.x` threads in block participate.
        init(&bar, blockDim.x);
        // Make initialized barrier visible in async proxy.
        cde::fence_proxy_async_shared_cta();
    }
    // Syncthreads so initialized barrier is visible to all threads.
    __syncthreads();

    barrier::arrival_token token;
    if (threadIdx.x == 0) {
        // Initiate bulk tensor copy.
        cde::cp_async_bulk_tensor_2d_global_to_shared(&smem, &tensor_map, x, y, bar);
        // Arrive on the barrier and tell how many bytes are expected to come in.
        token = cuda::device::barrier_arrive_tx(bar, 1, sizeof(smem));
    } else {
        // Other threads just arrive.
        token = bar.arrive();
    }
    // Wait for the data to have arrived.
    bar.wait(std::move(token));

    // Symbolically modify a value in shared memory.
    if(threadIdx.x == 0) {
        for(int i = 0; i < H; i ++) {
            for(int j = 0; j < W; j ++) {
                smem[i][j] = __hadd(smem[i][j], __float2half(0.1));
                // smem[i][j] += 0.1;
            }
        }
    }
    __syncthreads();

    if(threadIdx.x == 0 && blockIdx.x == 1) {
        printf("\ndata on device:\n");
        for(int i = 0; i < H; i ++) {
            for(int j = 0; j < W; j ++) {
                printf("%.2lf ", __half2float(smem[i][j]));
            }
            printf("\n");
        }
    }

    // Wait for shared memory writes to be visible to TMA engine.
    cde::fence_proxy_async_shared_cta();
    __syncthreads();
    // After syncthreads, writes by all threads are visible to TMA engine.

    // Initiate TMA transfer to copy shared memory to global memory
    if (threadIdx.x == 0) {
        cde::cp_async_bulk_tensor_2d_shared_to_global(&tensor_map, x, y, &smem);
        // Wait for TMA transfer to have finished reading shared memory.
        // Create a "bulk async-group" out of the previous bulk copy operation.
        cde::cp_async_bulk_commit_group();
        // Wait for the group to have completed reading from shared memory.
        cde::cp_async_bulk_wait_group_read<0>();
    }

    // Destroy barrier. This invalidates the memory region of the barrier. If
    // further computations were to take place in the kernel, this allows the
    // memory location of the shared memory barrier to be reused.
    if (threadIdx.x == 0) {
        (&bar)->~barrier();
    }
}