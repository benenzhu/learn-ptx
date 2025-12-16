#include <cuda_fp16.h>
#include "00_rtc.cu"
#include <cuda_pipeline.h>
#include <cuda/barrier>
#include <cuda/ptx>
using barrier = cuda::barrier<cuda::thread_scope_block>;
namespace cde = cuda::device::experimental;


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


