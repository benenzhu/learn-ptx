#include "00_rtc.cu"
#include <cuda/barrier>
#include <cuda/ptx>
#include <cuda_fp16.h>
#include <cuda_pipeline.h>
using barrier = cuda::barrier<cuda::thread_scope_block>;
namespace cde = cuda::device::experimental;

// #define WIDTH 32
// #define HEIGHT 32
// #define W 16
// #define H 16

// #include <cuda_bf16.h>

template <int HEIGHT, int WIDTH, int TB_SIZE>
__device__ inline void global_to_shared(uint32_t dst, const nv_bfloat16 *src,
                                        int src_stride, int tid) {
  // 16 = 4 * 4 = float4
  constexpr int num_elems = 16 / sizeof(nv_bfloat16);
  constexpr int num_iters = HEIGHT * WIDTH / (TB_SIZE * num_elems);
  static_assert(HEIGHT * WIDTH % (TB_SIZE * num_elems) == 0); // no elem left.
  static_assert(WIDTH % num_elems == 0);

  for (int iter = 0; iter < num_iters; iter++) {
    const int idx = (iter * TB_SIZE + tid) * num_elems;
    const int row = idx / WIDTH;
    const int col = idx % WIDTH;

    const uint32_t dst_addr =
        dst + (row * WIDTH + col) * sizeof(nv_bfloat16); // shard memory stride: (WIDTH, 1)
    const nv_bfloat16 *src_addr =
        src + (row * src_stride + col); // global memory stride: (src_stride, 1)
                                        // input src should be the final addr.
    asm volatile("cp.async.cg.shared.global [%0], [%1], 16;" ::"r"(dst_addr), "l"(src_addr));
  }
  asm volatile("cp.async.commit_group;");
  asm volatile("cp.async.wait_all;");
  __syncthreads();
}

extern "C" __global__ void
tma_2d_kernel(const __grid_constant__ CUtensorMap tensor_map) {
  // The destination shared memory buffer of a bulk tensor operation should be
  // 128 byte aligned.
  __shared__ __align__(128) half smem[W][H];

  // The top-left corner of the tile is indicated by the indices x and y。
  int x, y;
  if (blockIdx.x == 0) {
    x = 0, y = 0;
  } else {
    x = 16, y = 16;
  }

// Initialize shared memory barrier with the number of threads participating in
// the barrier.
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
    cde::cp_async_bulk_tensor_2d_global_to_shared(&smem, &tensor_map, x, y,
                                                  bar);
    // Arrive on the barrier and tell how many bytes are expected to come in.
    token = cuda::device::barrier_arrive_tx(bar, 1, sizeof(smem));
  } else {
    // Other threads just arrive.
    token = bar.arrive();
  }
  // Wait for the data to have arrived.
  bar.wait(std::move(token));

  // Symbolically modify a value in shared memory.
  if (threadIdx.x == 0) {
    for (int i = 0; i < H; i++) {
      for (int j = 0; j < W; j++) {
        smem[i][j] = __hadd(smem[i][j], __float2half(0.5));
        // smem[i][j] += 0.1;
      }
    }
  }
  __syncthreads();

  if (threadIdx.x == 0 && blockIdx.x == 0) {
    printf("\ndata on block %d:\n", blockIdx.x);
    print_mem(&smem[0][0]);
  }

  // if(threadIdx.x == 0 && blockIdx.x == 0) {
  //     printf("\ndata on block 0:\n");
  //     print_mem(&smem[0][0]);
  // }

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
#ifdef __CUDACC_RTC__
#else
int main() { cuTensorMapEncodeTiled() }
#endif
