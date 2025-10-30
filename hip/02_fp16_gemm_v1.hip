














#include <hip/hip_runtime.h>
#include <hip/hip_bf16.h>

__device__ __host__
constexpr int cdiv(int a, int b) { return (a + b - 1) / b; }
#ifndef PYTHON_CALL
constexpr int M = 4096;
constexpr int N = 4096;
constexpr int K = 4096;
constexpr int NUM_WARP_M = 2;
constexpr int NUM_WARP_N = 2;
constexpr int BLOCK_M = 128;
constexpr int BLOCK_N = 128;
constexpr int BLOCK_K = 64;
constexpr int SMEM_STRIDE = 64;
#endif
constexpr int MMA_M = 16;
constexpr int MMA_N = 16;
constexpr int MMA_K = 16;
constexpr int WARP_SIZE = 64;

using s16x4 = short __attribute__((__vector_size__(4 * sizeof(short))));
using s16x8 = short __attribute__((__vector_size__(8 * sizeof(short))));
using fp32x4 = float __attribute__((__vector_size__(4 * sizeof(float))));

constexpr int CTA_SIZE = NUM_WARP_M * NUM_WARP_N * WARP_SIZE;
template <int HEIGHT=BLOCK_M, int WIDTH=BLOCK_K, int TB_SIZE=CTA_SIZE, typename T=__hip_bfloat16>
__device__
void gmem_to_smem(T *dst, const T *src, int src_stride, int tid) {
  constexpr int multiplier/*8*/ = sizeof(float4) / sizeof(T);
  static_assert((HEIGHT * WIDTH) % (TB_SIZE * multiplier) == 0);
  constexpr int num_iters /*8*/ = (HEIGHT/*128*/ * WIDTH/*128*/) / (TB_SIZE/*256*/ * multiplier/*8*/);

  for (int i = 0; i < num_iters; i++) {
    const int idx = (i * TB_SIZE + tid) * multiplier;
    const int row = idx / WIDTH;
    const int col = idx % WIDTH;

    const float4 data = reinterpret_cast<const float4 *>(src + (row * src_stride + col))[0];
    reinterpret_cast<float4 *>(dst + (row * SMEM_STRIDE + col))[0] = data;
  }
}

__launch_bounds__(NUM_WARP_M * NUM_WARP_N * WARP_SIZE)
__global__
void fp16_gemm_full_NTN(
  const __hip_bfloat16 *A_gmem,
  const __hip_bfloat16 *B_gmem,
        __hip_bfloat16 *C_gmem
) {
  constexpr int WARP_M /*64*/ = BLOCK_M /*128*/ / NUM_WARP_M /*2*/;
  constexpr int WARP_N /*64*/ = BLOCK_N /*128*/ / NUM_WARP_N /*2*/;
  constexpr int TB_SIZE /*256*/ = NUM_WARP_M /*2*/ * NUM_WARP_N /*2*/ * WARP_SIZE /*64*/;

  const int tid = threadIdx.x;
  const int warp_id /*[0,4)*/  = tid /*[0,256)*/ / WARP_SIZE /*64*/;
  const int lane_id /*[0,64)*/ = tid /*[0,256)*/ % WARP_SIZE /*64*/;

  const int warp_id_m /*[0,2)*/ = warp_id / NUM_WARP_N /*2*/;
  const int warp_id_n /*[0,2)*/ = warp_id % NUM_WARP_N /*2*/;

  const int bid /*[0,1024)*/ = blockIdx.x;
  const int grid_m /*32*/ = cdiv(M /*4096*/, BLOCK_M /*128*/);
  const int grid_n /*32*/ = cdiv(N /*4096*/, BLOCK_N /*128*/);

  int bid_m /*[0,32)*/ = bid /*[0,1024)*/ / grid_n /*32*/; //TODO: pid remapping here.
  int bid_n /*[0,32)*/= bid % grid_n /*32*/;

  const int offset_m /*[0,4096)*/ = bid_m * BLOCK_M /*128*/;
  const int offset_n /*[0,4096)*/ = bid_n * BLOCK_N /*128*/;
  A_gmem += offset_m * K /*4096*/;
  B_gmem += offset_n * K /*4096*/;
  // To the position to calculate.
  C_gmem += (offset_m + warp_id_m * WARP_M /*256*/) * N /*4096*/ + (offset_n + warp_id_n /*[0,2)*/ * WARP_N /*256*/);

  // shared memory
  extern __shared__ __hip_bfloat16 smem[];
  __hip_bfloat16 *A_smem = smem;
  __hip_bfloat16 *B_smem = A_smem + BLOCK_M /*128*/ * SMEM_STRIDE /*64*/;

  // pre-compute offset for smem->rmem  // did this add some registers for cpp?
  // 128 * 64
  // / 256 = 32 per thread. 4 * float4
  __hip_bfloat16 *A_smem_thread, *B_smem_thread;
  {
    const int row = (warp_id_m /*[0,2)*/ * WARP_M /*64*/) + (lane_id /*[0,64)*/ % 16 /*16*/);
    const int col = (lane_id /*[0,64)*/ / 16 /*16*/) * 4 /*4*/;
    A_smem_thread = A_smem + row * SMEM_STRIDE + col;
  }
  {
    const int row = (warp_id_n * WARP_N) + (lane_id % 16);
    const int col = (lane_id / 16) * 4;
    B_smem_thread = B_smem + row * SMEM_STRIDE + col;
  }

  // register memory
  // do it this way to use mfma intrinsic
  s16x4 A_rmem[WARP_M /*64*/ / MMA_M /*16*/][BLOCK_K/*64*/ / MMA_K/*16*/];  // [4][4]
  s16x4 B_rmem[WARP_N / MMA_N][BLOCK_K / MMA_K];                            // [4][4]
  fp32x4 C_rmem[WARP_M / MMA_M][WARP_N / MMA_N] = {};                  // 4 * 4? only 16?

  const int num_k_iters = cdiv(K, BLOCK_K);
  for (int iter_k = 0; iter_k < num_k_iters; iter_k++) {
    // gmem->smem
    __syncthreads();
    gmem_to_smem<BLOCK_M, BLOCK_K, TB_SIZE>(A_smem, A_gmem, K, tid);
    gmem_to_smem<BLOCK_N, BLOCK_K, TB_SIZE>(B_smem, B_gmem, K, tid);
    A_gmem += BLOCK_K;
    B_gmem += BLOCK_K;

    // smem->rmem
    // K_L = 16 / (64 / (16 * 1)) = 16 / 4 = 4
    // A: lane (k / 4) * M + i
    // B: lane (k / 4) * N + j
    // TODO: use wider load?
    __syncthreads();
    for (int mma_id_m = 0; mma_id_m < WARP_M / MMA_M; mma_id_m++)
      for (int mma_id_k = 0; mma_id_k < BLOCK_K / MMA_K; mma_id_k++) {
        const int row = mma_id_m * MMA_M;
        const int col = mma_id_k * MMA_K;
        __hip_bfloat16 *addr = A_smem_thread + (row * SMEM_STRIDE + col);
        A_rmem[mma_id_m][mma_id_k] = reinterpret_cast<s16x4 *>(addr)[0];
      }
    for (int mma_id_n = 0; mma_id_n < WARP_N / MMA_N; mma_id_n++)
      for (int mma_id_k = 0; mma_id_k < BLOCK_K / MMA_K; mma_id_k++) {
        const int row = mma_id_n * MMA_N;
        const int col = mma_id_k * MMA_K;
        __hip_bfloat16 *addr = B_smem_thread + (row * SMEM_STRIDE + col);
        B_rmem[mma_id_n][mma_id_k] = reinterpret_cast<s16x4 *>(addr)[0];
      }

    // mma
    // https://github.com/ROCm/composable_kernel/blob/rocm-7.0.1/include/ck/utility/amd_xdlops.hpp
    // https://github.com/tile-ai/tilelang/blob/v0.1.6.post1/src/tl_templates/hip/gemm.h
    // TODO: swap A and B like in tilelang for better C layout
    for (int mma_id_m = 0; mma_id_m < WARP_M /*64*/ / MMA_M/*16*/; mma_id_m++)
      for (int mma_id_n = 0; mma_id_n < WARP_N /*64*/ / MMA_N /*16*/; mma_id_n++)
        for (int mma_id_k = 0; mma_id_k < BLOCK_K /*64*/ / MMA_K /*16*/; mma_id_k++)
          C_rmem[mma_id_m][mma_id_n] = __builtin_amdgcn_mfma_f32_16x16x16bf16_1k(A_rmem[mma_id_m][mma_id_k],
                                                                                 B_rmem[mma_id_n][mma_id_k],
                                                                                 C_rmem[mma_id_m][mma_id_n],
                                                                                 0, 0, 0);
  }

  // H = 4
  // B_I = ceil(64 / (N * M / H)) = ceil(64 / (16 * 16 / 4)) = 1
  // M_I = (64 / B_I) / N = 4
  // G = M / (H * M_I) = 1
  // item: (i % 4)
  // lane: (i / 4) * N + j
  __syncthreads();
  // 4 * 4 * 4 = 64 elements per warp. this can be optimized somehow.  But we can also enlarge K now..
  for (int mma_id_m = 0; mma_id_m < WARP_M / MMA_M; mma_id_m++)
    for (int mma_id_n = 0; mma_id_n < WARP_N / MMA_N; mma_id_n++) {
      const int row = mma_id_m * MMA_M + (lane_id / 16) * 4;
      const int col = mma_id_n * MMA_N + (lane_id % 16);

      fp32x4 data = C_rmem[mma_id_m][mma_id_n];
      C_gmem[(row + 0) * N + col] = __float2bfloat16(data[0]);
      C_gmem[(row + 1) * N + col] = __float2bfloat16(data[1]);
      C_gmem[(row + 2) * N + col] = __float2bfloat16(data[2]);
      C_gmem[(row + 3) * N + col] = __float2bfloat16(data[3]);
    }
}