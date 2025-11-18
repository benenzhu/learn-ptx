/***************************************************************************************************
 * Copyright (c) 2023 - 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are met:
 *
 * 1. Redistributions of source code must retain the above copyright notice, this
 * list of conditions and the following disclaimer.
 *
 * 2. Redistributions in binary form must reproduce the above copyright notice,
 * this list of conditions and the following disclaimer in the documentation
 * and/or other materials provided with the distribution.
 *
 * 3. Neither the name of the copyright holder nor the names of its
 * contributors may be used to endorse or promote products derived from
 * this software without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
 * AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
 * DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
 * FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
 * DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
 * SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
 * CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
 * OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
 * OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 *
 **************************************************************************************************/
//  #include <cstdlib>
//  #include <cstdio>
//  #include <cassert>
 
//  #include <thrust/host_vector.h>
//  #include <thrust/device_vector.h>
 
#include <cute/tensor.hpp>
 
//  #include "cutlass/util/print_error.hpp"
//  #include "cutlass/util/GPU_Clock.hpp"
//  #include "cutlass/util/helper_cuda.hpp"
 

using namespace cute;
__device__ void myprint() { printf("\033[0m\n"); }

template<typename T, typename... Types>
__device__ void myprint(const T& first, const Types&... args) {
    print(first); print(" ");
    myprint(args...);
}

#define PRINT(...) if(thread0()){ printf("\033[93m%d \033[94m", __LINE__); myprint(__VA_ARGS__);};

constexpr int m = Int<5120>{};
constexpr int n = Int<2048>{}; 
constexpr int k = Int<4096>{};
constexpr auto prob_shape = make_shape(m, n, k);
constexpr int ldA = k;
constexpr int ldB = k;
constexpr int ldC = m;
constexpr auto dA = make_stride(Int<ldA>{}, Int<1>{});                      // (dM, dK)
constexpr auto dB = make_stride(Int<ldB>{}, Int<1>{});                      // (dN, dK)
constexpr auto dC = make_stride(Int<1>{}, Int<ldC>{});                      // (dM, dN)

// Define CTA tile sizes (static)
constexpr auto bM = Int<128>{};
constexpr auto bN = Int<128>{};
constexpr auto bK = Int<  8>{};
constexpr auto cta_tiler = make_shape(bM, bN, bK);                   // (BLK_M, BLK_N, BLK_K)

// Define the smem layouts (static)
constexpr auto sA_layout = make_layout(make_shape(bM,bK), LayoutRight{});   // (m,k) -> smem_idx; k-major
constexpr auto sB_layout = make_layout(make_shape(bN,bK), LayoutRight{});   // (n,k) -> smem_idx; k-major
constexpr auto sC_layout = make_layout(make_shape(bM, bN));                 // (m,n) -> smem_idx; m-major

// Define the thread layouts (static)
constexpr auto tA = make_layout(make_shape(Int<32>{}, Int< 8>{}), LayoutRight{});  // (m,k) -> thr_idx; k-major
constexpr auto tB = make_layout(make_shape(Int<32>{}, Int< 8>{}), LayoutRight{});  // (n,k) -> thr_idx; k-major
constexpr auto tC = make_layout(make_shape(Int<16>{}, Int<16>{}));                 // (m,n) -> thr_idx; m-major
using AThreadLayout = decltype(tA);
using BThreadLayout = decltype(tB);
using CThreadLayout = decltype(tC);
using ASmemLayout = decltype(sA_layout);
using BSmemLayout = decltype(sB_layout);
using CSmemLayout = decltype(sC_layout);
using TA = float;
using TB = float;


TiledCopy copy_a = make_tiled_copy(Copy_Atom<UniversalCopy<TA>, TA>{},
    Layout<Shape<_32,_8>,Stride<_8,_1>>{}, // Thr layout 32x8 k-major
    Layout<Shape< _1,_1>>{});              // Val layout  1x1
TiledCopy copy_b = make_tiled_copy(Copy_Atom<UniversalCopy<TB>, TB>{},
    Layout<Shape<_32,_8>,Stride<_8,_1>>{}, // Thr layout 32x8 k-major
    Layout<Shape< _1,_1>>{});              // Val layout  1x1

// TUTORIAL: Construct TiledMMA to define the MMA_Atom to use and the
//           partitioning pattern to apply.
// Use a 1x1x1 FMA on the types TC += TA * TB. Each atom requires a single thread.
// Reproduce that atom 16x16x1 times (m-major) across threads so that we use 256 threads.

TiledMMA mma = make_tiled_mma(UniversalFMA<float, float, float>{},
 Layout<Shape<_16,_16,_1>>{});  // 16x16x1 TiledMMA

constexpr float alpha = 1.0;
constexpr float beta = 0.0;

constexpr auto shape_MNK = prob_shape;
//  template <class ProblemShape, class CtaTiler,
//            class TA, class AStride, class ASmemLayout, class AThreadLayout,
//            class TB, class BStride, class BSmemLayout, class BThreadLayout,
//            class TC, class CStride, class CSmemLayout, class CThreadLayout,
//            class Alpha, class Beta>
 __global__ static
 __launch_bounds__(decltype(size(CThreadLayout{}))::value)
 void
 cutlass_kernel_2( float const* A, float const* B, float* C)
 {
   using namespace cute;
 
//    // Preconditions
   CUTE_STATIC_ASSERT_V(rank(shape_MNK) == Int<3>{});                   // (M, N, K)
   CUTE_STATIC_ASSERT_V(rank(cta_tiler) == Int<3>{});                   // (BLK_M, BLK_N, BLK_K)
 
   CUTE_STATIC_ASSERT_V(size(copy_a) == size(mma));                     // NumThreads
   CUTE_STATIC_ASSERT_V(size(copy_b) == size(mma));                     // NumThreads
 
   static_assert(is_static<ASmemLayout>::value);
   static_assert(is_static<BSmemLayout>::value);
   static_assert(is_static<CSmemLayout>::value);
 
   CUTE_STATIC_ASSERT_V(size<0>(ASmemLayout{}) == size<0>(cta_tiler));  // BLK_M
   CUTE_STATIC_ASSERT_V(size<0>(CSmemLayout{}) == size<0>(cta_tiler));  // BLK_M
   CUTE_STATIC_ASSERT_V(size<0>(BSmemLayout{}) == size<1>(cta_tiler));  // BLK_N
   CUTE_STATIC_ASSERT_V(size<1>(CSmemLayout{}) == size<1>(cta_tiler));  // BLK_N
   CUTE_STATIC_ASSERT_V(size<1>(ASmemLayout{}) == size<2>(cta_tiler));  // BLK_K
   CUTE_STATIC_ASSERT_V(size<1>(BSmemLayout{}) == size<2>(cta_tiler));  // BLK_K
 
   CUTE_STATIC_ASSERT_V(congruent(select<0,2>(shape_MNK), dA));         // dA strides for shape MK
   CUTE_STATIC_ASSERT_V(congruent(select<1,2>(shape_MNK), dB));         // dB strides for shape NK
   CUTE_STATIC_ASSERT_V(congruent(select<0,1>(shape_MNK), dC));         // dC strides for shape MN
 
   //
   // Full and Tiled Tensors
   //
 
   // Represent the full tensors
   // ptr, shape, stride
   Tensor mA = make_tensor(make_gmem_ptr(A), select<0,2>(shape_MNK), dA); // (M,K)
   Tensor mB = make_tensor(make_gmem_ptr(B), select<1,2>(shape_MNK), dB); // (N,K)
   Tensor mC = make_tensor(make_gmem_ptr(C), select<0,1>(shape_MNK), dC); // (M,N)
   PRINT("mA", mA);
   PRINT("mB", mB);
   PRINT("mC", mC);
   // Get the appropriate blocks for this thread block
   auto cta_coord = make_coord(blockIdx.x, blockIdx.y, _);              // (m,n,k)
  /* cta_tiler: <128, 128, 8> */
  /* cta_coord: <blockIdx.x, blockIdx.y, _> */
   // 这里是怎么识别出来的, 用的后面的 Step?
   Tensor gA = local_tile(mA, cta_tiler, cta_coord, Step<_1, X,_1>{});  // (BLK_M,BLK_K,k)
   Tensor gB = local_tile(mB, cta_tiler, cta_coord, Step< X,_1,_1>{});  // (BLK_N,BLK_K,k)
   Tensor gC = local_tile(mC, cta_tiler, cta_coord, Step<_1,_1, X>{});  // (BLK_M,BLK_N)
                                                                        
   PRINT("cta_tiler", cta_tiler)
   PRINT("cta_coord", cta_coord);
   PRINT("gA", gA);
   PRINT("gB", gB);
   PRINT("gC", gC);

   // Shared memory buffers
   __shared__ TA smemA[cosize_v<ASmemLayout>];
   __shared__ TB smemB[cosize_v<BSmemLayout>];
   PRINT("ASmemLayout", sA_layout);
   PRINT("BSmemLayout", sB_layout);
   PRINT("cosize_v<sB_layout>", cosize_v<BSmemLayout>);
   Tensor sA = make_tensor(make_smem_ptr(smemA), sA_layout);            // (BLK_M,BLK_K)
   Tensor sB = make_tensor(make_smem_ptr(smemB), sB_layout);            // (BLK_N,BLK_K)  
                                                                        //
   //
   // Partition the copying of A and B tiles across the threads
   //
 
  // TUTORIAL: Example of partitioning via a TiledCopy
 
   ThrCopy thr_copy_a = copy_a.get_slice(threadIdx.x);
   Tensor tAgA = thr_copy_a.partition_S(gA);                            // (CPY,CPY_M,CPY_K,k)
   Tensor tAsA = thr_copy_a.partition_D(sA);                            // (CPY,CPY_M,CPY_K)
   // Allocate registers same shape/layout as partitioned data
   Tensor tArA = make_fragment_like(tAsA);                              // (CPY,CPY_M,CPY_K)
 
   ThrCopy thr_copy_b = copy_b.get_slice(threadIdx.x);
   Tensor tBgB = thr_copy_b.partition_S(gB);                            // (CPY,CPY_N,CPY_K,k)
   Tensor tBsB = thr_copy_b.partition_D(sB);                            // (CPY,CPY_N,CPY_K)
   // Allocate registers same shape/layout as partitioned data
   Tensor tBrB = make_fragment_like(tBsB);                              // (CPY,CPY_N,CPY_K)
 
   CUTE_STATIC_ASSERT_V(size<1>(tAgA) == size<1>(tAsA));                // CPY_M
   CUTE_STATIC_ASSERT_V(size<1>(tAgA) == size<1>(tArA));                // CPY_M
   CUTE_STATIC_ASSERT_V(size<2>(tAgA) == size<2>(tAsA));                // CPY_K
   CUTE_STATIC_ASSERT_V(size<2>(tAgA) == size<2>(tArA));                // CPY_K
   CUTE_STATIC_ASSERT_V(size<1>(tBgB) == size<1>(tBsB));                // CPY_N
   CUTE_STATIC_ASSERT_V(size<1>(tBgB) == size<1>(tBrB));                // CPY_N
   CUTE_STATIC_ASSERT_V(size<2>(tBgB) == size<2>(tBsB));                // CPY_K
   CUTE_STATIC_ASSERT_V(size<2>(tBgB) == size<2>(tBrB));                // CPY_K
 
   // Copy gmem to rmem for k_tile=0
   copy(copy_a, tAgA(_,_,_,0), tArA);
   copy(copy_b, tBgB(_,_,_,0), tBrB); 
   //
   // Define A/B partitioning and C accumulators
   //
 
  // TUTORIAL: Example of partitioning via a TiledMMA
 
  ThrMMA thr_mma = mma.get_slice(threadIdx.x);
  Tensor tCsA = thr_mma.partition_A(sA);                               // (MMA,MMA_M,MMA_K)
  Tensor tCsB = thr_mma.partition_B(sB);                               // (MMA,MMA_N,MMA_K)
  Tensor tCgC = thr_mma.partition_C(gC);                               // (MMA,MMA_M,MMA_N)
 
  // Allocate the accumulators -- same size as the projected data
  Tensor tCrC = thr_mma.make_fragment_C(tCgC);                         // (MMA,MMA_M,MMA_N)

  CUTE_STATIC_ASSERT_V(  shape(tCrC) ==   shape(tCgC));                // (MMA,MMA_M,MMA_N)
  CUTE_STATIC_ASSERT_V(size<1>(tCgC) == size<1>(tCsA));                // MMA_M
  CUTE_STATIC_ASSERT_V(size<2>(tCgC) == size<1>(tCsB));                // MMA_N
  CUTE_STATIC_ASSERT_V(size<2>(tCsA) == size<2>(tCsB));                // MMA_K
 
   // Clear the accumulators
   clear(tCrC);
 
 #if 1
   if(thread0()) {
     myprint("-------------------------------------");
     print("  mA : "); print(  mA); print("\n");
     print("  gA : "); print(  gA); print("\n");
     print("  sA : "); print(  sA); print("\n");
     print("tAgA : "); print(tAgA); print("\n");
     print("tAsA : "); print(tAsA); print("\n");
   }
 #endif
 
 #if 1
   if(thread0()) {
     print("  mB : "); print(  mB); print("\n");
     print("  gB : "); print(  gB); print("\n");
     print("  sB : "); print(  sB); print("\n");
     print("tBgB : "); print(tBgB); print("\n");
     print("tBsB : "); print(tBsB); print("\n");
   }
 #endif
 
 #if 1
   if(thread0()) {
     print("  mC : "); print(  mC); print("\n");
     print("  gC : "); print(  gC); print("\n");
     print("tCsA : "); print(tCsA); print("\n");
     print("tCsB : "); print(tCsB); print("\n");
     print("tCgC : "); print(tCgC); print("\n");
     print("tCrC : "); print(tCrC); print("\n");
     myprint("-------------------------------------");
   }
 #endif
 
 #if 1
 
  // TUTORIAL: Example of an inner loop that pipelines compute with reads
  //           from global memory by staging through register and shared memory.
  //   Data is read from global to registers, then to shared via the TiledCopy partitions
  //   gemm(.) operates on the shared memory directly via the TiledMMA partitions

  auto K_TILE_MAX = size<3>(tAgA);
   PRINT("K_TILE_MAX", K_TILE_MAX);
 
   for (int k_tile = 0; k_tile < K_TILE_MAX; ++k_tile)
   {
    // Copy rmem to smem with tA|tB thread-partitioned tensors
    __syncthreads();         // Wait for all threads to consume smem
    copy(tArA, tAsA);
    copy(tBrB, tBsB);
    __syncthreads();         // Wait for all threads to consume smem

    // Copy gmem to rmem for k_tile+1 with tA|tB thread-partitioned tensors
    int k_tile_next = (k_tile + 1 < K_TILE_MAX) ? k_tile + 1 : k_tile;
    copy(copy_a, tAgA(_,_,_,k_tile_next), tArA);
    copy(copy_b, tBgB(_,_,_,k_tile_next), tBrB);
    // TUTORIAL: The above call to copy(copy_a, tAgA(_,_,_,k_tile_next), tArA) is equivalent to
    //   CUTE_UNROLL
    //   for (int k = 0; k < size<1>(tCsA); ++k) {
    //     CUTE_UNROLL
    //     for (int m = 0; m < size<0>(tCrC); ++m) {
    //       copy_a.call(tAgA(_,m,k), tArA(_,m,k);
    //     }
    //   }

    // Compute gemm on mma-partitioned smem
    gemm(mma, tCsA, tCsB, tCrC);
    // TUTORIAL: The above call to gemm(tCsA, tCsB, tCrC) is equivalent to
    //   CUTE_UNROLL
    //   for (int k = 0; k < size<1>(tCsA); ++k) {
    //     CUTE_UNROLL
    //     for (int m = 0; m < size<0>(tCrC); ++m) {
    //       CUTE_UNROLL
    //       for (int n = 0; n < size<1>(tCrC); ++n) {
    //         mma.call(tCsA(_,m,k), tCsB(_,n,k), tCrC(_,m,n);
    //       }
    //     }
    //   }
   }
 
 #endif
 
   //
   // Epilogue
   //
 
   axpby(alpha, tCrC, beta, tCgC);
 
   // TUTORIAL: The above call to axpby(alpha, tCrC, beta, tCgC) is equivalent to
   //   CUTE_UNROLL
   //   for (int i = 0; i < size(tCrC); ++i) {
   //     tCgC(i) = alpha * tCrC(i) + beta * tCgC(i);
   //   }
 }
 
