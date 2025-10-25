#include <hip/hip_runtime.h>
__launch_bounds__(256)
__global__
void add_kernel(const int *A, const int *B, int *C, int M, int N) {

  __builtin_assume(N >= 2048);
  __builtin_assume(N <= 4096);
  __builtin_assume(N % 1024 == 0);  // 如果是256倍数
  // A, B, C are 2D matrices with shape M, N
  // we will use 1 threadblock to handle 1 row
  const int tid = threadIdx.x;
  const int bid = blockIdx.x;
  const int tb_size = blockDim.x;

  A += bid * N;
  B += bid * N;
  C += bid * N;

//printf("block_idx: %d thread_idx: %d   A:%p B:%p C:%p here...\n", blockIdx.x, threadIdx.x, A, B, C);

  for (int col = tid; col < N; col += tb_size)
    C[col] = A[col] + B[col];
}