#include <hip/hip_runtime.h>
// #include <hip/hip_bf16.h>
#include <hip/hip_fp16.h>
constexpr int M = 16;
constexpr int N = 16;
constexpr int K = 16;

constexpr int LDA = K;
constexpr int LDB = N;
constexpr int LDD = N;

constexpr int A_size = M * LDA;
constexpr int B_size = K * LDB;
constexpr int D_size = M * LDD;

__device__ void print_mem(const _Float16 *ptr, int row=16, int col=16){
    if(threadIdx.x == 0 && threadIdx.y == 0 && threadIdx.z == 0) { 
        for(int i = 0; i < row; i ++) {
            if(i % 8 == 0 && i != 0) {
                printf("\n");
            }
            for(int j = 0; j < col; j++) {
                if(j % 8 == 0 && j != 0) {
                    printf("  ");
                }
                
                float now = ptr[i*col+j];
                printf("%6.1lf ",  now);
            }
            printf("\n");
        }
    }
}


__global__ void fp16_gemm_16x16x16_NNN(const _Float16* A, const _Float16* B, _Float16* D)
{

  // This kernel computes a 16x16x16 matrix multiplication using a single wavefront.
  using float16x4 = __attribute__((__vector_size__(4 * sizeof(_Float16)))) _Float16;
  using floatx4 = __attribute__((__vector_size__(4 * sizeof(float)))) float;
  floatx4 d = {0}; // zero out 4 vanilla VGPRs

  /*
  One invocation of v_mfma_f32_16x16x16f16 accumulates the sum of 16 outer products,
  16 columns of A with 16 rows of B, into result matrix D (which is in AccVGPRs).
  So we need only one iteration to compute the full matrix D

  For both the 16 columns of A, and the 16 rows of B, we use a single VGPR pair.
  With 64 lanes, and 4 Float16 values per lane, that covers the 16 columns of A and 16
  rows of B.
  Matrix A is a 16 x 16 matrix that is stored in 2 VGPRs as follows:
    a[0] covers columns 0, 4, 8, and 12
    a[1] covers columns 1, 5, 9, and 13
    a[2] covers columns 2, 6, 10, and 14
    a[3] covers columns 3, 7, 11, and 15
    first 16 lanes of a[0] cover column 0 -  last 16 lanes of a[0] cover column 12
    first 16 lanes of a[1] cover column 1 -  last 16 lanes of a[1] cover column 13
    first 16 lanes of a[2] cover column 2 -  last 16 lanes of a[2] cover column 14
    first 16 lanes of a[3] cover column 3 -  last 16 lanes of a[3] cover column 15
  Matrix B is a 16 x 16 matrix that is stored in 2 VGPRs as follows:
    b[0] covers rows 0, 4, 8, and 12
    b[1] covers rows 1, 5, 9, and 13
    b[2] covers rows 2, 6, 10, and 14
    b[3] covers rows 3, 7, 11, and 15
    first 16 lanes of b[0] cover row 0 -  last 16 lanes of b[0] cover row 12
    first 16 lanes of b[1] cover row 1 -  last 16 lanes of b[1] cover row 13
    first 16 lanes of b[2] cover row 2 -  last 16 lanes of b[2] cover row 14
    first 16 lanes of b[3] cover row 3 -  last 16 lanes of b[3] cover row 15
  Note that A and B are in row-major order.

  This kernel is called with a single wavefront in dim3(16, 4) layout
  */
  if(threadIdx.x == 0 && threadIdx.y == 0 && threadIdx.z == 0){
    printf("Global A: \n");
    print_mem(A, 16, 16);
    printf("Global B: \n");
    print_mem(B, 16, 16);
  }

  float16x4 a;
  float16x4 b;
  for(int i = 0; i < 4; ++i){
    const int a_idx =  threadIdx.x * LDA      // consecutive threads cover 16 consecutive rows
                     + i                      // consecutive registers take consecutive columns
                     + threadIdx.y * 4;       // groups of 16 lanes skip 4 columns
    a[i] = A[a_idx];

    const int b_idx =  threadIdx.x            // consecutive threads cover 16 consecutive columns
                     + i * LDB                // consecutive registers take consecutive rows
                     + threadIdx.y * LDB * 4; // groups of 16 lanes skip 4 rows
    b[i] = B[b_idx];
  }

  d = __builtin_amdgcn_mfma_f32_16x16x16f16(a, b, d, 0, 0, 0);
  //                                        ^  ^  ^
  //D(=C)                                   |  |  C(=D)
  //                      16 columns of A---|  |--- 16 rows of B

  /*
  Matrix D is a 16 x 16 matrix that is stored in 4 AccVGPRs as follows:
    d[0] covers rows 0, 4, 8, and 12
    d[1] covers rows 1, 5, 9, and 13
    d[2] covers rows 2, 6, 10, and 14
    d[3] covers rows 3, 7, 11, and 15
    first 16 lanes of d[0] cover row 0 -  last 16 lanes of d[0] cover row 12
    first 16 lanes of d[1] cover row 1 -  last 16 lanes of d[1] cover row 13
    first 16 lanes of d[2] cover row 2 -  last 16 lanes of d[2] cover row 14
    first 16 lanes of d[3] cover row 3 -  last 16 lanes of d[3] cover row 15
  */
  for(int i = 0; i < 4; ++i){
    const int d_idx =  threadIdx.x            // consecutive threads cover 16 consecutive columns
                     + i * LDD                // consecutive registers take consecutive rows of 16 floats
                     + threadIdx.y * 4 * LDD; // groups of 16 lanes skip 4 rows

    D[d_idx] = d[i];
  }
  __syncthreads();
  if(threadIdx.x == 0 && threadIdx.y == 0 && threadIdx.z == 0){
    printf("Global D: \n");
    print_mem(D, 16, 16);
  }
}




__global__ void fp16_gemm_16x16x16_NTN(const _Float16* A, const _Float16* B, _Float16* D)
{

  using float16x4 = __attribute__((__vector_size__(4 * sizeof(_Float16)))) _Float16;
  using floatx4 = __attribute__((__vector_size__(4 * sizeof(float)))) float;
  floatx4 d = {0}; // zero out 4 vanilla VGPRs

  if(threadIdx.x == 0 && threadIdx.y == 0 && threadIdx.z == 0){
    printf("Global A  : \n");
    // print_mem(A, 16, 16);
    printf("Global B: \n");
    // print_mem(B, 16, 16);
  }

  float16x4 a;
  float16x4 b;
  for(int i = 0; i < 4; ++i){
    const int a_idx =  threadIdx.x * LDA      // consecutive threads cover 16 consecutive rows
                     + i                      // consecutive registers take consecutive columns
                     + threadIdx.y * 4;       // groups of 16 lanes skip 4 columns
    a[i] = A[a_idx];

    const int b_idx =  threadIdx.x * LDA           // consecutive threads cover 16 consecutive columns
                     + i                // consecutive registers take consecutive rows
                     + threadIdx.y * 4; // groups of 16 lanes skip 4 rows
    b[i] = B[b_idx];
  }

  d = __builtin_amdgcn_mfma_f32_16x16x16f16(a, b, d, 0, 0, 0);
  //                                        ^  ^  ^
  //D(=C)                                   |  |  C(=D)
  //                      16 columns of A---|  |--- 16 rows of B

  for(int i = 0; i < 4; ++i){
    const int d_idx =  threadIdx.x            // consecutive threads cover 16 consecutive columns
                     + i * LDD                // consecutive registers take consecutive rows of 16 floats
                     + threadIdx.y * 4 * LDD; // groups of 16 lanes skip 4 rows

    D[d_idx] = d[i];
  }
  __syncthreads();
  if(threadIdx.x == 0 && threadIdx.y == 0 && threadIdx.z == 0){
    printf("Global D: \n");
    print_mem(D, 16, 16);
  }
}