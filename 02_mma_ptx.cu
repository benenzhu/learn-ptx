#include <cuda_fp16.h>
#include "00_rtc.cu"
#include <cuda_pipeline.h>
#include <cuda/barrier>
#include <cuda/ptx>




// a: row major
// b: col major
// c: row major
// template <int elts, bool print>
constexpr int elts = 16 * 16;
constexpr bool print = true;

// mnk: 16 16 16
__global__ void mma_ptx_kernel(half *c_ptr, half *a_ptr, half *b_ptr, half *d_ptr) {
    int tid = threadIdx.x;
    constexpr int A_cols = 16;
    constexpr int B_cols = 16;

    // ldgsts
    __shared__ half smem_a[elts];
    __shared__ half smem_b[elts];

    // 16 * 16 / 8 = 32 一次正好拷贝完....
    half *src = a_ptr + tid / 2 * A_cols + tid % 2 * 8;
    half *dst = smem_a + tid * 8;
    uint32_t addr = __cvta_generic_to_shared(dst);
    asm("cp.async.cg.shared.global [%0], [%1], 16;\n"
        :
        :"r"(addr), "l"(src)
    );
    asm("cp.async.commit_group;\n"::);

    src = b_ptr + tid / 2 * B_cols + tid % 2 * 8;
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

    // ldmatrix 16 * 16 / 32 = 8 bfloat = float4
    uint32_t a_regs[4];
    uint32_t b_regs[4];
    src = smem_a + tid % 16 * 16 + tid / 16 * 8;

    addr = __cvta_generic_to_shared(src);
    asm("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];\n"
        :"=r"(a_regs[0]), "=r"(a_regs[1]), "=r"(a_regs[2]), "=r"(a_regs[3])
        :"r"(addr)
    );

    src = smem_b + tid % 16 * 16 + tid / 16 * 8;
    addr = __cvta_generic_to_shared(src);
    asm("ldmatrix.sync.aligned.x4.m8n8.shared.b16 {%0, %1, %2, %3}, [%4];\n"
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


