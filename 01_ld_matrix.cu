#include <cuda_fp16.h>
#include "00_rtc.cu"

__launch_bounds__(32)
__global__ void ld_matrix_kernel(half *d_ptr) {
    // 一个 total_row, total_col的矩阵,  加载他的左上角.
    constexpr int total_row = 36, total_col = 40;
    constexpr int elts = total_row * total_col;
    __shared__ half smem[elts];
    

    /// init and print smem
    int tid = threadIdx.x;
    if(tid == 0) {
        for(int i = 0; i < total_row * total_col; i++) smem[i] = __float2half(0.0);
        for(int i = 0; i < 16; i ++) {
            for(int j = 0; j < 16; j++){
                smem[i * total_col + j] = __float2half(i * 16 + j);
            }
        }
        print_mem(smem);
    }

    /// ldmatrix
    uint32_t regs[4];
    half *ptr = smem + tid % 16 * total_col + tid / 16 * 8;
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

    {
        int row = tid / 4;
        int col = tid % 4;
        reinterpret_cast<uint32_t*>(d_ptr)[row       * 8 + col] = regs[0];
        reinterpret_cast<uint32_t*>(d_ptr)[(row + 8) * 8 + col] = regs[1];
        reinterpret_cast<uint32_t*>(d_ptr)[row       * 8 + col + 4] = regs[2];
        reinterpret_cast<uint32_t*>(d_ptr)[(row + 8) * 8 + col + 4] = regs[3]; 
        __syncthreads();
        print_mem(d_ptr);
    }
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
