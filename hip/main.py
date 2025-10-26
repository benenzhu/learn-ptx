import torch
import time
import importlib
import rtc
importlib.reload(rtc)
from rtc import _compile_kernel


def get_kernel(kernel_name, file_name="00_add.hip"):
    tic = time.time()
    kernel = _compile_kernel(
        open(file_name, "r").read(),
        kernel_name=kernel_name,
        nvcc_options=[
            "-std=c++17", 
            "-Dhip_rtc",
            "-g", 
            "-save-temps"
            ],
        save_ptx=True,
    )
    toc = time.time()
    print(f"compile used {toc - tic}")
    return kernel





def test_add_kernel(): 
    M, N = 100, 2048
    A = torch.randint(-100, 100, (M, N), device="cuda", dtype=torch.int32)
    B = torch.randint(-100, 100, (M, N), device="cuda", dtype=torch.int32) 
    C = torch.empty_like(A)
    
    add_kernel = get_kernel("add_kernel", "00_add.hip")
    print("start_kernel here", A.data_ptr(), B.data_ptr(), C.data_ptr(), M, N)
    add_kernel((M, 1, 1), (256, 1, 1), (A, B, C, M, N))
    print("end_kernel here")
    torch.cuda.synchronize()
    print("synchronize done")
    torch.testing.assert_close(C, A + B)
    print("test passed")
    return C

# test_add_kernel()


def test_add_kernel_v2(): 
    M, N = 100, 2048
    A = torch.randint(-100, 100, (M, N), device="cuda", dtype=torch.int32)
    B = torch.randint(-100, 100, (M, N), device="cuda", dtype=torch.int32) 
    C = torch.empty_like(A)
    
    add_kernel = get_kernel("add_kernel", "00_add_2.hip")
    print("start_kernel here", A.data_ptr(), B.data_ptr(), C.data_ptr(), M, N)
    add_kernel((M, 1, 1), (256, 1, 1), (A, B, C, M, N))
    print("end_kernel here")
    torch.cuda.synchronize()
    print("synchronize done")
    torch.testing.assert_close(C, A + B)
    print("test passed")
    return C
    
# test_add_kernel_v2()

def test_bf16_matmul_NNN():
    matmul_kernel = get_kernel("fp16_gemm_16x16x16_NNN", "01_mfma.hip")  
    print(matmul_kernel)
    A = torch.arange(16*16, device="cuda").reshape(16, 16).half()*0.1
    B = torch.arange(16*16, device="cuda").reshape(16, 16).half()*0.1
    C = torch.zeros(16*16, device="cuda").reshape(16, 16).half()
    matmul_kernel((1,1,1), (16,4,1), (A, B, C))
    torch.cuda.synchronize()

test_bf16_matmul_NNN()


def test_bf16_matmul_NTN():
    matmul_kernel = get_kernel("fp16_gemm_16x16x16_NTN", "01_mfma.hip")  
    # print(matmul_kernel)
    A = torch.arange(16*16, device="cuda").reshape(16, 16).half()*0.1
    B = torch.arange(16*16, device="cuda").reshape(16, 16).half().transpose(0, 1).contiguous()*0.1
    C = torch.zeros(16*16, device="cuda").reshape(16, 16).half()
    matmul_kernel((1,1,1), (16,4,1), (A, B, C))
    torch.cuda.synchronize()
    
test_bf16_matmul_NTN()

def bf16_matmul_full_NTN():
    matmul_kernel = get_kernel("fp16_gemm_full_NTN", "02_fp16_gemm_v1.hip")
    M, N, K = 4096, 4096, 4096
    A = torch.randn(M, K, device="cuda").bfloat16()
    B = torch.randn(N, K, device="cuda").bfloat16()
    C = torch.zeros(M, N, device="cuda").bfloat16()
    
    
    NUM_WRAP_M = 2
    NUM_WARP_N = 2
    BLOCK_M = 128
    BLOCK_N = 128
    BLOCK_K = 64
    TB_SIZE = 64 * NUM_WRAP_M * NUM_WARP_N
    GRID_SIZE = ((M + BLOCK_M - 1) // BLOCK_M)* ((N + BLOCK_N - 1) // BLOCK_N)
    print(f"{GRID_SIZE=}, {TB_SIZE=}")
    shared_mem=(BLOCK_M + BLOCK_N) * BLOCK_K * 4
    matmul_kernel.set_shared_memory_config(shared_mem)
    matmul_kernel((GRID_SIZE,1,1), (TB_SIZE,1,1), (A, B, C, M, N, K), shared_mem=shared_mem)
    torch.cuda.synchronize()
    if not torch.allclose(C, A @ B.T, atol=1e-3, rtol=1e-3):
        print("C is not close to A @ B.T")
        print(C)
        print(A @ B.T)
        print(A)
        print(B)
        diff = C - A @ B.T
        print(C - A @ B.T)
        print("diff", (C - A @ B.T).max().item())
        max_diff_idx = diff.abs().argmax()
        max_diff_row = max_diff_idx // N
        max_diff_col = max_diff_idx % N
        print(f"Max diff at position ({max_diff_row}, {max_diff_col})")
        print(f"C[{max_diff_row}, {max_diff_col}] = {C[max_diff_row, max_diff_col]}")
        print(f"Expected = {(A @ B.T)[max_diff_row, max_diff_col]}")
        print(f"{diff.reshape(-1).sort()}")
        return diff
    torch.testing.assert_close(C, A @ B.T)
    
     
ret = bf16_matmul_full_NTN()


