from dataclasses import dataclass
import torch
import time
import importlib
import rtc
import os
os.environ["ROCPROF_COUNTER_COLLECTION"] = "1"

importlib.reload(rtc)
# import tritonblas
from tritonblas.matmul import persistent_matmul_lt
# importlib.reload(tritonblas)
# from tritonblas.matmul import persistent_matmul_lt
# importlib.reload(tritonblas.matmul)
from rtc import _compile_kernel, get_triton_gemm_NTN, my_assert_close


torch.set_printoptions(threshold=1000, edgeitems=3)     
def get_kernel(kernel_name, file_name="00_add.hip", config=None):
    tic = time.time()
    source = open(file_name, "r").read()
    if config is not None:
        defines = ["#define PYTHON_CALL\n"]
        for key, value in vars(config).items():
            defines.append(f"#define {key} {value}\n")
        # Check if source starts with non-empty lines and warn
        source_lines = source.split('\n')
        non_empty_prefix = []
        for line in source_lines:
            if line.strip():
                non_empty_prefix.append(line)
            else:
                break
        
        if non_empty_prefix:
            print(f"Warning: Replacing {len(non_empty_prefix)} non-empty line(s) at the beginning of source file")
            print(source_lines[:len(non_empty_prefix)])
        
        # Replace the first few lines with defines
        num_lines_to_replace = len(defines)
        remaining_source = '\n'.join(source_lines[num_lines_to_replace:])
        source = "".join(defines) + remaining_source
    print("".join(defines))   

    kernel = _compile_kernel(
        kernel_source=source,
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

# test_bf16_matmul_NNN()


def test_bf16_matmul_NTN():
    matmul_kernel = get_kernel("fp16_gemm_16x16x16_NTN", "01_mfma.hip")  
    # print(matmul_kernel)
    A = torch.arange(16*16, device="cuda").reshape(16, 16).half()*0.1
    B = torch.arange(16*16, device="cuda").reshape(16, 16).half().transpose(0, 1).contiguous()*0.1
    C = torch.zeros(16*16, device="cuda").reshape(16, 16).half()
    matmul_kernel((1,1,1), (16,4,1), (A, B, C))
    torch.cuda.synchronize()
    
# test_bf16_matmul_NTN()


def bench(f, A, B, C, check_correct=True):
    import inspect
    frame = inspect.currentframe().f_back
    name = frame.f_code.co_name
    from triton.testing import do_bench
    f()

    M, N, K = A.shape[0], B.shape[0], A.shape[1]
    torch.cuda.synchronize()
    right_output = torch.zeros_like(C)
    if M >= 512:
        A = A.T.contiguous()
        BT = B.T.contiguous()
        triton_fn = lambda: persistent_matmul_lt(A.T, BT, right_output, None)
    else:
        BT = B.T
        triton_fn = lambda: get_triton_gemm_NTN(A, B, right_output, M, N, K)
    if check_correct:
        ret = triton_fn()
        my_assert_close(C, right_output)
    if "ROCPROF_COUNTER_COLLECTION" in os.environ:
        print("ROC_PERF on fast return here.")
        return ret
    torch.cuda.synchronize()
    latency_ms = do_bench(f, warmup=100, rep=500)
    tflops = 2 * M * N * K / (latency_ms * 1e-3) * 1e-12
    print(f"{name}: {tflops:.2f} TFLOPS")
    latency_ms = do_bench(triton_fn, warmup=100, rep=500)
    tflops = 2 * M * N * K / (latency_ms * 1e-3) * 1e-12
    print(f"triton: \t{tflops:.2f} TFLOPS")
    return ret

@dataclass
class Bf16MatmulFullNTNConfig:
    M: int = 4096
    N: int = 4096
    K: int = 4096
    NUM_WARP_M: int = 2
    NUM_WARP_N: int = 2
    BLOCK_M: int = 128
    BLOCK_N: int = 128
    BLOCK_K: int = 64
    SMEM_STRIDE: int = 0
    def get_grid_size(self):
        return ((self.M + self.BLOCK_M - 1) // self.BLOCK_M)* ((self.N + self.BLOCK_N - 1) // self.BLOCK_N)
    def get_tb_size(self):
        return 64 * self.NUM_WARP_M * self.NUM_WARP_N
    def get_shared_mem(self):
        if not self.SMEM_STRIDE:
            self.SMEM_STRIDE = self.BLOCK_K
        return (self.BLOCK_M + self.BLOCK_N) * self.SMEM_STRIDE * 2


def get_inputNTN(M, N, K):
    if M <= 512:
        A = torch.arange(M*K, device="cuda").reshape(M, K).bfloat16().contiguous() * 0.1
        B = torch.arange(N*K, device="cuda").reshape(N, K).bfloat16().contiguous() * 0.1
    else:
        A = torch.randn(M, K, device="cuda").bfloat16().contiguous()
        B = torch.randn(N, K, device="cuda").bfloat16().contiguous() 
    C = torch.zeros(M, N, device="cuda").bfloat16().contiguous()
    return A, B, C


def bf16_matmul_full_NTN(M, N, K):
    A, B, C = get_inputNTN(M, N, K)
    config = Bf16MatmulFullNTNConfig(M=M, N=N, K=K)
    matmul_kernel = get_kernel("fp16_gemm_full_NTN", "02_fp16_gemm_v1.hip", config)
    TB_SIZE = config.get_tb_size()
    GRID_SIZE = config.get_grid_size()
    shared_mem=config.get_shared_mem()
    print(f"{GRID_SIZE=}, {TB_SIZE=}, {shared_mem=}")
    matmul_kernel.set_shared_memory_config(shared_mem)
    kernel_fn = lambda: matmul_kernel((GRID_SIZE,1,1), (TB_SIZE,1,1), (A, B, C, M, N, K), shared_mem=shared_mem)
    bench(kernel_fn, A, B, C)
    
# ret = bf16_matmul_full_NTN(4864, 4096, 4096)


def bf16_matmul_full_NTN_v2(M, N, K):
    A, B, C = get_inputNTN(M, N, K)
    config = Bf16MatmulFullNTNConfig(M=M, N=N, K=K, SMEM_STRIDE=64 + 8)
    matmul_kernel = get_kernel("fp16_gemm_full_NTN", "02_fp16_gemm_v1.hip", config)
    TB_SIZE = config.get_tb_size()
    GRID_SIZE = config.get_grid_size()
    shared_mem=config.get_shared_mem()
    print(f"{GRID_SIZE=}, {TB_SIZE=}, {shared_mem=}")
    matmul_kernel.set_shared_memory_config(shared_mem)
    kernel_fn = lambda: matmul_kernel((GRID_SIZE,1,1), (TB_SIZE,1,1), (A, B, C, M, N, K), shared_mem=shared_mem)
    
    bench(kernel_fn, A, B, C)
    
# ret = bf16_matmul_full_NTN_v2(4864, 4096, 4096)


def bf16_matmul_full_NTN_v3(M, N, K):
    A, B, C = get_inputNTN(M, N, K)
    config = Bf16MatmulFullNTNConfig(M=M, N=N, K=K, SMEM_STRIDE=64)
    matmul_kernel = get_kernel("fp16_gemm_full_NTN_v2", "02_fp16_gemm_v2.hip", config)
    TB_SIZE = config.get_tb_size()
    GRID_SIZE = config.get_grid_size()
    shared_mem=config.get_shared_mem()
    print(f"{GRID_SIZE=}, {TB_SIZE=}, {shared_mem=}")
    matmul_kernel.set_shared_memory_config(shared_mem)
    kernel_fn = lambda: matmul_kernel((GRID_SIZE,1,1), (TB_SIZE,1,1), (A, B, C, M, N, K), shared_mem=shared_mem)
    
    bench(kernel_fn, A, B, C)
    
# ret = bf16_matmul_full_NTN_v3(4864, 4096, 4096)

# print(list(os.environ.keys()))

def bf16_matmul_full_NTN_v2_opt1(M, N, K):
    A, B, C = get_inputNTN(M, N, K)
    

    config = Bf16MatmulFullNTNConfig(
        M=M, 
        N=N, 
        K=K, 
        NUM_WARP_M=2,
        NUM_WARP_N=4,
        BLOCK_M=256,
        BLOCK_N=256,
        BLOCK_K=64)
    matmul_kernel = get_kernel("fp16_gemm_full_NTN_v3", "02_fp16_gemm_full_NTN_v3.hip", config)
    TB_SIZE = config.get_tb_size()
    GRID_SIZE = config.get_grid_size()
    shared_mem=config.get_shared_mem()
    print(f"{GRID_SIZE=}, {TB_SIZE=}, {shared_mem=}")
    matmul_kernel.set_shared_memory_config(shared_mem)
    kernel_fn = lambda: matmul_kernel((GRID_SIZE,1,1), (TB_SIZE,1,1), (A, B, C, M, N, K), shared_mem=shared_mem)
    
    ret = bench(kernel_fn, A, B, C)
    return ret
    
# ret = bf16_matmul_full_NTN_v2_opt1(4864, 4096, 4096)
# ret = bf16_matmul_full_NTN_v2_opt1(256, 256, 64)


def bf16_matmul_full_NTN_v4(M, N, K):
    A, B, C = get_inputNTN(M, N, K)
    

    config = Bf16MatmulFullNTNConfig(
        M=M, 
        N=N, 
        K=K, 
        NUM_WARP_M=2,
        NUM_WARP_N=4,
        BLOCK_M=256,
        BLOCK_N=256,
        BLOCK_K=64)
    matmul_kernel = get_kernel("fp16_gemm_full_NTN_v4", "02_fp16_gemm_full_NTN_v4.hip", config)
    TB_SIZE = config.get_tb_size()
    GRID_SIZE = config.get_grid_size()
    shared_mem=config.get_shared_mem()
    print(f"{GRID_SIZE=}, {TB_SIZE=}, {shared_mem=}")
    matmul_kernel.set_shared_memory_config(shared_mem)
    kernel_fn = lambda: matmul_kernel((GRID_SIZE,1,1), (TB_SIZE,1,1), (A, B, C, M, N, K), shared_mem=shared_mem)
    
    ret = bench(kernel_fn, A, B, C)
    return ret
    
# ret = bf16_matmul_full_NTN_v2_opt1(4864, 4096, 4096)
# ret = bf16_matmul_full_NTN_v2_opt1(256, 256, 128)
ret = bf16_matmul_full_NTN_v4(256, 256, 64)
# ret = bf16_matmul_full_NTN_v4(64*4, 64*4, 128*4)






