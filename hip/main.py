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
    M, N = 100, 1024
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
    


test_add_kernel()


