import torch
from typing import Any, Dict
from rtc import _compile_kernel
import importlib
import rtc
import tma
importlib.reload(rtc)
importlib.reload(tma)
import triton.language as tl

import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

import time
print(torch.__version__)
from rtc import _check_cuda
import cuda.bindings.driver as cbd

def get_kernel(kernel_name, file_name="01_sm80_ptx.cu", save_ptx = False):
    tic = time.time()
    kernel = _compile_kernel(
        open(file_name, "r").read(),
        kernel_name=kernel_name,
        nvcc_options=[
            "-std=c++17", 
            "-D__CUDA_NO_HALF_OPERATORS__", 
            "-D__CUDA_NO_HALF_CONVERSIONS__", 
            "-D__CUDA_NO_BFLOAT16_CONVERSIONS__", 
            "-D__CUDA_NO_HALF2_OPERATORS__",
            "-I/A/cutlass/include",
            "-I/A/cutlass/examples/common",
            "-I/A/cutlass/build/include",
            "-I/A/cutlass/tools/util/include",
            "-I/usr/local/cuda/include",
            "-DCUTLASS_TEST_LEVEL=0",
            "-DCUTLASS_TEST_ENABLE_CACHED_RESULTS=1",
            "-DCUTLASS_CONV_UNIT_TEST_RIGOROUS_SIZE_ENABLED=1",
            "-DCUTLASS_DEBUG_TRACE_LEVEL=0",
            "-default-device",
        ],
        save_ptx=False,
    )
    toc = time.time()
    print("compile used time: ", toc - tic)
    return kernel


def test_async_cp_kernel():
    async_cp_kernel = get_kernel("async_cp_kernel")
    M = 16 * 16
    a = torch.arange(M, device="cuda").half()
    async_cp_kernel((1,1,1), (32,1,1), (a,))
    torch.cuda.synchronize()
    time.sleep(0.5)
 
# test_async_cp_kernel()

def test_ld_matrix_kernel():
    ld_matrix_kernel = get_kernel("ld_matrix_kernel")
    a = torch.zeros(16, 32, device="cuda").half()
    ld_matrix_kernel((1,1,1), (32,1,1), (a,))
    torch.cuda.synchronize()
    # time.sleep(0.5)
    
# test_ld_matrix_kernel()

def test_mma_ptx_kernel():
    mma_ptx_kernel = get_kernel("mma_ptx_kernel", file_name="02_mma_ptx.cu")
    a = torch.arange(16 * 16, device="cuda").half() * 0.1
    b = torch.arange(16 * 16, device="cuda").half() * 0.1
    c = torch.zeros(16 * 16 * 16, device="cuda").half()
    d = torch.matmul(a.reshape(16, 16), b.reshape(16, 16).T)
    mma_ptx_kernel((1,1,1), (32,1,1), (c, a, b, d))
    torch.cuda.synchronize()
    # time.sleep(0.5)

# test_mma_ptx_kernel()


def test_mma_wgmma_kernel():
    mma_wgmma_kernel = get_kernel("mma_wgmma_kernel", file_name="02_mma_ptx_v2.cu")
    a = torch.arange(16 * 16, device="cuda").half() * 0.1
    b = torch.arange(16 * 16, device="cuda").half() * 0.1
    c = torch.zeros(16 * 16 * 16, device="cuda").half()
    d = torch.matmul(a.reshape(16, 16), b.reshape(16, 16).T)
    mma_wgmma_kernel((1,1,1), (32,1,1), (c, a, b, d))
    torch.cuda.synchronize()
    time.sleep(0.5)
    
# test_mma_wgmma_kernel()
def test_tma_1d_kernel():
    tma_1d_kernel = get_kernel("tma_1d_kernel", file_name="02_mma_ptx.cu")
    warps_per_block = 4 
    threads_per_warps = 32
    elts_per_threads = 8
    elts = warps_per_block * threads_per_warps * elts_per_threads
    a = torch.arange(elts, device="cuda").half() * 0.1
    tma_1d_kernel((1,1,1), (warps_per_block * threads_per_warps, 1, 1), (a, a.numel()), shared_mem=2 * elts)
    torch.cuda.synchronize()
    time.sleep(0.5)

# test_tma_1d_kernel()


def test_tma_2d_kernel():
    tma_2d_kernel = get_kernel("tma_2d_kernel", file_name="04_tma_2d.cu")
    grid_size = (2, 1, 1)
    block_size = (4 * 32, 1, 1)
    WIDTH, HEIGHT=32, 32
    W, H = 16, 16
    input = torch.arange(WIDTH * HEIGHT, device="cuda").half()
    gmem_dims = (cbd.cuuint64_t(WIDTH), cbd.cuuint64_t(HEIGHT))
    gmem_strides = (cbd.cuuint64_t(WIDTH * input.element_size()),)
    box_stride = (cbd.cuuint32_t(W), cbd.cuuint32_t(H))

    tensor_dtype = tma.tmap_type_map[input.dtype]

    result, tensor_map = cbd.cuTensorMapEncodeTiled(
        tensor_dtype,
        2, # num dims.
        input.data_ptr(),
        gmem_dims, # 32, 32
        gmem_strides, # 32 * 2
        box_stride, # 16, 16
        (cbd.cuuint32_t(1),) * 2,
        cbd.CUtensorMapInterleave.CU_TENSOR_MAP_INTERLEAVE_NONE,
        tma.swizzle_type_map[0],
        # cbd.CUtensorMapL2promotion.CU_TENSOR_MAP_L2_PROMOTION_L2_256B,
        cbd.CUtensorMapL2promotion.CU_TENSOR_MAP_L2_PROMOTION_NONE,
        cbd.CUtensorMapFloatOOBfill.CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE, 
    )
    _check_cuda(result)
    print(f"grid_size : {grid_size}, block_size : {block_size}")
    tma_2d_kernel(grid_size, block_size, args=[tensor_map])
# test_tma_2d_kernel()







def test_build_cutlass_kernel():
    kernel = get_kernel("cutlass_kernel_2", file_name="05_cutlass.cu", save_ptx=True)
    print(kernel)
    m = 5120
    n = 2048
    k = 4096
    a = torch.randn(m, k, device="cuda").float()
    b = torch.randn(n, k, device="cuda").float()
    c = torch.zeros(m, n, device="cuda").float().T.contiguous().T
    
    dim_block = (16 * 16, 1, 1)
    dim_grid = (m // 128, n // 128, 1)
    kernel(dim_grid, dim_block , (a, b, c))
    
    diff = c - a@b.T
    print("result", c - a@b.T)
    print("max diff", diff.max())



test_build_cutlass_kernel()