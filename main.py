import torch
from typing import Optional, Union, Any, Dict
import ctypes
import sys
from rtc import _compile_kernel
import importlib
import rtc
importlib.reload(rtc)
from triton.testing import do_bench
from contextlib import nullcontext

import time
print(torch.__version__)
# with open("01_sm80_ptx.cu", "r") as f:
#     KERNEL_SOURCE = f.read()
# tic = time.time()
# kernel = _compile_kernel(
#     KERNEL_SOURCE,
#     kernel_name="add_kernel",
# )
# toc = time.time()

# print("compile used time: ", toc - tic)
from rtc import _get_cuda_runtime_library, _check_cuda
import cuda.bindings.driver as cbd

tmap_type_map: Dict[Any, str] = {
    torch.int8: cbd.CUtensorMapDataType.CU_TENSOR_MAP_DATA_TYPE_UINT8,
    torch.int16: cbd.CUtensorMapDataType.CU_TENSOR_MAP_DATA_TYPE_UINT16,
    torch.int32: cbd.CUtensorMapDataType.CU_TENSOR_MAP_DATA_TYPE_INT32,
    torch.int64: cbd.CUtensorMapDataType.CU_TENSOR_MAP_DATA_TYPE_INT64,
    torch.uint8: cbd.CUtensorMapDataType.CU_TENSOR_MAP_DATA_TYPE_UINT8,
    torch.uint16: cbd.CUtensorMapDataType.CU_TENSOR_MAP_DATA_TYPE_UINT16,
    torch.uint32: cbd.CUtensorMapDataType.CU_TENSOR_MAP_DATA_TYPE_UINT32,
    torch.uint64: cbd.CUtensorMapDataType.CU_TENSOR_MAP_DATA_TYPE_UINT64,
    torch.float32: cbd.CUtensorMapDataType.CU_TENSOR_MAP_DATA_TYPE_FLOAT32,
    torch.float16: cbd.CUtensorMapDataType.CU_TENSOR_MAP_DATA_TYPE_FLOAT16,
    torch.bfloat16: cbd.CUtensorMapDataType.CU_TENSOR_MAP_DATA_TYPE_BFLOAT16,
    torch.float8_e4m3fn: cbd.CUtensorMapDataType.CU_TENSOR_MAP_DATA_TYPE_UINT8,
    torch.float8_e4m3fnuz: cbd.CUtensorMapDataType.CU_TENSOR_MAP_DATA_TYPE_UINT8,
    torch.float8_e5m2: cbd.CUtensorMapDataType.CU_TENSOR_MAP_DATA_TYPE_UINT8,
    torch.float8_e5m2fnuz: cbd.CUtensorMapDataType.CU_TENSOR_MAP_DATA_TYPE_UINT8,
}

swizzle_type_map = {
    0: cbd.CUtensorMapSwizzle.CU_TENSOR_MAP_SWIZZLE_NONE,
    16: cbd.CUtensorMapSwizzle.CU_TENSOR_MAP_SWIZZLE_NONE,
    32: cbd.CUtensorMapSwizzle.CU_TENSOR_MAP_SWIZZLE_32B,
    64: cbd.CUtensorMapSwizzle.CU_TENSOR_MAP_SWIZZLE_64B,
    128: cbd.CUtensorMapSwizzle.CU_TENSOR_MAP_SWIZZLE_128B,
}

WIDTH, HEIGHT=32, 32
# t = torch.ones(HEIGHT, WIDTH, device="cuda").half()

# gmem_dims = (cbd.cuuint64_t(WIDTH), cbd.cuuint64_t(HEIGHT))
# gmem_strides = (cbd.cuuint64_t(WIDTH * t.element_size()),)
# smem_dims = (cbd.cuuint32_t(WIDTH), cbd.cuuint32_t(HEIGHT))

# tensor_dtype = tmap_type_map[t.dtype]
# ret = cbd.cuTensorMapEncodeTiled(
#     tensor_dtype,
#     2, # num dims.
#     t.data_ptr(),
#     gmem_dims,
#     gmem_strides,
#     smem_dims,
#     (cbd.cuuint32_t(1),) * 2,
#     cbd.CUtensorMapInterleave.CU_TENSOR_MAP_INTERLEAVE_NONE,
#     swizzle_type_map[0],
#     # cbd.CUtensorMapL2promotion.CU_TENSOR_MAP_L2_PROMOTION_L2_256B,
#     cbd.CUtensorMapL2promotion.CU_TENSOR_MAP_L2_PROMOTION_NONE,
#     cbd.CUtensorMapFloatOOBfill.CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE,
# )
# print(ret)
def get_mma_kernel():
    # with open("01_sm80_ptx.cu", "r") as f:
    #     KERNEL_SOURCE = f.read()
    tic = time.time()
    kernel = _compile_kernel(
        open("kernel_mma.cu", "r").read(),
        kernel_name="mma_kernel",
    )
    toc = time.time()
    print("compile used time: ", toc - tic)
    return kernel

# 1/0

def div_up(x):
    return (x + 255) // 256

def get_torch_prof_ctx(do_perf = False):
    ctx = torch.profiler.profile(
        activities=[torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA],
        record_shapes=True,
        with_stack=False if torch.cuda.device_count() == 1 else False,
    ) 
    return ctx if do_perf else nullcontext()



def bench():
    ctx =  get_torch_prof_ctx()
    with ctx:
        for exp in range(28, 30):
            M = 2 ** exp

            input1 = torch.randn(M, device="cuda")
            input2 = torch.randn(M, device="cuda")
            output = torch.empty(M, device="cuda")
            def call():
                return kernel((div_up(M//4),1,1), (256,1,1), (input1, input2, output, input1.numel()))
            call()

            torch.testing.assert_close(output, input1 + input2)
            
            tic = do_bench(call, warmup=100, rep=500)
            print("tic", tic)
            bandwidth_GB = M * 4 * 2 / (tic * 1e-3) / 1e3 / 1e3 /1e3 # KB -> MB -> GB
            print(f"M={M:10,}, bandwidth_GB={bandwidth_GB:10.2f} GB/s, ms: {tic:10.2f} ms, block_num: {div_up(M//4):10,}")

    if type(ctx) == torch.profiler.profile:
        ctx.export_chrome_trace(f"00.json")
        


def get_kernel(kernel_name, file_name="01_sm80_ptx.cu"):
    tic = time.time()
    kernel = _compile_kernel(
        open(file_name, "r").read(),
        kernel_name=kernel_name,
        nvcc_options=["-std=c++17", "-D__CUDA_NO_HALF_OPERATORS__", "-D__CUDA_NO_HALF_CONVERSIONS__", "-D__CUDA_NO_BFLOAT16_CONVERSIONS__", "-D__CUDA_NO_HALF2_OPERATORS__"],
        save_ptx=True,
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
    time.sleep(0.5)
    
# test_ld_matrix_kernel()

def test_mma_ptx_kernel():
    mma_ptx_kernel = get_kernel("mma_ptx_kernel", file_name="02_mma_ptx.cu")
    a = torch.arange(16 * 16, device="cuda").half() * 0.1
    b = torch.arange(16 * 16, device="cuda").half() * 0.1
    c = torch.zeros(16 * 16 * 16, device="cuda").half()
    d = torch.matmul(a.reshape(16, 16), b.reshape(16, 16).T)
    mma_ptx_kernel((1,1,1), (32,1,1), (c, a, b, d))
    torch.cuda.synchronize()
    time.sleep(0.5)

# test_mma_ptx_kernel()

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
    tma_2d_kernel = get_kernel("tma_2d_kernel", file_name="02_mma_ptx.cu")
    grid_size = (2, 1, 1)
    block_size = (4 * 32, 1, 1)
    WIDTH, HEIGHT=32, 32
    W, H = 16, 16
    input = torch.arange(WIDTH * HEIGHT, device="cuda").half()
    gmem_dims = (cbd.cuuint64_t(WIDTH), cbd.cuuint64_t(HEIGHT))
    gmem_strides = (cbd.cuuint64_t(WIDTH * input.element_size()),)
    box_stride = (cbd.cuuint32_t(W), cbd.cuuint32_t(H))

    tensor_dtype = tmap_type_map[input.dtype]

    result, tensor_map = cbd.cuTensorMapEncodeTiled(
        tensor_dtype,
        2, # num dims.
        input.data_ptr(),
        gmem_dims,
        gmem_strides,
        box_stride,
        (cbd.cuuint32_t(1),) * 2,
        cbd.CUtensorMapInterleave.CU_TENSOR_MAP_INTERLEAVE_NONE,
        swizzle_type_map[0],
        # cbd.CUtensorMapL2promotion.CU_TENSOR_MAP_L2_PROMOTION_L2_256B,
        cbd.CUtensorMapL2promotion.CU_TENSOR_MAP_L2_PROMOTION_NONE,
        cbd.CUtensorMapFloatOOBfill.CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE, 
    )
    _check_cuda(result)
    tma_2d_kernel(grid_size, block_size, args=[tensor_map])
test_tma_2d_kernel()






