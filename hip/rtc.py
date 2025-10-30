import torch
from typing import Optional, Union, Any
import ctypes
import sys
import time
import os
import triton
import triton.language as tl

def _get_gpu_rtc_compatible_flags() -> list[str]:
    """
    Get HIPCC/NVCC flags that are compatible with NVRTC compilation.

    Returns:
        List of HIPCC/NVCC flags that can be safely used with NVRTC.
    """
    from torch.utils.cpp_extension import COMMON_HIPCC_FLAGS, COMMON_NVCC_FLAGS

    nvrtc_unsupported_flags = {
        "--expt-relaxed-constexpr",
    }

    # Filter out unsupported flags
    compatible_flags = [
        flag for flag in COMMON_NVCC_FLAGS if flag not in nvrtc_unsupported_flags
    ]

    if torch.version.hip:
        compatible_flags.extend(COMMON_HIPCC_FLAGS)

    return compatible_flags

class _CudaModule:
    def __init__(self, module: ctypes.c_void_p) -> None:
        self._module = module
        self._kernels: dict[str, _CudaKernel] = {}

    def __getattr__(self, name: str) -> "_CudaKernel":
        if name in self._kernels:
            return self._kernels[name]

        # Import the CUDA library inside the method
        # pyrefly: ignore  # missing-module-attribute

        libcuda = _get_gpu_runtime_library()

        func = ctypes.c_void_p()
        try:
            _check_cuda(
                libcuda.cuModuleGetFunction(
                    ctypes.byref(func), self._module, name.encode("utf-8")
                )
            )
            kernel = _CudaKernel(func, self._module)
            self._kernels[name] = kernel
            return kernel

        except RuntimeError as err:
            raise AttributeError(f"No kernel named '{name}' in this module") from err
def _get_hiprtc_library() -> ctypes.CDLL:
    if sys.platform == "win32":
        version_str = "".join(["0", torch.version.hip[0], "0", torch.version.hip[2]])
        lib = ctypes.CDLL(f"hiprtc{version_str}.dll")
    else:
        lib = ctypes.CDLL("libhiprtc.so")

    # Provide aliases for HIP RTC functions to match NVRTC API
    lib.nvrtcGetErrorString = lib.hiprtcGetErrorString  # type: ignore[attr-defined]
    lib.nvrtcCreateProgram = lib.hiprtcCreateProgram  # type: ignore[attr-defined]
    lib.nvrtcDestroyProgram = lib.hiprtcDestroyProgram  # type: ignore[attr-defined]
    lib.nvrtcCompileProgram = lib.hiprtcCompileProgram  # type: ignore[attr-defined]
    lib.nvrtcGetPTXSize = lib.hiprtcGetCodeSize  # type: ignore[attr-defined]
    lib.nvrtcGetPTX = lib.hiprtcGetCode  # type: ignore[attr-defined]
    lib.nvrtcGetProgramLogSize = lib.hiprtcGetProgramLogSize  # type: ignore[attr-defined]
    lib.nvrtcGetProgramLog = lib.hiprtcGetProgramLog  # type: ignore[attr-defined]
    lib.nvrtcAddNameExpression = lib.hiprtcAddNameExpression  # type: ignore[attr-defined]
    lib.nvrtcGetLoweredName = lib.hiprtcGetLoweredName  # type: ignore[attr-defined]
    return lib


def _get_nvrtc_library() -> ctypes.CDLL:
    major_version = int(torch.version.cuda.split(".")[0])  # type: ignore[union-attr]
    if sys.platform == "win32":
        nvrtc_libs = [
            f"nvrtc64_{major_version}0_0.dll",
        ]
    else:
        nvrtc_libs = [
            f"libnvrtc.so.{major_version}",
            "libnvrtc.so",  # Fallback to unversioned
        ]
    for lib_name in nvrtc_libs:
        try:
            return ctypes.CDLL(lib_name)
        except OSError:
            continue
    raise OSError("Could not find any NVRTC library")


def _get_gpu_rtc_library() -> ctypes.CDLL:
    # Since PyTorch already loads the GPU RTC library, we can use the system library
    # which should be compatible with PyTorch's version
    if torch.version.hip:
        return _get_hiprtc_library()
    else:
        return _get_nvrtc_library()

def _get_hip_runtime_library() -> ctypes.CDLL:
    if False and sys.platform == "win32":
        lib = ctypes.CDLL(f"amdhip64_{torch.version.hip[0]}.dll")
    else:  # Unix-based systems
        lib = ctypes.CDLL("libamdhip64.so")
    lib.cuGetErrorString = lib.hipGetErrorString  # type: ignore[attr-defined]
    lib.cuModuleLoadData = lib.hipModuleLoadData  # type: ignore[attr-defined]
    lib.cuModuleGetFunction = lib.hipModuleGetFunction  # type: ignore[attr-defined]
    lib.cuLaunchKernel = lib.hipModuleLaunchKernel  # type: ignore[attr-defined]
    lib.cuFuncSetAttribute = lib.hipFuncSetAttribute  # type: ignore[attr-defined]
    return lib


def _get_gpu_runtime_library() -> ctypes.CDLL:
    if torch.version.hip:
        return _get_hip_runtime_library()
    else:
        assert False
        return _get_cuda_runtime_library()



def _check_cuda(result: int) -> None:
    if result == 0:
        return
    err_str = ctypes.c_char_p()
    libcuda = _get_gpu_runtime_library()  # Get reference to CUDA library
    libcuda.cuGetErrorString(result, ctypes.byref(err_str))
    error_message = (
        err_str.value.decode() if err_str.value is not None else "Unknown CUDA error"
    )
    raise RuntimeError(f"CUDA error: {error_message}")


def _nvrtc_compile(
    kernel_source: str,
    kernel_name: str,
    compute_capability: Optional[str] = None,
    cuda_include_dirs: Optional[list] = None,
    nvcc_options: Optional[list] = None,
    auto_pch: bool = False,
) -> tuple[bytes, str]:
    """
    Compiles a CUDA kernel using NVRTC and returns the PTX code.

    Args:
        kernel_source (str): The CUDA kernel source code as a string
        kernel_name (str): The name of the kernel function to compile
        compute_capability (str, None): The compute capability to target (e.g., "86").
                                           If None, will detect from current device.
        cuda_include_dirs (list, None): List of directories containing CUDA headers
        nvcc_options (list, None): Additional options to pass to NVRTC
        auto_pch (bool): Enable automatic precompiled headers (CUDA 12.8+)

    Returns:
        Tuple[bytes, str]: The compiled PTX code and mangled kernel name
    """
    # Ensure CUDA is initialized

    # Load NVRTC library
    libnvrtc = _get_gpu_rtc_library()

    # NVRTC constants
    NVRTC_SUCCESS = 0

    # Helper: check NVRTC errors
    def check_nvrtc(result: int) -> None:
        if result != NVRTC_SUCCESS:
            err_str = ctypes.c_char_p()
            libnvrtc.nvrtcGetErrorString(result, ctypes.byref(err_str))
            error_message = (
                err_str.value.decode()
                if err_str.value is not None
                else "Unknown CUDA error"
            )
            raise RuntimeError(f"CUDA error: {error_message}")

    # Convert source to bytes
    source_bytes = kernel_source.encode("utf-8")

    # Get compute capability if not provided
    if compute_capability is None:
        props = torch.cuda.get_device_properties(torch.cuda.current_device())
        if torch.version.hip:
            compute_capability = f"{props.gcnArchName}"
        else:
            compute_capability = f"{props.major}{props.minor}"

    # Prepare compilation options
    options = []
    if torch.version.hip:
        options.append(f"--offload-arch={compute_capability}".encode())
    else:
        options.append(f"--gpu-architecture=sm_{compute_capability}".encode())

    # Auto-detect and add CUDA include paths
    from torch.utils.cpp_extension import include_paths

    cuda_include_paths = include_paths("cuda")
    for cuda_path in cuda_include_paths:
        options.append(f"-I{cuda_path}".encode())

    # Add custom include directories
    if cuda_include_dirs:
        for directory in cuda_include_dirs:
            options.append(f"-I{directory}".encode())

    # Enable automatic precompiled headers (CUDA 12.8+)
    if auto_pch:
        assert str(torch.version.cuda) >= "12.8", "PCH requires CUDA 12.8+"
        if nvcc_options is None:
            nvcc_options = []
        nvcc_options.append("--pch")

    # Add custom NVCC options
    if nvcc_options:
        for option in nvcc_options:
            options.append(option.encode("utf-8"))

    nvrtc_compatible_flags = _get_gpu_rtc_compatible_flags()
    options.extend([flag.encode("utf-8") for flag in nvrtc_compatible_flags])

    # Convert options to C array
    num_options = len(options)
    options_array = (ctypes.c_char_p * num_options)(*options)

    # Create program
    prog = ctypes.c_void_p()
    check_nvrtc(
        libnvrtc.nvrtcCreateProgram(
            ctypes.byref(prog),
            source_bytes,
            f"{kernel_name}.cu".encode(),
            0,
            None,
            None,
        )
    )

    # Add kernel name, which can be a template expression
    c_kernel_name = kernel_name.encode("utf-8")
    check_nvrtc(libnvrtc.nvrtcAddNameExpression(prog, c_kernel_name))

    # Compile program
    res = libnvrtc.nvrtcCompileProgram(prog, num_options, options_array)

    # Handle compilation errors
    if res != NVRTC_SUCCESS:
        # Get log
        log_size = ctypes.c_size_t()
        libnvrtc.nvrtcGetProgramLogSize(prog, ctypes.byref(log_size))
        log = ctypes.create_string_buffer(log_size.value)
        libnvrtc.nvrtcGetProgramLog(prog, log)
        raise RuntimeError(f"Kernel compilation failed:\n{log.value.decode()}")

    # Get PTX
    ptx_size = ctypes.c_size_t()
    check_nvrtc(libnvrtc.nvrtcGetPTXSize(prog, ctypes.byref(ptx_size)))
    ptx = ctypes.create_string_buffer(ptx_size.value)
    check_nvrtc(libnvrtc.nvrtcGetPTX(prog, ptx))

    # Get mangled name
    c_mangled_name = ctypes.c_char_p()
    check_nvrtc(
        libnvrtc.nvrtcGetLoweredName(prog, c_kernel_name, ctypes.byref(c_mangled_name))
    )
    if c_mangled_name.value is not None:
        mangled_name = c_mangled_name.value.decode()  # make a copy
    else:
        mangled_name = ""

    libnvrtc.nvrtcDestroyProgram(ctypes.byref(prog))

    # For HIP, hipRTC generates raw CO binaries instead of PTX,
    # and for some reason, ".value" causes the string to be truncated,
    # likely due to the presence of '\0' in the string. So we use .raw instead.
    # print(ptx)
    ptx_bytes = ptx.raw if torch.version.hip else ptx.value
    return ptx_bytes, mangled_name

def _compile_kernel(
    kernel_source: str,
    kernel_name: str,
    compute_capability: Optional[str] = None,
    cuda_include_dirs: Optional[list] = None,
    nvcc_options: Optional[list] = None,
    save_ptx: bool = False,
):
    """
    Compiles a CUDA kernel using NVRTC and returns a callable function.

    This function is a wrapper for NVRTC that enables runtime compilation of CUDA kernels.
    Note that this returns a raw CUDA kernel that operates on raw memory pointers.
    To use this kernel as a proper PyTorch operator, you should wrap it following the guide at:
    pytorch.org/tutorials/advanced/python_custom_ops.html

    Args:
        kernel_source (str): The CUDA kernel source code as a string
        kernel_name (str): The name of the kernel function to compile
        compute_capability (str, optional): The compute capability to target (e.g., "86").
                                           If None, will detect from current device.
        cuda_include_dirs (list, optional): List of directories containing CUDA headers
        nvcc_options (list, optional): Additional options to pass to NVRTC

    Returns:
        callable: A Python function that can be called with PyTorch tensor arguments to execute the kernel

    Example:
        >>> # xdoctest: +SKIP
        >>> kernel_code = '''
        extern "C"
        __global__ void add_tensors(const float* a, const float* b, float* c, int n) {
            int i = threadIdx.x + blockIdx.x * blockDim.x;
            if (i < n)
                c[i] = a[i] + b[i];
        }
        '''
        >>> add_kernel = torch.cuda.compile_kernel(kernel_code, "add_tensors")
        >>> a = torch.randn(1024, device="cuda")
        >>> b = torch.randn(1024, device="cuda")
        >>> c = torch.empty_like(a)
        >>> add_kernel(grid=(4, 1, 1), block=(256, 1, 1), args=[a, b, c, a.numel()])
    """
    # from torch.cuda._utils import _cuda_load_module, _nvrtc_compile

    # Compile the kernel to PTX
    ptx, mangled_name = _nvrtc_compile(
        kernel_source,
        kernel_name,
        compute_capability,
        cuda_include_dirs,
        nvcc_options,
    )

    # Load the module and get the kernel
    result = _cuda_load_module(ptx, [mangled_name])

    if isinstance(result, dict):
        return result[mangled_name]
    else:
        # This branch shouldn't be executed if kernel_names is provided,
        # but MyPy needs this to understand type narrowing
        return getattr(result, mangled_name)
    


def _cuda_load_module(
    ptx: Union[str, bytes], kernel_names: Optional[list[str]] = None
) -> Union[_CudaModule, dict[str, "_CudaKernel"]]:
    """
    Loads a CUDA module from PTX code and returns a module object that can access kernels.

    Args:
        ptx (bytes or str): The PTX code to load
        kernel_names (list, optional): List of kernel names to extract from the module.
                                      If None, will return a module object with __getattr__.

    Returns:
        object: If kernel_names is None, returns a module object with __getattr__ to access kernels.
               If kernel_names is provided, returns a dict mapping kernel names to _CudaKernel objects.
    """
    # Ensure CUDA is initialized
    # import torch.cuda
    import torch.cuda

    # Load CUDA driver library
    libcuda = _get_gpu_runtime_library()

    # Convert PTX to bytes if it's a string
    if isinstance(ptx, str):
        ptx = ptx.encode("utf-8")

    # Load PTX module
    module = ctypes.c_void_p()
    # Get the current stream without directly importing torch.cuda at module level
    stream = torch.cuda.current_stream()
    with stream:
        _check_cuda(libcuda.cuModuleLoadData(ctypes.byref(module), ptx))

    if not kernel_names:
        return _CudaModule(module)

    # Return specific kernels
    kernels = {}
    for name in kernel_names:
        func = ctypes.c_void_p()
        _check_cuda(
            libcuda.cuModuleGetFunction(
                ctypes.byref(func), module, name.encode("utf-8")
            )
        )
        kernels[name] = _CudaKernel(func, module)
    return kernels


class _CudaKernel:
    """
    Represents a compiled CUDA kernel that can be called with PyTorch tensors.
    """

    def __init__(self, func: ctypes.c_void_p, module: ctypes.c_void_p) -> None:
        self.func = func
        self.module = module
        self._max_shared_mem_bytes = 0

    def __call__(
        self,
        grid: tuple[int, int, int] = (1, 1, 1),
        block: tuple[int, int, int] = (1, 1, 1),
        args: Optional[list] = None,
        shared_mem: int = 0,
        stream: Optional[Any] = None,
    ) -> None:
        """
        Call the compiled CUDA kernel

        Args:
            grid (tuple): Grid dimensions (grid_x, grid_y, grid_z)
            block (tuple): Block dimensions (block_x, block_y, block_z)
            args (list): List of arguments to pass to the kernel.
                         PyTorch tensor arguments will be automatically converted to pointers.
            shared_mem (int): Shared memory size in bytes
            stream (torch.cuda.Stream): CUDA stream to use. If None, uses current stream.
        """

        libcuda = torch.cuda._utils._get_gpu_runtime_library()

        if not args:
            args = []

        # Process arguments and convert tensors to pointers
        processed_args: list[ctypes.c_void_p] = []
        c_args = []

        for arg in args:
            if isinstance(arg, torch.Tensor):
                if not arg.is_cuda and not (arg.is_cpu and arg.is_pinned()):
                    raise ValueError(
                        "All tensor arguments must be CUDA tensors or pinned CPU tensors"
                    )
                # Get pointer to tensor data
                ptr = ctypes.c_void_p(arg.data_ptr())
                processed_args.append(ptr)
                c_args.append(ctypes.byref(ptr))
            elif isinstance(arg, int):
                # Convert integers to C int
                c_int = ctypes.c_int(arg)
                # Store the C int for reference keeping, not in processed_args
                c_args.append(ctypes.byref(c_int))
            elif isinstance(arg, float):
                # Python floats are doubles - use double by default
                c_double = ctypes.c_double(arg)
                # Store the C double for reference keeping, not in processed_args
                c_args.append(ctypes.byref(c_double))
            else:
                raise TypeError(f"Unsupported argument type: {type(arg)}")

        # Convert to array of void pointers
        c_args_array = (ctypes.c_void_p * len(c_args))()
        for i, arg in enumerate(c_args):
            c_args_array[i] = ctypes.cast(arg, ctypes.c_void_p)

        # Get the stream
        if stream is None:
            # Defer import to avoid circular imports
            stream = torch.cuda.current_stream()

        # Check if kernel requires large shared memory but hasn't been configured
        if shared_mem >= 48 * 1024 and (
            self._max_shared_mem_bytes == 0 or shared_mem > self._max_shared_mem_bytes
        ):
            configured_msg = (
                "not configured"
                if self._max_shared_mem_bytes == 0
                else f"only {self._max_shared_mem_bytes} bytes configured"
            )
            raise RuntimeError(
                f"Kernel requires {shared_mem} bytes of shared memory (>= 48KB), "
                f"but {configured_msg}. "
                "Call kernel.set_shared_memory_config(shared_mem) after compilation "
                "and before launching the kernel."
            )

        _check_cuda(
            libcuda.cuLaunchKernel(
                self.func,
                grid[0],
                grid[1],
                grid[2],
                block[0],
                block[1],
                block[2],
                shared_mem,
                stream._as_parameter_,
                c_args_array,
                None,
            )
        )

    def set_shared_memory_config(self, shared_mem_bytes: int) -> None:
        if shared_mem_bytes < 48 * 1024:
            # No configuration needed for <= 48KB, just update the value
            self._max_shared_mem_bytes = shared_mem_bytes
            return

        libcuda = _get_gpu_runtime_library()

        # Get device properties to validate against limits
        device_props = torch.cuda.get_device_properties()
        # HIP doesn't have shared_memory_per_block_optin in device properties, so we hard-code it here
        if torch.version.hip:
            # navi, CDNA1-CDNA3 allows a max of 64KB shared memory
            # CDNA4 allows a max of 160KB shared memory
            max_shared_mem = (
                65536 if device_props.gcnArchName != "gfx950" else 160 * 1024
            )
        else:
            max_shared_mem = getattr(
                device_props, "shared_memory_per_block_optin", 49152
            )

        if shared_mem_bytes > max_shared_mem:
            raise RuntimeError(
                f"Requested shared memory ({shared_mem_bytes} bytes) exceeds "
                f"device limit ({max_shared_mem} bytes). "
                "Consider reducing block size or shared memory usage."
            )

        # Set the function attribute once
        # https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__TYPES.html
        cudaFuncAttributeMaxDynamicSharedMemorySize = 8
        _check_cuda(
            libcuda.cuFuncSetAttribute(
                self.func,
                cudaFuncAttributeMaxDynamicSharedMemorySize,
                shared_mem_bytes,
            )
        )

        self._max_shared_mem_bytes = shared_mem_bytes



@triton.jit
def matmul_kernel(
        # Pointers to matrices
        a_ptr, b_ptr, c_ptr,
        # Matrix dimensions
        M: tl.constexpr, N: tl.constexpr, K: tl.constexpr,
        # The stride variables represent how much to increase the ptr by when moving by 1
        # element in a particular dimension. E.g. `stride_am` is how much to increase `a_ptr`
        # by to get the element one row down (A has M rows).
        stride_am: tl.constexpr, stride_ak: tl.constexpr,  #
        stride_bk: tl.constexpr, stride_bn: tl.constexpr,  #
        stride_cm: tl.constexpr, stride_cn: tl.constexpr,
        # Meta-parameters
        BLOCK_SIZE_M: tl.constexpr, BLOCK_SIZE_N: tl.constexpr, BLOCK_SIZE_K: tl.constexpr,  #
        GROUP_SIZE_M: tl.constexpr,  #
        ACTIVATION: tl.constexpr  #
):
    """Kernel for computing the matmul C = A x B.
    A has shape (M, K), B has shape (K, N) and C has shape (M, N)
    """
    # -----------------------------------------------------------
    # Map program ids `pid` to the block of C it should compute.
    # This is done in a grouped ordering to promote L2 data reuse.
    # See above `L2 Cache Optimizations` section for details.
    pid = tl.program_id(axis=0)
    num_pid_m = tl.cdiv(M, BLOCK_SIZE_M)
    num_pid_n = tl.cdiv(N, BLOCK_SIZE_N)
    num_pid_in_group = GROUP_SIZE_M * num_pid_n
    group_id = pid // num_pid_in_group
    first_pid_m = group_id * GROUP_SIZE_M
    group_size_m = min(num_pid_m - first_pid_m, GROUP_SIZE_M)
    pid_m = first_pid_m + ((pid % num_pid_in_group) % group_size_m)
    pid_n = (pid % num_pid_in_group) // group_size_m

    # -----------------------------------------------------------
    # Add some integer bound assumptions.
    # This helps to guide integer analysis in the backend to optimize
    # load/store offset address calculation
    tl.assume(pid_m >= 0)
    tl.assume(pid_n >= 0)
    tl.assume(stride_am > 0)
    tl.assume(stride_ak > 0)
    tl.assume(stride_bn > 0)
    tl.assume(stride_bk > 0)
    tl.assume(stride_cm > 0)
    tl.assume(stride_cn > 0)

    # ----------------------------------------------------------
    # Create pointers for the first blocks of A and B.
    # We will advance this pointer as we move in the K direction
    # and accumulate
    # `a_ptrs` is a block of [BLOCK_SIZE_M, BLOCK_SIZE_K] pointers
    # `b_ptrs` is a block of [BLOCK_SIZE_K, BLOCK_SIZE_N] pointers
    # See above `Pointer Arithmetic` section for details
    offs_am = (pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)) % M
    offs_bn = (pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)) % N
    offs_k = tl.arange(0, BLOCK_SIZE_K)
    a_ptrs = a_ptr + (offs_am[:, None] * stride_am + offs_k[None, :] * stride_ak)
    b_ptrs = b_ptr + (offs_k[:, None] * stride_bk + offs_bn[None, :] * stride_bn)

    # -----------------------------------------------------------
    # Iterate to compute a block of the C matrix.
    # We accumulate into a `[BLOCK_SIZE_M, BLOCK_SIZE_N]` block
    # of fp32 values for higher accuracy.
    # `accumulator` will be converted back to fp16 after the loop.
    accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
    for k in range(0, tl.cdiv(K, BLOCK_SIZE_K)):
        # Load the next block of A and B, generate a mask by checking the K dimension.
        # If it is out of bounds, set it to 0.
        a = tl.load(a_ptrs, mask=offs_k[None, :] < K - k * BLOCK_SIZE_K, other=0.0)
        b = tl.load(b_ptrs, mask=offs_k[:, None] < K - k * BLOCK_SIZE_K, other=0.0)
        # We accumulate along the K dimension.
        accumulator = tl.dot(a, b, accumulator)
        # Advance the ptrs to the next K block.
        a_ptrs += BLOCK_SIZE_K * stride_ak
        b_ptrs += BLOCK_SIZE_K * stride_bk
    # You can fuse arbitrary activation functions here
    # while the accumulator is still in FP32!
    c = accumulator.to(tl.bfloat16)

    # -----------------------------------------------------------
    # Write back the block of the output matrix C with masks.
    offs_cm = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_cn = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    c_ptrs = c_ptr + stride_cm * offs_cm[:, None] + stride_cn * offs_cn[None, :]
    c_mask = (offs_cm[:, None] < M) & (offs_cn[None, :] < N)
    tl.store(c_ptrs, c, mask=c_mask)
    
def get_triton_gemm_NTN(A, B, C, M, N, K):
    B = B.T
    BLOCK_M = 256
    BLOCK_N = 256
    BLOCK_K = 32
    grid = (triton.cdiv(M, BLOCK_M) * triton.cdiv(N, BLOCK_N),)
    matmul_kernel[grid](
        a_ptr = A, 
        b_ptr = B, 
        c_ptr = C, 
        M = M, 
        N = N, 
        K = K, 
        stride_am = A.stride(0), 
        stride_ak = A.stride(1), 
        stride_bk = B.stride(0), 
        stride_bn = B.stride(1), 
        stride_cm = C.stride(0), 
        stride_cn = C.stride(1), 
        BLOCK_SIZE_M = BLOCK_M, 
        BLOCK_SIZE_N = BLOCK_N, 
        BLOCK_SIZE_K = BLOCK_K, 
        GROUP_SIZE_M = 1, 
        ACTIVATION = None, 
    )
    

def my_assert_close(output, ref_output):
    if not torch.allclose(output, ref_output, atol=1e-3, rtol=1e-3):
        print("C is not close to A @ B")
        diff = output - ref_output
        print(diff)
        print("diff", (diff).max().item())
        # max_diff_idx = diff.abs().argmax()
        # max_diff_row = max_diff_idx // N
        # max_diff_col = max_diff_idx % N
        # print(f"Max diff at position ({max_diff_row}, {max_diff_col})")
        # print(f"C[{max_diff_row}, {max_diff_col}] = {C[max_diff_row, max_diff_col]}")
        # print(f"Expected = {right_output[max_diff_row, max_diff_col]}")
        print(f"{output=}")
        print(f"{ref_output=}")
        print(f"{diff.abs().mean()=}")
    
        torch.set_printoptions(threshold=1000, edgeitems=200, linewidth=200)     
        print(f"{diff.reshape(-1).sort()[0]=}")
        torch.set_printoptions(threshold=1000, edgeitems=3)     
        return diff
    print("test passed")
    return None
    # torch.testing.assert_close(C, right_output)
    # print(C, right_output)
