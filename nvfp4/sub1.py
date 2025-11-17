import torch
# Helper function to convert scale factor tensor to blocked format
def ceil_div(a, b):
    return (a + b - 1) // b
def to_blocked(input_matrix):
    rows, cols = input_matrix.shape

    # Please ensure rows and cols are multiples of 128 and 4 respectively
    n_row_blocks = ceil_div(rows, 128)
    n_col_blocks = ceil_div(cols, 4)

    padded = input_matrix
    blocks = padded.view(n_row_blocks, 128, n_col_blocks, 4).permute(0, 2, 1, 3)
    rearranged = blocks.reshape(-1, 4, 32, 4).transpose(1, 2).reshape(-1, 32, 16)

    return rearranged.flatten()
is_first = False
def custom_kernel(data):
    """
    PyTorch reference implementation of NVFP4 block-scaled GEMV.
    """
    a_ref, b_ref, sfa_ref_cpu, sfb_ref_cpu, _, _, c_ref = data
    global is_first
    # if not is_first:
    #     is_first = True
    print(f"a_ref: {a_ref.shape} {a_ref.stride()} {a_ref.dtype}")
    print(f"b_ref: {b_ref.shape} {b_ref.stride()} {b_ref.dtype}")
    print(f"sfa_ref_cpu: {sfa_ref_cpu.shape} {sfa_ref_cpu.stride()} {sfa_ref_cpu.dtype}")
    print(f"sfb_ref_cpu: {sfb_ref_cpu.shape} {sfb_ref_cpu.stride()} {sfb_ref_cpu.dtype}")
    
    # Get dimensions from MxNxL layout
    _, _, l = c_ref.shape

    # Call torch._scaled_mm to compute the GEMV result
    for l_idx in range(l):
        # Convert the scale factor tensor to blocked format
        scale_a = to_blocked(sfa_ref_cpu[:, :, l_idx])
        scale_b = to_blocked(sfb_ref_cpu[:, :, l_idx])
        # (m, k) @ (n, k).T -> (m, n)
        res = torch._scaled_mm(
            a_ref[:, :, l_idx],
            b_ref[:, :, l_idx].transpose(0, 1),
            scale_a.cuda(),
            scale_b.cuda(),
            bias=None,
            out_dtype=torch.float16,
        )
        c_ref[:, 0, l_idx] = res[:, 0]
    print("\n" * 3)
    return c_ref