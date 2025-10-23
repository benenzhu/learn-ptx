import torch
from triton.testing import do_bench
from contextlib import nullcontext
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