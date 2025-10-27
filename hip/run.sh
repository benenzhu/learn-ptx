rocprof-compute profile -n mm_v2a --no-roof --kernel fp16_gemm_full_NTN -- python main.py

rocprof-compute profile --no-roof -n pc_test -b 21 --kernel fp16_gemm_full_NTN --pc-sampling-method stochastic --pc-sampling-interval 1048576 -VVV -- python main.py 

rocprof-compute analyze -p workloads/pc_test/MI300X_A1/ --gui