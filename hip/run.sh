rocprof-compute profile -n mm_2 --no-roof -- python main.py

rocprof-compute profile --no-roof -n pc_test -b 21 --kernel fp16_gemm_full_NTN --pc-sampling-method stochastic --pc-sampling-interval 1048576 -VVV -- python main.py 

rocprof-compute analyze -p workloads/mm_2/MI300X_A1/ --gui