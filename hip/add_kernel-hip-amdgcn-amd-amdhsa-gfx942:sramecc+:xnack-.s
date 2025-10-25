	.file	0 "/root/learn-cuda/02c_matmul_hip/learn-ptx/hip" "add_kernel.cu" md5 0xac08225ccc767712b7f3551a3aa35d67
	.file	1 "/tmp/comgr-87174f/input" "add_kernel.cu"
	.file	2 "/tmp/comgr-87174f/include" "hiprtc_runtime.h"
	.amdgcn_target "amdgcn-amd-amdhsa--gfx942:sramecc+:xnack-"
	.amdhsa_code_object_version 6
	.text
	.protected	_Z10add_kernelPKiS0_Piii ; -- Begin function _Z10add_kernelPKiS0_Piii
	.globl	_Z10add_kernelPKiS0_Piii
	.p2align	8
	.type	_Z10add_kernelPKiS0_Piii,@function
_Z10add_kernelPKiS0_Piii:               ; @_Z10add_kernelPKiS0_Piii
.Lfunc_begin0:
	.cfi_sections .debug_frame
	.cfi_startproc
; %bb.0:
	.cfi_escape 0x0f, 0x04, 0x30, 0x36, 0xe9, 0x02 ; 
	.cfi_undefined 16
	.loc	2 2484 117 prologue_end         ; /tmp/comgr-87174f/include/hiprtc_runtime.h:2484:117
	s_load_dword s10, s[0:1], 0x1c
.Ltmp0:
	;DEBUG_VALUE: add_kernel:N <- [DW_OP_LLVM_poisoned] $sgpr10
	;DEBUG_VALUE: add_kernel:A <- [DW_OP_LLVM_poisoned] undef
	;DEBUG_VALUE: add_kernel:B <- [DW_OP_LLVM_poisoned] undef
	;DEBUG_VALUE: add_kernel:C <- [DW_OP_LLVM_poisoned] undef
	;DEBUG_VALUE: add_kernel:M <- undef
	;DEBUG_VALUE: add_kernel:tid <- [DW_OP_LLVM_poisoned] $vgpr0
	;DEBUG_VALUE: col <- [DW_OP_LLVM_poisoned] $vgpr0
	;DEBUG_VALUE: add_kernel:tb_size <- [DW_OP_LLVM_poisoned] undef
	;DEBUG_VALUE: add_kernel:A <- undef
	;DEBUG_VALUE: add_kernel:B <- undef
	;DEBUG_VALUE: add_kernel:C <- undef
	.loc	1 16 27                         ; /tmp/comgr-87174f/input/add_kernel.cu:16:27
	s_waitcnt lgkmcnt(0)
	v_cmp_gt_i32_e32 vcc, s10, v0
.Ltmp1:
	;DEBUG_VALUE: add_kernel:bid <- [DW_OP_LLVM_poisoned] undef
	.loc	1 16 3 is_stmt 0                ; /tmp/comgr-87174f/input/add_kernel.cu:16:3
	s_and_saveexec_b64 s[4:5], vcc
	s_cbranch_execz .LBB0_3
.Ltmp2:
; %bb.1:                                ; %.lr.ph.preheader
	;DEBUG_VALUE: add_kernel:N <- [DW_OP_LLVM_poisoned] $sgpr10
	;DEBUG_VALUE: add_kernel:tid <- [DW_OP_LLVM_poisoned] $vgpr0
	;DEBUG_VALUE: col <- [DW_OP_LLVM_poisoned] $vgpr0
	.loc	2 2492 116 is_stmt 1            ; /tmp/comgr-87174f/include/hiprtc_runtime.h:2492:116
	s_load_dword s3, s[0:1], 0x2c
.Ltmp3:
	.loc	2 2484 117                      ; /tmp/comgr-87174f/include/hiprtc_runtime.h:2484:117
	s_load_dwordx4 s[4:7], s[0:1], 0x0
	s_load_dwordx2 s[0:1], s[0:1], 0x10
.Ltmp4:
	.loc	1 10 12                         ; /tmp/comgr-87174f/input/add_kernel.cu:10:12
	s_mul_i32 s8, s10, s2
	.loc	1 10 5 is_stmt 0                ; /tmp/comgr-87174f/input/add_kernel.cu:10:5
	s_ashr_i32 s9, s8, 31
.Ltmp5:
	.loc	2 2492 116 is_stmt 1            ; /tmp/comgr-87174f/include/hiprtc_runtime.h:2492:116
	s_waitcnt lgkmcnt(0)
	s_and_b32 s11, s3, 0xffff
.Ltmp6:
	;DEBUG_VALUE: add_kernel:tb_size <- [DW_OP_LLVM_poisoned] $sgpr11
	.loc	1 16 3                          ; /tmp/comgr-87174f/input/add_kernel.cu:16:3
	v_lshlrev_b32_e32 v2, 2, v0
	v_mov_b32_e32 v3, 0
	s_mov_b32 s3, 0
	v_lshl_add_u64 v[2:3], s[8:9], 2, v[2:3]
	s_lshl_b32 s2, s11, 2
	s_mov_b64 s[8:9], 0
.Ltmp7:
.LBB0_2:                                ; %.lr.ph
                                        ; =>This Inner Loop Header: Depth=1
	;DEBUG_VALUE: add_kernel:N <- [DW_OP_LLVM_poisoned] $sgpr10
	;DEBUG_VALUE: col <- [DW_OP_LLVM_poisoned] $vgpr0
	;DEBUG_VALUE: add_kernel:tb_size <- [DW_OP_LLVM_poisoned] $sgpr11
	;DEBUG_VALUE: col <- [DW_OP_LLVM_poisoned] $vgpr0
	.loc	1 17 14                         ; /tmp/comgr-87174f/input/add_kernel.cu:17:14
	v_lshl_add_u64 v[4:5], s[4:5], 0, v[2:3]
	.loc	1 17 23 is_stmt 0               ; /tmp/comgr-87174f/input/add_kernel.cu:17:23
	v_lshl_add_u64 v[6:7], s[6:7], 0, v[2:3]
	.loc	1 17 14                         ; /tmp/comgr-87174f/input/add_kernel.cu:17:14
	global_load_dword v1, v[4:5], off
	.loc	1 17 23                         ; /tmp/comgr-87174f/input/add_kernel.cu:17:23
	global_load_dword v6, v[6:7], off
	.loc	1 16 36 is_stmt 1               ; /tmp/comgr-87174f/input/add_kernel.cu:16:36
	v_add_u32_e32 v0, s11, v0
.Ltmp8:
	.loc	1 16 27 is_stmt 0               ; /tmp/comgr-87174f/input/add_kernel.cu:16:27
	v_cmp_le_i32_e32 vcc, s10, v0
	.loc	1 17 12 is_stmt 1               ; /tmp/comgr-87174f/input/add_kernel.cu:17:12
	v_lshl_add_u64 v[4:5], s[0:1], 0, v[2:3]
.Ltmp9:
	;DEBUG_VALUE: col <- [DW_OP_LLVM_poisoned] $vgpr0
	.loc	1 16 27                         ; /tmp/comgr-87174f/input/add_kernel.cu:16:27
	v_lshl_add_u64 v[2:3], v[2:3], 0, s[2:3]
.Ltmp10:
	.loc	1 16 3 is_stmt 0                ; /tmp/comgr-87174f/input/add_kernel.cu:16:3
	s_or_b64 s[8:9], vcc, s[8:9]
.Ltmp11:
	.loc	1 17 21 is_stmt 1               ; /tmp/comgr-87174f/input/add_kernel.cu:17:21
	s_waitcnt vmcnt(0)
	v_add_u32_e32 v1, v6, v1
	.loc	1 17 12 is_stmt 0               ; /tmp/comgr-87174f/input/add_kernel.cu:17:12
	global_store_dword v[4:5], v1, off
.Ltmp12:
	.loc	1 16 3 is_stmt 1                ; /tmp/comgr-87174f/input/add_kernel.cu:16:3
	s_andn2_b64 exec, exec, s[8:9]
	s_cbranch_execnz .LBB0_2
.Ltmp13:
.LBB0_3:                                ; %Flow35
	;DEBUG_VALUE: add_kernel:N <- [DW_OP_LLVM_poisoned] $sgpr10
	.loc	1 18 1                          ; /tmp/comgr-87174f/input/add_kernel.cu:18:1
	s_endpgm
.Ltmp14:
	.section	.rodata,"a",@progbits
	.p2align	6, 0x0
	.amdhsa_kernel _Z10add_kernelPKiS0_Piii
		.amdhsa_group_segment_fixed_size 0
		.amdhsa_private_segment_fixed_size 0
		.amdhsa_kernarg_size 288
		.amdhsa_user_sgpr_count 2
		.amdhsa_user_sgpr_dispatch_ptr 0
		.amdhsa_user_sgpr_queue_ptr 0
		.amdhsa_user_sgpr_kernarg_segment_ptr 1
		.amdhsa_user_sgpr_dispatch_id 0
		.amdhsa_user_sgpr_kernarg_preload_length 0
		.amdhsa_user_sgpr_kernarg_preload_offset 0
		.amdhsa_user_sgpr_private_segment_size 0
		.amdhsa_uses_dynamic_stack 0
		.amdhsa_enable_private_segment 0
		.amdhsa_system_sgpr_workgroup_id_x 1
		.amdhsa_system_sgpr_workgroup_id_y 0
		.amdhsa_system_sgpr_workgroup_id_z 0
		.amdhsa_system_sgpr_workgroup_info 0
		.amdhsa_system_vgpr_workitem_id 0
		.amdhsa_next_free_vgpr 8
		.amdhsa_next_free_sgpr 12
		.amdhsa_accum_offset 8
		.amdhsa_reserve_vcc 1
		.amdhsa_float_round_mode_32 0
		.amdhsa_float_round_mode_16_64 0
		.amdhsa_float_denorm_mode_32 3
		.amdhsa_float_denorm_mode_16_64 3
		.amdhsa_dx10_clamp 1
		.amdhsa_ieee_mode 1
		.amdhsa_fp16_overflow 0
		.amdhsa_tg_split 0
		.amdhsa_exception_fp_ieee_invalid_op 0
		.amdhsa_exception_fp_denorm_src 0
		.amdhsa_exception_fp_ieee_div_zero 0
		.amdhsa_exception_fp_ieee_overflow 0
		.amdhsa_exception_fp_ieee_underflow 0
		.amdhsa_exception_fp_ieee_inexact 0
		.amdhsa_exception_int_div_zero 0
	.end_amdhsa_kernel
	.text
.Lfunc_end0:
	.size	_Z10add_kernelPKiS0_Piii, .Lfunc_end0-_Z10add_kernelPKiS0_Piii
	.cfi_endproc
                                        ; -- End function
	.set _Z10add_kernelPKiS0_Piii.num_vgpr, 8
	.set _Z10add_kernelPKiS0_Piii.num_agpr, 0
	.set _Z10add_kernelPKiS0_Piii.numbered_sgpr, 12
	.set _Z10add_kernelPKiS0_Piii.private_seg_size, 0
	.set _Z10add_kernelPKiS0_Piii.uses_vcc, 1
	.set _Z10add_kernelPKiS0_Piii.uses_flat_scratch, 0
	.set _Z10add_kernelPKiS0_Piii.has_dyn_sized_stack, 0
	.set _Z10add_kernelPKiS0_Piii.has_recursion, 0
	.set _Z10add_kernelPKiS0_Piii.has_indirect_call, 0
	.section	.AMDGPU.csdata,"",@progbits
; Kernel info:
; codeLenInByte = 184
; TotalNumSgprs: 18
; NumVgprs: 8
; NumAgprs: 0
; TotalNumVgprs: 8
; ScratchSize: 0
; MemoryBound: 0
; FloatMode: 240
; IeeeMode: 1
; LDSByteSize: 0 bytes/workgroup (compile time only)
; SGPRBlocks: 2
; VGPRBlocks: 0
; NumSGPRsForWavesPerEU: 18
; NumVGPRsForWavesPerEU: 8
; AccumOffset: 8
; Occupancy: 8
; WaveLimiterHint : 0
; COMPUTE_PGM_RSRC2:SCRATCH_EN: 0
; COMPUTE_PGM_RSRC2:USER_SGPR: 2
; COMPUTE_PGM_RSRC2:TRAP_HANDLER: 0
; COMPUTE_PGM_RSRC2:TGID_X_EN: 1
; COMPUTE_PGM_RSRC2:TGID_Y_EN: 0
; COMPUTE_PGM_RSRC2:TGID_Z_EN: 0
; COMPUTE_PGM_RSRC2:TIDIG_COMP_CNT: 0
; COMPUTE_PGM_RSRC3_GFX90A:ACCUM_OFFSET: 1
; COMPUTE_PGM_RSRC3_GFX90A:TG_SPLIT: 0
	.text
	.p2alignl 6, 3212836864
	.fill 256, 4, 3212836864
	.section	.AMDGPU.gpr_maximums,"",@progbits
	.set amdgpu.max_num_vgpr, 0
	.set amdgpu.max_num_agpr, 0
	.set amdgpu.max_num_sgpr, 0
	.text
	.type	_ZL20__amdgcn_name_expr_1,@object ; @_ZL20__amdgcn_name_expr_1
	.data
	.globl	_ZL20__amdgcn_name_expr_1
	.p2align	4, 0x0
_ZL20__amdgcn_name_expr_1:
	.quad	.str
	.quad	_Z10add_kernelPKiS0_Piii
	.size	_ZL20__amdgcn_name_expr_1, 16

	.type	.str,@object                    ; @.str
	.section	.rodata.str1.1,"aMS",@progbits,1
.str:
	.asciz	"add_kernel"
	.size	.str, 11

	.type	__hip_cuid_ab25958aa643f7a5,@object ; @__hip_cuid_ab25958aa643f7a5
	.section	.bss,"aw",@nobits
	.globl	__hip_cuid_ab25958aa643f7a5
__hip_cuid_ab25958aa643f7a5:
	.byte	0                               ; 0x0
	.size	__hip_cuid_ab25958aa643f7a5, 1

	.section	.debug_loclists,"",@progbits
	.long	.Ldebug_list_header_end0-.Ldebug_list_header_start0 ; Length
.Ldebug_list_header_start0:
	.short	5                               ; Version
	.byte	8                               ; Address size
	.byte	0                               ; Segment selector size
	.long	4                               ; Offset entry count
.Lloclists_table_base0:
	.long	.Ldebug_loc0-.Lloclists_table_base0
	.long	.Ldebug_loc1-.Lloclists_table_base0
	.long	.Ldebug_loc2-.Lloclists_table_base0
	.long	.Ldebug_loc3-.Lloclists_table_base0
.Ldebug_loc0:
	.byte	4                               ; DW_LLE_offset_pair
	.uleb128 .Ltmp0-.Lfunc_begin0           ;   starting offset
	.uleb128 .Lfunc_end0-.Lfunc_begin0      ;   ending offset
	.byte	2                               ; Loc expr size
	.byte	144                             ; DW_OP_regx
	.byte	42                              ; 42
	.byte	0                               ; DW_LLE_end_of_list
.Ldebug_loc1:
	.byte	4                               ; DW_LLE_offset_pair
	.uleb128 .Ltmp0-.Lfunc_begin0           ;   starting offset
	.uleb128 .Ltmp7-.Lfunc_begin0           ;   ending offset
	.byte	3                               ; Loc expr size
	.byte	144                             ; DW_OP_regx
	.byte	128                             ; 2560
	.byte	20                              ; 
	.byte	0                               ; DW_LLE_end_of_list
.Ldebug_loc2:
	.byte	4                               ; DW_LLE_offset_pair
	.uleb128 .Ltmp0-.Lfunc_begin0           ;   starting offset
	.uleb128 .Ltmp8-.Lfunc_begin0           ;   ending offset
	.byte	3                               ; Loc expr size
	.byte	144                             ; DW_OP_regx
	.byte	128                             ; 2560
	.byte	20                              ; 
	.byte	4                               ; DW_LLE_offset_pair
	.uleb128 .Ltmp9-.Lfunc_begin0           ;   starting offset
	.uleb128 .Ltmp13-.Lfunc_begin0          ;   ending offset
	.byte	3                               ; Loc expr size
	.byte	144                             ; DW_OP_regx
	.byte	128                             ; 2560
	.byte	20                              ; 
	.byte	0                               ; DW_LLE_end_of_list
.Ldebug_loc3:
	.byte	4                               ; DW_LLE_offset_pair
	.uleb128 .Ltmp6-.Lfunc_begin0           ;   starting offset
	.uleb128 .Ltmp13-.Lfunc_begin0          ;   ending offset
	.byte	2                               ; Loc expr size
	.byte	144                             ; DW_OP_regx
	.byte	43                              ; 43
	.byte	0                               ; DW_LLE_end_of_list
.Ldebug_list_header_end0:
	.section	.debug_abbrev,"",@progbits
	.byte	1                               ; Abbreviation Code
	.byte	17                              ; DW_TAG_compile_unit
	.byte	1                               ; DW_CHILDREN_yes
	.byte	37                              ; DW_AT_producer
	.byte	37                              ; DW_FORM_strx1
	.byte	19                              ; DW_AT_language
	.byte	5                               ; DW_FORM_data2
	.byte	3                               ; DW_AT_name
	.byte	37                              ; DW_FORM_strx1
	.byte	114                             ; DW_AT_str_offsets_base
	.byte	23                              ; DW_FORM_sec_offset
	.byte	16                              ; DW_AT_stmt_list
	.byte	23                              ; DW_FORM_sec_offset
	.byte	27                              ; DW_AT_comp_dir
	.byte	37                              ; DW_FORM_strx1
	.byte	17                              ; DW_AT_low_pc
	.byte	27                              ; DW_FORM_addrx
	.byte	18                              ; DW_AT_high_pc
	.byte	6                               ; DW_FORM_data4
	.byte	115                             ; DW_AT_addr_base
	.byte	23                              ; DW_FORM_sec_offset
	.byte	116                             ; DW_AT_rnglists_base
	.byte	23                              ; DW_FORM_sec_offset
	.ascii	"\214\001"                      ; DW_AT_loclists_base
	.byte	23                              ; DW_FORM_sec_offset
	.byte	0                               ; EOM(1)
	.byte	0                               ; EOM(2)
	.byte	2                               ; Abbreviation Code
	.byte	52                              ; DW_TAG_variable
	.byte	0                               ; DW_CHILDREN_no
	.byte	3                               ; DW_AT_name
	.byte	37                              ; DW_FORM_strx1
	.byte	73                              ; DW_AT_type
	.byte	19                              ; DW_FORM_ref4
	.byte	63                              ; DW_AT_external
	.byte	25                              ; DW_FORM_flag_present
	.byte	58                              ; DW_AT_decl_file
	.byte	11                              ; DW_FORM_data1
	.byte	59                              ; DW_AT_decl_line
	.byte	11                              ; DW_FORM_data1
	.ascii	"\215|"                         ; DW_AT_LLVM_memory_space
	.byte	6                               ; DW_FORM_data4
	.byte	2                               ; DW_AT_location
	.byte	24                              ; DW_FORM_exprloc
	.byte	110                             ; DW_AT_linkage_name
	.byte	37                              ; DW_FORM_strx1
	.byte	0                               ; EOM(1)
	.byte	0                               ; EOM(2)
	.byte	3                               ; Abbreviation Code
	.byte	1                               ; DW_TAG_array_type
	.byte	1                               ; DW_CHILDREN_yes
	.byte	73                              ; DW_AT_type
	.byte	19                              ; DW_FORM_ref4
	.byte	0                               ; EOM(1)
	.byte	0                               ; EOM(2)
	.byte	4                               ; Abbreviation Code
	.byte	33                              ; DW_TAG_subrange_type
	.byte	0                               ; DW_CHILDREN_no
	.byte	73                              ; DW_AT_type
	.byte	19                              ; DW_FORM_ref4
	.byte	55                              ; DW_AT_count
	.byte	11                              ; DW_FORM_data1
	.byte	0                               ; EOM(1)
	.byte	0                               ; EOM(2)
	.byte	5                               ; Abbreviation Code
	.byte	15                              ; DW_TAG_pointer_type
	.byte	0                               ; DW_CHILDREN_no
	.byte	73                              ; DW_AT_type
	.byte	19                              ; DW_FORM_ref4
	.byte	0                               ; EOM(1)
	.byte	0                               ; EOM(2)
	.byte	6                               ; Abbreviation Code
	.byte	38                              ; DW_TAG_const_type
	.byte	0                               ; DW_CHILDREN_no
	.byte	0                               ; EOM(1)
	.byte	0                               ; EOM(2)
	.byte	7                               ; Abbreviation Code
	.byte	36                              ; DW_TAG_base_type
	.byte	0                               ; DW_CHILDREN_no
	.byte	3                               ; DW_AT_name
	.byte	37                              ; DW_FORM_strx1
	.byte	11                              ; DW_AT_byte_size
	.byte	11                              ; DW_FORM_data1
	.byte	62                              ; DW_AT_encoding
	.byte	11                              ; DW_FORM_data1
	.byte	0                               ; EOM(1)
	.byte	0                               ; EOM(2)
	.byte	8                               ; Abbreviation Code
	.byte	15                              ; DW_TAG_pointer_type
	.byte	0                               ; DW_CHILDREN_no
	.byte	0                               ; EOM(1)
	.byte	0                               ; EOM(2)
	.byte	9                               ; Abbreviation Code
	.byte	19                              ; DW_TAG_structure_type
	.byte	1                               ; DW_CHILDREN_yes
	.byte	54                              ; DW_AT_calling_convention
	.byte	11                              ; DW_FORM_data1
	.byte	3                               ; DW_AT_name
	.byte	37                              ; DW_FORM_strx1
	.byte	11                              ; DW_AT_byte_size
	.byte	11                              ; DW_FORM_data1
	.byte	58                              ; DW_AT_decl_file
	.byte	11                              ; DW_FORM_data1
	.byte	59                              ; DW_AT_decl_line
	.byte	5                               ; DW_FORM_data2
	.byte	0                               ; EOM(1)
	.byte	0                               ; EOM(2)
	.byte	10                              ; Abbreviation Code
	.byte	13                              ; DW_TAG_member
	.byte	0                               ; DW_CHILDREN_no
	.byte	3                               ; DW_AT_name
	.byte	37                              ; DW_FORM_strx1
	.byte	73                              ; DW_AT_type
	.byte	19                              ; DW_FORM_ref4
	.byte	58                              ; DW_AT_decl_file
	.byte	11                              ; DW_FORM_data1
	.byte	59                              ; DW_AT_decl_line
	.byte	5                               ; DW_FORM_data2
	.byte	56                              ; DW_AT_data_member_location
	.byte	11                              ; DW_FORM_data1
	.byte	0                               ; EOM(1)
	.byte	0                               ; EOM(2)
	.byte	11                              ; Abbreviation Code
	.byte	46                              ; DW_TAG_subprogram
	.byte	1                               ; DW_CHILDREN_yes
	.byte	3                               ; DW_AT_name
	.byte	37                              ; DW_FORM_strx1
	.byte	58                              ; DW_AT_decl_file
	.byte	11                              ; DW_FORM_data1
	.byte	59                              ; DW_AT_decl_line
	.byte	5                               ; DW_FORM_data2
	.byte	60                              ; DW_AT_declaration
	.byte	25                              ; DW_FORM_flag_present
	.byte	100                             ; DW_AT_object_pointer
	.byte	19                              ; DW_FORM_ref4
	.byte	63                              ; DW_AT_external
	.byte	25                              ; DW_FORM_flag_present
	.byte	0                               ; EOM(1)
	.byte	0                               ; EOM(2)
	.byte	12                              ; Abbreviation Code
	.byte	5                               ; DW_TAG_formal_parameter
	.byte	0                               ; DW_CHILDREN_no
	.byte	73                              ; DW_AT_type
	.byte	19                              ; DW_FORM_ref4
	.byte	52                              ; DW_AT_artificial
	.byte	25                              ; DW_FORM_flag_present
	.byte	0                               ; EOM(1)
	.byte	0                               ; EOM(2)
	.byte	13                              ; Abbreviation Code
	.byte	5                               ; DW_TAG_formal_parameter
	.byte	0                               ; DW_CHILDREN_no
	.byte	73                              ; DW_AT_type
	.byte	19                              ; DW_FORM_ref4
	.byte	0                               ; EOM(1)
	.byte	0                               ; EOM(2)
	.byte	14                              ; Abbreviation Code
	.byte	22                              ; DW_TAG_typedef
	.byte	0                               ; DW_CHILDREN_no
	.byte	73                              ; DW_AT_type
	.byte	19                              ; DW_FORM_ref4
	.byte	3                               ; DW_AT_name
	.byte	37                              ; DW_FORM_strx1
	.byte	58                              ; DW_AT_decl_file
	.byte	11                              ; DW_FORM_data1
	.byte	59                              ; DW_AT_decl_line
	.byte	5                               ; DW_FORM_data2
	.byte	0                               ; EOM(1)
	.byte	0                               ; EOM(2)
	.byte	15                              ; Abbreviation Code
	.byte	57                              ; DW_TAG_namespace
	.byte	1                               ; DW_CHILDREN_yes
	.byte	3                               ; DW_AT_name
	.byte	37                              ; DW_FORM_strx1
	.byte	0                               ; EOM(1)
	.byte	0                               ; EOM(2)
	.byte	16                              ; Abbreviation Code
	.byte	36                              ; DW_TAG_base_type
	.byte	0                               ; DW_CHILDREN_no
	.byte	3                               ; DW_AT_name
	.byte	37                              ; DW_FORM_strx1
	.byte	62                              ; DW_AT_encoding
	.byte	11                              ; DW_FORM_data1
	.byte	11                              ; DW_AT_byte_size
	.byte	11                              ; DW_FORM_data1
	.byte	0                               ; EOM(1)
	.byte	0                               ; EOM(2)
	.byte	17                              ; Abbreviation Code
	.byte	46                              ; DW_TAG_subprogram
	.byte	0                               ; DW_CHILDREN_no
	.byte	110                             ; DW_AT_linkage_name
	.byte	37                              ; DW_FORM_strx1
	.byte	3                               ; DW_AT_name
	.byte	37                              ; DW_FORM_strx1
	.byte	58                              ; DW_AT_decl_file
	.byte	11                              ; DW_FORM_data1
	.byte	59                              ; DW_AT_decl_line
	.byte	5                               ; DW_FORM_data2
	.byte	73                              ; DW_AT_type
	.byte	19                              ; DW_FORM_ref4
	.byte	32                              ; DW_AT_inline
	.byte	33                              ; DW_FORM_implicit_const
	.byte	1
	.byte	0                               ; EOM(1)
	.byte	0                               ; EOM(2)
	.byte	18                              ; Abbreviation Code
	.byte	46                              ; DW_TAG_subprogram
	.byte	0                               ; DW_CHILDREN_no
	.byte	110                             ; DW_AT_linkage_name
	.byte	37                              ; DW_FORM_strx1
	.byte	3                               ; DW_AT_name
	.byte	37                              ; DW_FORM_strx1
	.byte	58                              ; DW_AT_decl_file
	.byte	11                              ; DW_FORM_data1
	.byte	59                              ; DW_AT_decl_line
	.byte	5                               ; DW_FORM_data2
	.byte	73                              ; DW_AT_type
	.byte	19                              ; DW_FORM_ref4
	.byte	60                              ; DW_AT_declaration
	.byte	25                              ; DW_FORM_flag_present
	.byte	63                              ; DW_AT_external
	.byte	25                              ; DW_FORM_flag_present
	.byte	0                               ; EOM(1)
	.byte	0                               ; EOM(2)
	.byte	19                              ; Abbreviation Code
	.byte	46                              ; DW_TAG_subprogram
	.byte	1                               ; DW_CHILDREN_yes
	.byte	110                             ; DW_AT_linkage_name
	.byte	37                              ; DW_FORM_strx1
	.byte	3                               ; DW_AT_name
	.byte	37                              ; DW_FORM_strx1
	.byte	58                              ; DW_AT_decl_file
	.byte	11                              ; DW_FORM_data1
	.byte	59                              ; DW_AT_decl_line
	.byte	5                               ; DW_FORM_data2
	.byte	73                              ; DW_AT_type
	.byte	19                              ; DW_FORM_ref4
	.byte	60                              ; DW_AT_declaration
	.byte	25                              ; DW_FORM_flag_present
	.byte	100                             ; DW_AT_object_pointer
	.byte	19                              ; DW_FORM_ref4
	.byte	63                              ; DW_AT_external
	.byte	25                              ; DW_FORM_flag_present
	.byte	0                               ; EOM(1)
	.byte	0                               ; EOM(2)
	.byte	20                              ; Abbreviation Code
	.byte	38                              ; DW_TAG_const_type
	.byte	0                               ; DW_CHILDREN_no
	.byte	73                              ; DW_AT_type
	.byte	19                              ; DW_FORM_ref4
	.byte	0                               ; EOM(1)
	.byte	0                               ; EOM(2)
	.byte	21                              ; Abbreviation Code
	.byte	46                              ; DW_TAG_subprogram
	.byte	0                               ; DW_CHILDREN_no
	.byte	71                              ; DW_AT_specification
	.byte	19                              ; DW_FORM_ref4
	.byte	32                              ; DW_AT_inline
	.byte	33                              ; DW_FORM_implicit_const
	.byte	1
	.byte	0                               ; EOM(1)
	.byte	0                               ; EOM(2)
	.byte	22                              ; Abbreviation Code
	.byte	46                              ; DW_TAG_subprogram
	.byte	1                               ; DW_CHILDREN_yes
	.byte	17                              ; DW_AT_low_pc
	.byte	27                              ; DW_FORM_addrx
	.byte	18                              ; DW_AT_high_pc
	.byte	6                               ; DW_FORM_data4
	.byte	122                             ; DW_AT_call_all_calls
	.byte	25                              ; DW_FORM_flag_present
	.byte	110                             ; DW_AT_linkage_name
	.byte	37                              ; DW_FORM_strx1
	.byte	3                               ; DW_AT_name
	.byte	37                              ; DW_FORM_strx1
	.byte	58                              ; DW_AT_decl_file
	.byte	11                              ; DW_FORM_data1
	.byte	59                              ; DW_AT_decl_line
	.byte	11                              ; DW_FORM_data1
	.byte	63                              ; DW_AT_external
	.byte	25                              ; DW_FORM_flag_present
	.byte	0                               ; EOM(1)
	.byte	0                               ; EOM(2)
	.byte	23                              ; Abbreviation Code
	.byte	5                               ; DW_TAG_formal_parameter
	.byte	0                               ; DW_CHILDREN_no
	.byte	3                               ; DW_AT_name
	.byte	37                              ; DW_FORM_strx1
	.byte	58                              ; DW_AT_decl_file
	.byte	11                              ; DW_FORM_data1
	.byte	59                              ; DW_AT_decl_line
	.byte	11                              ; DW_FORM_data1
	.byte	73                              ; DW_AT_type
	.byte	19                              ; DW_FORM_ref4
	.byte	0                               ; EOM(1)
	.byte	0                               ; EOM(2)
	.byte	24                              ; Abbreviation Code
	.byte	5                               ; DW_TAG_formal_parameter
	.byte	0                               ; DW_CHILDREN_no
	.byte	2                               ; DW_AT_location
	.byte	34                              ; DW_FORM_loclistx
	.byte	3                               ; DW_AT_name
	.byte	37                              ; DW_FORM_strx1
	.byte	58                              ; DW_AT_decl_file
	.byte	11                              ; DW_FORM_data1
	.byte	59                              ; DW_AT_decl_line
	.byte	11                              ; DW_FORM_data1
	.byte	73                              ; DW_AT_type
	.byte	19                              ; DW_FORM_ref4
	.byte	0                               ; EOM(1)
	.byte	0                               ; EOM(2)
	.byte	25                              ; Abbreviation Code
	.byte	52                              ; DW_TAG_variable
	.byte	0                               ; DW_CHILDREN_no
	.byte	2                               ; DW_AT_location
	.byte	34                              ; DW_FORM_loclistx
	.byte	3                               ; DW_AT_name
	.byte	37                              ; DW_FORM_strx1
	.byte	58                              ; DW_AT_decl_file
	.byte	11                              ; DW_FORM_data1
	.byte	59                              ; DW_AT_decl_line
	.byte	11                              ; DW_FORM_data1
	.byte	73                              ; DW_AT_type
	.byte	19                              ; DW_FORM_ref4
	.byte	0                               ; EOM(1)
	.byte	0                               ; EOM(2)
	.byte	26                              ; Abbreviation Code
	.byte	52                              ; DW_TAG_variable
	.byte	0                               ; DW_CHILDREN_no
	.byte	3                               ; DW_AT_name
	.byte	37                              ; DW_FORM_strx1
	.byte	58                              ; DW_AT_decl_file
	.byte	11                              ; DW_FORM_data1
	.byte	59                              ; DW_AT_decl_line
	.byte	11                              ; DW_FORM_data1
	.byte	73                              ; DW_AT_type
	.byte	19                              ; DW_FORM_ref4
	.byte	0                               ; EOM(1)
	.byte	0                               ; EOM(2)
	.byte	27                              ; Abbreviation Code
	.byte	29                              ; DW_TAG_inlined_subroutine
	.byte	1                               ; DW_CHILDREN_yes
	.byte	49                              ; DW_AT_abstract_origin
	.byte	19                              ; DW_FORM_ref4
	.byte	85                              ; DW_AT_ranges
	.byte	35                              ; DW_FORM_rnglistx
	.byte	88                              ; DW_AT_call_file
	.byte	11                              ; DW_FORM_data1
	.byte	89                              ; DW_AT_call_line
	.byte	11                              ; DW_FORM_data1
	.byte	87                              ; DW_AT_call_column
	.byte	11                              ; DW_FORM_data1
	.byte	0                               ; EOM(1)
	.byte	0                               ; EOM(2)
	.byte	28                              ; Abbreviation Code
	.byte	29                              ; DW_TAG_inlined_subroutine
	.byte	0                               ; DW_CHILDREN_no
	.byte	49                              ; DW_AT_abstract_origin
	.byte	19                              ; DW_FORM_ref4
	.byte	85                              ; DW_AT_ranges
	.byte	35                              ; DW_FORM_rnglistx
	.byte	88                              ; DW_AT_call_file
	.byte	11                              ; DW_FORM_data1
	.byte	89                              ; DW_AT_call_line
	.byte	5                               ; DW_FORM_data2
	.byte	87                              ; DW_AT_call_column
	.byte	11                              ; DW_FORM_data1
	.byte	0                               ; EOM(1)
	.byte	0                               ; EOM(2)
	.byte	29                              ; Abbreviation Code
	.byte	11                              ; DW_TAG_lexical_block
	.byte	1                               ; DW_CHILDREN_yes
	.byte	85                              ; DW_AT_ranges
	.byte	35                              ; DW_FORM_rnglistx
	.byte	0                               ; EOM(1)
	.byte	0                               ; EOM(2)
	.byte	30                              ; Abbreviation Code
	.byte	8                               ; DW_TAG_imported_declaration
	.byte	0                               ; DW_CHILDREN_no
	.byte	58                              ; DW_AT_decl_file
	.byte	11                              ; DW_FORM_data1
	.byte	59                              ; DW_AT_decl_line
	.byte	5                               ; DW_FORM_data2
	.byte	24                              ; DW_AT_import
	.byte	19                              ; DW_FORM_ref4
	.byte	0                               ; EOM(1)
	.byte	0                               ; EOM(2)
	.byte	0                               ; EOM(3)
	.section	.debug_info,"",@progbits
.Lcu_begin0:
	.long	.Ldebug_info_end0-.Ldebug_info_start0 ; Length of Unit
.Ldebug_info_start0:
	.short	5                               ; DWARF version number
	.byte	1                               ; DWARF Unit Type
	.byte	8                               ; Address Size (in bytes)
	.long	.debug_abbrev                   ; Offset Into Abbrev. Section
	.byte	1                               ; Abbrev [1] 0xc:0x21c DW_TAG_compile_unit
	.byte	0                               ; DW_AT_producer
	.short	33                              ; DW_AT_language
	.byte	1                               ; DW_AT_name
	.long	.Lstr_offsets_base0             ; DW_AT_str_offsets_base
	.long	.Lline_table_start0             ; DW_AT_stmt_list
	.byte	2                               ; DW_AT_comp_dir
	.byte	1                               ; DW_AT_low_pc
	.long	.Lfunc_end0-.Lfunc_begin0       ; DW_AT_high_pc
	.long	.Laddr_table_base0              ; DW_AT_addr_base
	.long	.Lrnglists_table_base0          ; DW_AT_rnglists_base
	.long	.Lloclists_table_base0          ; DW_AT_loclists_base
	.byte	2                               ; Abbrev [2] 0x2b:0x16 DW_TAG_variable
	.byte	3                               ; DW_AT_name
	.long	65                              ; DW_AT_type
                                        ; DW_AT_external
	.byte	1                               ; DW_AT_decl_file
	.byte	19                              ; DW_AT_decl_line
	.long	1                               ; DW_AT_LLVM_memory_space
	.byte	8                               ; DW_AT_location
	.byte	161
	.byte	0
	.byte	159
	.byte	148
	.byte	8
	.byte	48
	.byte	233
	.byte	2
	.byte	5                               ; DW_AT_linkage_name
	.byte	3                               ; Abbrev [3] 0x41:0xc DW_TAG_array_type
	.long	77                              ; DW_AT_type
	.byte	4                               ; Abbrev [4] 0x46:0x6 DW_TAG_subrange_type
	.long	83                              ; DW_AT_type
	.byte	2                               ; DW_AT_count
	.byte	0                               ; End Of Children Mark
	.byte	5                               ; Abbrev [5] 0x4d:0x5 DW_TAG_pointer_type
	.long	82                              ; DW_AT_type
	.byte	6                               ; Abbrev [6] 0x52:0x1 DW_TAG_const_type
	.byte	7                               ; Abbrev [7] 0x53:0x4 DW_TAG_base_type
	.byte	4                               ; DW_AT_name
	.byte	8                               ; DW_AT_byte_size
	.byte	7                               ; DW_AT_encoding
	.byte	8                               ; Abbrev [8] 0x57:0x1 DW_TAG_pointer_type
	.byte	9                               ; Abbrev [9] 0x58:0x44 DW_TAG_structure_type
	.byte	5                               ; DW_AT_calling_convention
	.byte	13                              ; DW_AT_name
	.byte	12                              ; DW_AT_byte_size
	.byte	2                               ; DW_AT_decl_file
	.short	2476                            ; DW_AT_decl_line
	.byte	10                              ; Abbrev [10] 0x5f:0xa DW_TAG_member
	.byte	6                               ; DW_AT_name
	.long	156                             ; DW_AT_type
	.byte	2                               ; DW_AT_decl_file
	.short	2477                            ; DW_AT_decl_line
	.byte	0                               ; DW_AT_data_member_location
	.byte	10                              ; Abbrev [10] 0x69:0xa DW_TAG_member
	.byte	11                              ; DW_AT_name
	.long	156                             ; DW_AT_type
	.byte	2                               ; DW_AT_decl_file
	.short	2478                            ; DW_AT_decl_line
	.byte	4                               ; DW_AT_data_member_location
	.byte	10                              ; Abbrev [10] 0x73:0xa DW_TAG_member
	.byte	12                              ; DW_AT_name
	.long	156                             ; DW_AT_type
	.byte	2                               ; DW_AT_decl_file
	.short	2479                            ; DW_AT_decl_line
	.byte	8                               ; DW_AT_data_member_location
	.byte	11                              ; Abbrev [11] 0x7d:0x1e DW_TAG_subprogram
	.byte	13                              ; DW_AT_name
	.byte	2                               ; DW_AT_decl_file
	.short	2480                            ; DW_AT_decl_line
                                        ; DW_AT_declaration
	.long	134                             ; DW_AT_object_pointer
                                        ; DW_AT_external
	.byte	12                              ; Abbrev [12] 0x86:0x5 DW_TAG_formal_parameter
	.long	181                             ; DW_AT_type
                                        ; DW_AT_artificial
	.byte	13                              ; Abbrev [13] 0x8b:0x5 DW_TAG_formal_parameter
	.long	156                             ; DW_AT_type
	.byte	13                              ; Abbrev [13] 0x90:0x5 DW_TAG_formal_parameter
	.long	156                             ; DW_AT_type
	.byte	13                              ; Abbrev [13] 0x95:0x5 DW_TAG_formal_parameter
	.long	156                             ; DW_AT_type
	.byte	0                               ; End Of Children Mark
	.byte	0                               ; End Of Children Mark
	.byte	14                              ; Abbrev [14] 0x9c:0x9 DW_TAG_typedef
	.long	167                             ; DW_AT_type
	.byte	10                              ; DW_AT_name
	.byte	2                               ; DW_AT_decl_file
	.short	1569                            ; DW_AT_decl_line
	.byte	15                              ; Abbrev [15] 0xa5:0xc DW_TAG_namespace
	.byte	7                               ; DW_AT_name
	.byte	14                              ; Abbrev [14] 0xa7:0x9 DW_TAG_typedef
	.long	177                             ; DW_AT_type
	.byte	9                               ; DW_AT_name
	.byte	2                               ; DW_AT_decl_file
	.short	1427                            ; DW_AT_decl_line
	.byte	0                               ; End Of Children Mark
	.byte	16                              ; Abbrev [16] 0xb1:0x4 DW_TAG_base_type
	.byte	8                               ; DW_AT_name
	.byte	7                               ; DW_AT_encoding
	.byte	4                               ; DW_AT_byte_size
	.byte	5                               ; Abbrev [5] 0xb5:0x5 DW_TAG_pointer_type
	.long	88                              ; DW_AT_type
	.byte	17                              ; Abbrev [17] 0xba:0xa DW_TAG_subprogram
	.byte	14                              ; DW_AT_linkage_name
	.byte	15                              ; DW_AT_name
	.byte	2                               ; DW_AT_decl_file
	.short	2484                            ; DW_AT_decl_line
	.long	177                             ; DW_AT_type
                                        ; DW_AT_inline
	.byte	9                               ; Abbrev [9] 0xc4:0x3a DW_TAG_structure_type
	.byte	5                               ; DW_AT_calling_convention
	.byte	24                              ; DW_AT_name
	.byte	1                               ; DW_AT_byte_size
	.byte	2                               ; DW_AT_decl_file
	.short	2499                            ; DW_AT_decl_line
	.byte	18                              ; Abbrev [18] 0xcb:0xa DW_TAG_subprogram
	.byte	16                              ; DW_AT_linkage_name
	.byte	17                              ; DW_AT_name
	.byte	2                               ; DW_AT_decl_file
	.short	2500                            ; DW_AT_decl_line
	.long	177                             ; DW_AT_type
                                        ; DW_AT_declaration
                                        ; DW_AT_external
	.byte	18                              ; Abbrev [18] 0xd5:0xa DW_TAG_subprogram
	.byte	18                              ; DW_AT_linkage_name
	.byte	19                              ; DW_AT_name
	.byte	2                               ; DW_AT_decl_file
	.short	2501                            ; DW_AT_decl_line
	.long	177                             ; DW_AT_type
                                        ; DW_AT_declaration
                                        ; DW_AT_external
	.byte	18                              ; Abbrev [18] 0xdf:0xa DW_TAG_subprogram
	.byte	20                              ; DW_AT_linkage_name
	.byte	21                              ; DW_AT_name
	.byte	2                               ; DW_AT_decl_file
	.short	2502                            ; DW_AT_decl_line
	.long	177                             ; DW_AT_type
                                        ; DW_AT_declaration
                                        ; DW_AT_external
	.byte	19                              ; Abbrev [19] 0xe9:0x14 DW_TAG_subprogram
	.byte	22                              ; DW_AT_linkage_name
	.byte	23                              ; DW_AT_name
	.byte	2                               ; DW_AT_decl_file
	.short	2503                            ; DW_AT_decl_line
	.long	254                             ; DW_AT_type
                                        ; DW_AT_declaration
	.long	247                             ; DW_AT_object_pointer
                                        ; DW_AT_external
	.byte	12                              ; Abbrev [12] 0xf7:0x5 DW_TAG_formal_parameter
	.long	263                             ; DW_AT_type
                                        ; DW_AT_artificial
	.byte	0                               ; End Of Children Mark
	.byte	0                               ; End Of Children Mark
	.byte	14                              ; Abbrev [14] 0xfe:0x9 DW_TAG_typedef
	.long	88                              ; DW_AT_type
	.byte	13                              ; DW_AT_name
	.byte	2                               ; DW_AT_decl_file
	.short	2482                            ; DW_AT_decl_line
	.byte	5                               ; Abbrev [5] 0x107:0x5 DW_TAG_pointer_type
	.long	268                             ; DW_AT_type
	.byte	20                              ; Abbrev [20] 0x10c:0x5 DW_TAG_const_type
	.long	196                             ; DW_AT_type
	.byte	21                              ; Abbrev [21] 0x111:0x5 DW_TAG_subprogram
	.long	203                             ; DW_AT_specification
                                        ; DW_AT_inline
	.byte	17                              ; Abbrev [17] 0x116:0xa DW_TAG_subprogram
	.byte	25                              ; DW_AT_linkage_name
	.byte	26                              ; DW_AT_name
	.byte	2                               ; DW_AT_decl_file
	.short	2492                            ; DW_AT_decl_line
	.long	177                             ; DW_AT_type
                                        ; DW_AT_inline
	.byte	9                               ; Abbrev [9] 0x120:0x3a DW_TAG_structure_type
	.byte	5                               ; DW_AT_calling_convention
	.byte	31                              ; DW_AT_name
	.byte	1                               ; DW_AT_byte_size
	.byte	2                               ; DW_AT_decl_file
	.short	2511                            ; DW_AT_decl_line
	.byte	18                              ; Abbrev [18] 0x127:0xa DW_TAG_subprogram
	.byte	27                              ; DW_AT_linkage_name
	.byte	17                              ; DW_AT_name
	.byte	2                               ; DW_AT_decl_file
	.short	2512                            ; DW_AT_decl_line
	.long	177                             ; DW_AT_type
                                        ; DW_AT_declaration
                                        ; DW_AT_external
	.byte	18                              ; Abbrev [18] 0x131:0xa DW_TAG_subprogram
	.byte	28                              ; DW_AT_linkage_name
	.byte	19                              ; DW_AT_name
	.byte	2                               ; DW_AT_decl_file
	.short	2513                            ; DW_AT_decl_line
	.long	177                             ; DW_AT_type
                                        ; DW_AT_declaration
                                        ; DW_AT_external
	.byte	18                              ; Abbrev [18] 0x13b:0xa DW_TAG_subprogram
	.byte	29                              ; DW_AT_linkage_name
	.byte	21                              ; DW_AT_name
	.byte	2                               ; DW_AT_decl_file
	.short	2514                            ; DW_AT_decl_line
	.long	177                             ; DW_AT_type
                                        ; DW_AT_declaration
                                        ; DW_AT_external
	.byte	19                              ; Abbrev [19] 0x145:0x14 DW_TAG_subprogram
	.byte	30                              ; DW_AT_linkage_name
	.byte	23                              ; DW_AT_name
	.byte	2                               ; DW_AT_decl_file
	.short	2515                            ; DW_AT_decl_line
	.long	254                             ; DW_AT_type
                                        ; DW_AT_declaration
	.long	339                             ; DW_AT_object_pointer
                                        ; DW_AT_external
	.byte	12                              ; Abbrev [12] 0x153:0x5 DW_TAG_formal_parameter
	.long	346                             ; DW_AT_type
                                        ; DW_AT_artificial
	.byte	0                               ; End Of Children Mark
	.byte	0                               ; End Of Children Mark
	.byte	5                               ; Abbrev [5] 0x15a:0x5 DW_TAG_pointer_type
	.long	351                             ; DW_AT_type
	.byte	20                              ; Abbrev [20] 0x15f:0x5 DW_TAG_const_type
	.long	288                             ; DW_AT_type
	.byte	21                              ; Abbrev [21] 0x164:0x5 DW_TAG_subprogram
	.long	295                             ; DW_AT_specification
                                        ; DW_AT_inline
	.byte	22                              ; Abbrev [22] 0x169:0x82 DW_TAG_subprogram
	.byte	1                               ; DW_AT_low_pc
	.long	.Lfunc_end0-.Lfunc_begin0       ; DW_AT_high_pc
                                        ; DW_AT_call_all_calls
	.byte	36                              ; DW_AT_linkage_name
	.byte	37                              ; DW_AT_name
	.byte	1                               ; DW_AT_decl_file
	.byte	3                               ; DW_AT_decl_line
                                        ; DW_AT_external
	.byte	23                              ; Abbrev [23] 0x173:0x8 DW_TAG_formal_parameter
	.byte	43                              ; DW_AT_name
	.byte	1                               ; DW_AT_decl_file
	.byte	3                               ; DW_AT_decl_line
	.long	541                             ; DW_AT_type
	.byte	23                              ; Abbrev [23] 0x17b:0x8 DW_TAG_formal_parameter
	.byte	44                              ; DW_AT_name
	.byte	1                               ; DW_AT_decl_file
	.byte	3                               ; DW_AT_decl_line
	.long	541                             ; DW_AT_type
	.byte	23                              ; Abbrev [23] 0x183:0x8 DW_TAG_formal_parameter
	.byte	45                              ; DW_AT_name
	.byte	1                               ; DW_AT_decl_file
	.byte	3                               ; DW_AT_decl_line
	.long	546                             ; DW_AT_type
	.byte	23                              ; Abbrev [23] 0x18b:0x8 DW_TAG_formal_parameter
	.byte	46                              ; DW_AT_name
	.byte	1                               ; DW_AT_decl_file
	.byte	3                               ; DW_AT_decl_line
	.long	532                             ; DW_AT_type
	.byte	24                              ; Abbrev [24] 0x193:0x9 DW_TAG_formal_parameter
	.byte	0                               ; DW_AT_location
	.byte	38                              ; DW_AT_name
	.byte	1                               ; DW_AT_decl_file
	.byte	3                               ; DW_AT_decl_line
	.long	532                             ; DW_AT_type
	.byte	25                              ; Abbrev [25] 0x19c:0x9 DW_TAG_variable
	.byte	1                               ; DW_AT_location
	.byte	40                              ; DW_AT_name
	.byte	1                               ; DW_AT_decl_file
	.byte	6                               ; DW_AT_decl_line
	.long	536                             ; DW_AT_type
	.byte	25                              ; Abbrev [25] 0x1a5:0x9 DW_TAG_variable
	.byte	3                               ; DW_AT_location
	.byte	42                              ; DW_AT_name
	.byte	1                               ; DW_AT_decl_file
	.byte	8                               ; DW_AT_decl_line
	.long	536                             ; DW_AT_type
	.byte	26                              ; Abbrev [26] 0x1ae:0x8 DW_TAG_variable
	.byte	47                              ; DW_AT_name
	.byte	1                               ; DW_AT_decl_file
	.byte	7                               ; DW_AT_decl_line
	.long	536                             ; DW_AT_type
	.byte	27                              ; Abbrev [27] 0x1b6:0x14 DW_TAG_inlined_subroutine
	.long	273                             ; DW_AT_abstract_origin
	.byte	0                               ; DW_AT_ranges
	.byte	1                               ; DW_AT_call_file
	.byte	6                               ; DW_AT_call_line
	.byte	19                              ; DW_AT_call_column
	.byte	28                              ; Abbrev [28] 0x1bf:0xa DW_TAG_inlined_subroutine
	.long	186                             ; DW_AT_abstract_origin
	.byte	0                               ; DW_AT_ranges
	.byte	2                               ; DW_AT_call_file
	.short	2500                            ; DW_AT_call_line
	.byte	160                             ; DW_AT_call_column
	.byte	0                               ; End Of Children Mark
	.byte	29                              ; Abbrev [29] 0x1ca:0xc DW_TAG_lexical_block
	.byte	1                               ; DW_AT_ranges
	.byte	25                              ; Abbrev [25] 0x1cc:0x9 DW_TAG_variable
	.byte	2                               ; DW_AT_location
	.byte	41                              ; DW_AT_name
	.byte	1                               ; DW_AT_decl_file
	.byte	16                              ; DW_AT_decl_line
	.long	532                             ; DW_AT_type
	.byte	0                               ; End Of Children Mark
	.byte	27                              ; Abbrev [27] 0x1d6:0x14 DW_TAG_inlined_subroutine
	.long	356                             ; DW_AT_abstract_origin
	.byte	2                               ; DW_AT_ranges
	.byte	1                               ; DW_AT_call_file
	.byte	8                               ; DW_AT_call_line
	.byte	23                              ; DW_AT_call_column
	.byte	28                              ; Abbrev [28] 0x1df:0xa DW_TAG_inlined_subroutine
	.long	278                             ; DW_AT_abstract_origin
	.byte	2                               ; DW_AT_ranges
	.byte	2                               ; DW_AT_call_file
	.short	2512                            ; DW_AT_call_line
	.byte	160                             ; DW_AT_call_column
	.byte	0                               ; End Of Children Mark
	.byte	0                               ; End Of Children Mark
	.byte	15                              ; Abbrev [15] 0x1eb:0x13 DW_TAG_namespace
	.byte	32                              ; DW_AT_name
	.byte	30                              ; Abbrev [30] 0x1ed:0x8 DW_TAG_imported_declaration
	.byte	2                               ; DW_AT_decl_file
	.short	2746                            ; DW_AT_decl_line
	.long	510                             ; DW_AT_import
	.byte	30                              ; Abbrev [30] 0x1f5:0x8 DW_TAG_imported_declaration
	.byte	2                               ; DW_AT_decl_file
	.short	2747                            ; DW_AT_decl_line
	.long	523                             ; DW_AT_import
	.byte	0                               ; End Of Children Mark
	.byte	14                              ; Abbrev [14] 0x1fe:0x9 DW_TAG_typedef
	.long	519                             ; DW_AT_type
	.byte	34                              ; DW_AT_name
	.byte	2                               ; DW_AT_decl_file
	.short	2742                            ; DW_AT_decl_line
	.byte	16                              ; Abbrev [16] 0x207:0x4 DW_TAG_base_type
	.byte	33                              ; DW_AT_name
	.byte	5                               ; DW_AT_encoding
	.byte	8                               ; DW_AT_byte_size
	.byte	14                              ; Abbrev [14] 0x20b:0x9 DW_TAG_typedef
	.long	519                             ; DW_AT_type
	.byte	35                              ; DW_AT_name
	.byte	2                               ; DW_AT_decl_file
	.short	2744                            ; DW_AT_decl_line
	.byte	16                              ; Abbrev [16] 0x214:0x4 DW_TAG_base_type
	.byte	39                              ; DW_AT_name
	.byte	5                               ; DW_AT_encoding
	.byte	4                               ; DW_AT_byte_size
	.byte	20                              ; Abbrev [20] 0x218:0x5 DW_TAG_const_type
	.long	532                             ; DW_AT_type
	.byte	5                               ; Abbrev [5] 0x21d:0x5 DW_TAG_pointer_type
	.long	536                             ; DW_AT_type
	.byte	5                               ; Abbrev [5] 0x222:0x5 DW_TAG_pointer_type
	.long	532                             ; DW_AT_type
	.byte	0                               ; End Of Children Mark
.Ldebug_info_end0:
	.section	.debug_rnglists,"",@progbits
	.long	.Ldebug_list_header_end1-.Ldebug_list_header_start1 ; Length
.Ldebug_list_header_start1:
	.short	5                               ; Version
	.byte	8                               ; Address size
	.byte	0                               ; Segment selector size
	.long	3                               ; Offset entry count
.Lrnglists_table_base0:
	.long	.Ldebug_ranges0-.Lrnglists_table_base0
	.long	.Ldebug_ranges1-.Lrnglists_table_base0
	.long	.Ldebug_ranges2-.Lrnglists_table_base0
.Ldebug_ranges0:
	.byte	4                               ; DW_RLE_offset_pair
	.uleb128 .Lfunc_begin0-.Lfunc_begin0    ;   starting offset
	.uleb128 .Ltmp0-.Lfunc_begin0           ;   ending offset
	.byte	4                               ; DW_RLE_offset_pair
	.uleb128 .Ltmp3-.Lfunc_begin0           ;   starting offset
	.uleb128 .Ltmp4-.Lfunc_begin0           ;   ending offset
	.byte	0                               ; DW_RLE_end_of_list
.Ldebug_ranges1:
	.byte	4                               ; DW_RLE_offset_pair
	.uleb128 .Ltmp0-.Lfunc_begin0           ;   starting offset
	.uleb128 .Ltmp2-.Lfunc_begin0           ;   ending offset
	.byte	4                               ; DW_RLE_offset_pair
	.uleb128 .Ltmp6-.Lfunc_begin0           ;   starting offset
	.uleb128 .Ltmp13-.Lfunc_begin0          ;   ending offset
	.byte	0                               ; DW_RLE_end_of_list
.Ldebug_ranges2:
	.byte	4                               ; DW_RLE_offset_pair
	.uleb128 .Ltmp2-.Lfunc_begin0           ;   starting offset
	.uleb128 .Ltmp3-.Lfunc_begin0           ;   ending offset
	.byte	4                               ; DW_RLE_offset_pair
	.uleb128 .Ltmp5-.Lfunc_begin0           ;   starting offset
	.uleb128 .Ltmp6-.Lfunc_begin0           ;   ending offset
	.byte	0                               ; DW_RLE_end_of_list
.Ldebug_list_header_end1:
	.section	.debug_str_offsets,"",@progbits
	.long	196                             ; Length of String Offsets Set
	.short	5
	.short	0
.Lstr_offsets_base0:
	.section	.debug_str,"MS",@progbits,1
.Linfo_string0:
	.asciz	"AMD clang version 20.0.0git (https://github.com/RadeonOpenCompute/llvm-project roc-7.0.2 25385 0dda3adf56766e0aac0d03173ced3759e1ffecbc)" ; string offset=0
.Linfo_string1:
	.asciz	"add_kernel.cu"                 ; string offset=137
.Linfo_string2:
	.asciz	"/root/learn-cuda/02c_matmul_hip/learn-ptx/hip" ; string offset=151
.Linfo_string3:
	.asciz	"__amdgcn_name_expr_1"          ; string offset=197
.Linfo_string4:
	.asciz	"__ARRAY_SIZE_TYPE__"           ; string offset=218
.Linfo_string5:
	.asciz	"_ZL20__amdgcn_name_expr_1"     ; string offset=238
.Linfo_string6:
	.asciz	"x"                             ; string offset=264
.Linfo_string7:
	.asciz	"__hip_internal"                ; string offset=266
.Linfo_string8:
	.asciz	"unsigned int"                  ; string offset=281
.Linfo_string9:
	.asciz	"uint32_t"                      ; string offset=294
.Linfo_string10:
	.asciz	"__hip_uint32_t"                ; string offset=303
.Linfo_string11:
	.asciz	"y"                             ; string offset=318
.Linfo_string12:
	.asciz	"z"                             ; string offset=320
.Linfo_string13:
	.asciz	"dim3"                          ; string offset=322
.Linfo_string14:
	.asciz	"_ZL22__hip_get_thread_idx_xv"  ; string offset=327
.Linfo_string15:
	.asciz	"__hip_get_thread_idx_x"        ; string offset=356
.Linfo_string16:
	.asciz	"_ZN25__hip_builtin_threadIdx_t7__get_xEv" ; string offset=379
.Linfo_string17:
	.asciz	"__get_x"                       ; string offset=420
.Linfo_string18:
	.asciz	"_ZN25__hip_builtin_threadIdx_t7__get_yEv" ; string offset=428
.Linfo_string19:
	.asciz	"__get_y"                       ; string offset=469
.Linfo_string20:
	.asciz	"_ZN25__hip_builtin_threadIdx_t7__get_zEv" ; string offset=477
.Linfo_string21:
	.asciz	"__get_z"                       ; string offset=518
.Linfo_string22:
	.asciz	"_ZNK25__hip_builtin_threadIdx_tcv4dim3Ev" ; string offset=526
.Linfo_string23:
	.asciz	"operator dim3"                 ; string offset=567
.Linfo_string24:
	.asciz	"__hip_builtin_threadIdx_t"     ; string offset=581
.Linfo_string25:
	.asciz	"_ZL21__hip_get_block_dim_xv"   ; string offset=607
.Linfo_string26:
	.asciz	"__hip_get_block_dim_x"         ; string offset=635
.Linfo_string27:
	.asciz	"_ZN24__hip_builtin_blockDim_t7__get_xEv" ; string offset=657
.Linfo_string28:
	.asciz	"_ZN24__hip_builtin_blockDim_t7__get_yEv" ; string offset=697
.Linfo_string29:
	.asciz	"_ZN24__hip_builtin_blockDim_t7__get_zEv" ; string offset=737
.Linfo_string30:
	.asciz	"_ZNK24__hip_builtin_blockDim_tcv4dim3Ev" ; string offset=777
.Linfo_string31:
	.asciz	"__hip_builtin_blockDim_t"      ; string offset=817
.Linfo_string32:
	.asciz	"std"                           ; string offset=842
.Linfo_string33:
	.asciz	"long"                          ; string offset=846
.Linfo_string34:
	.asciz	"ptrdiff_t"                     ; string offset=851
.Linfo_string35:
	.asciz	"clock_t"                       ; string offset=861
.Linfo_string36:
	.asciz	"_Z10add_kernelPKiS0_Piii"      ; string offset=869
.Linfo_string37:
	.asciz	"add_kernel"                    ; string offset=894
.Linfo_string38:
	.asciz	"N"                             ; string offset=905
.Linfo_string39:
	.asciz	"int"                           ; string offset=907
.Linfo_string40:
	.asciz	"tid"                           ; string offset=911
.Linfo_string41:
	.asciz	"col"                           ; string offset=915
.Linfo_string42:
	.asciz	"tb_size"                       ; string offset=919
.Linfo_string43:
	.asciz	"A"                             ; string offset=927
.Linfo_string44:
	.asciz	"B"                             ; string offset=929
.Linfo_string45:
	.asciz	"C"                             ; string offset=931
.Linfo_string46:
	.asciz	"M"                             ; string offset=933
.Linfo_string47:
	.asciz	"bid"                           ; string offset=935
	.section	.debug_str_offsets,"",@progbits
	.long	.Linfo_string0
	.long	.Linfo_string1
	.long	.Linfo_string2
	.long	.Linfo_string3
	.long	.Linfo_string4
	.long	.Linfo_string5
	.long	.Linfo_string6
	.long	.Linfo_string7
	.long	.Linfo_string8
	.long	.Linfo_string9
	.long	.Linfo_string10
	.long	.Linfo_string11
	.long	.Linfo_string12
	.long	.Linfo_string13
	.long	.Linfo_string14
	.long	.Linfo_string15
	.long	.Linfo_string16
	.long	.Linfo_string17
	.long	.Linfo_string18
	.long	.Linfo_string19
	.long	.Linfo_string20
	.long	.Linfo_string21
	.long	.Linfo_string22
	.long	.Linfo_string23
	.long	.Linfo_string24
	.long	.Linfo_string25
	.long	.Linfo_string26
	.long	.Linfo_string27
	.long	.Linfo_string28
	.long	.Linfo_string29
	.long	.Linfo_string30
	.long	.Linfo_string31
	.long	.Linfo_string32
	.long	.Linfo_string33
	.long	.Linfo_string34
	.long	.Linfo_string35
	.long	.Linfo_string36
	.long	.Linfo_string37
	.long	.Linfo_string38
	.long	.Linfo_string39
	.long	.Linfo_string40
	.long	.Linfo_string41
	.long	.Linfo_string42
	.long	.Linfo_string43
	.long	.Linfo_string44
	.long	.Linfo_string45
	.long	.Linfo_string46
	.long	.Linfo_string47
	.section	.debug_addr,"",@progbits
	.long	.Ldebug_addr_end0-.Ldebug_addr_start0 ; Length of contribution
.Ldebug_addr_start0:
	.short	5                               ; DWARF version number
	.byte	8                               ; Address size
	.byte	0                               ; Segment selector size
.Laddr_table_base0:
	.quad	_ZL20__amdgcn_name_expr_1
	.quad	.Lfunc_begin0
.Ldebug_addr_end0:
	.ident	"AMD clang version 20.0.0git (https://github.com/RadeonOpenCompute/llvm-project roc-7.0.2 25385 0dda3adf56766e0aac0d03173ced3759e1ffecbc)"
	.section	".note.GNU-stack","",@progbits
	.addrsig
	.addrsig_sym _Z10add_kernelPKiS0_Piii
	.addrsig_sym _ZL20__amdgcn_name_expr_1
	.addrsig_sym __hip_cuid_ab25958aa643f7a5
	.amdgpu_metadata
---
amdhsa.kernels:
  - .agpr_count:     0
    .args:
      - .address_space:  global
        .offset:         0
        .size:           8
        .value_kind:     global_buffer
      - .address_space:  global
        .offset:         8
        .size:           8
        .value_kind:     global_buffer
      - .address_space:  global
        .offset:         16
        .size:           8
        .value_kind:     global_buffer
      - .offset:         24
        .size:           4
        .value_kind:     by_value
      - .offset:         28
        .size:           4
        .value_kind:     by_value
      - .offset:         32
        .size:           4
        .value_kind:     hidden_block_count_x
      - .offset:         36
        .size:           4
        .value_kind:     hidden_block_count_y
      - .offset:         40
        .size:           4
        .value_kind:     hidden_block_count_z
      - .offset:         44
        .size:           2
        .value_kind:     hidden_group_size_x
      - .offset:         46
        .size:           2
        .value_kind:     hidden_group_size_y
      - .offset:         48
        .size:           2
        .value_kind:     hidden_group_size_z
      - .offset:         50
        .size:           2
        .value_kind:     hidden_remainder_x
      - .offset:         52
        .size:           2
        .value_kind:     hidden_remainder_y
      - .offset:         54
        .size:           2
        .value_kind:     hidden_remainder_z
      - .offset:         72
        .size:           8
        .value_kind:     hidden_global_offset_x
      - .offset:         80
        .size:           8
        .value_kind:     hidden_global_offset_y
      - .offset:         88
        .size:           8
        .value_kind:     hidden_global_offset_z
      - .offset:         96
        .size:           2
        .value_kind:     hidden_grid_dims
    .group_segment_fixed_size: 0
    .kernarg_segment_align: 8
    .kernarg_segment_size: 288
    .language:       OpenCL C
    .language_version:
      - 2
      - 0
    .max_flat_workgroup_size: 1024
    .name:           _Z10add_kernelPKiS0_Piii
    .private_segment_fixed_size: 0
    .sgpr_count:     18
    .sgpr_spill_count: 0
    .symbol:         _Z10add_kernelPKiS0_Piii.kd
    .uniform_work_group_size: 1
    .uses_dynamic_stack: false
    .vgpr_count:     8
    .vgpr_spill_count: 0
    .wavefront_size: 64
amdhsa.target:   'amdgcn-amd-amdhsa--gfx942:sramecc+:xnack-'
amdhsa.version:
  - 1
  - 2
...

	.end_amdgpu_metadata
	.section	.debug_line,"",@progbits
.Lline_table_start0:
