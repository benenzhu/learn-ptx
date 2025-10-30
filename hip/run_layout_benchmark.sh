#!/bin/bash

echo "========================================"
echo "LDS Layout Benchmark"
echo "========================================"
echo ""

RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

# Compile
echo -e "${YELLOW}Compiling benchmark...${NC}"
hipcc -O3 -save-temps \
      --offload-arch=gfx942 \
      -o lds_layout_benchmark \
      lds_layout_benchmark.hip

if [ $? -ne 0 ]; then
    echo -e "${RED}Compilation failed!${NC}"
    exit 1
fi

echo -e "${GREEN}Compilation successful!${NC}"
echo ""

# Analyze assembly
ASM_FILE="lds_layout_benchmark-hip-amdgcn-amd-amdhsa-gfx942.s"
if [ -f "$ASM_FILE" ]; then
    echo -e "${YELLOW}Analyzing generated assembly...${NC}"
    echo "================================================"
    
    # Count different DS instructions
    echo -e "${BLUE}DS Instruction counts:${NC}"
    
    ds_read2st64=$(grep -c "ds_read2st64_b64" "$ASM_FILE" 2>/dev/null || echo "0")
    ds_read2=$(grep -c "ds_read2_b64" "$ASM_FILE" 2>/dev/null || echo "0")
    ds_read_b128=$(grep -c "ds_read_b128" "$ASM_FILE" 2>/dev/null || echo "0")
    ds_read_b64=$(grep -c "ds_read_b64" "$ASM_FILE" 2>/dev/null || echo "0")
    ds_read=$(grep -c "^\s*ds_read" "$ASM_FILE" 2>/dev/null || echo "0")
    
    printf "  %-25s: %4d\n" "ds_read2st64_b64" "$ds_read2st64"
    printf "  %-25s: %4d\n" "ds_read2_b64" "$ds_read2"
    printf "  %-25s: %4d\n" "ds_read_b128" "$ds_read_b128"
    printf "  %-25s: %4d\n" "ds_read_b64" "$ds_read_b64"
    printf "  %-25s: %4d\n" "Total ds_read*" "$ds_read"
    
    echo "================================================"
    echo ""
fi

# Run benchmark
echo -e "${YELLOW}Running benchmark...${NC}"
echo "================================================"
./lds_layout_benchmark
echo "================================================"

# Show assembly details if requested
if [ "$1" == "--show-asm" ]; then
    echo ""
    echo -e "${YELLOW}Detailed Assembly Analysis:${NC}"
    echo "================================================"
    
    if [ -f "$ASM_FILE" ]; then
        echo -e "\n${BLUE}compact_layout kernel:${NC}"
        sed -n '/_Z22benchmark_compact/,/_Z22benchmark_strided/p' "$ASM_FILE" | grep "ds_read" | head -20
        
        echo -e "\n${BLUE}strided_layout kernel:${NC}"
        sed -n '/_Z22benchmark_strided/,/_Z24benchmark_transpose/p' "$ASM_FILE" | grep "ds_read" | head -20
        
        echo -e "\n${BLUE}transpose_layout kernel:${NC}"
        sed -n '/_Z24benchmark_transpose/,/\.size/p' "$ASM_FILE" | grep "ds_read" | head -20
    fi
fi

echo ""
echo -e "${GREEN}Benchmark complete!${NC}"

