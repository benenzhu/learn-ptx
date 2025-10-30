#!/bin/bash

echo "========================================"
echo "LDS Instruction Microbenchmark"
echo "========================================"
echo ""

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Compile
echo -e "${YELLOW}Compiling benchmark...${NC}"
hipcc -O3 -save-temps \
      --offload-arch=gfx942 \
      -o lds_instruction_benchmark \
      lds_instruction_benchmark.hip

if [ $? -ne 0 ]; then
    echo -e "${RED}Compilation failed!${NC}"
    exit 1
fi

echo -e "${GREEN}Compilation successful!${NC}"
echo ""

# Check if assembly was generated
if [ -f "lds_instruction_benchmark-hip-amdgcn-amd-amdhsa-gfx942.s" ]; then
    echo -e "${YELLOW}Checking generated assembly for instructions...${NC}"
    
    echo -n "  ds_read2st64_b64 count: "
    grep -c "ds_read2st64_b64" lds_instruction_benchmark-hip-amdgcn-amd-amdhsa-gfx942.s || echo "0"
    
    echo -n "  ds_read2_b64 count:     "
    grep -c "ds_read2_b64" lds_instruction_benchmark-hip-amdgcn-amd-amdhsa-gfx942.s || echo "0"
    
    echo ""
fi

# Run benchmark
echo -e "${YELLOW}Running benchmark...${NC}"
echo "========================================"
echo ""

./lds_instruction_benchmark

echo ""
echo -e "${GREEN}Benchmark complete!${NC}"

# Optional: show some assembly snippets
if [ "$1" == "--show-asm" ]; then
    echo ""
    echo "========================================"
    echo "Assembly snippets:"
    echo "========================================"
    
    echo ""
    echo "--- ds_read2st64_b64 usage ---"
    grep -A 2 -B 2 "ds_read2st64_b64" lds_instruction_benchmark-hip-amdgcn-amd-amdhsa-gfx942.s | head -20
    
    echo ""
    echo "--- ds_read2_b64 usage ---"
    grep -A 2 -B 2 "ds_read2_b64" lds_instruction_benchmark-hip-amdgcn-amd-amdhsa-gfx942.s | head -20
fi

