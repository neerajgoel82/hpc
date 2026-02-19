#!/bin/bash

# HPC Monorepo Setup Script
# Checks for required tools and dependencies

set -e

echo "==================================="
echo "HPC Monorepo Setup"
echo "==================================="
echo ""

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Function to check if a command exists
command_exists() {
    command -v "$1" >/dev/null 2>&1
}

# Function to print status
print_status() {
    if [ $1 -eq 0 ]; then
        echo -e "${GREEN}✓${NC} $2"
    else
        echo -e "${RED}✗${NC} $2"
    fi
}

# Track overall status
all_good=true

echo "Checking C Development Tools..."
echo "-----------------------------------"

# Check GCC
if command_exists gcc; then
    gcc_version=$(gcc --version | head -n1)
    print_status 0 "GCC found: $gcc_version"
else
    print_status 1 "GCC not found"
    echo -e "${YELLOW}  Install: brew install gcc (macOS) or apt-get install build-essential (Ubuntu)${NC}"
    all_good=false
fi

# Check Make
if command_exists make; then
    make_version=$(make --version | head -n1)
    print_status 0 "Make found: $make_version"
else
    print_status 1 "Make not found"
    all_good=false
fi

echo ""
echo "Checking C++ Development Tools..."
echo "-----------------------------------"

# Check G++
if command_exists g++; then
    gpp_version=$(g++ --version | head -n1)
    print_status 0 "G++ found: $gpp_version"
else
    print_status 1 "G++ not found"
    echo -e "${YELLOW}  Install: brew install gcc (macOS) or apt-get install build-essential (Ubuntu)${NC}"
    all_good=false
fi

echo ""
echo "Checking CUDA Development Tools..."
echo "-----------------------------------"

# Check NVCC
if command_exists nvcc; then
    nvcc_version=$(nvcc --version | grep "release" | awk '{print $5}')
    print_status 0 "NVCC found: version $nvcc_version"
else
    print_status 1 "NVCC not found (optional if no local GPU)"
    echo -e "${YELLOW}  Install: https://developer.nvidia.com/cuda-downloads${NC}"
fi

# Check nvidia-smi
if command_exists nvidia-smi; then
    print_status 0 "nvidia-smi found"
    echo -e "${YELLOW}  GPU Info:${NC}"
    nvidia-smi --query-gpu=name,compute_cap --format=csv,noheader 2>/dev/null || echo "  Could not query GPU"
else
    print_status 1 "nvidia-smi not found (optional if no local GPU)"
fi

echo ""
echo "Checking Python Development Tools..."
echo "-----------------------------------"

# Check Python
if command_exists python3; then
    python_version=$(python3 --version)
    print_status 0 "Python found: $python_version"

    # Check if version is 3.9+
    py_major=$(python3 -c 'import sys; print(sys.version_info.major)')
    py_minor=$(python3 -c 'import sys; print(sys.version_info.minor)')
    if [ "$py_major" -ge 3 ] && [ "$py_minor" -ge 9 ]; then
        print_status 0 "Python version is 3.9+ ✓"
    else
        print_status 1 "Python version is less than 3.9"
        echo -e "${YELLOW}  Recommended: Python 3.9 or later${NC}"
    fi
else
    print_status 1 "Python3 not found"
    echo -e "${YELLOW}  Install: brew install python3 (macOS) or apt-get install python3 (Ubuntu)${NC}"
    all_good=false
fi

# Check pip
if command_exists pip3; then
    pip_version=$(pip3 --version)
    print_status 0 "pip3 found: $pip_version"
else
    print_status 1 "pip3 not found"
    echo -e "${YELLOW}  Install: python3 -m ensurepip --upgrade${NC}"
fi

echo ""
echo "Checking Optional Tools..."
echo "-----------------------------------"

# Check CMake
if command_exists cmake; then
    cmake_version=$(cmake --version | head -n1)
    print_status 0 "CMake found: $cmake_version"
else
    print_status 1 "CMake not found (optional, but recommended for C++)"
    echo -e "${YELLOW}  Install: brew install cmake (macOS) or apt-get install cmake (Ubuntu)${NC}"
fi

# Check Git
if command_exists git; then
    git_version=$(git --version)
    print_status 0 "Git found: $git_version"
else
    print_status 1 "Git not found"
    echo -e "${YELLOW}  Install: brew install git (macOS) or apt-get install git (Ubuntu)${NC}"
fi

# Check Valgrind (memory checker)
if command_exists valgrind; then
    valgrind_version=$(valgrind --version)
    print_status 0 "Valgrind found: $valgrind_version"
else
    print_status 1 "Valgrind not found (optional, useful for C/C++ debugging)"
    echo -e "${YELLOW}  Install: brew install valgrind (macOS) or apt-get install valgrind (Ubuntu)${NC}"
fi

echo ""
echo "==================================="
if [ "$all_good" = true ]; then
    echo -e "${GREEN}All required tools are installed!${NC}"
    echo ""
    echo "Next steps:"
    echo "  1. Explore language directories: c/, cpp/, cuda/, python/"
    echo "  2. Read the main README.md"
    echo "  3. Start with a learning path from docs/"
else
    echo -e "${YELLOW}Some tools are missing. Install them to get started.${NC}"
fi
echo "==================================="
