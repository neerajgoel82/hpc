# CUDA Local Development - Complete Guide

**67 CUDA programs** organized in 9 phases for learning GPU programming with your local NVIDIA GPU.

---

## Quick Start

```bash
# 1. Verify CUDA installation
nvcc --version
nvidia-smi

# 2. Compile and run first program
cd phase1
nvcc 01_hello_world.cu -o hello && ./hello

# 3. Follow the phases sequentially
```

---

## Prerequisites

### Hardware
- NVIDIA GPU with compute capability 3.0+ (7.0+ recommended)
- Check compatibility: https://developer.nvidia.com/cuda-gpus

### Software
- CUDA Toolkit 11.0+ (12.0+ recommended)
- NVIDIA Driver (latest stable)
- C/C++ Compiler: gcc 7.0+ (Linux), MSVC 2019+ (Windows)

---

## Installation Guide

### Linux (Ubuntu/Debian)

#### Step 1: Verify GPU
```bash
lspci | grep -i nvidia
```

#### Step 2: Install NVIDIA Driver
```bash
# Update packages
sudo apt update

# Install recommended driver
sudo ubuntu-drivers autoinstall

# Or install specific version
sudo apt install nvidia-driver-525

# Reboot
sudo reboot

# Verify
nvidia-smi
```

#### Step 3: Install CUDA Toolkit
```bash
# Using package manager (recommended)
wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/cuda-keyring_1.1-1_all.deb
sudo dpkg -i cuda-keyring_1.1-1_all.deb
sudo apt-get update
sudo apt-get install cuda-toolkit-12-3

# Or use runfile installer for more control
wget https://developer.download.nvidia.com/compute/cuda/12.3.0/local_installers/cuda_12.3.0_545.23.06_linux.run
sudo sh cuda_12.3.0_545.23.06_linux.run
```

#### Step 4: Set Environment Variables
Add to `~/.bashrc` or `~/.zshrc`:
```bash
export PATH=/usr/local/cuda/bin:$PATH
export LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH
```

Reload:
```bash
source ~/.bashrc
```

#### Step 5: Verify Installation
```bash
nvcc --version
nvidia-smi

# Test compile
nvcc phase1/01_hello_world.cu -o test && ./test
```

---

### Windows

#### Step 1: Install Visual Studio
- Download Visual Studio 2019 or 2022 Community Edition
- Install with "Desktop development with C++" workload
- Ensure MSVC compiler is installed

#### Step 2: Install NVIDIA Driver
- Visit https://www.nvidia.com/Download/index.aspx
- Download and install the latest driver for your GPU
- Reboot

#### Step 3: Install CUDA Toolkit
- Download from https://developer.nvidia.com/cuda-downloads
- Run the installer (cuda_12.x.x_win10.exe)
- Select components: driver, toolkit, documentation
- Follow installation wizard
- Installer will set PATH automatically

#### Step 4: Verify Installation
```cmd
nvcc --version
nvidia-smi

# Test compile
nvcc phase1\01_hello_world.cu -o test.exe
test.exe
```

---

### macOS (Limited Support)

**⚠️ Important**: NVIDIA stopped CUDA support for macOS after CUDA 10.2.

- **Apple Silicon Macs (M1/M2/M3)**: CUDA not available, use Google Colab instead
- **Intel Macs with NVIDIA GPUs**: Can use CUDA 10.2 (outdated, not recommended)

**Recommendation**: Use Google Colab for CUDA development on Mac.

---

## Directory Structure

```
local/
├── README.md           # This file
│
├── phase1/            # 7 programs - Foundations
│   ├── 01_hello_world.cu
│   ├── 02_device_query.cu
│   └── ...
│
├── phase2/            # 7 programs - Memory Management
├── phase3/            # 9 programs - Optimization
├── phase4/            # 7 programs - Advanced Memory
├── phase5/            # 7 programs - Advanced Algorithms
├── phase6/            # 7 programs - Streams & Concurrency
├── phase7/            # 6 programs - Performance Engineering
├── phase8/            # 10 programs - Real Applications
└── phase9/            # 7 programs - Modern CUDA
```

**Total**: 67 CUDA programs across 9 phases

---

## Compilation Guide

### Basic Compilation
```bash
nvcc program.cu -o program
./program
```

### With Compute Capability
```bash
# For Volta/Turing/Ampere GPUs
nvcc -arch=sm_70 program.cu -o program

# For specific GPU architectures
nvcc -arch=sm_80 program.cu -o program  # Ampere (A100, RTX 30xx)
nvcc -arch=sm_89 program.cu -o program  # Ada (RTX 40xx)
```

### With CUDA Libraries
```bash
# cuBLAS
nvcc -arch=sm_70 program.cu -o program -lcublas

# cuFFT
nvcc -arch=sm_70 program.cu -o program -lcufft

# cuSPARSE
nvcc -arch=sm_70 program.cu -o program -lcusparse

# cuRAND
nvcc -arch=sm_70 program.cu -o program -lcurand
```

### With Optimization and Debugging
```bash
# Optimized build
nvcc -O3 -arch=sm_70 program.cu -o program

# Debug build
nvcc -g -G -arch=sm_70 program.cu -o program
```

### Dynamic Parallelism (Phase 9)
```bash
nvcc -arch=sm_35 -rdc=true program.cu -o program -lcudadevrt
```

---

## Compute Capability Reference

Choose the right `-arch` flag for your GPU:

| Architecture | Capability | Example GPUs |
|--------------|-----------|--------------|
| Pascal | sm_60, sm_61 | GTX 1080, Tesla P100 |
| Volta | sm_70 | Tesla V100, Titan V |
| Turing | sm_75 | RTX 2080, T4, Quadro RTX |
| Ampere | sm_80, sm_86 | A100, RTX 3090, RTX 3060 |
| Ada Lovelace | sm_89 | RTX 4090, RTX 4080 |
| Hopper | sm_90 | H100 |

**Find your GPU's capability**:
```bash
cd phase1
nvcc 02_device_query.cu -o query && ./query
```

---

## Learning Path

### Phase 1: Foundations (Week 1-2)
- Hello World, Device Query
- Vector and Matrix Addition
- Thread indexing patterns

**Compile & Run**:
```bash
cd phase1
nvcc 03_vector_add.cu -o vector_add
./vector_add
```

### Phase 2: Memory Management (Week 3-4)
- Host-device transfers
- Shared memory basics
- Unified memory

### Phase 3: Optimization (Week 5-6)
- Tiled matrix multiplication
- Warp shuffle operations
- Parallel reduction

### Phase 4: Advanced Memory (Week 7-8)
- Texture and constant memory
- Zero-copy memory
- Atomic operations

### Phase 5: Advanced Algorithms (Week 9-10)
- Optimized GEMM
- cuBLAS integration
- Sorting algorithms

### Phase 6: Streams & Concurrency (Week 11)
- CUDA streams
- Asynchronous operations
- Multi-GPU basics

### Phase 7: Performance Engineering (Week 12-13)
- Profiling with Nsight
- Kernel fusion
- Debugging techniques

### Phase 8: Real Applications (Week 14-15)
- cuFFT, cuSPARSE, cuRAND
- Image processing
- N-body simulation
- Neural networks

### Phase 9: Modern CUDA (Week 16+)
- Dynamic parallelism
- CUDA graphs
- Tensor cores (WMMA)

**Total Learning Time**: 16+ weeks

---

## Profiling and Debugging

### Nsight Compute (Kernel Profiler)
```bash
# Basic profile
ncu ./program

# Detailed metrics
ncu --set full ./program

# Export report
ncu --export report ./program
```

### Nsight Systems (System Profiler)
```bash
# Profile application
nsys profile ./program

# With statistics
nsys profile --stats=true ./program

# Export for GUI
nsys profile -o report ./program
```

### CUDA Memcheck / Compute Sanitizer
```bash
# Check memory errors
cuda-memcheck ./program

# Modern tool (CUDA 11.4+)
compute-sanitizer ./program

# Check race conditions
compute-sanitizer --tool racecheck ./program
```

### CUDA GDB Debugger
```bash
# Compile with debug symbols
nvcc -g -G program.cu -o program

# Run debugger
cuda-gdb ./program

# Common commands:
# (cuda-gdb) break main
# (cuda-gdb) run
# (cuda-gdb) cuda thread
# (cuda-gdb) cuda kernel
```

---

## Common Issues and Solutions

### Issue: nvcc: command not found
**Cause**: CUDA bin directory not in PATH

**Solution**:
```bash
# Add to ~/.bashrc
export PATH=/usr/local/cuda/bin:$PATH
source ~/.bashrc
```

### Issue: CUDA driver version mismatch
**Cause**: Driver version incompatible with CUDA toolkit

**Solution**: Update NVIDIA driver to match toolkit version
```bash
sudo apt update
sudo apt upgrade nvidia-driver-XXX
```

### Issue: Kernel launch fails silently
**Cause**: No error checking

**Solution**: Always check for errors
```cuda
cudaError_t err = cudaGetLastError();
if (err != cudaSuccess) {
    printf("CUDA error: %s\n", cudaGetErrorString(err));
}
```

### Issue: undefined reference to cudaXXX
**Cause**: Missing CUDA runtime library

**Solution**: Link with `-lcudart`
```bash
nvcc program.cu -o program -lcudart
```

### Issue: Out of memory errors
**Cause**: Allocating too much GPU memory

**Solution**: Check available memory
```cuda
size_t free, total;
cudaMemGetInfo(&free, &total);
printf("Free: %zu MB, Total: %zu MB\n", free/1024/1024, total/1024/1024);
```

---

## Advantages of Local Development

✅ **No session time limits** - Work as long as you need
✅ **Full GPU control** - All features accessible
✅ **Better performance** - No network latency
✅ **Professional tools** - Nsight Compute, Nsight Systems
✅ **Multi-GPU support** - Program multiple GPUs
✅ **Offline development** - No internet required
✅ **Your own pace** - No resource sharing

---

## IDE Setup

### VS Code (Recommended)
1. Install VS Code
2. Install extensions:
   - C/C++ (Microsoft)
   - NVIDIA Nsight Visual Studio Code Edition
3. Configure tasks.json for compilation
4. Use integrated terminal for running programs

### CLion
- Built-in CUDA support
- Good for larger projects
- Excellent debugger integration

### Visual Studio (Windows)
- Install NVIDIA Nsight Visual Studio Edition
- Full IDE integration
- Best debugging experience on Windows

### Command Line
- Lightweight and fast
- Use with Vim/Emacs
- Good for learning CUDA fundamentals

---

## System Requirements

### Minimum
- NVIDIA GPU (compute capability 3.0+)
- CUDA Toolkit 10.0+
- 2GB GPU memory
- gcc 5.0+ or MSVC 2017+

### Recommended
- NVIDIA GPU (compute capability 7.0+)
- CUDA Toolkit 12.0+
- 4GB+ GPU memory
- gcc 9.0+ or MSVC 2019+
- 8GB+ system RAM

### For All Features
- Compute capability 7.0+ for tensor cores (Phase 9)
- Compute capability 3.5+ for dynamic parallelism (Phase 9)
- Multiple GPUs for multi-GPU programming (Phase 6)

---

## Verification Script

Create `check_setup.sh`:
```bash
#!/bin/bash

echo "=== CUDA Installation Check ==="
echo

echo "1. NVIDIA Driver:"
nvidia-smi --query-gpu=driver_version --format=csv,noheader
echo

echo "2. CUDA Compiler:"
nvcc --version | grep "release"
echo

echo "3. GPU Information:"
nvidia-smi --query-gpu=name,compute_cap,memory.total --format=csv
echo

echo "4. Test Compilation:"
cd phase1
if nvcc 01_hello_world.cu -o test_compile 2>/dev/null; then
    echo "✓ Compilation successful"
    ./test_compile
    rm test_compile
else
    echo "✗ Compilation failed"
fi
```

Run:
```bash
chmod +x check_setup.sh
./check_setup.sh
```

---

## Next Steps

1. **Verify installation**: Run the verification script above
2. **Test compilation**: Build and run `phase1/01_hello_world.cu`
3. **Query your GPU**: Run `phase1/02_device_query.cu`
4. **Start learning**: Follow phases 1-9 sequentially
5. **Use profiling tools**: Essential in Phase 7 and beyond

---

## Resources

### Official Documentation
- [CUDA Programming Guide](https://docs.nvidia.com/cuda/cuda-c-programming-guide/)
- [CUDA Installation Guide](https://docs.nvidia.com/cuda/cuda-installation-guide-linux/)
- [cuBLAS Documentation](https://docs.nvidia.com/cuda/cublas/)
- [cuFFT Documentation](https://docs.nvidia.com/cuda/cufft/)
- [Nsight Compute](https://docs.nvidia.com/nsight-compute/)
- [Nsight Systems](https://docs.nvidia.com/nsight-systems/)

### Community
- [NVIDIA Developer Forums](https://forums.developer.nvidia.com/)
- [Stack Overflow - CUDA Tag](https://stackoverflow.com/questions/tagged/cuda)
- [CUDA by Example (Book)](http://www.cudabyexample.com/)

### Tools
- [CUDA Toolkit Download](https://developer.nvidia.com/cuda-downloads)
- [GPU Compatibility List](https://developer.nvidia.com/cuda-gpus)
- [Nsight Tools Download](https://developer.nvidia.com/nsight-visual-studio-edition)

---

## Getting Help

If you encounter issues:

1. **Check this README** - Common solutions above
2. **Read error messages** - CUDA errors are usually descriptive
3. **Search NVIDIA Forums** - Someone likely had the same issue
4. **Check GPU compatibility** - Verify your GPU supports the features
5. **Update drivers** - Keep NVIDIA drivers up to date
6. **Stack Overflow** - Search the CUDA tag

---

**Ready to start?** Verify your installation and compile `phase1/01_hello_world.cu`! 🚀
