# Local CUDA Development Setup

This guide will help you set up CUDA development on your local machine with an NVIDIA GPU.

## Prerequisites

- NVIDIA GPU (check compatibility at https://developer.nvidia.com/cuda-gpus)
- Appropriate OS: Linux (Ubuntu/RHEL), Windows 10/11, or macOS (limited support)
- Administrator/root access for installation

## Installation Steps

### Linux (Ubuntu/Debian)

#### 1. Verify GPU
```bash
lspci | grep -i nvidia
```

#### 2. Install NVIDIA Driver
```bash
# Update package list
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

#### 3. Install CUDA Toolkit
```bash
# Download from NVIDIA (replace with current version)
wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/cuda-keyring_1.1-1_all.deb
sudo dpkg -i cuda-keyring_1.1-1_all.deb
sudo apt-get update
sudo apt-get install cuda-toolkit-12-3

# Or use runfile installer for more control
wget https://developer.download.nvidia.com/compute/cuda/12.3.0/local_installers/cuda_12.3.0_545.23.06_linux.run
sudo sh cuda_12.3.0_545.23.06_linux.run
```

#### 4. Set Environment Variables
Add to `~/.bashrc` or `~/.zshrc`:
```bash
export PATH=/usr/local/cuda/bin:$PATH
export LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH
```

Reload:
```bash
source ~/.bashrc
```

#### 5. Verify Installation
```bash
nvcc --version
nvidia-smi
```

### Windows

#### 1. Install Visual Studio
- Download Visual Studio 2019 or 2022 Community Edition
- Install with "Desktop development with C++" workload

#### 2. Install NVIDIA Driver
- Visit https://www.nvidia.com/Download/index.aspx
- Download and install the latest driver for your GPU

#### 3. Install CUDA Toolkit
- Download from https://developer.nvidia.com/cuda-downloads
- Run the installer
- Select components (driver, toolkit, samples, documentation)
- Follow installation wizard

#### 4. Verify Installation
```cmd
nvcc --version
nvidia-smi
```

### macOS

**Note**: NVIDIA stopped CUDA support for macOS after CUDA 10.2. For newer Macs with Apple Silicon, CUDA is not available. Use Colab instead.

For older Macs with NVIDIA GPUs:
- Install CUDA 10.2 (last supported version)
- Limited functionality
- Not recommended for learning

## Post-Installation

### Compile Test Program

Create `test.cu`:
```cuda
#include <stdio.h>

__global__ void hello() {
    printf("Hello from GPU!\n");
}

int main() {
    hello<<<1, 1>>>();
    cudaDeviceSynchronize();
    return 0;
}
```

Compile and run:
```bash
nvcc test.cu -o test
./test
```

### Install Development Tools

#### CMake
```bash
# Linux
sudo apt install cmake

# Windows
# Download from https://cmake.org/download/
```

#### Nsight Tools
Included with CUDA Toolkit:
- **Nsight Compute**: Kernel profiler (`ncu`)
- **Nsight Systems**: System profiler (`nsys`)

Verify:
```bash
ncu --version
nsys --version
```

## Common Issues

### Issue: CUDA driver version mismatch
**Solution**: Update NVIDIA driver to match CUDA toolkit version

### Issue: nvcc: command not found
**Solution**: Add CUDA bin directory to PATH

### Issue: undefined reference to cudaXXX
**Solution**: Link with `-lcudart` flag:
```bash
nvcc program.cu -o program -lcudart
```

### Issue: Kernel launch fails silently
**Solution**: Check for errors:
```cuda
cudaError_t err = cudaGetLastError();
if (err != cudaSuccess) {
    printf("CUDA error: %s\n", cudaGetErrorString(err));
}
```

## Recommended Setup

### IDE Options
- **VS Code**: Install NVIDIA Nsight Visual Studio Code Edition
- **CLion**: CUDA support built-in
- **Visual Studio**: Install NVIDIA Nsight Visual Studio Edition
- **Vim/Emacs**: Use with command-line tools

### Version Control
```bash
# Initialize git in projects
git init
git add .
git commit -m "Initial commit"
```

## System Check Script

Save as `check_cuda.sh`:
```bash
#!/bin/bash

echo "=== CUDA Installation Check ==="
echo

echo "1. NVIDIA Driver:"
nvidia-smi --version
echo

echo "2. CUDA Compiler:"
nvcc --version
echo

echo "3. GPU Information:"
nvidia-smi
echo

echo "4. Compute Capabilities:"
if [ -f "../projects/02-device-query/device_query" ]; then
    ../projects/02-device-query/device_query
else
    echo "Build device_query project first"
fi
```

Run:
```bash
chmod +x check_cuda.sh
./check_cuda.sh
```

## Next Steps

1. Verify installation with system check
2. Build and run `projects/01-hello-world`
3. Build and run `projects/02-device-query`
4. Start following the curriculum

## Resources

- [CUDA Installation Guide](https://docs.nvidia.com/cuda/cuda-installation-guide-linux/)
- [CUDA Quick Start Guide](https://docs.nvidia.com/cuda/cuda-quick-start-guide/)
- [NVIDIA Developer Forums](https://forums.developer.nvidia.com/)

## Getting Help

If you encounter issues:
1. Check NVIDIA documentation
2. Search NVIDIA Developer Forums
3. Check CUDA tag on Stack Overflow
4. Verify GPU compatibility and driver versions
