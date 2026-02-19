# CUDA Learning with Local GPU

This directory contains all resources for learning CUDA programming with a local NVIDIA GPU.

## Prerequisites

- NVIDIA GPU (compute capability 3.0 or higher recommended)
- CUDA Toolkit installed
- C/C++ compiler (gcc/g++ on Linux, MSVC on Windows)
- CMake (optional but recommended)

## Quick Start

1. **Verify installation**: Run `nvcc --version` and `nvidia-smi`
2. **Read setup guide**: `docs/SETUP.md`
3. **Build first project**: Start with `projects/01-hello-world/`

## Directory Structure

```
local/
├── README.md                          # This file
│
├── projects/                          # Sample CUDA projects
│   ├── 01-hello-world/               # Basic kernel launch
│   ├── 02-device-query/              # Query GPU properties
│   ├── 03-vector-add/                # Parallel vector addition
│   ├── 04-matrix-add/                # 2D thread blocks
│   └── ...                           # More projects following curriculum
│
├── common/                           # Shared utilities
│   ├── helper_cuda.h                 # CUDA helper functions
│   ├── helper_string.h               # String utilities
│   ├── timer.h                       # Timing utilities
│   └── utils.h                       # General utilities
│
└── docs/                            # Documentation
    ├── SETUP.md                      # Installation guide
    ├── COMPILATION.md                # Build instructions
    ├── PROFILING.md                  # Using Nsight tools
    └── DEBUGGING.md                  # Debugging CUDA code
```

## Building Projects

Each project directory contains:
- `*.cu` - CUDA source files
- `Makefile` or `CMakeLists.txt` - Build configuration
- `README.md` - Project description and learning objectives

### Using Make
```bash
cd projects/01-hello-world
make
./hello_world
```

### Using CMake
```bash
cd projects/01-hello-world
mkdir build && cd build
cmake ..
make
./hello_world
```

### Manual Compilation
```bash
nvcc hello_world.cu -o hello_world
./hello_world
```

## System Requirements

### Minimum
- NVIDIA GPU with compute capability 3.0+
- CUDA Toolkit 10.0+
- 2GB GPU memory
- gcc 5.0+ or MSVC 2017+

### Recommended
- NVIDIA GPU with compute capability 7.0+ (Volta or newer)
- CUDA Toolkit 11.0+
- 4GB+ GPU memory
- gcc 7.0+ or MSVC 2019+

## Advantages of Local Development

- No session time limits
- Full control over GPU
- Better performance (no network latency)
- Access to all GPU features
- Use professional profiling tools (Nsight Compute, Nsight Systems)
- Multi-GPU programming
- Integration with local development workflow

## Getting Started

### 1. Verify Installation
```bash
# Check CUDA compiler
nvcc --version

# Check GPU
nvidia-smi

# Check compute capability
cd projects/02-device-query
make && ./device_query
```

### 2. Build and Run Hello World
```bash
cd projects/01-hello-world
make
./hello_world
```

### 3. Follow the Curriculum
Work through projects sequentially, using the curriculum as a guide.

## Profiling and Debugging

### Nsight Compute (Kernel Profiler)
```bash
ncu ./your_program
ncu --set full ./your_program
```

### Nsight Systems (System Profiler)
```bash
nsys profile ./your_program
nsys profile --stats=true ./your_program
```

### CUDA Memcheck
```bash
cuda-memcheck ./your_program
compute-sanitizer ./your_program
```

### CUDA GDB
```bash
cuda-gdb ./your_program
```

## Common Issues

See `docs/TROUBLESHOOTING.md` for solutions to common problems.

## Next Steps

Once your GPU is set up:
1. Complete Phase 1-2 projects (foundations)
2. Use profiling tools extensively in Phase 7
3. Experiment with multi-GPU in Phase 6
4. Try advanced features in Phase 9

## Ready to Build?

Head to `projects/01-hello-world/` and start coding!
