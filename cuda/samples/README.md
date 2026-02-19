# CUDA Learning Repository

This repository contains a comprehensive curriculum and sample projects for learning CUDA programming from beginner to advanced level.

## Repository Structure

```
cuda-samples/
├── colab/              # Google Colab-based learning (no local GPU needed)
│   ├── notebooks/      # Jupyter notebooks for Colab
│   ├── projects/       # Sample projects optimized for Colab
│   └── docs/          # Colab-specific documentation
│
├── local/              # Local GPU development (requires NVIDIA GPU)
│   ├── projects/       # Sample projects for local execution
│   ├── common/         # Shared utilities and helpers
│   └── docs/          # Local setup documentation
│
└── README.md          # This file
```

## Getting Started

### Option 1: Learning with Google Colab (No GPU Required)
If you don't have a local NVIDIA GPU, start here:
1. Read `colab/SETUP_WITHOUT_LOCAL_GPU.md` for setup instructions
2. Follow `colab/CUDA_LEARNING_CURRICULUM.md` for the learning path
3. Use the Colab notebooks in `colab/notebooks/`

### Option 2: Learning with Local GPU
If you have an NVIDIA GPU:
1. Check `local/docs/SETUP.md` for installation instructions
2. Follow the curriculum adapted for local development
3. Build and run projects in `local/projects/`

## Curriculum Overview

The curriculum is divided into 9 phases covering 16+ weeks:
- **Phase 1-2**: Foundations & Memory Management
- **Phase 3-4**: Optimization & Synchronization
- **Phase 5**: Advanced Algorithms
- **Phase 6**: Streams & Multi-GPU
- **Phase 7**: Performance Engineering
- **Phase 8**: Real-World Applications
- **Phase 9**: Modern CUDA Features

## Prerequisites

- Strong C/C++ programming skills
- Basic understanding of computer architecture
- Familiarity with pointers, memory management, and parallel concepts (helpful but not required)

## Resources

- [CUDA Programming Guide](https://docs.nvidia.com/cuda/cuda-c-programming-guide/)
- [CUDA Best Practices](https://docs.nvidia.com/cuda/cuda-c-best-practices-guide/)
- [NVIDIA Developer Forums](https://forums.developer.nvidia.com/)

## Contributing

This is a personal learning repository. Feel free to fork and adapt for your own learning journey.

## License

Educational use only.
