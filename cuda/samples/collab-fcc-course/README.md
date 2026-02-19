# FreeCodeCamp CUDA Course - Google Colab Edition

This directory contains Jupyter notebooks adapted for Google Colab from the
[FreeCodeCamp CUDA Programming Course](https://www.youtube.com/watch?v=86FAWCzIe_4).

## Quick Start

1. **Open Google Colab**: Go to [colab.research.google.com](https://colab.research.google.com)
2. **Upload a Notebook**: Upload any `.ipynb` file from this directory
3. **Enable GPU**:
   - Click **Runtime** → **Change runtime type**
   - Select **T4 GPU** as Hardware accelerator
   - Click **Save**
4. **Run the Notebook**: Execute cells sequentially

## Directory Structure

```
collab-fcc-course/
├── module5/     # Writing Your First Kernels
├── module6/     # CUDA APIs (cuBLAS, cuDNN)
├── module7/     # Optimizing Matrix Multiplication
├── module8/     # Triton
├── module9/     # PyTorch CUDA Extensions
├── INDEX.md     # Complete list of all notebooks
└── README.md    # This file
```

## Prerequisites

- Basic Python programming
- Understanding of parallel computing concepts (helpful)
- Google account (for Colab access)

## Course Modules

### Module 5: Writing Your First Kernels
Learn CUDA basics: thread indexing, memory management, kernels, and streams.

### Module 6: CUDA APIs
Leverage optimized libraries like cuBLAS and cuDNN for production performance.

### Module 7: Optimizing Matrix Multiplication
Deep dive into performance optimization techniques and memory hierarchy.

### Module 8: Triton
Explore high-level GPU programming with OpenAI's Triton language.

### Module 9: PyTorch CUDA Extensions
Create custom CUDA operations integrated with PyTorch.

## Learning Path

1. Start with Module 5, Notebook 1 (CUDA Indexing)
2. Work through notebooks sequentially
3. Complete exercises in each notebook
4. Reference the original video course for detailed explanations
5. Experiment with code modifications

## Notes

- **Free GPU Access**: Colab provides free T4 GPU access (with usage limits)
- **Session Limits**: Free tier has 12-hour session limits
- **Save Your Work**: Download modified notebooks to avoid losing changes
- **Runtime Disconnects**: Colab may disconnect after inactivity

## Attribution

- **Original Course**: FreeCodeCamp CUDA Programming Course
- **Course Author**: Check the original repository for author information
- **Notebooks**: Converted for educational use with Google Colab

## Additional Resources

- [CUDA Programming Guide](https://docs.nvidia.com/cuda/cuda-c-programming-guide/)
- [CUDA Best Practices](https://docs.nvidia.com/cuda/cuda-c-best-practices-guide/)
- [Course Discord/Community]: Check the original course for community links

---

**Happy GPU Programming!** 🚀
