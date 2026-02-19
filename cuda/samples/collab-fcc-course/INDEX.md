# FreeCodeCamp CUDA Course - Colab Notebooks

This directory contains Google Colab-compatible notebooks converted from the
[FreeCodeCamp CUDA Course](https://www.youtube.com/watch?v=86FAWCzIe_4).

## How to Use

1. Upload notebooks to Google Colab or open them directly from this repository
2. Enable GPU: Runtime → Change runtime type → T4 GPU
3. Run the setup cells in each notebook
4. Follow along with the video course

---

## Module 5: Writing Your First Kernels

1. [01 Cuda Basics 01 Idxing](module5/01_CUDA_Basics_01_idxing.ipynb)
2. [02 Kernels 00 Vector Add V1](module5/02_Kernels_00_vector_add_v1.ipynb)
3. [02 Kernels 01 Vector Add V2](module5/02_Kernels_01_vector_add_v2.ipynb)
4. [02 Kernels 02 Matmul](module5/02_Kernels_02_matmul.ipynb)
5. [03 Profiling 00 Nvtx Matmul](module5/03_Profiling_00_nvtx_matmul.ipynb)
6. [03 Profiling 01 Naive Matmul](module5/03_Profiling_01_naive_matmul.ipynb)
7. [03 Profiling 02 Tiled Matmul](module5/03_Profiling_02_tiled_matmul.ipynb)
8. [04 Atomics 00 Atomicadd](module5/04_Atomics_00_atomicAdd.ipynb)
9. [05 Streams 01 Stream Basics](module5/05_Streams_01_stream_basics.ipynb)
10. [05 Streams 02 Stream Advanced](module5/05_Streams_02_stream_advanced.ipynb)

## Module 6: CUDA APIs (cuBLAS, cuDNN)

1. [01 Cublas 01 Cublas 01 Hgemm Sgemm](module6/01_CUBLAS_01_cuBLAS_01_Hgemm_Sgemm.ipynb)
2. [01 Cublas 02 Cublaslt 01 Ltmatmul](module6/01_CUBLAS_02_cuBLASLt_01_LtMatmul.ipynb)
3. [01 Cublas 02 Cublaslt 02 Compare](module6/01_CUBLAS_02_cuBLASLt_02_compare.ipynb)
4. [01 Cublas 03 Cublasxt 01 Demo](module6/01_CUBLAS_03_cuBLASXt_01_demo.ipynb)
5. [01 Cublas 03 Cublasxt 02 Compare](module6/01_CUBLAS_03_cuBLASXt_02_compare.ipynb)
6. [02 Cudnn 00 Tanh](module6/02_CUDNN_00_Tanh.ipynb)
7. [02 Cudnn 01 Conv2D Nchw](module6/02_CUDNN_01_Conv2d_NCHW.ipynb)
8. [02 Cudnn 02 Compare Conv](module6/02_CUDNN_02_compare_conv.ipynb)
9. [Optional Cutlass Compare](module6/optional_CUTLASS_compare.ipynb)

## Module 7: Optimizing Matrix Multiplication

1. [Unrolling Example](module7/unrolling_example.ipynb)

## Module 8: Triton

1. [Triton Vector Addition](module8/01_triton_vector_add.ipynb) - Introduction to Triton with benchmarking
2. [CUDA Softmax Comparison](module8/02_softmax.ipynb) - CUDA softmax implementation
3. [Triton Softmax](module8/03_triton_softmax.ipynb) - Softmax in Triton with numerical stability

## Module 9: PyTorch CUDA Extensions

1. [CUDA Extension (Generated)](module9/polynomial_cuda.ipynb) - Basic polynomial activation
2. [PyTorch Extension Demo](module9/pytorch_extension_demo.ipynb) - Complete PyTorch integration guide

---

## Course Resources

- **YouTube Course**: [CUDA Programming on FreeCodeCamp](https://www.youtube.com/watch?v=86FAWCzIe_4)
- **Original Repository**: [Infatoshi/cuda-course](https://github.com/Infatoshi/cuda-course)
- **Your Fork**: Use your forked repository to track changes

---

## Tips for Learning

1. **Run Every Cell**: Don't just read - execute and experiment
2. **Try Exercises**: The exercises reinforce your understanding
3. **Modify Code**: Change parameters and observe the effects
4. **Take Notes**: Use the Notes section at the end of each notebook
5. **Ask Questions**: Use the course Discord or forums if you get stuck

Happy GPU Programming! 🚀
