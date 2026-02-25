# Phase 8: Real Applications (Weeks 14-15)

## Overview
Phase 8 brings everything together. You'll build complete real-world applications using CUDA libraries and all techniques learned. These projects demonstrate how CUDA accelerates actual problems in science, engineering, graphics, and finance.

## Notebooks in This Phase

### 41_cufft_signal_processing.ipynb
**Duration**: 2 hours
**Learning Objectives**:
- Use cuFFT library for FFT computations
- Implement signal processing algorithms
- Understand frequency domain operations
- Compare CPU vs GPU FFT performance

**Key Concepts**:
- Fast Fourier Transform (FFT)
- cuFFT plan creation and execution
- 1D, 2D, and 3D FFTs
- Real-to-complex and complex-to-complex transforms
- Batched FFTs

**Applications**:
- Audio processing
- Image filtering in frequency domain
- Spectral analysis

**Performance**: 10-50x faster than CPU FFT

---

### 42_cusparse_sparse_matrices.ipynb
**Duration**: 2 hours
**Learning Objectives**:
- Use cuSPARSE for sparse matrix operations
- Implement sparse matrix-vector multiplication (SpMV)
- Understand sparse matrix formats
- Apply to large-scale linear systems

**Key Concepts**:
- Sparse matrix formats: COO, CSR, CSC
- cuSPARSE API
- Sparse matrix-vector multiply (SpMV)
- Sparse matrix-matrix multiply (SpMM)
- Iterative solvers

**Applications**:
- Graph algorithms
- Finite element analysis
- Scientific computing
- Machine learning (sparse features)

---

### 43_curand_monte_carlo.ipynb
**Duration**: 2 hours
**Learning Objectives**:
- Use cuRAND for random number generation
- Implement Monte Carlo simulations
- Understand PRNG algorithms
- Parallel random number generation challenges

**Key Concepts**:
- cuRAND generator types (XORWOW, MRG32k3a)
- Host vs device API
- Random number distributions
- Quasi-random sequences (Sobol)

**Applications**:
- Monte Carlo integration
- Risk analysis
- Physics simulations
- Sampling algorithms

**Performance**: Generate 1-10 billion random numbers/second

---

### 44_image_processing.ipynb
**Duration**: 2.5 hours
**Learning Objectives**:
- Implement image processing kernels
- Apply convolution, blur, edge detection
- Use texture memory for images
- Optimize for 2D access patterns

**Key Concepts**:
- 2D convolution with shared memory
- Gaussian blur
- Sobel edge detection
- Image filters (sharpen, emboss)
- Color space conversions

**Implementations**:
- Gaussian blur filter
- Sobel edge detector
- Histogram equalization
- Morphological operations

**Performance**: Process 4K images at 60+ FPS

---

### 45_ray_tracer.ipynb ⭐ KEY NOTEBOOK
**Duration**: 3-4 hours
**Learning Objectives**:
- Build a complete ray tracing renderer
- Implement ray-sphere intersection
- Add lighting, shadows, reflections
- Optimize rendering performance

**Key Concepts**:
- Ray tracing algorithm
- Ray-object intersection tests
- Phong lighting model
- Recursive reflections
- BVH acceleration structures

**Implementations**:
- Basic ray tracer with spheres
- Lighting (ambient, diffuse, specular)
- Shadows and reflections
- Anti-aliasing

**Performance**: Render 1080p images in milliseconds

---

### 46_nbody_simulation.ipynb ⭐ KEY NOTEBOOK
**Duration**: 3 hours
**Learning Objectives**:
- Simulate N-body gravitational systems
- Implement O(n²) all-pairs interactions
- Optimize with shared memory tiling
- Visualize particle dynamics

**Key Concepts**:
- N-body force calculation
- Gravitational physics
- Shared memory tiling for O(n²) algorithms
- Time integration (leapfrog, RK4)
- Visualization of results

**Implementations**:
- Naive O(n²) implementation
- Tiled implementation with shared memory
- Energy conservation checks
- Galaxy collision simulation

**Performance**: Simulate 10K-100K particles in real-time

---

### 47_neural_network_forward_pass.ipynb
**Duration**: 2.5 hours
**Learning Objectives**:
- Implement neural network inference
- Compute matrix operations for layers
- Implement activation functions
- Compare with cuDNN performance

**Key Concepts**:
- Dense (fully-connected) layers
- Activation functions (ReLU, sigmoid, tanh)
- Batch matrix multiplication
- Network inference pipeline

**Implementations**:
- Forward pass for multilayer perceptron
- Batch inference
- Custom activation kernels
- Performance optimization

---

### 48_molecular_dynamics.ipynb
**Duration**: 2.5 hours
**Learning Objectives**:
- Simulate molecular dynamics
- Implement Lennard-Jones potential
- Apply periodic boundary conditions
- Calculate thermodynamic properties

**Key Concepts**:
- Lennard-Jones potential
- Verlet integration
- Periodic boundary conditions
- Cutoff distances
- Neighbor lists

**Applications**:
- Protein folding
- Material science
- Chemistry simulations

---

### 49_option_pricing_monte_carlo.ipynb
**Duration**: 2 hours
**Learning Objectives**:
- Price financial options with Monte Carlo
- Implement Black-Scholes model
- Generate price paths
- Calculate option Greeks

**Key Concepts**:
- Black-Scholes model
- Geometric Brownian motion
- Monte Carlo option pricing
- Variance reduction techniques
- European vs American options

**Applications**:
- Financial derivatives pricing
- Risk management
- Portfolio optimization

**Performance**: Price millions of options in seconds

---

## Learning Path

```
41-cuFFT          42-cuSPARSE      43-cuRAND
        ↓               ↓               ↓
44-image-processing  45-ray-tracer ⭐  46-nbody ⭐
        ↓               ↓               ↓
47-neural-network  48-molecular-dynamics  49-option-pricing
```

**Suggested order**: Libraries first (41-43), then applications (44-49)

## Prerequisites
- Completed Phases 1-7
- Comfortable with all CUDA techniques
- Domain knowledge helpful but not required
- Strong programming skills

## Success Criteria

By the end of Phase 8, you should be able to:
- [ ] Use CUDA libraries (cuFFT, cuSPARSE, cuRAND) effectively
- [ ] Build complete applications from scratch
- [ ] Apply appropriate optimization techniques
- [ ] Compare GPU vs CPU performance
- [ ] Debug complex applications
- [ ] Choose the right tools for the problem
- [ ] Implement real-world algorithms
- [ ] Deliver production-quality code

## Performance Expectations

| Application | Speedup vs CPU | Performance |
|-------------|----------------|-------------|
| **FFT (cuFFT)** | 10-50x | TFLOPs |
| **SpMV (cuSPARSE)** | 5-20x | 100+ GB/s |
| **Monte Carlo** | 50-100x | Billions of samples/sec |
| **Image Processing** | 20-50x | 4K @ 60 FPS |
| **Ray Tracing** | 30-100x | 1080p in ms |
| **N-body** | 50-200x | 100K particles real-time |
| **Neural Network** | 10-30x | 1000+ inferences/sec |
| **Molecular Dynamics** | 20-100x | μs per timestep |

## CUDA Libraries Overview

### cuFFT - Fast Fourier Transform
```cuda
cufftHandle plan;
cufftPlan1d(&plan, N, CUFFT_C2C, 1);
cufftExecC2C(plan, d_data, d_data, CUFFT_FORWARD);
cufftDestroy(plan);
```

**Use cases**: Signal processing, image filtering, PDE solvers

### cuSPARSE - Sparse Linear Algebra
```cuda
cusparseHandle_t handle;
cusparseCreate(&handle);
cusparseSpMV(handle, CUSPARSE_OPERATION_NON_TRANSPOSE,
             &alpha, matA, vecX, &beta, vecY,
             CUDA_R_32F, CUSPARSE_SPMV_ALG_DEFAULT, buffer);
cusparseDestroy(handle);
```

**Use cases**: Graphs, FEM, iterative solvers, machine learning

### cuRAND - Random Number Generation
```cuda
curandGenerator_t gen;
curandCreateGenerator(&gen, CURAND_RNG_PSEUDO_DEFAULT);
curandSetPseudoRandomGeneratorSeed(gen, 1234ULL);
curandGenerateUniform(gen, d_data, N);
curandDestroyGenerator(gen);
```

**Use cases**: Monte Carlo, sampling, stochastic algorithms

## Application Patterns

### 1. Iterative Solver Pattern
```
Initialize data
while (!converged) {
    Compute step
    Update state
    Check convergence
}
Output results
```

### 2. Monte Carlo Pattern
```
Generate random samples (cuRAND)
For each sample:
    Simulate one path
    Compute payoff
Aggregate results
Calculate statistics
```

### 3. Image Processing Pattern
```
Load image to texture memory
For each pixel:
    Sample neighborhood
    Apply filter
    Write result
Post-process and save
```

## Common Pitfalls

1. **Library Initialization Overhead**
   ```cuda
   // Initialize once, use many times
   // Don't create/destroy handles in hot path
   ```

2. **Data Layout Mismatch**
   ```cuda
   // cuBLAS uses column-major (Fortran)
   // cuFFT has specific layout requirements
   // Always check library documentation
   ```

3. **Precision Issues**
   ```cuda
   // Use float for performance, double for accuracy
   // Validate numerical results
   // Be aware of accumulation errors
   ```

4. **Memory Management**
   ```cuda
   // Large applications need careful memory planning
   // Use unified memory or explicit transfers wisely
   ```

## Integration Tips

### Using Multiple Libraries
```cuda
// Libraries can share the same data
cublasHandle_t cublas_handle;
cusparseHandle_t cusparse_handle;
cublasCreate(&cublas_handle);
cusparseCreate(&cusparse_handle);

// Use on same device memory
cublasSgemv(cublas_handle, ...); // Dense ops
cusparseSpMV(cusparse_handle, ...); // Sparse ops
```

### Pipeline Applications
```cuda
// Stage 1: Preprocessing (cuRAND, image ops)
// Stage 2: Main computation (cuBLAS, cuFFT)
// Stage 3: Post-processing (reduction, analysis)
```

## Time Estimate
- **Fast pace**: 2 weeks (4-5 hours/day)
- **Moderate pace**: 3 weeks (2-3 hours/day)
- **Relaxed pace**: 4 weeks (1-2 hours/day)

## Additional Resources

### CUDA Libraries
- [cuFFT Documentation](https://docs.nvidia.com/cuda/cufft/)
- [cuSPARSE Documentation](https://docs.nvidia.com/cuda/cusparse/)
- [cuRAND Documentation](https://docs.nvidia.com/cuda/curand/)
- [cuBLAS Documentation](https://docs.nvidia.com/cuda/cublas/)

### Domain-Specific
- Ray Tracing: "Ray Tracing in One Weekend" (Peter Shirley)
- N-body: "The Art of Molecular Dynamics Simulation" (Rapaport)
- Neural Networks: "Deep Learning" (Goodfellow et al.)
- Finance: "Monte Carlo Methods in Financial Engineering" (Glasserman)

### Case Studies
- [GPU Applications Catalog](https://www.nvidia.com/en-us/data-center/gpu-accelerated-applications/)

## Practice Projects

1. **3D Volume Rendering**: Ray casting through volumetric data
2. **Fluid Simulation**: Navier-Stokes solver with CUDA
3. **Genetic Algorithm**: Population-based optimization
4. **Protein Folding**: Molecular dynamics simulation
5. **Real-time Video Processing**: Multi-stage pipeline

## Production Deployment Tips

### 1. Error Handling
```cuda
// Robust error checking for all library calls
#define CUBLAS_CHECK(call) /* check status */
#define CUFFT_CHECK(call) /* check status */
```

### 2. Performance Monitoring
```cuda
// Add timing and profiling
// Log performance metrics
// Set performance baselines
```

### 3. Testing
```cuda
// Validate against CPU reference
// Test edge cases
// Stress test with large inputs
```

## Next Phase

Once comfortable with Phase 8, move to:
**Phase 9: Modern CUDA** - Learn cutting-edge features including dynamic parallelism, CUDA graphs, tensor cores, and mixed precision.

**Path**: `../phase9/README.md`

---

**Pro Tip**: These applications demonstrate the power of GPU computing. Pick ones aligned with your interests and dive deep!

## Questions to Test Your Understanding

1. When should you use cuFFT vs custom FFT kernels?
2. What sparse matrix format is best for SpMV?
3. How do you ensure quality random numbers on GPU?
4. What optimization techniques are critical for image processing?
5. How does ray tracing benefit from GPU parallelism?
6. What's the computational complexity of N-body simulation?
7. How do you implement neural network layers efficiently?
8. What are the challenges in molecular dynamics simulation?

If you can build and optimize complete applications using CUDA libraries, you're ready for Phase 9!
