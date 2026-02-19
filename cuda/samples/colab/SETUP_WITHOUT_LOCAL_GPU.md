# Learning CUDA Without a Local GPU

Don't have an NVIDIA GPU? No problem! You can still learn and practice CUDA programming using cloud-based solutions.

---

## Option 1: Google Colab (Free & Easiest)

### Setup Steps

1. **Go to Google Colab**: https://colab.research.google.com
2. **Create a new notebook**
3. **Enable GPU**:
   - Click `Runtime` → `Change runtime type`
   - Select `T4 GPU` or `V100` from Hardware accelerator
   - Click `Save`

### Writing CUDA in Colab

**Method 1: Using %%cu magic (Inline CUDA)**

```python
# Load the CUDA extension
!pip install git+https://github.com/andreinechaev/nvcc4jupyter.git
%load_ext nvcc4jupyter
```

```cuda
%%cu
#include <stdio.h>

__global__ void hello() {
    printf("Hello from GPU thread %d!\n", threadIdx.x);
}

int main() {
    hello<<<1, 10>>>();
    cudaDeviceSynchronize();
    return 0;
}
```

**Method 2: Write and compile .cu files**

```python
# Write CUDA code to file
%%writefile hello.cu
#include <stdio.h>

__global__ void hello() {
    printf("Hello from block %d, thread %d\n", blockIdx.x, threadIdx.x);
}

int main() {
    hello<<<2, 4>>>();
    cudaDeviceSynchronize();
    return 0;
}
```

```python
# Compile and run
!nvcc hello.cu -o hello
!./hello
```

**Method 3: Check GPU availability**

```python
# Verify GPU
!nvidia-smi

# Check CUDA version
!nvcc --version

# Device query
%%writefile device_query.cu
#include <stdio.h>

int main() {
    int deviceCount;
    cudaGetDeviceCount(&deviceCount);

    printf("Number of CUDA devices: %d\n\n", deviceCount);

    for (int i = 0; i < deviceCount; i++) {
        cudaDeviceProp prop;
        cudaGetDeviceProperties(&prop, i);

        printf("Device %d: %s\n", i, prop.name);
        printf("  Compute Capability: %d.%d\n", prop.major, prop.minor);
        printf("  Total Global Memory: %.2f GB\n", prop.totalGlobalMem / 1e9);
        printf("  Multiprocessors: %d\n", prop.multiProcessorCount);
        printf("  Max Threads per Block: %d\n", prop.maxThreadsPerBlock);
        printf("  Max Threads per Multiprocessor: %d\n", prop.maxThreadsPerMultiProcessor);
        printf("  Warp Size: %d\n", prop.warpSize);
        printf("  Memory Clock Rate: %.2f GHz\n", prop.memoryClockRate / 1e6);
        printf("  Memory Bus Width: %d bits\n", prop.memoryBusWidth);
        printf("  L2 Cache Size: %.2f MB\n", prop.l2CacheSize / 1e6);
    }

    return 0;
}

!nvcc device_query.cu -o device_query && ./device_query
```

### Google Colab Template Notebook Structure

```python
# Cell 1: Setup
!pip install git+https://github.com/andreinechaev/nvcc4jupyter.git
%load_ext nvcc4jupyter
!nvidia-smi

# Cell 2: Your CUDA code
%%cu
// Your CUDA kernel here

# Cell 3: Compile and run (alternative method)
%%writefile myprogram.cu
// CUDA code
// ...

!nvcc myprogram.cu -o myprogram
!./myprogram

# Cell 4: Profile (optional)
!nvprof ./myprogram
```

### Limitations of Colab
- 12-hour session limit (disconnects after)
- GPU availability not guaranteed during peak hours
- Limited to T4 GPU (free tier), but sufficient for learning
- Files are deleted when session ends (download important work)
- Cannot install persistent CUDA toolkit modifications

### Tips for Using Colab
- Save notebooks to Google Drive (File → Save a copy in Drive)
- Download compiled programs before session ends
- Use `!cp file /content/drive/MyDrive/` to save to Drive
- Mount Google Drive: `from google.colab import drive; drive.mount('/content/drive')`

---

## Option 2: Paperspace Gradient (Free Tier Available)

### Setup
1. Go to https://www.paperspace.com/gradient
2. Sign up for free tier
3. Create a new notebook
4. Select a GPU machine (Free tier includes limited GPU hours)

### Advantages over Colab
- Better GPU options
- Longer session times
- Persistent storage
- Better for larger projects

### Limitations
- Free tier has limited GPU hours per month
- May require payment for extended use

---

## Option 3: AWS EC2 (Pay-per-use)

### Cost-effective approach:
- **g4dn.xlarge**: ~$0.50/hour (T4 GPU)
- **p3.2xlarge**: ~$3/hour (V100 GPU)

### Setup Steps
1. Create AWS account
2. Launch EC2 instance with Deep Learning AMI
3. Select instance with GPU (g4dn, p3, p2 series)
4. SSH into instance
5. CUDA toolkit pre-installed

### Cost Optimization
- Use spot instances (70% cheaper)
- Stop instance when not in use (only pay for storage)
- Set billing alerts

---

## Option 4: NVIDIA DLI Courses

- Some courses include GPU access
- Structured learning with hands-on labs
- Certificate upon completion
- Usually $90-200 per course

---

## Option 5: Kaggle Notebooks

- Free GPU access (NVIDIA T4)
- 30 hours/week GPU quota
- Similar to Colab but with different limits
- Good for learning and experimentation

### Setup
1. Go to https://www.kaggle.com
2. Create account
3. Create new notebook
4. Enable GPU accelerator in settings

---

## Recommended Workflow for This Curriculum

### Phase 1-2 (Foundations): Google Colab
- Perfect for learning basics
- Write simple kernels
- Experiment with memory operations
- No cost

### Phase 3-5 (Optimization): Colab or Kaggle
- Profile with nvprof
- More complex algorithms
- Still works within free tier limits

### Phase 6-7 (Performance Engineering): Consider Paid Option
- Longer sessions needed for profiling
- More GPU memory for larger problems
- Paperspace or AWS spot instances recommended
- Alternatively: batch your work into 12-hour Colab sessions

### Phase 8-9 (Advanced Topics): Paid Cloud
- Multi-GPU requires cloud infrastructure
- Use AWS/GCP for multi-GPU experiments
- Consider GPU instances with NVLink

---

## Setting Up Development Environment

### Local Development (Write code locally, run in cloud)

**Option A: VS Code with Remote Development**
```bash
# Install VS Code
# Install "Remote - SSH" extension
# Connect to cloud GPU instance
# Edit code with full IDE features
```

**Option B: Local editing + Colab execution**
```bash
# Write .cu files locally
# Upload to Colab
# Compile and run in Colab
# Download results
```

**Option C: Git workflow**
```bash
# Write code locally
# Commit to GitHub
# Clone in Colab/cloud instance
# Run and test
# Commit results back
```

---

## Sample Colab Notebook for Getting Started

Create a notebook with these cells:

```python
# === CELL 1: Setup ===
# Install CUDA extension
!pip install git+https://github.com/andreinechaev/nvcc4jupyter.git
%load_ext nvcc4jupyter

# Check GPU
!nvidia-smi
!nvcc --version
```

```python
# === CELL 2: Hello World ===
%%cu
#include <stdio.h>

__global__ void hello() {
    printf("Hello from block %d, thread %d\n", blockIdx.x, threadIdx.x);
}

int main() {
    printf("Launching kernel...\n");
    hello<<<2, 4>>>();
    cudaDeviceSynchronize();
    printf("Done!\n");
    return 0;
}
```

```python
# === CELL 3: Vector Addition ===
%%writefile vector_add.cu
#include <stdio.h>
#include <stdlib.h>

__global__ void vectorAdd(float *a, float *b, float *c, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        c[idx] = a[idx] + b[idx];
    }
}

int main() {
    int n = 1000000;
    size_t size = n * sizeof(float);

    // Allocate host memory
    float *h_a = (float*)malloc(size);
    float *h_b = (float*)malloc(size);
    float *h_c = (float*)malloc(size);

    // Initialize
    for (int i = 0; i < n; i++) {
        h_a[i] = i;
        h_b[i] = i * 2;
    }

    // Allocate device memory
    float *d_a, *d_b, *d_c;
    cudaMalloc(&d_a, size);
    cudaMalloc(&d_b, size);
    cudaMalloc(&d_c, size);

    // Copy to device
    cudaMemcpy(d_a, h_a, size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, h_b, size, cudaMemcpyHostToDevice);

    // Launch kernel
    int blockSize = 256;
    int gridSize = (n + blockSize - 1) / blockSize;
    vectorAdd<<<gridSize, blockSize>>>(d_a, d_b, d_c, n);

    // Copy result back
    cudaMemcpy(h_c, d_c, size, cudaMemcpyDeviceToHost);

    // Verify
    for (int i = 0; i < 10; i++) {
        printf("c[%d] = %.2f (expected %.2f)\n", i, h_c[i], h_a[i] + h_b[i]);
    }

    // Cleanup
    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_c);
    free(h_a);
    free(h_b);
    free(h_c);

    return 0;
}

!nvcc vector_add.cu -o vector_add
!./vector_add
```

```python
# === CELL 4: Timing Example ===
%%writefile timed_vector_add.cu
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

__global__ void vectorAdd(float *a, float *b, float *c, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        c[idx] = a[idx] + b[idx];
    }
}

void vectorAddCPU(float *a, float *b, float *c, int n) {
    for (int i = 0; i < n; i++) {
        c[i] = a[i] + b[i];
    }
}

int main() {
    int n = 10000000;
    size_t size = n * sizeof(float);

    float *h_a = (float*)malloc(size);
    float *h_b = (float*)malloc(size);
    float *h_c = (float*)malloc(size);

    for (int i = 0; i < n; i++) {
        h_a[i] = i;
        h_b[i] = i * 2;
    }

    // CPU timing
    clock_t start = clock();
    vectorAddCPU(h_a, h_b, h_c, n);
    clock_t end = clock();
    double cpu_time = ((double)(end - start)) / CLOCKS_PER_SEC;

    // GPU timing
    float *d_a, *d_b, *d_c;
    cudaMalloc(&d_a, size);
    cudaMalloc(&d_b, size);
    cudaMalloc(&d_c, size);

    cudaEvent_t gpu_start, gpu_stop;
    cudaEventCreate(&gpu_start);
    cudaEventCreate(&gpu_stop);

    cudaMemcpy(d_a, h_a, size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, h_b, size, cudaMemcpyHostToDevice);

    cudaEventRecord(gpu_start);
    int blockSize = 256;
    int gridSize = (n + blockSize - 1) / blockSize;
    vectorAdd<<<gridSize, blockSize>>>(d_a, d_b, d_c, n);
    cudaEventRecord(gpu_stop);

    cudaEventSynchronize(gpu_stop);
    float gpu_time;
    cudaEventElapsedTime(&gpu_time, gpu_start, gpu_stop);

    printf("CPU Time: %.4f seconds\n", cpu_time);
    printf("GPU Time: %.4f seconds\n", gpu_time / 1000.0);
    printf("Speedup: %.2fx\n", cpu_time / (gpu_time / 1000.0));

    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_c);
    free(h_a);
    free(h_b);
    free(h_c);

    return 0;
}

!nvcc timed_vector_add.cu -o timed_vector_add
!./timed_vector_add
```

---

## Monthly Cost Estimate

### Free Options
- Google Colab: $0 (with limitations)
- Kaggle: $0 (30 hrs/week)
- Total: **$0/month**

### Budget Learning (~$20-50/month)
- Paperspace free tier + occasional paid hours
- AWS spot instances (10-20 hours/month)
- Good for serious learning

### Professional Practice (~$100-200/month)
- Dedicated cloud GPU instances
- Multi-GPU access
- Production-level learning

---

## Getting Started Today

1. **Create Google Colab account** (free, 5 minutes)
2. **Copy the sample notebook structure above**
3. **Enable GPU in runtime settings**
4. **Run the hello world example**
5. **Start with Phase 1 of the curriculum**

You don't need to own a GPU to become proficient in CUDA programming. Many professional CUDA developers use cloud GPUs for development and testing!

---

## Next Steps

Would you like me to:
1. Create a ready-to-use Colab notebook template for Phase 1?
2. Set up a workflow for syncing this repo with Colab?
3. Create scripts to easily upload/download code between local and cloud?
4. Provide more detailed cost comparisons for cloud options?
