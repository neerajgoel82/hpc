# Google Colab Tips and Best Practices

## Essential Tips for CUDA Learning on Colab

### 1. Enabling GPU Runtime

**Every time you start a new notebook:**
- Click `Runtime` → `Change runtime type`
- Select `T4 GPU` or `V100` from Hardware accelerator dropdown
- Click `Save`

**Verify GPU is enabled:**
```python
!nvidia-smi
```

### 2. Session Management

**Session Limits:**
- Free tier: 12-hour maximum session
- Idle timeout: 90 minutes of inactivity
- GPU availability not guaranteed during peak hours

**Best Practices:**
- Save work frequently
- Download important results before 12-hour limit
- Use Google Drive for persistence

### 3. Saving Your Work

**Option A: Save to Google Drive**
```python
# Mount Google Drive
from google.colab import drive
drive.mount('/content/drive')

# Save files
!cp myfile.cu /content/drive/MyDrive/cuda-learning/
```

**Option B: Download Locally**
```python
from google.colab import files
files.download('output.txt')
```

**Option C: Git Integration**
```python
# Configure git (first time only)
!git config --global user.email "your-email@example.com"
!git config --global user.name "Your Name"

# Clone your repo
!git clone https://github.com/yourusername/cuda-samples.git

# Make changes, commit, push
!cd cuda-samples && git add . && git commit -m "Update" && git push
```

### 4. Writing CUDA Code in Colab

**Method 1: Using %%cu magic (recommended for learning)**
```python
# Install extension (once per session)
!pip install git+https://github.com/andreinechaev/nvcc4jupyter.git
%load_ext nvcc4jupyter

# Write CUDA code directly
%%cu
#include <stdio.h>

__global__ void kernel() {
    printf("Hello from GPU!\n");
}

int main() {
    kernel<<<1, 10>>>();
    cudaDeviceSynchronize();
    return 0;
}
```

**Method 2: Write to file and compile**
```python
%%writefile program.cu
#include <stdio.h>
// ... your CUDA code ...

# Compile
!nvcc program.cu -o program

# Run
!./program
```

**Method 3: Multi-file projects**
```python
# Create directory structure
!mkdir -p myproject
%%writefile myproject/kernel.cu
// kernel code

%%writefile myproject/main.cu
// main code

# Compile multiple files
!nvcc myproject/kernel.cu myproject/main.cu -o myproject/app
!./myproject/app
```

### 5. Timing and Performance Measurement

**Using CUDA Events:**
```cuda
cudaEvent_t start, stop;
cudaEventCreate(&start);
cudaEventCreate(&stop);

cudaEventRecord(start);
// ... kernel execution ...
cudaEventRecord(stop);

cudaEventSynchronize(stop);
float milliseconds = 0;
cudaEventElapsedTime(&milliseconds, start, stop);
printf("Time: %.2f ms\n", milliseconds);
```

**Using Python time module:**
```python
import time
start = time.time()
# ... run CUDA program ...
end = time.time()
print(f"Time: {end - start:.4f} seconds")
```

### 6. Installing Additional Libraries

```python
# cuBLAS, cuFFT, etc. are already included with CUDA toolkit
# Install Thrust (usually pre-installed)
# Install CUB (usually pre-installed)

# Install additional Python packages
!pip install matplotlib numpy pandas

# Check CUDA libraries
!ls /usr/local/cuda/lib64/
```

### 7. Profiling on Colab

**Using nvprof (deprecated but still useful):**
```bash
!nvprof ./program
```

**Using Nsight Compute:**
```bash
!ncu --set full ./program
!ncu --target-processes all ./program
```

**Using Nsight Systems:**
```bash
!nsys profile --stats=true ./program
```

### 8. Debugging

**Check for CUDA errors:**
```cuda
#define CHECK_CUDA_ERROR(call) { \
    cudaError_t err = call; \
    if (err != cudaSuccess) { \
        printf("CUDA error: %s:%d, %s\n", __FILE__, __LINE__, \
               cudaGetErrorString(err)); \
        exit(1); \
    } \
}

// Use it
CHECK_CUDA_ERROR(cudaMalloc(&d_ptr, size));
```

**Using cuda-memcheck (compute-sanitizer):**
```bash
!compute-sanitizer ./program
```

### 9. Working with Data Files

**Upload files:**
```python
from google.colab import files
uploaded = files.upload()
```

**Download from URL:**
```python
!wget https://example.com/data.txt
```

**Use Google Drive:**
```python
from google.colab import drive
drive.mount('/content/drive')
# Access files at /content/drive/MyDrive/
```

### 10. Optimizing Compilation

**Specify compute capability:**
```bash
# For T4 GPU (compute capability 7.5)
!nvcc -arch=sm_75 program.cu -o program

# For multiple architectures
!nvcc -gencode arch=compute_75,code=sm_75 program.cu -o program
```

**Enable optimizations:**
```bash
!nvcc -O3 program.cu -o program
```

**Fast math:**
```bash
!nvcc --use_fast_math program.cu -o program
```

### 11. Handling Long-Running Tasks

**Prevent idle disconnection:**
```javascript
// Run this in browser console (F12)
function KeepAlive(){
    console.log("Keeping alive");
    document.querySelector("colab-connect-button").click();
}
setInterval(KeepAlive, 60000); // Every 60 seconds
```

**Break work into chunks:**
- Complete one phase per session
- Save intermediate results
- Use checkpoints for long computations

### 12. GPU Availability Issues

**Check GPU type:**
```python
!nvidia-smi --query-gpu=name --format=csv,noheader
```

**If no GPU available:**
- Try different times (off-peak hours)
- Consider Colab Pro ($10/month) for priority access
- Use Kaggle as alternative (30 hrs/week free)

### 13. Best Practices for Learning

1. **One concept per notebook**: Keep notebooks focused
2. **Add markdown explanations**: Document your learning
3. **Include expected output**: Comment what results should be
4. **Copy before experimenting**: Duplicate notebooks to try variations
5. **Use version control**: Commit working code to git

### 14. Common Pitfalls

**Forgetting to sync:**
```cuda
// Always sync after kernel launch when checking results
kernel<<<grid, block>>>();
cudaDeviceSynchronize();  // Don't forget this!
```

**Not checking errors:**
```cuda
// After every CUDA call
cudaError_t err = cudaMalloc(&d_ptr, size);
if (err != cudaSuccess) {
    printf("Error: %s\n", cudaGetErrorString(err));
}
```

**Memory leaks:**
```cuda
// Always free allocated memory
cudaFree(d_ptr);
```

### 15. Useful Colab Shortcuts

- `Ctrl/Cmd + S`: Save notebook
- `Ctrl/Cmd + Enter`: Run current cell
- `Shift + Enter`: Run cell and move to next
- `Ctrl/Cmd + M B`: Insert cell below
- `Ctrl/Cmd + M D`: Delete cell

### 16. Recommended Notebook Structure

```python
# === CELL 1: Setup ===
!pip install git+https://github.com/andreinechaev/nvcc4jupyter.git
%load_ext nvcc4jupyter
!nvidia-smi

# === CELL 2: Helper Functions ===
# Define reusable code

# === CELL 3: Main Code ===
%%cu
// Your CUDA kernel

# === CELL 4: Experiments ===
# Try variations

# === CELL 5: Verification ===
# Check results

# === CELL 6: Performance ===
# Benchmarks and profiling

# === CELL 7: Cleanup ===
# Clear variables if needed
```

## Summary

- Always verify GPU is enabled
- Save work frequently (Drive or git)
- Use %%cu magic for quick prototyping
- Profile your code
- Check for CUDA errors
- Break long tasks into manageable sessions

Happy learning!
