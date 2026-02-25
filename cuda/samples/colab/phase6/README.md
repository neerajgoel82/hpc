# Phase 6: Streams & Concurrency (Week 11)

## Overview
Phase 6 teaches asynchronous CUDA programming. You'll learn to overlap computation with memory transfers, use multiple streams for parallelism, and program multi-GPU systems for maximum throughput.

## Notebooks in This Phase

### 30_cuda_streams_basics.ipynb
**Duration**: 2 hours
**Learning Objectives**:
- Create and use CUDA streams
- Understand asynchronous execution
- Overlap kernel execution with transfers
- Manage stream synchronization

**Key Concepts**:
- Default stream (stream 0)
- Non-blocking streams
- Stream synchronization
- `cudaStreamSynchronize()` vs `cudaDeviceSynchronize()`
- Stream callbacks

**Implementations**:
- Sequential execution (no streams)
- Concurrent execution with streams
- Overlapping compute and transfers

---

### 31_async_data_pipeline.ipynb ⭐ KEY NOTEBOOK
**Duration**: 3 hours
**Learning Objectives**:
- Build asynchronous data pipeline
- Hide transfer latency with computation
- Use pinned memory for async transfers
- Achieve maximum throughput

**Key Concepts**:
- Pinned memory requirement for async
- Pipeline pattern: Load → Compute → Store
- Multiple buffers for pipelining
- Effective bandwidth calculation
- PCIe bandwidth utilization

**Implementations**:
- Synchronous pipeline (baseline)
- 2-stage async pipeline
- 3+ stage deep pipeline
- Performance analysis

**Performance**: 2-3x speedup with pipelining

---

### 32_cuda_events_and_timing.ipynb
**Duration**: 1.5 hours
**Learning Objectives**:
- Use CUDA events for precise timing
- Measure kernel execution time accurately
- Synchronize streams with events
- Profile async operations

**Key Concepts**:
- `cudaEvent_t` creation and recording
- Event synchronization
- Event-based timing (microsecond precision)
- Inter-stream dependencies
- Event overhead

**Usage**:
```cuda
cudaEvent_t start, stop;
cudaEventCreate(&start);
cudaEventCreate(&stop);
cudaEventRecord(start, stream);
kernel<<<grid, block, 0, stream>>>();
cudaEventRecord(stop, stream);
cudaEventSynchronize(stop);
float ms;
cudaEventElapsedTime(&ms, start, stop);
```

---

### 33_multi_gpu_basics.ipynb
**Duration**: 2 hours
**Learning Objectives**:
- Program multiple GPUs
- Select and manage different devices
- Partition work across GPUs
- Understand multi-GPU architecture

**Key Concepts**:
- `cudaSetDevice()` for device selection
- Per-device contexts
- Device-to-device transfers
- Work partitioning strategies
- Load balancing

**Implementations**:
- Single GPU baseline
- Multi-GPU data parallel execution
- Result aggregation

---

### 34_peer_to_peer_transfers.ipynb
**Duration**: 2 hours
**Learning Objectives**:
- Enable P2P (peer-to-peer) transfers
- Transfer data directly between GPUs
- Understand NVLink vs PCIe topology
- Optimize multi-GPU communication

**Key Concepts**:
- `cudaDeviceEnablePeerAccess()`
- Direct GPU-to-GPU transfers
- NVLink high-speed interconnect
- Topology awareness
- Bandwidth measurements

**Performance**:
- PCIe P2P: 10-12 GB/s
- NVLink P2P: 50-300 GB/s (depending on generation)

---

### 35_nccl_collective_operations.ipynb
**Duration**: 2 hours
**Learning Objectives**:
- Use NCCL for multi-GPU communication
- Implement collective operations
- Scale to many GPUs efficiently
- Understand communication patterns

**Key Concepts**:
- NCCL library (NVIDIA Collective Communications Library)
- Collectives: AllReduce, Broadcast, Reduce, AllGather
- In-place vs out-of-place operations
- Communication rings and trees
- Integration with training frameworks

**Implementations**:
- AllReduce for gradient aggregation
- Broadcast for weight distribution
- Multi-GPU data parallelism

---

## Learning Path

```
30-streams-basics
        ↓
31-async-pipeline ⭐
        ↓
32-events-timing
        ↓
33-multi-gpu-basics
        ↓
34-p2p-transfers
        ↓
35-nccl-collectives
```

## Prerequisites
- Completed Phases 1-5
- Understanding of asynchronous programming concepts
- Familiarity with pipeline patterns
- Access to multi-GPU system (for notebooks 33-35)

## Success Criteria

By the end of Phase 6, you should be able to:
- [ ] Use CUDA streams for asynchronous execution
- [ ] Build efficient data pipelines
- [ ] Measure performance with CUDA events
- [ ] Program multiple GPUs
- [ ] Enable and use P2P transfers
- [ ] Use NCCL for multi-GPU communication
- [ ] Overlap computation and data transfers
- [ ] Optimize multi-GPU applications

## Asynchronous Execution Model

### Stream Execution Rules:
1. Operations in same stream execute in order
2. Operations in different streams may execute concurrently
3. Default stream (stream 0) synchronizes with all other streams
4. Kernel launches are asynchronous
5. Memory copies can be async if using pinned memory

### Pipeline Pattern:
```
Time →
Stream 1: [Transfer1] [Compute1] [Transfer1_back]
Stream 2:             [Transfer2] [Compute2] [Transfer2_back]
Stream 3:                         [Transfer3] [Compute3] [Transfer3_back]

Result: 3x overlap = 3x speedup (ideally)
```

## Performance Expectations

### With Streams & Pipelining:
- **2-3x speedup** for compute-bound kernels with transfers
- **Near-zero transfer overhead** when fully overlapped
- **Effective bandwidth**: 90-100% of theoretical

### Multi-GPU:
- **Linear scaling** (ideally N×speedup for N GPUs)
- **Actual**: 80-95% scaling due to communication overhead
- **NVLink**: Better scaling than PCIe-only

## Common Pitfalls

1. **Pageable Memory for Async Copies**
   ```cuda
   // BAD - async copy with pageable memory is synchronous!
   float* h_data = (float*)malloc(size);
   cudaMemcpyAsync(d_data, h_data, size, H2D, stream);  // Actually sync!
   
   // GOOD - use pinned memory
   float* h_data;
   cudaHostAlloc(&h_data, size, cudaHostAllocDefault);
   cudaMemcpyAsync(d_data, h_data, size, H2D, stream);  // Truly async
   ```

2. **Default Stream Blocking**
   ```cuda
   // Default stream (NULL) synchronizes with all streams!
   // Use explicit streams for concurrency
   cudaStream_t stream1, stream2;
   cudaStreamCreate(&stream1);
   cudaStreamCreate(&stream2);
   ```

3. **Insufficient Overlap**
   ```cuda
   // Need enough work to hide transfer latency
   // If kernel is too fast, no benefit from pipelining
   ```

4. **P2P Not Enabled**
   ```cuda
   // Must explicitly enable P2P
   int canAccessPeer;
   cudaDeviceCanAccessPeer(&canAccessPeer, dev1, dev2);
   if (canAccessPeer) {
       cudaSetDevice(dev1);
       cudaDeviceEnablePeerAccess(dev2, 0);
   }
   ```

## Best Practices

### 1. Stream Management
```cuda
// Create streams at initialization
cudaStream_t streams[NUM_STREAMS];
for (int i = 0; i < NUM_STREAMS; i++) {
    cudaStreamCreate(&streams[i]);
}

// Use throughout application
kernel<<<grid, block, 0, streams[i]>>>();

// Destroy at cleanup
for (int i = 0; i < NUM_STREAMS; i++) {
    cudaStreamDestroy(streams[i]);
}
```

### 2. Pipeline Pattern
```cuda
for (int i = 0; i < N; i++) {
    int s = i % NUM_STREAMS;
    cudaMemcpyAsync(d_in[s], h_in[i], size, H2D, streams[s]);
    kernel<<<grid, block, 0, streams[s]>>>(d_in[s], d_out[s]);
    cudaMemcpyAsync(h_out[i], d_out[s], size, D2H, streams[s]);
}
```

### 3. Multi-GPU Work Distribution
```cuda
int n = cudaGetDeviceCount();
for (int dev = 0; dev < n; dev++) {
    cudaSetDevice(dev);
    int start = (N / n) * dev;
    int end = (dev == n-1) ? N : (N / n) * (dev + 1);
    kernel<<<grid, block>>>(d_data[dev], start, end);
}
```

## Time Estimate
- **Fast pace**: 1 week (3-4 hours/day)
- **Moderate pace**: 1.5 weeks (2 hours/day)
- **Relaxed pace**: 2 weeks (1-1.5 hours/day)

**Note**: Multi-GPU notebooks require multi-GPU hardware

## Additional Resources

### NVIDIA Documentation
- [CUDA C Programming Guide](https://docs.nvidia.com/cuda/cuda-c-programming-guide/) - Chapter 3.2.5 (Streams)
- [NCCL Documentation](https://docs.nvidia.com/deeplearning/nccl/)
- [Multi-GPU Programming Guide](https://docs.nvidia.com/cuda/cuda-c-best-practices-guide/index.html#multi-gpu)

### Papers & Talks
- "Optimizing CUDA Applications for Maxwell" (NVIDIA)
- "NCCL: Fast Multi-GPU Collective Communications" (NVIDIA)

## Practice Exercises

1. **Image processing pipeline**: Load → Process → Save with 3 streams
2. **Multi-GPU GEMM**: Partition matrix across GPUs
3. **Data parallel training**: Implement AllReduce for gradients
4. **Video processing**: Real-time pipeline with streams
5. **Monte Carlo simulation**: Distribute across multiple GPUs

## Debugging Tips

### Check async copy behavior:
```bash
# If using pageable memory, copy will be synchronous
# Use cudaHostAlloc for truly async copies
```

### Profile stream concurrency:
```bash
nvprof --print-gpu-trace program
# Look for overlapping operations
```

### Check P2P support:
```cuda
cudaDeviceCanAccessPeer(&canAccess, dev1, dev2);
printf("P2P support: %d\n", canAccess);
```

### Visualize with Nsight Systems:
```bash
nsys profile -o report ./program
# Open report.qdrep in Nsight Systems GUI
# See stream timeline visualization
```

## Next Phase

Once comfortable with Phase 6, move to:
**Phase 7: Performance Engineering** - Learn profiling tools, debugging techniques, kernel fusion, and advanced optimization strategies.

**Path**: `../phase7/README.md`

---

**Pro Tip**: Asynchronous execution and multi-GPU programming are essential for production applications. Master streams and you'll unlock 2-3x performance gains!

## Questions to Test Your Understanding

1. What's the difference between cudaDeviceSynchronize() and cudaStreamSynchronize()?
2. Why must you use pinned memory for async copies?
3. How do you overlap kernel execution with memory transfers?
4. What is P2P and when is it available?
5. What are NCCL collective operations?
6. How do you partition work across multiple GPUs?
7. What is the default stream and why should you avoid it?
8. How do you measure kernel execution time accurately?

If you can build an efficient async pipeline and program multiple GPUs, you're ready for Phase 7!
