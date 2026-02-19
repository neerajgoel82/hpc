# Module 11: Multithreading and Concurrency

## Overview
Learn CPU multithreading with C++11/14/17 threading features. Understanding parallel execution on CPUs prepares you for GPU parallel programming.

## Topics Covered

### Threading Basics
- **std::thread** - Creating threads
- **join()** and **detach()**
- Thread lifetime
- Passing arguments to threads
- Thread IDs
- Hardware concurrency

### Mutexes and Locks
- **std::mutex** - Basic mutual exclusion
- **std::recursive_mutex** - Reentrant mutex
- **std::timed_mutex** - Timeout support
- **std::shared_mutex** - Reader-writer lock (C++17)

### Lock Guards
- **std::lock_guard** - RAII lock
- **std::unique_lock** - Flexible lock
- **std::scoped_lock** - Multiple mutex lock (C++17)
- Avoiding deadlocks
- Lock ordering

### Condition Variables
- **std::condition_variable**
- wait() and notify()
- **notify_one()** vs **notify_all()**
- Spurious wakeups
- Producer-consumer pattern

### Atomics
- **std::atomic** - Atomic operations
- Atomic types (atomic_int, atomic_bool, etc.)
- **fetch_add**, **fetch_sub**, etc.
- Compare-and-swap operations
- Lock-free programming

### Memory Ordering
- **std::memory_order** - Memory consistency
- **memory_order_relaxed**
- **memory_order_acquire** / **memory_order_release**
- **memory_order_seq_cst** (default)
- When to use each

### Futures and Promises
- **std::future** - Asynchronous result
- **std::promise** - Set future value
- **std::async** - Launch async tasks
- **std::packaged_task**
- Launch policies (async, deferred)

### Thread Synchronization
- Barriers
- Latches (C++20)
- Semaphores (C++20)
- Coordination patterns

### Thread Safety
- Race conditions
- Data races
- Deadlocks
- Livelocks
- Thread-safe design patterns

### Thread-Local Storage
- **thread_local** keyword
- Per-thread data
- Use cases

### Common Parallel Patterns
- **Data parallelism** - Same operation on different data
- **Task parallelism** - Different operations in parallel
- **Pipeline parallelism** - Stages of processing
- **Fork-join pattern**
- **Map-reduce pattern**

## Why This Matters for GPU

### Understanding Parallelism
- CPU threading concepts transfer to GPU
- Race conditions happen on GPU too
- Synchronization principles apply
- GPU has thousands of parallel threads!

### Key Differences: CPU vs GPU
| Concept | CPU Threading | GPU Computing |
|---------|---------------|---------------|
| Threads | 4-32 threads | Thousands-millions |
| Model | Task parallel | Data parallel |
| Sync | Mutex, locks | __syncthreads() |
| Atomics | std::atomic | atomicAdd() |
| Memory | Shared memory | Global/shared memory |

### Parallel Thinking
- Identifying parallelizable work
- Understanding dependencies
- Avoiding race conditions
- Coordinating parallel work
- These skills directly apply to GPU

### CPU + GPU Hybrid
- Use CPU threads to manage multiple GPUs
- Async GPU operations with CPU threads
- Overlapping CPU and GPU work
- Stream-based GPU concurrency

### Example: Multi-GPU with Threads
```cpp
std::vector<std::thread> gpu_threads;
for (int gpu = 0; gpu < num_gpus; ++gpu) {
    gpu_threads.emplace_back([gpu]() {
        cudaSetDevice(gpu);
        // Launch kernel on this GPU
    });
}
for (auto& t : gpu_threads) {
    t.join();
}
```

## Coming Soon

Comprehensive examples including:
- Thread creation and management
- Mutex and lock patterns
- Atomic operations
- Producer-consumer implementation
- Thread pools
- Parallel algorithm patterns
- Real-world threading scenarios

## Estimated Time
20-25 hours

## Prerequisites
Complete Modules 1-10 first.

**Note**: Challenging but essential for understanding parallel computing!