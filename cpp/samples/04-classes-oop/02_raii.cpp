/*
 * MODULE 4: Classes and Object-Oriented Programming
 * File: 02_raii.cpp
 *
 * TOPIC: RAII (Resource Acquisition Is Initialization)
 *
 * CONCEPTS:
 * - RAII: Resources are acquired in constructor, released in destructor
 * - Automatic resource management (no manual cleanup needed)
 * - Exception safety through automatic cleanup
 * - Common resources: memory, files, locks, GPU memory
 *
 * GPU RELEVANCE:
 * RAII is CRITICAL for GPU programming:
 * - Automatically free GPU memory (cudaMalloc/cudaFree)
 * - Release OpenGL buffers and textures
 * - Manage CUDA streams and events
 * - Prevent resource leaks that crash GPU applications
 *
 * COMPILE: g++ -std=c++17 -o raii 02_raii.cpp
 */

#include <iostream>
#include <string>
#include <cstring>
#include <fstream>

// Example 1: Simple RAII for dynamic memory
class DynamicArray {
private:
    int* data;
    size_t size;

public:
    // Constructor: Acquire resource (allocate memory)
    DynamicArray(size_t n) : size(n) {
        std::cout << "Allocating array of size " << n << "\n";
        data = new int[size];
        // Initialize to zero
        for (size_t i = 0; i < size; ++i) {
            data[i] = 0;
        }
    }

    // Destructor: Release resource (free memory)
    ~DynamicArray() {
        std::cout << "Deallocating array of size " << size << "\n";
        delete[] data;
        data = nullptr;
    }

    // Prevent copying (we'll learn proper copying in next file)
    DynamicArray(const DynamicArray&) = delete;
    DynamicArray& operator=(const DynamicArray&) = delete;

    // Access elements
    int& operator[](size_t index) {
        return data[index];
    }

    const int& operator[](size_t index) const {
        return data[index];
    }

    size_t getSize() const { return size; }

    void print() const {
        std::cout << "Array contents: [";
        for (size_t i = 0; i < size; ++i) {
            std::cout << data[i];
            if (i < size - 1) std::cout << ", ";
        }
        std::cout << "]\n";
    }
};

// Example 2: RAII for file handling
class FileWriter {
private:
    std::ofstream file;
    std::string filename;

public:
    // Constructor: Open file
    FileWriter(const std::string& fname) : filename(fname) {
        std::cout << "Opening file: " << filename << "\n";
        file.open(filename);
        if (!file.is_open()) {
            std::cerr << "Failed to open file!\n";
        }
    }

    // Destructor: Close file automatically
    ~FileWriter() {
        if (file.is_open()) {
            std::cout << "Closing file: " << filename << "\n";
            file.close();
        }
    }

    // Delete copy operations
    FileWriter(const FileWriter&) = delete;
    FileWriter& operator=(const FileWriter&) = delete;

    void writeLine(const std::string& line) {
        if (file.is_open()) {
            file << line << "\n";
        }
    }

    bool isOpen() const {
        return file.is_open();
    }
};

// Example 3: Simulated GPU Memory Manager (RAII pattern for GPU)
class GPUMemory {
private:
    void* devicePtr;
    size_t bytes;

    // Simulate cudaMalloc
    void* allocateGPUMemory(size_t size) {
        std::cout << "  [GPU] Allocating " << size << " bytes on device\n";
        // In real CUDA: cudaMalloc(&devicePtr, size);
        return malloc(size);  // Simulation only
    }

    // Simulate cudaFree
    void freeGPUMemory(void* ptr) {
        std::cout << "  [GPU] Freeing " << bytes << " bytes from device\n";
        // In real CUDA: cudaFree(ptr);
        free(ptr);  // Simulation only
    }

public:
    // Constructor: Allocate GPU memory
    GPUMemory(size_t size) : bytes(size) {
        std::cout << "[GPU RAII] Constructor allocating GPU memory\n";
        devicePtr = allocateGPUMemory(bytes);
    }

    // Destructor: Free GPU memory automatically
    ~GPUMemory() {
        std::cout << "[GPU RAII] Destructor freeing GPU memory\n";
        if (devicePtr) {
            freeGPUMemory(devicePtr);
            devicePtr = nullptr;
        }
    }

    // Delete copy operations to prevent double-free
    GPUMemory(const GPUMemory&) = delete;
    GPUMemory& operator=(const GPUMemory&) = delete;

    void* getPtr() const { return devicePtr; }
    size_t getSize() const { return bytes; }

    // Simulate copying data to GPU
    void copyToDevice(const void* hostPtr, size_t size) {
        if (size <= bytes) {
            std::cout << "  [GPU] Copying " << size << " bytes to device\n";
            // In real CUDA: cudaMemcpy(devicePtr, hostPtr, size, cudaMemcpyHostToDevice);
            memcpy(devicePtr, hostPtr, size);  // Simulation only
        }
    }

    // Simulate copying data from GPU
    void copyFromDevice(void* hostPtr, size_t size) {
        if (size <= bytes) {
            std::cout << "  [GPU] Copying " << size << " bytes from device\n";
            // In real CUDA: cudaMemcpy(hostPtr, devicePtr, size, cudaMemcpyDeviceToHost);
            memcpy(hostPtr, devicePtr, size);  // Simulation only
        }
    }
};

// Example 4: Scope Guard pattern
class ScopeGuard {
private:
    std::string operation;

public:
    ScopeGuard(const std::string& op) : operation(op) {
        std::cout << ">>> Entering: " << operation << "\n";
    }

    ~ScopeGuard() {
        std::cout << "<<< Exiting: " << operation << "\n";
    }
};

// Demonstrate RAII benefits
void demonstrateRAII() {
    std::cout << "\n=== RAII with Dynamic Array ===\n";

    {
        ScopeGuard guard("Array scope");
        DynamicArray arr(5);

        // Use the array
        for (size_t i = 0; i < arr.getSize(); ++i) {
            arr[i] = i * 10;
        }
        arr.print();

        // No need to manually delete!
        // Destructor called automatically when leaving scope
    }

    std::cout << "\nArray automatically cleaned up!\n";
}

void demonstrateFileRAII() {
    std::cout << "\n=== RAII with File Handling ===\n";

    {
        ScopeGuard guard("File scope");
        FileWriter writer("output.txt");

        if (writer.isOpen()) {
            writer.writeLine("Line 1: RAII ensures file is closed");
            writer.writeLine("Line 2: Even if exceptions occur");
            writer.writeLine("Line 3: No need to manually close");
        }

        // File automatically closed when leaving scope
    }

    std::cout << "\nFile automatically closed!\n";
}

void demonstrateGPURAII() {
    std::cout << "\n=== RAII with Simulated GPU Memory ===\n";

    {
        ScopeGuard guard("GPU scope");

        // Allocate GPU memory using RAII
        GPUMemory gpuBuffer(1024);

        // Prepare some host data
        int hostData[] = {1, 2, 3, 4, 5};

        // Copy to "GPU"
        gpuBuffer.copyToDevice(hostData, sizeof(hostData));

        // Do some work...
        std::cout << "  [GPU] Processing data on device...\n";

        // Copy back from "GPU"
        int resultData[5];
        gpuBuffer.copyFromDevice(resultData, sizeof(resultData));

        std::cout << "  [GPU] Results: ";
        for (int i = 0; i < 5; ++i) {
            std::cout << resultData[i] << " ";
        }
        std::cout << "\n";

        // GPU memory automatically freed when leaving scope!
    }

    std::cout << "\nGPU memory automatically freed!\n";
}

// Demonstrate exception safety with RAII
void demonstrateExceptionSafety() {
    std::cout << "\n=== RAII Exception Safety ===\n";

    try {
        DynamicArray arr(10);
        arr[0] = 100;

        std::cout << "Array created and used\n";

        // Simulate an error
        // throw std::runtime_error("Simulated error");

        // Even if exception occurs, destructor is called!
    }
    catch (const std::exception& e) {
        std::cout << "Exception caught: " << e.what() << "\n";
    }

    std::cout << "Array was properly cleaned up despite potential exception\n";
}

int main() {
    std::cout << "=== MODULE 4: RAII Pattern Demo ===\n";

    demonstrateRAII();
    demonstrateFileRAII();
    demonstrateGPURAII();
    demonstrateExceptionSafety();

    std::cout << "\n=== Program Complete ===\n";
    return 0;
}

/*
 * KEY CONCEPTS DEMONSTRATED:
 *
 * 1. RAII PRINCIPLE:
 *    - Resource acquisition = initialization (constructor)
 *    - Resource release = cleanup (destructor)
 *    - Automatic, deterministic cleanup
 *
 * 2. BENEFITS:
 *    - No manual cleanup needed (can't forget to free)
 *    - Exception safe (cleanup happens even with exceptions)
 *    - Clear ownership semantics
 *    - Prevents resource leaks
 *
 * 3. COMMON RAII RESOURCES:
 *    - Dynamic memory (new/delete)
 *    - File handles (open/close)
 *    - Locks (acquire/release)
 *    - GPU memory (cudaMalloc/cudaFree)
 *
 * TRY THIS:
 * 1. Create a LockGuard class that acquires/releases a mutex
 * 2. Add error checking to GPUMemory (handle allocation failures)
 * 3. Create a BufferPool class that reuses GPU memory buffers
 * 4. Implement a TextureRAII class for OpenGL textures
 * 5. Add logging to track all resource acquisitions and releases
 * 6. Uncomment the throw statement to see exception safety in action
 *
 * COMMON MISTAKES:
 * - Forgetting to delete copy constructor/assignment (can cause double-free)
 * - Not checking if resource acquisition succeeded
 * - Mixing RAII with manual resource management
 * - Not making destructor virtual in base classes
 *
 * GPU CONNECTION:
 * - RAII is the GOLD STANDARD for GPU memory management
 * - Prevents memory leaks that crash GPU applications
 * - Modern libraries (Thrust, cuDNN) use RAII extensively
 * - Example: cudaMalloc in constructor, cudaFree in destructor
 *
 * REAL CUDA EXAMPLE PATTERN:
 *
 * class CudaMemory {
 *     void* d_ptr;
 *     size_t bytes;
 * public:
 *     CudaMemory(size_t n) : bytes(n) {
 *         cudaMalloc(&d_ptr, bytes);
 *     }
 *     ~CudaMemory() {
 *         cudaFree(d_ptr);
 *     }
 *     // ... methods ...
 * };
 */
