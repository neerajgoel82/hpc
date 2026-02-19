/*
 * MODULE 10: Exception Handling
 * File: 01_exception_handling.cpp
 *
 * TOPIC: Try/Catch and Exception Classes
 *
 * COMPILE: g++ -std=c++17 -o exception_handling 01_exception_handling.cpp
 */

#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

// Custom exception class
class GPUException : public std::runtime_error {
private:
    int errorCode;

public:
    GPUException(const std::string& msg, int code)
        : std::runtime_error(msg), errorCode(code) {}

    int getErrorCode() const { return errorCode; }
};

// Function that may throw
float safeDivide(float a, float b) {
    if (b == 0.0f) {
        throw std::invalid_argument("Division by zero!");
    }
    return a / b;
}

// Vector bounds checking
float safeVectorAccess(const std::vector<float>& vec, size_t index) {
    if (index >= vec.size()) {
        throw std::out_of_range("Index out of bounds: " + std::to_string(index));
    }
    return vec[index];
}

// Simulated GPU operation
void uploadToGPU(const float* data, size_t size) {
    if (data == nullptr) {
        throw std::invalid_argument("Null pointer passed to GPU");
    }

    if (size == 0) {
        throw GPUException("Empty data buffer", 1001);
    }

    if (size > 1000000) {
        throw GPUException("Buffer too large for GPU", 1002);
    }

    std::cout << "  [GPU] Uploaded " << size << " elements\n";
}

int main() {
    std::cout << "=== Exception Handling Demo ===\n\n";

    // Basic try-catch
    std::cout << "=== Basic Exception ===\n";
    try {
        float result = safeDivide(10.0f, 0.0f);
        std::cout << "Result: " << result << "\n";
    }
    catch (const std::invalid_argument& e) {
        std::cout << "Caught exception: " << e.what() << "\n";
    }
    std::cout << "\n";

    // Out of range exception
    std::cout << "=== Out of Range ===\n";
    try {
        std::vector<float> data{1, 2, 3};
        float value = safeVectorAccess(data, 10);
        std::cout << "Value: " << value << "\n";
    }
    catch (const std::out_of_range& e) {
        std::cout << "Caught exception: " << e.what() << "\n";
    }
    std::cout << "\n";

    // Custom exception
    std::cout << "=== Custom GPU Exception ===\n";
    try {
        float data[10];
        uploadToGPU(data, 2000000);  // Too large
    }
    catch (const GPUException& e) {
        std::cout << "GPU Error: " << e.what() << "\n";
        std::cout << "Error code: " << e.getErrorCode() << "\n";
    }
    std::cout << "\n";

    // Multiple catch blocks
    std::cout << "=== Multiple Catch Blocks ===\n";
    try {
        uploadToGPU(nullptr, 100);
    }
    catch (const GPUException& e) {
        std::cout << "GPU-specific error: " << e.what() << "\n";
    }
    catch (const std::invalid_argument& e) {
        std::cout << "Invalid argument: " << e.what() << "\n";
    }
    catch (const std::exception& e) {
        std::cout << "Generic exception: " << e.what() << "\n";
    }
    catch (...) {
        std::cout << "Unknown exception\n";
    }
    std::cout << "\n";

    // RAII with exceptions (automatic cleanup)
    std::cout << "=== RAII Exception Safety ===\n";
    try {
        std::vector<float> buffer(1000);
        // Even if exception thrown, vector is cleaned up
        throw std::runtime_error("Something went wrong");
    }
    catch (const std::exception& e) {
        std::cout << "Exception: " << e.what() << "\n";
        std::cout << "Buffer automatically cleaned up!\n";
    }

    std::cout << "\nGPU CONNECTION:\n";
    std::cout << "  - CUDA error checking with exceptions\n";
    std::cout << "  - OpenGL error handling\n";
    std::cout << "  - Resource cleanup with RAII\n";
    std::cout << "  - Robust GPU applications\n";

    std::cout << "\nNOTE: GPU device code cannot use exceptions!\n";
    std::cout << "      Use error codes on device, exceptions on host.\n";

    return 0;
}

/*
 * TRY THIS:
 * 1. Create TextureLoadException class
 * 2. Add shader compilation error handling
 * 3. Implement CUDA error checking wrapper
 * 4. Create exception-safe texture manager
 * 5. Add nested try-catch blocks
 *
 * IMPORTANT:
 * - GPU device code (CUDA kernels) CANNOT use exceptions
 * - Use exceptions for host-side error handling
 * - Use error codes for device-side errors
 */
