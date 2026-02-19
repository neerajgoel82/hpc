// 05_preprocessor.cpp
// Understanding the C++ preprocessor
// Compile: g++ -std=c++17 -o preprocessor 05_preprocessor.cpp
// See preprocessed output: g++ -E 05_preprocessor.cpp

#include <iostream>

// ===== Simple Macros =====
#define PI 3.14159f
#define ARRAY_SIZE 100
#define MAX_THREADS 256

// ===== Function-like Macros =====
#define SQUARE(x) ((x) * (x))
#define MAX(a, b) ((a) > (b) ? (a) : (b))
#define MIN(a, b) ((a) < (b) ? (a) : (b))

// ===== Conditional Compilation =====
#define DEBUG_MODE 1
#define GPU_ENABLED 1

// ===== Multi-line Macros =====
#define PRINT_VAR(var) \
    std::cout << #var << " = " << (var) << std::endl

// ===== Stringification (#) =====
#define TO_STRING(x) #x
#define STRINGIFY(x) TO_STRING(x)

// ===== Token Pasting (##) =====
#define CONCAT(a, b) a##b

// ===== Predefined Macros =====
void showInfo() {
    std::cout << "File: " << __FILE__ << std::endl;
    std::cout << "Line: " << __LINE__ << std::endl;
    std::cout << "Date: " << __DATE__ << std::endl;
    std::cout << "Time: " << __TIME__ << std::endl;
    std::cout << "C++ Standard: " << __cplusplus << std::endl;
}

// ===== Conditional Compilation Examples =====
#ifdef DEBUG_MODE
    #define LOG(msg) std::cout << "[DEBUG] " << msg << std::endl
#else
    #define LOG(msg)  // Do nothing in release mode
#endif

#if GPU_ENABLED
    #define DEVICE_CODE __device__
    #define KERNEL_LAUNCH(grid, block) <<<grid, block>>>
#else
    #define DEVICE_CODE
    #define KERNEL_LAUNCH(grid, block)
#endif

// ===== Platform Detection =====
#ifdef __APPLE__
    #define PLATFORM "macOS"
#elif defined(_WIN32) || defined(_WIN64)
    #define PLATFORM "Windows"
#elif defined(__linux__)
    #define PLATFORM "Linux"
#else
    #define PLATFORM "Unknown"
#endif

// ===== Error and Warning Directives =====
#if ARRAY_SIZE < 10
    #error "ARRAY_SIZE must be at least 10"
#endif

#ifndef PI
    #warning "PI not defined, using default"
    #define PI 3.14159f
#endif

int main() {
    // ===== Using Macros =====
    std::cout << "=== Basic Macros ===" << std::endl;
    std::cout << "PI = " << PI << std::endl;
    std::cout << "ARRAY_SIZE = " << ARRAY_SIZE << std::endl;
    std::cout << "MAX_THREADS = " << MAX_THREADS << std::endl;
    std::cout << std::endl;

    // ===== Function-like Macros =====
    std::cout << "=== Function Macros ===" << std::endl;
    int x = 5;
    std::cout << "SQUARE(5) = " << SQUARE(x) << std::endl;
    std::cout << "MAX(10, 20) = " << MAX(10, 20) << std::endl;
    std::cout << "MIN(10, 20) = " << MIN(10, 20) << std::endl;
    std::cout << std::endl;

    // ===== Stringification =====
    std::cout << "=== Stringification ===" << std::endl;
    std::cout << TO_STRING(Hello World) << std::endl;  // Becomes "Hello World"
    PRINT_VAR(x);  // Prints "x = 5"
    std::cout << std::endl;

    // ===== Token Pasting =====
    std::cout << "=== Token Pasting ===" << std::endl;
    int var1 = 10, var2 = 20;
    std::cout << "var1 = " << var1 << std::endl;
    std::cout << "CONCAT(var, 1) = " << CONCAT(var, 1) << std::endl;
    std::cout << "CONCAT(var, 2) = " << CONCAT(var, 2) << std::endl;
    std::cout << std::endl;

    // ===== Predefined Macros =====
    std::cout << "=== Predefined Macros ===" << std::endl;
    showInfo();
    std::cout << "Platform: " << PLATFORM << std::endl;
    std::cout << std::endl;

    // ===== Conditional Compilation =====
    std::cout << "=== Conditional Compilation ===" << std::endl;
    LOG("This message only appears in debug mode");

    #ifdef DEBUG_MODE
        std::cout << "Debug mode is ON" << std::endl;
    #else
        std::cout << "Debug mode is OFF" << std::endl;
    #endif

    #if GPU_ENABLED
        std::cout << "GPU support enabled" << std::endl;
    #else
        std::cout << "GPU support disabled" << std::endl;
    #endif

    return 0;
}

/*
LEARNING NOTES:

PREPROCESSOR:
- Runs before compilation
- Text substitution
- No type checking
- No scoping rules

MACROS vs CONST/CONSTEXPR:
Prefer constexpr over macros:
#define PI 3.14f           // Macro (old style)
constexpr float PI = 3.14f // Modern C++ (better!)

MACRO PITFALLS:
#define SQUARE(x) x * x
SQUARE(2 + 3)  // Expands to: 2 + 3 * 2 + 3 = 11 (NOT 25!)

Always use parentheses:
#define SQUARE(x) ((x) * (x))

CONDITIONAL COMPILATION USES:
- Debug vs release builds
- Platform-specific code
- Feature toggles
- API versioning

COMMON DIRECTIVES:
#include    - Include files
#define     - Define macro
#undef      - Undefine macro
#ifdef      - If defined
#ifndef     - If not defined
#if         - If condition
#elif       - Else if
#else       - Else
#endif      - End if
#error      - Compilation error
#warning    - Compilation warning
#pragma     - Compiler-specific

PREDEFINED MACROS:
__FILE__      - Current filename
__LINE__      - Current line number
__DATE__      - Compilation date
__TIME__      - Compilation time
__cplusplus   - C++ standard version
__FUNCTION__  - Current function name

GPU/CUDA RELEVANCE:
- CUDA uses preprocessor extensively
- __CUDA_ARCH__ for architecture detection
- Conditional compilation for device/host code
- Debug macros for GPU error checking

CUDA EXAMPLE:
#ifdef __CUDACC__
    #define DEVICE __device__
    #define HOST __host__
#else
    #define DEVICE
    #define HOST
#endif

DEVICE float compute(float x) {
    return x * x;
}

MODERN ALTERNATIVES:
- constexpr instead of #define constants
- inline functions instead of function macros
- if constexpr instead of #ifdef (C++17)
- Attributes instead of some #pragmas

TRY THIS:
1. Create DEBUG_PRINT macro that only prints in debug mode
2. Use #error to enforce minimum C++ standard
3. Create a macro to time code execution
4. Use ## to create variable names programmatically
5. Compare macro vs constexpr for constants
*/

#undef PI  // Undefine PI
#undef DEBUG_MODE
#undef GPU_ENABLED
