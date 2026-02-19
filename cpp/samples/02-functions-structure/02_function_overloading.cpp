// 02_function_overloading.cpp
// Function overloading: same name, different parameters
// Compile: g++ -std=c++17 -o function_overloading 02_function_overloading.cpp

#include <iostream>
#include <string>

// ===== Basic Overloading =====
// Different number of parameters
int add(int a, int b) {
    return a + b;
}

int add(int a, int b, int c) {
    return a + b + c;
}

// ===== Different Parameter Types =====
float add(float a, float b) {
    return a + b;
}

double add(double a, double b) {
    return a + b;
}

// ===== Print Overloading (different types) =====
void print(int value) {
    std::cout << "Integer: " << value << std::endl;
}

void print(float value) {
    std::cout << "Float: " << value << std::endl;
}

void print(const std::string& value) {
    std::cout << "String: " << value << std::endl;
}

void print(const char* value) {
    std::cout << "C-string: " << value << std::endl;
}

// ===== Array Processing Overloads =====
void process(int* array, int size) {
    std::cout << "Processing int array of size " << size << std::endl;
    for (int i = 0; i < size; i++) {
        array[i] *= 2;
    }
}

void process(float* array, int size) {
    std::cout << "Processing float array of size " << size << std::endl;
    for (int i = 0; i < size; i++) {
        array[i] *= 2.0f;
    }
}

// ===== Vector Math Overloading =====
struct Vec3 {
    float x, y, z;
};

// Compute length
float length(const Vec3& v) {
    return std::sqrt(v.x * v.x + v.y * v.y + v.z * v.z);
}

// Compute length squared (faster, no sqrt)
float lengthSquared(const Vec3& v) {
    return v.x * v.x + v.y * v.y + v.z * v.z;
}

// Distance between two points
float distance(const Vec3& a, const Vec3& b) {
    Vec3 diff = {a.x - b.x, a.y - b.y, a.z - b.z};
    return length(diff);
}

// Distance in 2D (overload ignoring z)
float distance(float x1, float y1, float x2, float y2) {
    float dx = x2 - x1;
    float dy = y2 - y1;
    return std::sqrt(dx * dx + dy * dy);
}

// ===== Default Arguments (alternative to overloading) =====
void log(const std::string& message, int level = 0, bool timestamp = false) {
    if (timestamp) {
        std::cout << "[TIME] ";
    }
    std::cout << "[Level " << level << "] " << message << std::endl;
}

// ===== Const Overloading (advanced) =====
class Array {
private:
    int data[10] = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9};
public:
    // Non-const version: allows modification
    int& get(int index) {
        std::cout << "Non-const get" << std::endl;
        return data[index];
    }

    // Const version: read-only
    const int& get(int index) const {
        std::cout << "Const get" << std::endl;
        return data[index];
    }
};

int main() {
    // ===== Basic Overloading =====
    std::cout << "add(5, 3) = " << add(5, 3) << std::endl;
    std::cout << "add(5, 3, 2) = " << add(5, 3, 2) << std::endl;
    std::cout << "add(5.5f, 3.2f) = " << add(5.5f, 3.2f) << std::endl;
    std::cout << std::endl;

    // ===== Type-based Overloading =====
    print(42);
    print(3.14f);
    print(std::string("Hello"));
    print("World");
    std::cout << std::endl;

    // ===== Array Processing =====
    int intArray[] = {1, 2, 3, 4, 5};
    float floatArray[] = {1.0f, 2.0f, 3.0f};

    process(intArray, 5);
    std::cout << "Int array: ";
    for (int i = 0; i < 5; i++) std::cout << intArray[i] << " ";
    std::cout << std::endl;

    process(floatArray, 3);
    std::cout << "Float array: ";
    for (int i = 0; i < 3; i++) std::cout << floatArray[i] << " ";
    std::cout << std::endl << std::endl;

    // ===== Vector Math =====
    Vec3 v1 = {3.0f, 4.0f, 0.0f};
    Vec3 v2 = {6.0f, 8.0f, 0.0f};

    std::cout << "Length of v1: " << length(v1) << std::endl;
    std::cout << "Distance between v1 and v2: " << distance(v1, v2) << std::endl;
    std::cout << "2D distance: " << distance(3.0f, 4.0f, 6.0f, 8.0f) << std::endl;
    std::cout << std::endl;

    // ===== Default Arguments =====
    log("Starting program");                    // Uses defaults
    log("Warning message", 1);                  // Custom level
    log("Error with timestamp", 2, true);       // All parameters
    std::cout << std::endl;

    // ===== Const Overloading =====
    Array arr;
    const Array constArr;

    arr.get(0) = 100;  // Calls non-const version, can modify
    int value = constArr.get(0);  // Calls const version, read-only
    std::cout << "Value: " << value << std::endl;

    return 0;
}

/*
LEARNING NOTES:

FUNCTION OVERLOADING:
- Same function name, different parameters
- Compiler chooses based on arguments
- Different number of parameters OR different types
- Return type alone is NOT enough to overload

OVERLOAD RESOLUTION:
1. Exact match
2. Promotions (int to long, float to double)
3. Standard conversions
4. User-defined conversions

CANNOT OVERLOAD ON:
- Return type only
- Parameter names
- typedef (same underlying type)

DEFAULT ARGUMENTS:
- Alternative to overloading
- Specify from right to left
- Must be in declaration, not definition (if separate)

WHEN TO USE:
- Overloading: Different operations on different types
- Defaults: Same operation with optional parameters

AMBIGUITY:
Watch out for ambiguous overloads:
void func(int x);
void func(float x);
func(3.14);  // Ambiguous! Is it int or float?

GPU RELEVANCE:
- GPU libraries overload functions for different types
- CUDA kernels can be overloaded (template specialization preferred)
- Vector math libraries use heavy overloading
- cuBLAS functions overloaded for float/double/complex

CUDA EXAMPLE:
// Thrust library uses overloading
thrust::reduce(vec_int.begin(), vec_int.end());    // int version
thrust::reduce(vec_float.begin(), vec_float.end()); // float version

TRY THIS:
1. Create overloaded max() for int, float, double
2. Overload a function that works with 2D and 3D vectors
3. What happens if you call add(3, 4.5)? Why?
4. Create min() with default parameters for multiple values
5. Overload a print() function for arrays of different types
*/
