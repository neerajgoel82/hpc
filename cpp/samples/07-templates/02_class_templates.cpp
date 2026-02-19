/*
 * MODULE 7: Templates
 * File: 02_class_templates.cpp
 *
 * TOPIC: Class Templates for Generic Data Structures
 *
 * COMPILE: g++ -std=c++17 -o class_templates 02_class_templates.cpp
 */

#include <iostream>
#include <cstring>
#include <stdexcept>

// Example 1: Generic Array class
template <typename T, int SIZE>
class Array {
private:
    T data[SIZE];

public:
    Array() {
        for (int i = 0; i < SIZE; ++i) {
            data[i] = T();  // Default initialize
        }
    }

    T& operator[](int index) {
        if (index < 0 || index >= SIZE) {
            throw std::out_of_range("Index out of bounds");
        }
        return data[index];
    }

    const T& operator[](int index) const {
        if (index < 0 || index >= SIZE) {
            throw std::out_of_range("Index out of bounds");
        }
        return data[index];
    }

    int size() const { return SIZE; }

    void fill(const T& value) {
        for (int i = 0; i < SIZE; ++i) {
            data[i] = value;
        }
    }

    void print() const {
        std::cout << "[";
        for (int i = 0; i < SIZE; ++i) {
            std::cout << data[i];
            if (i < SIZE - 1) std::cout << ", ";
        }
        std::cout << "]";
    }
};

// Example 2: Generic Pair class
template <typename T1, typename T2>
class Pair {
public:
    T1 first;
    T2 second;

    Pair() : first(), second() {}
    Pair(const T1& f, const T2& s) : first(f), second(s) {}

    void print() const {
        std::cout << "(" << first << ", " << second << ")";
    }
};

// Example 3: Generic Stack
template <typename T>
class Stack {
private:
    T* data;
    int capacity;
    int top;

public:
    Stack(int cap = 10) : capacity(cap), top(-1) {
        data = new T[capacity];
    }

    ~Stack() {
        delete[] data;
    }

    void push(const T& value) {
        if (top >= capacity - 1) {
            throw std::overflow_error("Stack overflow");
        }
        data[++top] = value;
    }

    T pop() {
        if (top < 0) {
            throw std::underflow_error("Stack underflow");
        }
        return data[top--];
    }

    T& peek() {
        if (top < 0) {
            throw std::underflow_error("Stack empty");
        }
        return data[top];
    }

    bool isEmpty() const { return top < 0; }
    int size() const { return top + 1; }
};

// Example 4: Generic 3D Vector
template <typename T>
class Vec3 {
public:
    T x, y, z;

    Vec3(T x_ = 0, T y_ = 0, T z_ = 0) : x(x_), y(y_), z(z_) {}

    Vec3 operator+(const Vec3& v) const {
        return Vec3(x + v.x, y + v.y, z + v.z);
    }

    Vec3 operator*(T scalar) const {
        return Vec3(x * scalar, y * scalar, z * scalar);
    }

    T dot(const Vec3& v) const {
        return x * v.x + y * v.y + z * v.z;
    }

    void print() const {
        std::cout << "Vec3(" << x << ", " << y << ", " << z << ")";
    }
};

// Example 5: GPU Buffer simulation (type-generic)
template <typename T>
class GPUBuffer {
private:
    T* hostData;
    T* deviceData;  // Simulated
    size_t count;

public:
    GPUBuffer(size_t n) : count(n) {
        hostData = new T[count];
        deviceData = new T[count];  // In real CUDA: cudaMalloc
        std::cout << "  [GPU] Allocated buffer for " << count
                  << " elements of size " << sizeof(T) << "\n";
    }

    ~GPUBuffer() {
        delete[] hostData;
        delete[] deviceData;  // In real CUDA: cudaFree
        std::cout << "  [GPU] Freed buffer\n";
    }

    void set(size_t index, const T& value) {
        hostData[index] = value;
    }

    T get(size_t index) const {
        return hostData[index];
    }

    void copyToDevice() {
        std::memcpy(deviceData, hostData, count * sizeof(T));
        std::cout << "  [GPU] Copied " << count << " elements to device\n";
    }

    void copyFromDevice() {
        std::memcpy(hostData, deviceData, count * sizeof(T));
        std::cout << "  [GPU] Copied " << count << " elements from device\n";
    }

    size_t size() const { return count; }
};

int main() {
    std::cout << "=== MODULE 7: Class Templates Demo ===\n";

    // Array template
    std::cout << "\n=== Generic Array ===\n";
    Array<int, 5> intArray;
    intArray.fill(10);
    intArray[2] = 99;
    std::cout << "Int array: ";
    intArray.print();
    std::cout << "\n";

    Array<float, 3> floatArray;
    floatArray[0] = 1.1f;
    floatArray[1] = 2.2f;
    floatArray[2] = 3.3f;
    std::cout << "Float array: ";
    floatArray.print();
    std::cout << "\n";

    // Pair template
    std::cout << "\n=== Generic Pair ===\n";
    Pair<int, std::string> p1(42, "answer");
    std::cout << "Pair: ";
    p1.print();
    std::cout << "\n";

    Pair<float, float> p2(3.14f, 2.71f);
    std::cout << "Pair: ";
    p2.print();
    std::cout << "\n";

    // Stack template
    std::cout << "\n=== Generic Stack ===\n";
    Stack<int> intStack;
    intStack.push(10);
    intStack.push(20);
    intStack.push(30);
    std::cout << "Stack size: " << intStack.size() << "\n";
    std::cout << "Popping: " << intStack.pop() << "\n";
    std::cout << "Peek: " << intStack.peek() << "\n";

    // Vector template
    std::cout << "\n=== Generic Vector ===\n";
    Vec3<float> vf(1.0f, 2.0f, 3.0f);
    Vec3<int> vi(1, 2, 3);

    std::cout << "Float vector: ";
    vf.print();
    std::cout << "\n";

    std::cout << "Int vector: ";
    vi.print();
    std::cout << "\n";

    // GPU buffer template
    std::cout << "\n=== Generic GPU Buffer ===\n";
    GPUBuffer<float> floatBuffer(100);
    floatBuffer.set(0, 3.14f);
    floatBuffer.set(1, 2.71f);
    floatBuffer.copyToDevice();
    floatBuffer.copyFromDevice();

    GPUBuffer<Vec3<float>> vectorBuffer(50);
    std::cout << "Can store complex types too!\n";

    std::cout << "\nGPU CONNECTION: CUDA Thrust containers are templated!\n";
    std::cout << "thrust::device_vector<float>, thrust::device_vector<int>, etc.\n";

    return 0;
}

/*
 * TRY THIS:
 * 1. Add Matrix<T, ROWS, COLS> template class
 * 2. Implement Queue<T> template
 * 3. Create SmartPointer<T> template
 * 4. Add bounds-checking to GPUBuffer
 * 5. Implement generic min/max heap
 */
