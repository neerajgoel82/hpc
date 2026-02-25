// 01_pointers_basics.cpp
// Understanding pointers - the foundation of memory management
// Compile: g++ -std=c++17 -o pointers_basics 01_pointers_basics.cpp

#include <iostream>

int main() {
    // ===== Basic Variable and Address =====
    int x = 42;
    std::cout << "x = " << x << std::endl;
    std::cout << "Address of x = " << &x << std::endl;  // & = address-of operator
    std::cout << std::endl;

    // ===== Pointer Declaration and Initialization =====
    int* ptr = &x;  // ptr holds the address of x
    std::cout << "ptr = " << ptr << " (address it holds)" << std::endl;
    std::cout << "*ptr = " << *ptr << " (value at that address)" << std::endl;  // * = dereference
    std::cout << std::endl;

    // ===== Modifying Through Pointer =====
    *ptr = 100;  // Change value at address ptr points to
    std::cout << "After *ptr = 100:" << std::endl;
    std::cout << "x = " << x << std::endl;  // x is now 100!
    std::cout << "*ptr = " << *ptr << std::endl;
    std::cout << std::endl;

    // ===== Multiple Pointers to Same Variable =====
    int* ptr2 = &x;
    std::cout << "ptr = " << ptr << std::endl;
    std::cout << "ptr2 = " << ptr2 << std::endl;  // Same address
    std::cout << "Both point to same location" << std::endl;
    std::cout << std::endl;

    // ===== Null Pointer =====
    int* nullPtr = nullptr;  // Modern C++ (use nullptr, not NULL or 0)
    std::cout << "nullPtr = " << nullPtr << std::endl;

    if (nullPtr == nullptr) {
        std::cout << "nullPtr is null (points to nothing)" << std::endl;
    }

    // NEVER dereference a null pointer!
    // *nullPtr = 5;  // CRASH! Segmentation fault
    std::cout << std::endl;

    // ===== Pointer Arithmetic =====
    int arr[] = {10, 20, 30, 40, 50};
    int* arrPtr = arr;  // Array name decays to pointer

    std::cout << "Array using pointer arithmetic:" << std::endl;
    for (int i = 0; i < 5; i++) {
        std::cout << "arrPtr + " << i << " = " << (arrPtr + i)
                  << ", value = " << *(arrPtr + i) << std::endl;
    }
    std::cout << std::endl;

    // ===== Pointer to Pointer =====
    int value = 42;
    int* ptr1 = &value;
    int** ptrToPtr = &ptr1;  // Pointer to pointer

    std::cout << "value = " << value << std::endl;
    std::cout << "*ptr1 = " << *ptr1 << std::endl;
    std::cout << "**ptrToPtr = " << **ptrToPtr << std::endl;
    std::cout << std::endl;

    // ===== Const Pointers =====
    int a = 10, b = 20;

    const int* ptr3 = &a;      // Pointer to const int
    // *ptr3 = 15;             // ERROR: can't modify value
    ptr3 = &b;                 // OK: can change where it points

    int* const ptr4 = &a;      // Const pointer to int
    *ptr4 = 15;                // OK: can modify value
    // ptr4 = &b;              // ERROR: can't change where it points

    const int* const ptr5 = &a;  // Const pointer to const int
    // *ptr5 = 15;             // ERROR: can't modify value
    // ptr5 = &b;              // ERROR: can't change where it points

    // ===== Sizeof Pointer =====
    std::cout << "Size of int: " << sizeof(int) << " bytes" << std::endl;
    std::cout << "Size of int*: " << sizeof(int*) << " bytes" << std::endl;
    std::cout << "Size of double*: " << sizeof(double*) << " bytes" << std::endl;
    std::cout << "All pointers same size (8 bytes on 64-bit)" << std::endl;

    return 0;
}

/*
LEARNING NOTES:

POINTER BASICS:
int x = 42;        // Variable
int* ptr = &x;     // Pointer holds address of x
*ptr               // Dereference: get value at address

OPERATORS:
& = address-of     Get memory address of variable
* = dereference    Access value at address

DECLARATION:
int* ptr;          // Pointer to int
float* ptr;        // Pointer to float
char* ptr;         // Pointer to char

NULLPTR:
- Modern C++ uses nullptr
- Avoid NULL (C-style) or 0
- Always check before dereferencing
- Dereferencing nullptr = crash

POINTER ARITHMETIC:
ptr + 1           // Next element (adds sizeof(type))
ptr++             // Move to next element
ptr - 1           // Previous element
ptr1 - ptr2       // Distance between pointers

CONST POINTERS (read right-to-left):
const int* ptr        // Pointer to const int (can't modify value)
int* const ptr        // Const pointer (can't change address)
const int* const ptr  // Both const

WHY POINTERS MATTER:
1. Dynamic memory allocation
2. Passing large data efficiently
3. Data structures (linked lists, trees)
4. Low-level memory manipulation
5. Function pointers
6. GPU memory management

GPU RELEVANCE - CRITICAL!:
- cudaMalloc returns pointer to GPU memory
- GPU kernels receive pointers as parameters
- Host and device pointers are different!
- Memory transfers use pointers
- Understanding pointers is ESSENTIAL for GPU

CUDA EXAMPLE:
float* d_data;  // Pointer to device memory
cudaMalloc(&d_data, size * sizeof(float));
kernel<<<grid, block>>>(d_data);  // Pass pointer to kernel
cudaFree(d_data);

COMMON MISTAKES:
✗ Dereferencing null pointer
✗ Dangling pointers (pointing to freed memory)
✗ Memory leaks (forgetting to free)
✗ Buffer overflows (accessing past array end)
✗ Uninitialized pointers

TRY THIS:
1. Create two int variables and swap them using pointers
2. What happens if you dereference an uninitialized pointer?
3. Create a function that returns a pointer to the larger of two ints
4. Use pointer arithmetic to print array backwards
5. What's the difference between ptr++ and (*ptr)++?
*/
