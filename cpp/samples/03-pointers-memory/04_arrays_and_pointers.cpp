// 04_arrays_and_pointers.cpp
// Arrays and their relationship with pointers
// Compile: g++ -std=c++17 -o arrays_and_pointers 04_arrays_and_pointers.cpp

#include <iostream>

void printArray(int* arr, int size) {
    for (int i = 0; i < size; i++) {
        std::cout << arr[i] << " ";
    }
    std::cout << std::endl;
}

void fillArray(float* arr, int size, float value) {
    for (int i = 0; i < size; i++) {
        arr[i] = value + i;
    }
}

int main() {
    // ===== Array Basics =====
    int arr[5] = {10, 20, 30, 40, 50};
    std::cout << "Array elements: ";
    for (int i = 0; i < 5; i++) {
        std::cout << arr[i] << " ";
    }
    std::cout << std::endl;
    std::cout << "Array name 'arr': " << arr << " (address of first element)" << std::endl;
    std::cout << "&arr[0]: " << &arr[0] << " (same address)" << std::endl;
    std::cout << std::endl;

    // ===== Array and Pointer Equivalence =====
    int* ptr = arr;  // Array name decays to pointer
    std::cout << "arr[2] = " << arr[2] << std::endl;
    std::cout << "ptr[2] = " << ptr[2] << std::endl;  // Same thing!
    std::cout << "*(arr + 2) = " << *(arr + 2) << std::endl;  // Same thing!
    std::cout << "*(ptr + 2) = " << *(ptr + 2) << std::endl;  // Same thing!
    std::cout << std::endl;

    // ===== Pointer Arithmetic =====
    std::cout << "Pointer arithmetic:" << std::endl;
    for (int i = 0; i < 5; i++) {
        std::cout << "ptr + " << i << " = " << (ptr + i)
                  << ", value = " << *(ptr + i) << std::endl;
    }
    std::cout << std::endl;

    // ===== Sizeof Array vs Pointer =====
    std::cout << "sizeof(arr) = " << sizeof(arr) << " bytes" << std::endl;
    std::cout << "sizeof(ptr) = " << sizeof(ptr) << " bytes" << std::endl;
    std::cout << "Array size = " << sizeof(arr) / sizeof(arr[0]) << " elements" << std::endl;
    std::cout << std::endl;

    // ===== Passing Arrays to Functions =====
    int data[] = {1, 2, 3, 4, 5};
    std::cout << "Printing array via function: ";
    printArray(data, 5);
    std::cout << std::endl;

    // ===== Dynamic Array =====
    int size = 10;
    float* dynArray = new float[size];
    fillArray(dynArray, size, 1.0f);

    std::cout << "Dynamic array: ";
    for (int i = 0; i < size; i++) {
        std::cout << dynArray[i] << " ";
    }
    std::cout << std::endl;
    delete[] dynArray;
    std::cout << std::endl;

    // ===== 2D Array (Array of Arrays) =====
    int matrix[3][4] = {
        {1, 2, 3, 4},
        {5, 6, 7, 8},
        {9, 10, 11, 12}
    };

    std::cout << "2D Array:" << std::endl;
    for (int i = 0; i < 3; i++) {
        for (int j = 0; j < 4; j++) {
            std::cout << matrix[i][j] << " ";
        }
        std::cout << std::endl;
    }
    std::cout << std::endl;

    // ===== Dynamic 2D Array =====
    int rows = 3, cols = 4;
    int** matrix2d = new int*[rows];
    for (int i = 0; i < rows; i++) {
        matrix2d[i] = new int[cols];
    }

    // Fill matrix
    int value = 1;
    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < cols; j++) {
            matrix2d[i][j] = value++;
        }
    }

    // Print matrix
    std::cout << "Dynamic 2D array:" << std::endl;
    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < cols; j++) {
            std::cout << matrix2d[i][j] << " ";
        }
        std::cout << std::endl;
    }

    // Free matrix
    for (int i = 0; i < rows; i++) {
        delete[] matrix2d[i];
    }
    delete[] matrix2d;

    return 0;
}

/*
LEARNING NOTES:

ARRAY DECAY:
int arr[10];
int* ptr = arr;  // Array name decays to pointer to first element

ARRAY ACCESS:
arr[i] is equivalent to *(arr + i)
ptr[i] is equivalent to *(ptr + i)

POINTER ARITHMETIC:
ptr + i        // Points to ith element after ptr
ptr++          // Move to next element
ptr += n       // Move forward n elements

SIZEOF:
sizeof(arr)    // Total array size in bytes
sizeof(ptr)    // Size of pointer (8 bytes on 64-bit)
// Cannot get array size from pointer alone!

PASSING ARRAYS:
void func(int arr[], int size);  // Array notation
void func(int* arr, int size);   // Pointer notation
// Both are equivalent! Array size must be passed separately

2D ARRAYS:
Static: int arr[rows][cols];
Dynamic: Allocate array of pointers, then each row

CONTIGUOUS MEMORY:
Arrays are contiguous in memory
Critical for cache performance
Critical for GPU memory access!

GPU RELEVANCE:
- GPU operates on arrays of data
- Understanding pointer arithmetic essential
- 1D arrays map to 1D grids
- 2D arrays map to 2D grids
- Row-major vs column-major order matters

CUDA ARRAY ACCESS:
int idx = blockIdx.x * blockDim.x + threadIdx.x;
output[idx] = input[idx] * 2;  // Pointer arithmetic!

MEMORY LAYOUT:
arr[i][j] in row-major (C/C++/CUDA):
Memory: [row0][row1][row2]...

Column-major (Fortran/MATLAB):
Memory: [col0][col1][col2]...

TRY THIS:
1. Create function that reverses array in-place using pointers
2. Implement 2D array as single 1D array (more efficient!)
3. Calculate memory address of arr[i][j] manually
4. Create dynamic 3D array
5. Compare performance: array vs pointer access
*/
