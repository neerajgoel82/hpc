// 02_references.cpp
// References: aliases for variables
// Compile: g++ -std=c++17 -o references 02_references.cpp

#include <iostream>

void swapByValue(int a, int b) {
    int temp = a;
    a = b;
    b = temp;
    // Only swaps copies, originals unchanged
}

void swapByPointer(int* a, int* b) {
    int temp = *a;
    *a = *b;
    *b = temp;
}

void swapByReference(int& a, int& b) {
    int temp = a;
    a = b;
    b = temp;
}

void modifyArray(int* arr, int size) {
    for (int i = 0; i < size; i++) {
        arr[i] *= 2;
    }
}

void printValue(const int& value) {
    std::cout << "Value: " << value << std::endl;
    // value = 10;  // ERROR: const reference
}

int& getElement(int* arr, int index) {
    return arr[index];  // Returns reference to array element
}

int main() {
    // ===== Reference Basics =====
    int x = 10;
    int& ref = x;  // ref is an alias for x

    std::cout << "x = " << x << std::endl;
    std::cout << "ref = " << ref << std::endl;

    ref = 20;  // Modifying ref modifies x
    std::cout << "After ref = 20:" << std::endl;
    std::cout << "x = " << x << std::endl;
    std::cout << std::endl;

    // ===== References vs Pointers =====
    int a = 5;
    int& refA = a;    // Reference
    int* ptrA = &a;   // Pointer

    std::cout << "Value via reference: " << refA << std::endl;
    std::cout << "Value via pointer: " << *ptrA << std::endl;
    std::cout << "Address of a: " << &a << std::endl;
    std::cout << "Address via reference: " << &refA << std::endl;  // Same as &a
    std::cout << "Pointer value: " << ptrA << std::endl;  // Same as &a
    std::cout << std::endl;

    // ===== Swap Comparison =====
    int m = 10, n = 20;
    std::cout << "Before swap: m = " << m << ", n = " << n << std::endl;

    swapByValue(m, n);
    std::cout << "After swapByValue: m = " << m << ", n = " << n << std::endl;

    swapByPointer(&m, &n);
    std::cout << "After swapByPointer: m = " << m << ", n = " << n << std::endl;

    swapByReference(m, n);
    std::cout << "After swapByReference: m = " << m << ", n = " << n << std::endl;
    std::cout << std::endl;

    // ===== Const Reference =====
    int value = 42;
    const int& constRef = value;

    std::cout << "constRef = " << constRef << std::endl;
    value = 100;  // Can modify original
    std::cout << "constRef after modifying value = " << constRef << std::endl;
    // constRef = 50;  // ERROR: can't modify through const reference
    std::cout << std::endl;

    // ===== Reference to Array Element =====
    int arr[] = {1, 2, 3, 4, 5};
    int& elem = arr[2];  // Reference to arr[2]

    std::cout << "arr[2] = " << arr[2] << std::endl;
    elem = 100;
    std::cout << "After elem = 100, arr[2] = " << arr[2] << std::endl;
    std::cout << std::endl;

    // ===== Returning Reference =====
    int data[] = {10, 20, 30};
    getElement(data, 1) = 200;  // Can modify through returned reference
    std::cout << "After getElement(data, 1) = 200:" << std::endl;
    std::cout << "data[1] = " << data[1] << std::endl;

    return 0;
}

/*
LEARNING NOTES:

REFERENCES:
- Alias for existing variable
- Must be initialized when declared
- Cannot be null (unlike pointers)
- Cannot be reassigned to refer to different variable
- Cleaner syntax than pointers

DECLARATION:
int x = 10;
int& ref = x;     // ref is reference to x
// int& ref;      // ERROR: must initialize

REFERENCES VS POINTERS:

References:
✓ Cleaner syntax
✓ Cannot be null
✓ Cannot be reassigned
✓ Must be initialized
✓ Implicit dereferencing

Pointers:
✓ Can be null
✓ Can be reassigned
✓ Optional initialization
✓ Explicit dereferencing (*)
✓ Pointer arithmetic

WHEN TO USE:
- References: Function parameters, return values
- Pointers: Dynamic memory, nullable values, arrays

CONST REFERENCE:
const int& ref = value;
- Read-only access
- No copy made (efficient)
- Perfect for function parameters

FUNCTION PARAMETERS:
void func(int x)         // Pass by value (copy)
void func(int* x)        // Pass by pointer (can be null)
void func(int& x)        // Pass by reference (cleaner)
void func(const int& x)  // Const reference (efficient, read-only)

RETURNING REFERENCES:
- Can return reference to modify values
- NEVER return reference to local variable!
int& bad() {
    int x = 10;
    return x;  // DANGER: x destroyed after return!
}

GPU RELEVANCE:
- GPU kernels use pointers, not references
- References useful in host code
- Understanding both critical for GPU programming
- CPU code preparing GPU data uses references

RVALUE REFERENCES (C++11):
int&& rref = 10;  // Can bind to temporary
- Used in move semantics (Module 9)
- Advanced topic

TRY THIS:
1. Create a function that increments a value using reference
2. What happens if you try: int& ref; (without initialization)?
3. Create a function that returns largest element by reference
4. Compare performance: pass large struct by value vs const reference
5. Can you create a reference to a reference?
*/
