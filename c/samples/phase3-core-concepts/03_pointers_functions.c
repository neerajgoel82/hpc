/*
 * File: 03_pointers_functions.c
 * Topic: Pointers and Functions
 *
 * Pointers enable functions to modify variables, return multiple values,
 * and work with data more efficiently.
 *
 * Key Concepts:
 * - Pass by value vs pass by reference
 * - Returning pointers from functions
 * - Function pointers
 * - Callback functions
 */

#include <stdio.h>

// Function prototypes
void swapByValue(int a, int b);
void swapByReference(int *a, int *b);
void modifyValue(int *ptr);
void getDivisionResult(int dividend, int divisor, int *quotient, int *remainder);
int* getArrayElement(int arr[], int index);
void applyOperation(int *arr, int size, void (*operation)(int*));
void doubleValue(int *val);
void squareValue(int *val);

int main() {
    // Pass by value - doesn't modify original
    printf("=== Pass by Value ===\n");
    int x = 10, y = 20;
    printf("Before swap: x = %d, y = %d\n", x, y);
    swapByValue(x, y);
    printf("After swap: x = %d, y = %d (unchanged!)\n", x, y);

    // Pass by reference - modifies original
    printf("\n=== Pass by Reference (Pointers) ===\n");
    x = 10; y = 20;
    printf("Before swap: x = %d, y = %d\n", x, y);
    swapByReference(&x, &y);
    printf("After swap: x = %d, y = %d (swapped!)\n", x, y);

    // Modifying value through pointer
    printf("\n=== Modifying Through Pointer ===\n");
    int value = 5;
    printf("Before: value = %d\n", value);
    modifyValue(&value);
    printf("After: value = %d\n", value);

    // Returning multiple values
    printf("\n=== Returning Multiple Values ===\n");
    int quotient, remainder;
    getDivisionResult(17, 5, &quotient, &remainder);
    printf("17 / 5 = %d remainder %d\n", quotient, remainder);

    // Returning pointer from function
    printf("\n=== Returning Pointer ===\n");
    int arr[] = {10, 20, 30, 40, 50};
    int *element_ptr = getArrayElement(arr, 2);
    printf("Element at index 2: %d\n", *element_ptr);

    *element_ptr = 99;  // Modify through returned pointer
    printf("After modification, arr[2] = %d\n", arr[2]);

    // Function pointers
    printf("\n=== Function Pointers ===\n");
    int numbers[] = {1, 2, 3, 4, 5};
    int size = 5;

    printf("Original array: ");
    for (int i = 0; i < size; i++) printf("%d ", numbers[i]);
    printf("\n");

    // Pass doubleValue function as parameter
    applyOperation(numbers, size, doubleValue);
    printf("After doubling: ");
    for (int i = 0; i < size; i++) printf("%d ", numbers[i]);
    printf("\n");

    // Pass squareValue function as parameter
    applyOperation(numbers, size, squareValue);
    printf("After squaring: ");
    for (int i = 0; i < size; i++) printf("%d ", numbers[i]);
    printf("\n");

    // Direct function pointer usage
    printf("\n=== Direct Function Pointer Usage ===\n");
    void (*func_ptr)(int*);  // Declare function pointer

    int num = 10;
    printf("Original: %d\n", num);

    func_ptr = doubleValue;  // Point to doubleValue
    func_ptr(&num);
    printf("After doubleValue: %d\n", num);

    func_ptr = squareValue;  // Point to squareValue
    func_ptr(&num);
    printf("After squareValue: %d\n", num);

    return 0;
}

// Pass by value - receives copies
void swapByValue(int a, int b) {
    int temp = a;
    a = b;
    b = temp;
    printf("Inside swapByValue: a = %d, b = %d\n", a, b);
    // Changes are local, don't affect original variables
}

// Pass by reference - receives addresses
void swapByReference(int *a, int *b) {
    int temp = *a;
    *a = *b;
    *b = temp;
    // Changes affect original variables through pointers
}

// Modify value through pointer
void modifyValue(int *ptr) {
    *ptr = *ptr * 10;
}

// Return multiple values using pointers
void getDivisionResult(int dividend, int divisor, int *quotient, int *remainder) {
    *quotient = dividend / divisor;
    *remainder = dividend % divisor;
}

// Return pointer to array element
int* getArrayElement(int arr[], int index) {
    return &arr[index];  // Return address of element
}

// Higher-order function - takes function as parameter
void applyOperation(int *arr, int size, void (*operation)(int*)) {
    for (int i = 0; i < size; i++) {
        operation(&arr[i]);  // Call the function passed as parameter
    }
}

// Operations to be passed as callbacks
void doubleValue(int *val) {
    *val = *val * 2;
}

void squareValue(int *val) {
    *val = *val * *val;
}

/*
 * PASS BY VALUE vs PASS BY REFERENCE:
 *
 * Pass by Value:
 * - Function receives copy of the value
 * - Changes don't affect original
 * - void func(int x) { x = 10; }
 *
 * Pass by Reference (using pointers):
 * - Function receives address of variable
 * - Changes affect original variable
 * - void func(int *x) { *x = 10; }
 *
 * WHY USE POINTERS IN FUNCTIONS?
 * 1. Modify original variables
 * 2. Return multiple values
 * 3. Avoid copying large data structures
 * 4. Work with arrays efficiently
 * 5. Implement callbacks and higher-order functions
 *
 * FUNCTION POINTER SYNTAX:
 * return_type (*pointer_name)(parameter_types);
 *
 * Example:
 * void (*func_ptr)(int*);  // Pointer to function taking int* and returning void
 * func_ptr = myFunction;   // Assign function to pointer
 * func_ptr(&x);            // Call function through pointer
 *
 * IMPORTANT WARNINGS:
 * - Don't return pointer to local variable (it's destroyed!)
 * - Always validate pointer parameters (check for NULL)
 * - Be careful with function pointer types (must match exactly)
 *
 * EXERCISES:
 * 1. Write a function that finds min and max using pointers
 * 2. Create a function to sort array using pointers
 * 3. Implement a calculator with function pointers for operations
 * 4. Write a function that returns pointer to largest array element
 * 5. Create array of function pointers for a menu system
 * 6. Implement map function that applies operation to each element
 * 7. Write function that filters array based on a predicate function
 */
