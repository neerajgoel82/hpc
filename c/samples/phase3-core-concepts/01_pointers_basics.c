/*
 * File: 01_pointers_basics.c
 * Topic: Introduction to Pointers
 *
 * Pointers are one of the most powerful and important features of C.
 * A pointer is a variable that stores the memory address of another variable.
 *
 * Key Concepts:
 * - What is a pointer?
 * - Pointer declaration
 * - Address-of operator (&)
 * - Dereference operator (*)
 * - NULL pointers
 */

#include <stdio.h>

int main() {
    // Understanding memory addresses
    printf("=== Memory Addresses ===\n");
    int num = 42;
    printf("Value of num: %d\n", num);
    printf("Address of num: %p\n", (void*)&num);
    printf("Size of num: %zu bytes\n", sizeof(num));

    // Pointer declaration and initialization
    printf("\n=== Pointer Basics ===\n");
    int value = 100;
    int *ptr;  // Declare a pointer to int

    ptr = &value;  // ptr now holds the address of value

    printf("value = %d\n", value);
    printf("Address of value = %p\n", (void*)&value);
    printf("ptr = %p (stores address of value)\n", (void*)ptr);
    printf("*ptr = %d (value at that address)\n", *ptr);

    // Dereferencing - accessing value through pointer
    printf("\n=== Dereferencing ===\n");
    int x = 10;
    int *p = &x;

    printf("x = %d\n", x);
    printf("*p = %d (accessing x through pointer)\n", *p);

    *p = 20;  // Modify x through pointer
    printf("After *p = 20:\n");
    printf("x = %d (changed!)\n", x);
    printf("*p = %d\n", *p);

    // Multiple pointers to same variable
    printf("\n=== Multiple Pointers ===\n");
    int number = 50;
    int *ptr1 = &number;
    int *ptr2 = &number;

    printf("number = %d\n", number);
    printf("*ptr1 = %d\n", *ptr1);
    printf("*ptr2 = %d\n", *ptr2);

    *ptr1 = 60;
    printf("\nAfter *ptr1 = 60:\n");
    printf("number = %d\n", number);
    printf("*ptr2 = %d (also changed!)\n", *ptr2);

    // Pointer to pointer
    printf("\n=== Pointer to Pointer ===\n");
    int val = 100;
    int *ptr_to_val = &val;
    int **ptr_to_ptr = &ptr_to_val;

    printf("val = %d\n", val);
    printf("*ptr_to_val = %d\n", *ptr_to_val);
    printf("**ptr_to_ptr = %d\n", **ptr_to_ptr);

    // NULL pointer
    printf("\n=== NULL Pointer ===\n");
    int *null_ptr = NULL;
    printf("null_ptr = %p\n", (void*)null_ptr);

    if (null_ptr == NULL) {
        printf("Pointer is NULL (not pointing to anything)\n");
    }

    // Pointer arithmetic introduction
    printf("\n=== Pointer Size ===\n");
    printf("Size of int: %zu bytes\n", sizeof(int));
    printf("Size of int*: %zu bytes\n", sizeof(int*));
    printf("Size of char*: %zu bytes\n", sizeof(char*));
    printf("Size of double*: %zu bytes\n", sizeof(double*));
    printf("(All pointer types have same size on same architecture)\n");

    // Practical example: Swap using pointers
    printf("\n=== Practical Example: Swap Function ===\n");
    int a = 5, b = 10;
    printf("Before swap: a = %d, b = %d\n", a, b);

    // Swap using pointers
    int *pa = &a;
    int *pb = &b;
    int temp = *pa;
    *pa = *pb;
    *pb = temp;

    printf("After swap: a = %d, b = %d\n", a, b);

    // Pointer safety
    printf("\n=== Pointer Safety ===\n");
    int *safe_ptr = NULL;

    // Always check before dereferencing
    if (safe_ptr != NULL) {
        printf("Value: %d\n", *safe_ptr);
    } else {
        printf("Cannot dereference NULL pointer\n");
    }

    return 0;
}

/*
 * POINTER SYNTAX:
 * int *ptr;     - Declares a pointer to int
 * ptr = &var;   - Assigns address of var to ptr
 * *ptr          - Dereferences ptr (accesses the value)
 * &var          - Gets address of var
 *
 * VISUAL REPRESENTATION:
 * int num = 42;
 * int *ptr = &num;
 *
 * Memory:
 * +----------+----------+
 * | num: 42  |  ptr: ---|----> points to num
 * +----------+----------+
 * Address:    0x1000      0x2000
 *
 * WHY USE POINTERS?
 * 1. Pass large data efficiently (pass address, not copy)
 * 2. Modify variables in functions (pass by reference)
 * 3. Dynamic memory allocation
 * 4. Data structures (linked lists, trees)
 * 5. Direct memory access
 *
 * COMMON MISTAKES:
 * - Dereferencing uninitialized pointers
 * - Dereferencing NULL pointers
 * - Losing the address (pointer goes out of scope)
 * - Confusing * in declaration vs dereferencing
 *
 * EXERCISES:
 * 1. Write a program to swap two numbers using pointers
 * 2. Create a function that takes a pointer and doubles the value
 * 3. Use pointers to find max of two numbers
 * 4. Write a function that uses pointers to swap characters
 * 5. Create pointers to different data types and print their sizes
 */
