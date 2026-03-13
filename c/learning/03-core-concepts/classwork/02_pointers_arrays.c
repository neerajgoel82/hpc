/*
 * File: 02_pointers_arrays.c
 * Topic: Pointers and Arrays
 *
 * Arrays and pointers have a close relationship in C. An array name
 * is essentially a pointer to its first element.
 *
 * Key Concepts:
 * - Array name as pointer
 * - Pointer arithmetic
 * - Array access using pointers
 * - Pointers vs arrays
 * - Passing arrays to functions
 */

#include <stdio.h>

void printArrayUsingPointer(int *arr, int size);
void modifyArray(int *arr, int size);
int sumArray(int *arr, int size);

int main() {
    // Array and pointer relationship
    printf("=== Array Name as Pointer ===\n");
    int arr[] = {10, 20, 30, 40, 50};

    printf("arr = %p (address of first element)\n", (void*)arr);
    printf("&arr[0] = %p (same as arr)\n", (void*)&arr[0]);
    printf("*arr = %d (first element)\n", *arr);
    printf("arr[0] = %d (same as *arr)\n", arr[0]);

    // Pointer arithmetic
    printf("\n=== Pointer Arithmetic ===\n");
    int numbers[] = {100, 200, 300, 400, 500};
    int *ptr = numbers;  // Points to first element

    printf("ptr points to: %d\n", *ptr);
    printf("ptr + 1 points to: %d\n", *(ptr + 1));
    printf("ptr + 2 points to: %d\n", *(ptr + 2));

    // Incrementing pointer
    printf("\nIncrementing pointer:\n");
    ptr = numbers;  // Reset to start
    for (int i = 0; i < 5; i++) {
        printf("*ptr = %d (address: %p)\n", *ptr, (void*)ptr);
        ptr++;  // Move to next element
    }

    // Array access using pointers
    printf("\n=== Array Access Using Pointers ===\n");
    int data[] = {5, 10, 15, 20, 25};
    int *p = data;

    printf("Using array notation:\n");
    for (int i = 0; i < 5; i++) {
        printf("data[%d] = %d\n", i, data[i]);
    }

    printf("\nUsing pointer notation:\n");
    for (int i = 0; i < 5; i++) {
        printf("*(p + %d) = %d\n", i, *(p + i));
    }

    printf("\nUsing pointer arithmetic:\n");
    p = data;  // Reset pointer
    for (int i = 0; i < 5; i++) {
        printf("*p = %d\n", *p);
        p++;
    }

    // Equivalences
    printf("\n=== Pointer-Array Equivalences ===\n");
    int vals[] = {1, 2, 3};
    printf("vals[1] is same as *(vals + 1): %d = %d\n", vals[1], *(vals + 1));
    printf("&vals[1] is same as (vals + 1): %p = %p\n",
           (void*)&vals[1], (void*)(vals + 1));

    // Passing arrays to functions
    printf("\n=== Passing Arrays to Functions ===\n");
    int array[] = {10, 20, 30, 40, 50};
    int size = sizeof(array) / sizeof(array[0]);

    printf("Original array: ");
    printArrayUsingPointer(array, size);

    printf("Sum: %d\n", sumArray(array, size));

    modifyArray(array, size);
    printf("After modification: ");
    printArrayUsingPointer(array, size);

    // Pointer arithmetic with different types
    printf("\n=== Pointer Arithmetic with Different Types ===\n");
    int int_arr[] = {1, 2, 3};
    char char_arr[] = {'A', 'B', 'C'};

    int *ip = int_arr;
    char *cp = char_arr;

    printf("int pointer: %p -> %p (diff: %ld bytes)\n",
           (void*)ip, (void*)(ip + 1), (char*)(ip + 1) - (char*)ip);
    printf("char pointer: %p -> %p (diff: %ld bytes)\n",
           (void*)cp, (void*)(cp + 1), (cp + 1) - cp);

    // Pointer to array vs array of pointers
    printf("\n=== Pointer to Array ===\n");
    int my_array[5] = {1, 2, 3, 4, 5};
    int (*ptr_to_array)[5] = &my_array;  // Pointer to entire array

    printf("First element via pointer to array: %d\n", (*ptr_to_array)[0]);
    printf("Second element: %d\n", (*ptr_to_array)[1]);

    // String as character array
    printf("\n=== Strings and Pointers ===\n");
    char str[] = "Hello";
    char *str_ptr = str;

    printf("Using array: %s\n", str);
    printf("Using pointer: %s\n", str_ptr);

    printf("Characters using pointer:\n");
    while (*str_ptr != '\0') {
        printf("%c ", *str_ptr);
        str_ptr++;
    }
    printf("\n");

    return 0;
}

void printArrayUsingPointer(int *arr, int size) {
    for (int i = 0; i < size; i++) {
        printf("%d ", *(arr + i));
    }
    printf("\n");
}

void modifyArray(int *arr, int size) {
    for (int i = 0; i < size; i++) {
        arr[i] *= 2;  // Double each element
    }
}

int sumArray(int *arr, int size) {
    int sum = 0;
    for (int i = 0; i < size; i++) {
        sum += *(arr + i);
    }
    return sum;
}

/*
 * KEY RELATIONSHIPS:
 * arr[i]  ==  *(arr + i)
 * &arr[i] ==  (arr + i)
 * arr     ==  &arr[0]
 *
 * POINTER ARITHMETIC:
 * ptr + 1 moves forward by sizeof(*ptr) bytes
 * ptr - 1 moves backward by sizeof(*ptr) bytes
 * ptr1 - ptr2 gives number of elements between them
 *
 * IMPORTANT NOTES:
 * - Array name is a constant pointer (cannot be reassigned)
 * - When passing array to function, it decays to pointer
 * - Size information is lost when array decays to pointer
 * - Always pass size separately when passing arrays
 *
 * DIFFERENCES: ARRAYS VS POINTERS:
 * - Array: allocates space, cannot be reassigned
 * - Pointer: holds address, can be reassigned
 * - sizeof(array) gives total size
 * - sizeof(pointer) gives pointer size
 *
 * EXERCISES:
 * 1. Write a function to reverse an array using pointers
 * 2. Find maximum element in array using pointers
 * 3. Copy one array to another using pointers
 * 4. Implement string length function using pointers
 * 5. Search for element in array using pointers
 * 6. Concatenate two arrays using pointers
 * 7. Remove duplicates from array using pointers
 */
