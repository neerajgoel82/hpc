/*
 * File: 05_dynamic_memory.c
 * Topic: Dynamic Memory Allocation
 *
 * Dynamic memory allows you to allocate memory at runtime rather than
 * compile time. This is essential for flexible data structures.
 *
 * Key Concepts:
 * - malloc() - allocate memory
 * - calloc() - allocate and initialize to zero
 * - realloc() - resize allocated memory
 * - free() - deallocate memory
 * - Memory leaks and how to avoid them
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

int main() {
    // malloc - allocates uninitialized memory
    printf("=== malloc() ===\n");
    int *ptr = (int*)malloc(5 * sizeof(int));

    if (ptr == NULL) {
        printf("Memory allocation failed!\n");
        return 1;
    }

    printf("Memory allocated for 5 integers\n");

    // Assign values
    for (int i = 0; i < 5; i++) {
        ptr[i] = (i + 1) * 10;
    }

    printf("Values: ");
    for (int i = 0; i < 5; i++) {
        printf("%d ", ptr[i]);
    }
    printf("\n");

    // Always free allocated memory
    free(ptr);
    printf("Memory freed\n");

    // calloc - allocates and initializes to zero
    printf("\n=== calloc() ===\n");
    int *zeros = (int*)calloc(5, sizeof(int));

    if (zeros == NULL) {
        printf("Memory allocation failed!\n");
        return 1;
    }

    printf("Values (initialized to 0): ");
    for (int i = 0; i < 5; i++) {
        printf("%d ", zeros[i]);
    }
    printf("\n");

    free(zeros);

    // realloc - resize allocated memory
    printf("\n=== realloc() ===\n");
    int *arr = (int*)malloc(3 * sizeof(int));

    if (arr == NULL) {
        printf("Allocation failed!\n");
        return 1;
    }

    // Initialize
    for (int i = 0; i < 3; i++) {
        arr[i] = i + 1;
    }

    printf("Original array (size 3): ");
    for (int i = 0; i < 3; i++) {
        printf("%d ", arr[i]);
    }
    printf("\n");

    // Resize to hold 6 integers
    int *new_arr = (int*)realloc(arr, 6 * sizeof(int));

    if (new_arr == NULL) {
        printf("Reallocation failed!\n");
        free(arr);  // Free original memory
        return 1;
    }

    arr = new_arr;  // Update pointer

    // Fill new elements
    for (int i = 3; i < 6; i++) {
        arr[i] = i + 1;
    }

    printf("Resized array (size 6): ");
    for (int i = 0; i < 6; i++) {
        printf("%d ", arr[i]);
    }
    printf("\n");

    free(arr);

    // Dynamic array from user input
    printf("\n=== Dynamic Array from User Input ===\n");
    int n;
    printf("How many numbers do you want to enter? ");
    scanf("%d", &n);

    int *numbers = (int*)malloc(n * sizeof(int));

    if (numbers == NULL) {
        printf("Allocation failed!\n");
        return 1;
    }

    printf("Enter %d numbers:\n", n);
    for (int i = 0; i < n; i++) {
        printf("Number %d: ", i + 1);
        scanf("%d", &numbers[i]);
    }

    printf("You entered: ");
    for (int i = 0; i < n; i++) {
        printf("%d ", numbers[i]);
    }
    printf("\n");

    // Calculate average
    float sum = 0;
    for (int i = 0; i < n; i++) {
        sum += numbers[i];
    }
    printf("Average: %.2f\n", sum / n);

    free(numbers);

    // Dynamic string
    printf("\n=== Dynamic String ===\n");
    char *str = (char*)malloc(50 * sizeof(char));

    if (str == NULL) {
        printf("Allocation failed!\n");
        return 1;
    }

    strcpy(str, "Hello, Dynamic Memory!");
    printf("String: %s\n", str);

    free(str);

    // Dynamic 2D array
    printf("\n=== Dynamic 2D Array ===\n");
    int rows = 3, cols = 4;

    // Allocate array of pointers
    int **matrix = (int**)malloc(rows * sizeof(int*));

    if (matrix == NULL) {
        printf("Allocation failed!\n");
        return 1;
    }

    // Allocate each row
    for (int i = 0; i < rows; i++) {
        matrix[i] = (int*)malloc(cols * sizeof(int));
        if (matrix[i] == NULL) {
            printf("Allocation failed!\n");
            // Free previously allocated rows
            for (int j = 0; j < i; j++) {
                free(matrix[j]);
            }
            free(matrix);
            return 1;
        }
    }

    // Fill matrix
    int value = 1;
    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < cols; j++) {
            matrix[i][j] = value++;
        }
    }

    // Print matrix
    printf("Matrix:\n");
    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < cols; j++) {
            printf("%3d", matrix[i][j]);
        }
        printf("\n");
    }

    // Free 2D array
    for (int i = 0; i < rows; i++) {
        free(matrix[i]);
    }
    free(matrix);

    printf("\n=== Memory Management Best Practices ===\n");
    printf("1. Always check if allocation succeeded (ptr != NULL)\n");
    printf("2. Always free() what you malloc()\n");
    printf("3. Set pointer to NULL after freeing\n");
    printf("4. Don't access memory after freeing\n");
    printf("5. Don't lose the pointer (memory leak!)\n");
    printf("6. Use tools like valgrind to detect leaks\n");

    return 0;
}

/*
 * DYNAMIC MEMORY FUNCTIONS:
 *
 * malloc(size):
 * - Allocates 'size' bytes
 * - Returns pointer to allocated memory
 * - Memory is uninitialized
 * - Returns NULL if allocation fails
 *
 * calloc(count, size):
 * - Allocates memory for 'count' elements of 'size' bytes each
 * - Initializes all bytes to zero
 * - Returns pointer or NULL
 *
 * realloc(ptr, new_size):
 * - Resizes previously allocated memory
 * - May move memory to new location
 * - Preserves existing data
 * - Returns pointer to (possibly new) location or NULL
 *
 * free(ptr):
 * - Deallocates memory previously allocated
 * - Pointer becomes invalid after free
 * - Don't free the same pointer twice
 * - OK to free(NULL)
 *
 * COMMON MISTAKES:
 * 1. Memory leak - forgetting to free()
 * 2. Dangling pointer - using pointer after free()
 * 3. Double free - freeing same pointer twice
 * 4. Buffer overflow - writing beyond allocated size
 * 5. Not checking for allocation failure
 *
 * WHEN TO USE DYNAMIC MEMORY:
 * - Size not known at compile time
 * - Need large amounts of memory
 * - Need memory to persist beyond function scope
 * - Building data structures (linked lists, trees)
 * - Need to resize data structures
 *
 * EXERCISES:
 * 1. Create dynamic array of structures
 * 2. Implement dynamic string concatenation
 * 3. Build a resizable array (like vector in C++)
 * 4. Create function that returns dynamically allocated array
 * 5. Implement dynamic matrix multiplication
 * 6. Build a simple memory pool allocator
 * 7. Create a program that demonstrates a memory leak
 * 8. Fix the memory leak from previous exercise
 */
