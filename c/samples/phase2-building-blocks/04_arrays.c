/*
 * File: 04_arrays.c
 * Topic: Arrays in C
 *
 * Arrays are collections of elements of the same type stored in
 * contiguous memory locations. They allow you to work with multiple
 * related values efficiently.
 *
 * Key Concepts:
 * - Array declaration and initialization
 * - Accessing array elements
 * - Array bounds
 * - Passing arrays to functions
 * - Common array operations
 */

#include <stdio.h>

// Function prototypes
void printArray(int arr[], int size);
int findMax(int arr[], int size);
int findMin(int arr[], int size);
float calculateAverage(int arr[], int size);
void reverseArray(int arr[], int size);

int main() {
    // Array declaration and initialization
    printf("=== Array Declaration ===\n");

    // Method 1: Declare size and initialize
    int numbers[5] = {10, 20, 30, 40, 50};

    // Method 2: Compiler determines size
    int scores[] = {85, 92, 78, 95, 88};

    // Method 3: Partial initialization (rest are 0)
    int values[5] = {1, 2};  // {1, 2, 0, 0, 0}

    // Accessing array elements (0-indexed)
    printf("numbers[0] = %d\n", numbers[0]);
    printf("numbers[2] = %d\n", numbers[2]);
    printf("numbers[4] = %d\n", numbers[4]);

    // Modifying array elements
    printf("\n=== Modifying Arrays ===\n");
    numbers[2] = 35;
    printf("After modification, numbers[2] = %d\n", numbers[2]);

    // Iterating through array
    printf("\n=== Printing Array with Loop ===\n");
    printf("Numbers: ");
    for (int i = 0; i < 5; i++) {
        printf("%d ", numbers[i]);
    }
    printf("\n");

    // Using helper function
    printf("\nScores: ");
    printArray(scores, 5);

    // Array size calculation
    printf("\n=== Array Size ===\n");
    int my_array[] = {1, 2, 3, 4, 5, 6, 7, 8};
    int array_size = sizeof(my_array) / sizeof(my_array[0]);
    printf("Array size: %d elements\n", array_size);

    // Finding maximum and minimum
    printf("\n=== Finding Max and Min ===\n");
    int data[] = {45, 23, 89, 12, 67, 34, 91};
    int size = sizeof(data) / sizeof(data[0]);
    printf("Array: ");
    printArray(data, size);
    printf("Maximum: %d\n", findMax(data, size));
    printf("Minimum: %d\n", findMin(data, size));
    printf("Average: %.2f\n", calculateAverage(data, size));

    // Reversing array
    printf("\n=== Reversing Array ===\n");
    int to_reverse[] = {1, 2, 3, 4, 5};
    int rev_size = sizeof(to_reverse) / sizeof(to_reverse[0]);
    printf("Original: ");
    printArray(to_reverse, rev_size);
    reverseArray(to_reverse, rev_size);
    printf("Reversed: ");
    printArray(to_reverse, rev_size);

    // Reading array from user
    printf("\n=== User Input Array ===\n");
    int user_array[5];
    printf("Enter 5 numbers:\n");
    for (int i = 0; i < 5; i++) {
        printf("Element %d: ", i + 1);
        scanf("%d", &user_array[i]);
    }
    printf("You entered: ");
    printArray(user_array, 5);

    return 0;
}

// Print array elements
void printArray(int arr[], int size) {
    for (int i = 0; i < size; i++) {
        printf("%d ", arr[i]);
    }
    printf("\n");
}

// Find maximum element
int findMax(int arr[], int size) {
    int max = arr[0];
    for (int i = 1; i < size; i++) {
        if (arr[i] > max) {
            max = arr[i];
        }
    }
    return max;
}

// Find minimum element
int findMin(int arr[], int size) {
    int min = arr[0];
    for (int i = 1; i < size; i++) {
        if (arr[i] < min) {
            min = arr[i];
        }
    }
    return min;
}

// Calculate average
float calculateAverage(int arr[], int size) {
    int sum = 0;
    for (int i = 0; i < size; i++) {
        sum += arr[i];
    }
    return (float)sum / size;
}

// Reverse array in place
void reverseArray(int arr[], int size) {
    for (int i = 0; i < size / 2; i++) {
        // Swap elements
        int temp = arr[i];
        arr[i] = arr[size - 1 - i];
        arr[size - 1 - i] = temp;
    }
}

/*
 * IMPORTANT NOTES:
 * - Array indices start at 0
 * - Array size is fixed at compile time
 * - No automatic bounds checking (be careful!)
 * - Arrays are passed to functions by reference (modifications persist)
 * - sizeof(array) gives total bytes, not number of elements
 *
 * COMMON MISTAKES:
 * - Accessing out of bounds: arr[10] when size is 10
 * - Forgetting to pass array size to functions
 * - Trying to return an array directly from a function
 *
 * EXERCISES:
 * 1. Write a function to search for an element in an array (linear search)
 * 2. Implement bubble sort to sort an array
 * 3. Find the second largest element in an array
 * 4. Copy one array to another
 * 5. Merge two sorted arrays into one sorted array
 * 6. Remove duplicates from an array
 * 7. Rotate array to the right by k positions
 * 8. Find all pairs in array that sum to a given value
 */
