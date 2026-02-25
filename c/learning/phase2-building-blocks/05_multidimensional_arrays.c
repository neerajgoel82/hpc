/*
 * File: 05_multidimensional_arrays.c
 * Topic: Multi-dimensional Arrays
 *
 * Multi-dimensional arrays are arrays of arrays. Most commonly used
 * are 2D arrays (matrices), but you can have 3D, 4D, etc.
 *
 * Key Concepts:
 * - 2D array declaration and initialization
 * - Accessing 2D array elements
 * - Matrix operations
 * - Passing 2D arrays to functions
 */

#include <stdio.h>

// Function prototypes
void print2DArray(int rows, int cols, int arr[rows][cols]);
void printMatrix(int rows, int cols, int matrix[rows][cols]);
void addMatrices(int rows, int cols, int a[rows][cols], int b[rows][cols], int result[rows][cols]);
void transposeMatrix(int rows, int cols, int matrix[rows][cols], int transposed[cols][rows]);

int main() {
    // 2D Array declaration and initialization
    printf("=== 2D Array Declaration ===\n");

    // Method 1: Full initialization
    int matrix[3][3] = {
        {1, 2, 3},
        {4, 5, 6},
        {7, 8, 9}
    };

    // Method 2: Partial initialization
    int grid[3][4] = {
        {1, 2},
        {3, 4, 5},
        {6}
    };  // Rest filled with 0

    // Method 3: Flat initialization
    int table[2][3] = {10, 20, 30, 40, 50, 60};

    // Accessing elements
    printf("\n=== Accessing 2D Array Elements ===\n");
    printf("matrix[0][0] = %d\n", matrix[0][0]);
    printf("matrix[1][2] = %d\n", matrix[1][2]);
    printf("matrix[2][2] = %d\n", matrix[2][2]);

    // Printing 2D array
    printf("\n=== Printing 2D Array ===\n");
    printf("Matrix:\n");
    printMatrix(3, 3, matrix);

    printf("\nGrid:\n");
    printMatrix(3, 4, grid);

    // Modifying elements
    printf("\n=== Modifying 2D Array ===\n");
    matrix[1][1] = 99;
    printf("After setting matrix[1][1] = 99:\n");
    printMatrix(3, 3, matrix);

    // Reading 2D array from user
    printf("\n=== User Input for 2D Array ===\n");
    int user_matrix[2][2];
    printf("Enter elements for 2x2 matrix:\n");
    for (int i = 0; i < 2; i++) {
        for (int j = 0; j < 2; j++) {
            printf("Element [%d][%d]: ", i, j);
            scanf("%d", &user_matrix[i][j]);
        }
    }
    printf("\nYou entered:\n");
    printMatrix(2, 2, user_matrix);

    // Matrix addition
    printf("\n=== Matrix Addition ===\n");
    int mat1[2][2] = {{1, 2}, {3, 4}};
    int mat2[2][2] = {{5, 6}, {7, 8}};
    int sum[2][2];

    printf("Matrix 1:\n");
    printMatrix(2, 2, mat1);
    printf("\nMatrix 2:\n");
    printMatrix(2, 2, mat2);

    addMatrices(2, 2, mat1, mat2, sum);
    printf("\nSum:\n");
    printMatrix(2, 2, sum);

    // Matrix transpose
    printf("\n=== Matrix Transpose ===\n");
    int original[2][3] = {{1, 2, 3}, {4, 5, 6}};
    int transposed[3][2];

    printf("Original (2x3):\n");
    printMatrix(2, 3, original);

    transposeMatrix(2, 3, original, transposed);
    printf("\nTransposed (3x2):\n");
    printMatrix(3, 2, transposed);

    // 3D Array example
    printf("\n=== 3D Array ===\n");
    int cube[2][2][2] = {
        {{1, 2}, {3, 4}},
        {{5, 6}, {7, 8}}
    };

    printf("3D Array (2x2x2):\n");
    for (int i = 0; i < 2; i++) {
        printf("Layer %d:\n", i);
        for (int j = 0; j < 2; j++) {
            for (int k = 0; k < 2; k++) {
                printf("%d ", cube[i][j][k]);
            }
            printf("\n");
        }
    }

    return 0;
}

// Print 2D array as matrix
void printMatrix(int rows, int cols, int matrix[rows][cols]) {
    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < cols; j++) {
            printf("%4d", matrix[i][j]);
        }
        printf("\n");
    }
}

// Add two matrices
void addMatrices(int rows, int cols, int a[rows][cols], int b[rows][cols], int result[rows][cols]) {
    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < cols; j++) {
            result[i][j] = a[i][j] + b[i][j];
        }
    }
}

// Transpose matrix
void transposeMatrix(int rows, int cols, int matrix[rows][cols], int transposed[cols][rows]) {
    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < cols; j++) {
            transposed[j][i] = matrix[i][j];
        }
    }
}

/*
 * MEMORY LAYOUT:
 * 2D array int arr[3][4] is stored as:
 * arr[0][0], arr[0][1], arr[0][2], arr[0][3],
 * arr[1][0], arr[1][1], arr[1][2], arr[1][3],
 * arr[2][0], arr[2][1], arr[2][2], arr[2][3]
 * (Row-major order)
 *
 * IMPORTANT NOTES:
 * - int arr[3][4] means 3 rows and 4 columns
 * - First index is row, second is column
 * - Total elements = rows * columns
 * - When passing to functions, all dimensions except first must be specified
 *
 * EXERCISES:
 * 1. Write a function to multiply two matrices
 * 2. Find the sum of diagonal elements in a square matrix
 * 3. Check if a matrix is symmetric
 * 4. Find the largest element in a 2D array
 * 5. Rotate a matrix 90 degrees clockwise
 * 6. Print a matrix in spiral order
 * 7. Check if a matrix is an identity matrix
 * 8. Implement tic-tac-toe using a 3x3 array
 */
