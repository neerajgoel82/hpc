/*
 * File: 04_input_output.c
 * Topic: Input and Output in C
 *
 * Learn how to interact with users by reading input and displaying output.
 *
 * Key Concepts:
 * - printf() for formatted output
 * - scanf() for formatted input
 * - Common pitfalls with scanf()
 * - Format specifiers
 */

#include <stdio.h>

int main() {
    // Output with printf()
    printf("=== Basic Output with printf() ===\n");
    printf("This is a simple message.\n");
    printf("You can print numbers: %d\n", 42);
    printf("And decimals: %.2f\n", 3.14159);

    // Multiple values in one printf()
    printf("Name: %s, Age: %d, Height: %.1fcm\n", "John", 25, 175.5);

    // Input with scanf()
    printf("\n=== Getting Input with scanf() ===\n");

    int age;
    printf("Enter your age: ");
    scanf("%d", &age);  // Note the & (address-of operator)
    printf("You are %d years old.\n", age);

    // Reading floating-point numbers
    float height;
    printf("Enter your height in cm: ");
    scanf("%f", &height);
    printf("Your height is %.1f cm.\n", height);

    // Reading characters
    char grade;
    printf("Enter your grade (A-F): ");
    scanf(" %c", &grade);  // Note the space before %c to skip whitespace
    printf("Your grade is: %c\n", grade);

    // Reading multiple values
    int day, month, year;
    printf("Enter a date (DD MM YYYY): ");
    scanf("%d %d %d", &day, &month, &year);
    printf("Date entered: %02d/%02d/%d\n", day, month, year);

    // Formatted output examples
    printf("\n=== Formatted Output Examples ===\n");
    int num = 42;
    float pi = 3.14159265;

    printf("Default: %d\n", num);
    printf("Width 5: %5d\n", num);           // Right-aligned in 5 spaces
    printf("Width 5, left-aligned: %-5d|\n", num);

    printf("\nFloating-point formatting:\n");
    printf("Default: %f\n", pi);
    printf("2 decimals: %.2f\n", pi);
    printf("5 decimals: %.5f\n", pi);
    printf("Width 10, 2 decimals: %10.2f|\n", pi);

    return 0;
}

/*
 * IMPORTANT NOTES ABOUT scanf():
 * - Always use & before variable names (except for arrays/strings)
 * - scanf() leaves the newline character in the buffer
 * - Use a space before %c to skip whitespace: " %c"
 * - Check return value of scanf() in production code
 *
 * FORMAT SPECIFIERS:
 * %d   - int
 * %f   - float
 * %lf  - double (for scanf)
 * %c   - char
 * %s   - string
 * %02d - int with leading zeros, width 2
 * %.2f - float with 2 decimal places
 *
 * EXERCISES:
 * 1. Write a program that reads two numbers and prints their sum
 * 2. Create a simple calculator that reads two numbers and an operator (+, -, *, /)
 * 3. Read a temperature in Fahrenheit and convert to Celsius
 *    Formula: C = (F - 32) * 5/9
 * 4. Read a person's name (first and last) and age, then print a greeting
 * 5. Create a program that reads 3 test scores and calculates the average
 */
