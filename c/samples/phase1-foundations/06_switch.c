/*
 * File: 06_switch.c
 * Topic: Switch Statement
 *
 * The switch statement is used for multi-way decision making.
 * It's cleaner than multiple if-else statements when checking
 * a single variable against multiple constant values.
 *
 * Key Concepts:
 * - switch statement syntax
 * - case labels
 * - break statement
 * - default case
 * - Fall-through behavior
 */

#include <stdio.h>

int main() {
    // Basic switch statement
    printf("=== Basic Switch Statement ===\n");
    int day = 3;

    switch (day) {
        case 1:
            printf("Monday\n");
            break;
        case 2:
            printf("Tuesday\n");
            break;
        case 3:
            printf("Wednesday\n");
            break;
        case 4:
            printf("Thursday\n");
            break;
        case 5:
            printf("Friday\n");
            break;
        case 6:
            printf("Saturday\n");
            break;
        case 7:
            printf("Sunday\n");
            break;
        default:
            printf("Invalid day number\n");
    }

    // Switch with characters
    printf("\n=== Switch with Characters ===\n");
    char grade = 'B';

    switch (grade) {
        case 'A':
            printf("Excellent!\n");
            break;
        case 'B':
            printf("Good job!\n");
            break;
        case 'C':
            printf("Satisfactory.\n");
            break;
        case 'D':
            printf("Needs improvement.\n");
            break;
        case 'F':
            printf("Failed. Please study harder.\n");
            break;
        default:
            printf("Invalid grade.\n");
    }

    // Fall-through behavior (intentional)
    printf("\n=== Fall-through Behavior ===\n");
    int month = 2;

    printf("Month %d has ", month);
    switch (month) {
        case 1: case 3: case 5: case 7: case 8: case 10: case 12:
            printf("31 days.\n");
            break;
        case 4: case 6: case 9: case 11:
            printf("30 days.\n");
            break;
        case 2:
            printf("28 or 29 days.\n");
            break;
        default:
            printf("Invalid month.\n");
    }

    // Practical example: Simple calculator
    printf("\n=== Practical Example: Calculator ===\n");
    float num1, num2, result;
    char operator;

    printf("Enter first number: ");
    scanf("%f", &num1);
    printf("Enter operator (+, -, *, /): ");
    scanf(" %c", &operator);
    printf("Enter second number: ");
    scanf("%f", &num2);

    switch (operator) {
        case '+':
            result = num1 + num2;
            printf("%.2f + %.2f = %.2f\n", num1, num2, result);
            break;
        case '-':
            result = num1 - num2;
            printf("%.2f - %.2f = %.2f\n", num1, num2, result);
            break;
        case '*':
            result = num1 * num2;
            printf("%.2f * %.2f = %.2f\n", num1, num2, result);
            break;
        case '/':
            if (num2 != 0) {
                result = num1 / num2;
                printf("%.2f / %.2f = %.2f\n", num1, num2, result);
            } else {
                printf("Error: Division by zero!\n");
            }
            break;
        default:
            printf("Error: Invalid operator!\n");
    }

    // Menu-driven program example
    printf("\n=== Menu Example ===\n");
    int choice;

    printf("\n----- Menu -----\n");
    printf("1. Print Hello\n");
    printf("2. Print Goodbye\n");
    printf("3. Print Your Name\n");
    printf("4. Exit\n");
    printf("Enter your choice: ");
    scanf("%d", &choice);

    switch (choice) {
        case 1:
            printf("\nHello!\n");
            break;
        case 2:
            printf("\nGoodbye!\n");
            break;
        case 3:
            printf("\nMy name is C Program\n");
            break;
        case 4:
            printf("\nExiting...\n");
            break;
        default:
            printf("\nInvalid choice!\n");
    }

    return 0;
}

/*
 * IMPORTANT NOTES:
 * - switch only works with integer types and characters
 * - Cannot use strings or floating-point numbers
 * - break is crucial to prevent fall-through
 * - default is optional but recommended
 * - case values must be constants, not variables
 *
 * EXERCISES:
 * 1. Create a program that converts numbers 1-12 to month names
 * 2. Build a menu-driven program with 5 different options
 * 3. Write a vowel/consonant checker using switch
 * 4. Create a unit converter (km to miles, kg to pounds, etc.)
 * 5. Make a simple ATM menu (check balance, deposit, withdraw, exit)
 */
