/*
 * File: 01_hello_world.c
 * Topic: Your First C Program
 *
 * This is the traditional "Hello World" program - the first program
 * every C programmer writes. It demonstrates the basic structure of
 * a C program.
 *
 * Key Concepts:
 * - #include directive (bringing in standard library)
 * - main() function (entry point of every C program)
 * - printf() function (for output)
 * - return statement (exit status)
 *
 * To Run:
 * - Press Cmd+Shift+B and select "Build and Run"
 * - Or use terminal: gcc 01_hello_world.c -o hello && ./hello
 */

#include <stdio.h>  // Standard Input/Output library

// Every C program starts executing from the main() function
int main() {
    // printf() displays text to the console
    printf("Hello, World!\n");

    // return 0 indicates successful program execution
    // Non-zero values typically indicate errors
    return 0;
}

/*
 * EXERCISES:
 * 1. Change the message to print your name
 * 2. Add another printf() line to print "Welcome to C Programming!"
 * 3. Try removing the \n and see what happens
 * 4. What happens if you remove 'return 0;'? (Compile with -Wall)
 */
