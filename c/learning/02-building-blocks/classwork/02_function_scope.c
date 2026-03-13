/*
 * File: 02_function_scope.c
 * Topic: Variable Scope and Lifetime
 *
 * Understanding where variables can be accessed and how long they exist.
 *
 * Key Concepts:
 * - Local variables
 * - Global variables
 * - Function parameters
 * - Static variables
 * - Scope rules
 */

#include <stdio.h>

// Global variable - accessible from all functions
int global_count = 0;

// Function prototypes
void incrementGlobal(void);
void demonstrateLocal(void);
void demonstrateStatic(void);
int functionWithParameter(int param);

int main() {
    // Local variable - only accessible in main()
    int local_var = 10;

    printf("=== Global Variables ===\n");
    printf("Initial global_count: %d\n", global_count);
    incrementGlobal();
    printf("After incrementGlobal(): %d\n", global_count);
    incrementGlobal();
    printf("After incrementGlobal(): %d\n", global_count);

    printf("\n=== Local Variables ===\n");
    printf("local_var in main: %d\n", local_var);
    demonstrateLocal();
    demonstrateLocal();  // local_var in function doesn't persist

    printf("\n=== Static Variables ===\n");
    demonstrateStatic();
    demonstrateStatic();
    demonstrateStatic();  // static variable persists!

    printf("\n=== Function Parameters ===\n");
    int original = 5;
    printf("Before function call: %d\n", original);
    int result = functionWithParameter(original);
    printf("After function call: original = %d, result = %d\n", original, result);
    printf("(original didn't change - pass by value!)\n");

    return 0;
}

void incrementGlobal(void) {
    global_count++;
    printf("Inside incrementGlobal: global_count = %d\n", global_count);
}

void demonstrateLocal(void) {
    // Local variable - created each time function is called
    int local_var = 0;
    local_var++;
    printf("Local variable: %d\n", local_var);
    // local_var is destroyed when function returns
}

void demonstrateStatic(void) {
    // Static variable - retains value between function calls
    static int static_var = 0;
    static_var++;
    printf("Static variable: %d\n", static_var);
    // static_var persists between function calls
}

int functionWithParameter(int param) {
    // param is a local copy of the argument
    param = param * 2;
    printf("Inside function: param = %d\n", param);
    return param;
    // Changing param doesn't affect the original variable
}

/*
 * SCOPE RULES:
 *
 * Local Variables:
 * - Declared inside a function or block
 * - Only accessible within that function/block
 * - Created when function is called
 * - Destroyed when function returns
 *
 * Global Variables:
 * - Declared outside all functions
 * - Accessible from all functions
 * - Exist for entire program lifetime
 * - Use sparingly - can make code hard to understand
 *
 * Static Variables:
 * - Declared with 'static' keyword
 * - Retain value between function calls
 * - Only accessible within their scope
 * - Initialized only once
 *
 * Function Parameters:
 * - Act like local variables
 * - Receive copies of arguments (pass by value)
 * - Changes don't affect original variables
 *
 * BEST PRACTICES:
 * 1. Prefer local variables when possible
 * 2. Use global variables sparingly
 * 3. Use static when you need persistence
 * 4. Use meaningful variable names
 * 5. Keep scope as limited as possible
 *
 * EXERCISES:
 * 1. Create a function that counts how many times it's been called (use static)
 * 2. Write a program with global and local variables with the same name
 * 3. Create a function that uses a static variable to generate unique IDs
 * 4. Experiment with nested blocks and variable scope
 */
