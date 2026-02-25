// 04_control_flow.cpp
// Control flow: making decisions and repeating actions
// Compile: g++ -std=c++17 -o control_flow 04_control_flow.cpp

#include <iostream>

int main() {
    // ===== IF/ELSE Statements =====
    int temperature = 25;

    if (temperature > 30) {
        std::cout << "It's hot!" << std::endl;
    } else if (temperature > 20) {
        std::cout << "It's pleasant!" << std::endl;
    } else {
        std::cout << "It's cold!" << std::endl;
    }

    // Ternary operator (compact if/else)
    std::string weather = (temperature > 25) ? "warm" : "cool";
    std::cout << "Weather is " << weather << std::endl;

    // ===== SWITCH Statement =====
    int day = 3;

    switch (day) {
        case 1:
            std::cout << "Monday" << std::endl;
            break;  // Without break, execution continues to next case!
        case 2:
            std::cout << "Tuesday" << std::endl;
            break;
        case 3:
            std::cout << "Wednesday" << std::endl;
            break;
        case 4:
        case 5:  // Multiple cases can share code
            std::cout << "Thursday or Friday" << std::endl;
            break;
        default:  // Optional: handles all other cases
            std::cout << "Weekend!" << std::endl;
            break;
    }

    // ===== FOR Loop =====
    std::cout << "\nCounting 0 to 4:" << std::endl;
    for (int i = 0; i < 5; i++) {
        std::cout << i << " ";
    }
    std::cout << std::endl;

    // Loop in reverse
    std::cout << "Countdown: ";
    for (int i = 5; i > 0; i--) {
        std::cout << i << " ";
    }
    std::cout << "Liftoff!" << std::endl;

    // ===== WHILE Loop =====
    std::cout << "\nWhile loop (power of 2):" << std::endl;
    int power = 1;
    while (power <= 128) {
        std::cout << power << " ";
        power *= 2;
    }
    std::cout << std::endl;

    // ===== DO-WHILE Loop =====
    // Executes at least once (condition checked at end)
    int value = 10;
    do {
        std::cout << "Value: " << value << std::endl;
        value--;
    } while (value > 10);  // False on first check, but body ran once!

    // ===== BREAK and CONTINUE =====
    std::cout << "\nBreak example (stop at 5):" << std::endl;
    for (int i = 0; i < 10; i++) {
        if (i == 5) {
            break;  // Exit the loop immediately
        }
        std::cout << i << " ";
    }
    std::cout << std::endl;

    std::cout << "Continue example (skip even numbers):" << std::endl;
    for (int i = 0; i < 10; i++) {
        if (i % 2 == 0) {
            continue;  // Skip rest of loop body, go to next iteration
        }
        std::cout << i << " ";
    }
    std::cout << std::endl;

    // ===== Nested Loops =====
    std::cout << "\nMultiplication table (3x3):" << std::endl;
    for (int row = 1; row <= 3; row++) {
        for (int col = 1; col <= 3; col++) {
            std::cout << row * col << "\t";
        }
        std::cout << std::endl;
    }

    return 0;
}

/*
LEARNING NOTES:
- if/else: Make decisions based on conditions
- switch: Multiple conditions on same variable (faster than many if/else)
- for: Best when you know number of iterations
- while: Best when condition-based looping
- do-while: Like while, but always runs at least once
- break: Exit loop immediately
- continue: Skip to next iteration
- Comparison operators: ==, !=, <, >, <=, >=
- Logical operators: && (and), || (or), ! (not)

GPU RELEVANCE:
- Kernels often have loops over data
- Understanding loop patterns helps optimize GPU code
- Break/continue work in GPU kernels too
- Nested loops map to 2D/3D grid patterns in CUDA

TRY THIS:
1. Write a loop to print the first 10 Fibonacci numbers
2. Use a switch to convert a number (1-7) to day name
3. Create a nested loop to print a triangle pattern:
   *
   **
   ***
   ****
4. What happens if you remove 'break' from a switch case?
5. Write a loop that finds the first number divisible by both 7 and 13
*/
