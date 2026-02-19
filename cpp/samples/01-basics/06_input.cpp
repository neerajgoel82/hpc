// 06_input.cpp
// Getting input from users
// Compile: g++ -std=c++17 -o input 06_input.cpp

#include <iostream>
#include <string>
#include <limits>  // For numeric_limits

int main() {
    // ===== Basic Input with cin =====
    std::cout << "What is your name? ";
    std::string name;
    std::cin >> name;  // Reads until whitespace (space, tab, newline)
    std::cout << "Hello, " << name << "!" << std::endl;

    // ===== Reading Numbers =====
    std::cout << "Enter your age: ";
    int age;
    std::cin >> age;
    std::cout << "You are " << age << " years old." << std::endl;

    // ===== Multiple Inputs =====
    std::cout << "Enter width and height: ";
    int width, height;
    std::cin >> width >> height;  // Chain multiple inputs
    std::cout << "Area: " << width * height << std::endl;

    // ===== Reading Full Lines (with spaces) =====
    // Need to clear the buffer first (leftover newline from previous cin)
    std::cin.ignore(std::numeric_limits<std::streamsize>::max(), '\n');

    std::cout << "Enter your full address: ";
    std::string address;
    std::getline(std::cin, address);  // Reads entire line including spaces
    std::cout << "Address: " << address << std::endl;

    // ===== Reading Characters =====
    std::cout << "Enter a single character: ";
    char letter;
    std::cin >> letter;
    std::cout << "You entered: " << letter << std::endl;

    // ===== Reading Floating Point =====
    std::cout << "Enter a decimal number: ";
    float decimal;
    std::cin >> decimal;
    std::cout << "You entered: " << decimal << std::endl;

    // ===== Input Validation Example =====
    std::cout << "\n=== Input Validation ===" << std::endl;
    std::cout << "Enter a positive number: ";
    int number;

    if (std::cin >> number) {  // Check if input succeeded
        if (number > 0) {
            std::cout << "Valid input: " << number << std::endl;
        } else {
            std::cout << "Number must be positive!" << std::endl;
        }
    } else {
        std::cout << "Invalid input! Not a number." << std::endl;
        std::cin.clear();  // Clear error state
        std::cin.ignore(std::numeric_limits<std::streamsize>::max(), '\n');  // Discard bad input
    }

    // ===== Simple Calculator Example =====
    std::cout << "\n=== Simple Calculator ===" << std::endl;
    std::cout << "Enter first number: ";
    double num1;
    std::cin >> num1;

    std::cout << "Enter operator (+, -, *, /): ";
    char op;
    std::cin >> op;

    std::cout << "Enter second number: ";
    double num2;
    std::cin >> num2;

    double result;
    bool valid = true;

    switch (op) {
        case '+':
            result = num1 + num2;
            break;
        case '-':
            result = num1 - num2;
            break;
        case '*':
            result = num1 * num2;
            break;
        case '/':
            if (num2 != 0) {
                result = num1 / num2;
            } else {
                std::cout << "Error: Division by zero!" << std::endl;
                valid = false;
            }
            break;
        default:
            std::cout << "Error: Invalid operator!" << std::endl;
            valid = false;
    }

    if (valid) {
        std::cout << num1 << " " << op << " " << num2 << " = " << result << std::endl;
    }

    return 0;
}

/*
LEARNING NOTES:
- std::cin >> variable: Reads input (stops at whitespace)
- std::getline(std::cin, string): Reads entire line including spaces
- std::cin.ignore(): Clears input buffer
- Check if (std::cin >> var) to validate input succeeded
- std::cin.clear() and std::cin.ignore() to recover from bad input

INPUT OPERATORS:
- >> : Extraction operator (skips leading whitespace)
- getline(): Gets entire line

COMMON ISSUES:
- cin >> leaves newline in buffer (use ignore() before getline())
- Failed input puts cin in error state (use clear() to reset)
- Wrong data type causes input to fail

GPU RELEVANCE:
- User input typically happens on CPU/host side
- Configuration parameters, file paths, sizes often come from user input
- GPU kernel parameters might be derived from user input

TRY THIS:
1. Create a program that asks for temperature in Celsius and converts to Fahrenheit
2. Read 3 numbers and find the largest
3. Ask for user's birth year and calculate their age
4. Create a guessing game (program picks random number, user guesses)
5. Handle invalid input gracefully (what if user enters letters instead of numbers?)

EXAMPLE RUN:
What is your name? John
Hello, John!
Enter your age: 25
You are 25 years old.
*/
