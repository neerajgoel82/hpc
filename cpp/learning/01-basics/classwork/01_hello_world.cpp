// 01_hello_world.cpp
// Your first C++ program
// Compile: g++ -std=c++17 -o hello 01_hello_world.cpp
// Run: ./hello

#include <iostream>

int main() {
    std::cout << "Hello, GPU Programming!" << std::endl;
    return 0;
}

/*
LEARNING NOTES:
- #include <iostream>: Imports input/output stream library
- int main(): Entry point of every C++ program
- std::cout: Standard output stream (like print in Python)
- << operator: Inserts data into the stream
- std::endl: Ends line and flushes buffer
- return 0: Indicates successful execution

TRY THIS:
1. Change the message
2. Add multiple cout statements
3. Remove std::endl and observe the difference
*/
