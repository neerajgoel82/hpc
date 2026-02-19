// 05_strings.cpp
// Working with strings in C++
// Compile: g++ -std=c++17 -o strings 05_strings.cpp

#include <iostream>
#include <string>

int main() {
    // ===== Creating Strings =====
    std::string greeting = "Hello";
    std::string name("GPU Programmer");  // Constructor syntax
    std::string empty;  // Empty string ""

    std::cout << greeting << ", " << name << "!" << std::endl;

    // ===== String Concatenation =====
    std::string message = greeting + " " + name;  // Using + operator
    std::cout << message << std::endl;

    message += "!";  // Append to existing string
    std::cout << message << std::endl;

    // ===== String Length =====
    std::cout << "Length of message: " << message.length() << std::endl;
    std::cout << "Is empty string empty? " << (empty.empty() ? "yes" : "no") << std::endl;

    // ===== Accessing Characters =====
    std::string word = "CUDA";
    std::cout << "First letter: " << word[0] << std::endl;  // 'C'
    std::cout << "Last letter: " << word[word.length() - 1] << std::endl;  // 'A'

    // Safer access with .at() - throws exception if out of bounds
    std::cout << "Second letter: " << word.at(1) << std::endl;  // 'U'

    // ===== Modifying Strings =====
    word[0] = 'K';  // Change first letter
    std::cout << "Modified: " << word << std::endl;  // "KUDA"

    // ===== String Comparison =====
    std::string s1 = "apple";
    std::string s2 = "banana";
    std::string s3 = "apple";

    if (s1 == s3) {
        std::cout << s1 << " equals " << s3 << std::endl;
    }

    if (s1 < s2) {  // Lexicographic comparison
        std::cout << s1 << " comes before " << s2 << std::endl;
    }

    // ===== Substrings =====
    std::string full = "GPU Programming";
    std::string sub = full.substr(0, 3);  // Extract "GPU" (start at 0, length 3)
    std::cout << "Substring: " << sub << std::endl;

    // ===== Finding in Strings =====
    std::string text = "Learning C++ for GPU";
    size_t pos = text.find("GPU");  // Returns position of "GPU"

    if (pos != std::string::npos) {  // npos means "not found"
        std::cout << "Found 'GPU' at position: " << pos << std::endl;
    }

    if (text.find("CUDA") == std::string::npos) {
        std::cout << "'CUDA' not found in text" << std::endl;
    }

    // ===== String Methods =====
    std::string demo = "   spaces   ";

    // Get C-string (char array) - needed for some C APIs
    const char* c_str = demo.c_str();
    std::cout << "C-string: [" << c_str << "]" << std::endl;

    // Insert and erase
    std::string test = "Hello World";
    test.insert(5, " Beautiful");  // Insert at position 5
    std::cout << test << std::endl;

    test.erase(5, 10);  // Remove 10 characters starting at position 5
    std::cout << test << std::endl;

    // ===== Converting Numbers to Strings =====
    int num = 42;
    std::string num_str = std::to_string(num);
    std::cout << "Number as string: " << num_str << std::endl;

    // Converting Strings to Numbers
    std::string number_text = "123";
    int converted = std::stoi(number_text);  // string to int
    std::cout << "String to int: " << converted << std::endl;

    float float_val = std::stof("3.14");  // string to float
    std::cout << "String to float: " << float_val << std::endl;

    // ===== Iterating Over String =====
    std::string word2 = "GPU";
    std::cout << "Letters: ";
    for (size_t i = 0; i < word2.length(); i++) {
        std::cout << word2[i] << " ";
    }
    std::cout << std::endl;

    // Modern C++ range-based for loop (more on this in Module 9)
    std::cout << "Letters (range-based): ";
    for (char c : word2) {
        std::cout << c << " ";
    }
    std::cout << std::endl;

    return 0;
}

/*
LEARNING NOTES:
- std::string is C++'s string class (safer than C-style char arrays)
- Strings are mutable (can be changed after creation)
- String indices start at 0
- .length() and .size() are equivalent
- .find() returns std::string::npos if not found
- .substr(start, length) extracts a substring
- Can convert between strings and numbers with std::to_string, std::stoi, etc.

C-STRING vs C++ STRING:
- C-string: char arr[] = "hello";  (char array, null-terminated)
- C++ string: std::string s = "hello";  (safer, easier to use)
- Use .c_str() to get C-string from std::string when needed

GPU RELEVANCE:
- String processing typically done on CPU before GPU execution
- Filenames, configuration, and logging use strings
- GPU kernel names and parameters often manipulated as strings in host code

TRY THIS:
1. Create a program that asks user's name and greets them
2. Count how many times letter 'a' appears in a string
3. Reverse a string (hint: loop backwards)
4. Check if a word is a palindrome (reads same forwards and backwards)
5. Split a sentence into words (find spaces)
*/
