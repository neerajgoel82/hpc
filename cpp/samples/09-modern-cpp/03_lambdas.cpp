/*
 * MODULE 9: Modern C++
 * File: 03_lambdas.cpp
 *
 * TOPIC: Lambda Expressions
 *
 * COMPILE: g++ -std=c++17 -o lambdas 03_lambdas.cpp
 */

#include <iostream>
#include <vector>
#include <algorithm>
#include <functional>

int main() {
    std::cout << "=== Lambda Expressions Demo ===\n\n";

    // Basic lambda
    auto add = [](int a, int b) { return a + b; };
    std::cout << "add(3, 4) = " << add(3, 4) << "\n\n";

    // Lambda with STL algorithms
    std::vector<int> numbers{1, 2, 3, 4, 5};

    std::cout << "=== Transform (multiply by 2) ===\n";
    std::transform(numbers.begin(), numbers.end(), numbers.begin(),
                   [](int x) { return x * 2; });
    for (int x : numbers) std::cout << x << " ";
    std::cout << "\n\n";

    // Capture by value
    int multiplier = 3;
    auto multiplyByN = [multiplier](int x) { return x * multiplier; };
    std::cout << "Multiply 5 by " << multiplier << " = " << multiplyByN(5) << "\n\n";

    // Capture by reference
    int sum = 0;
    std::for_each(numbers.begin(), numbers.end(), [&sum](int x) {
        sum += x;
    });
    std::cout << "Sum: " << sum << "\n\n";

    // Generic lambda (C++14)
    auto print = [](const auto& val) {
        std::cout << val << " ";
    };
    print(42);
    print(3.14);
    print("hello");
    std::cout << "\n\n";

    // GPU-style operations with lambdas
    std::cout << "=== GPU-Style Parallel Operations ===\n";
    std::vector<float> positions{0, 1, 2, 3, 4};

    // Update positions (like GPU kernel)
    std::transform(positions.begin(), positions.end(), positions.begin(),
                   [](float x) { return x + 0.1f; });

    std::cout << "Updated positions: ";
    std::for_each(positions.begin(), positions.end(), print);
    std::cout << "\n\n";

    // Filter with lambda
    std::cout << "=== Filter (count even numbers) ===\n";
    std::vector<int> nums{1, 2, 3, 4, 5, 6, 7, 8};
    int evenCount = std::count_if(nums.begin(), nums.end(),
                                    [](int x) { return x % 2 == 0; });
    std::cout << "Even count: " << evenCount << "\n";

    std::cout << "\nGPU CONNECTION:\n";
    std::cout << "  - Thrust library uses lambdas extensively\n";
    std::cout << "  - CUDA device lambdas (C++11 in CUDA)\n";
    std::cout << "  - Inline kernel definitions\n";

    return 0;
}

/*
 * TRY THIS:
 * 1. Use lambda for custom sorting
 * 2. Implement map-reduce with lambdas
 * 3. Create callback system with lambdas
 * 4. Use mutable lambda to track state
 * 5. Implement event handlers with lambdas
 */
