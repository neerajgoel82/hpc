/*
 * MODULE 8: Standard Template Library (STL)
 * File: 01_vector_algorithms.cpp
 *
 * TOPIC: std::vector and STL Algorithms
 *
 * COMPILE: g++ -std=c++17 -o vector_algorithms 01_vector_algorithms.cpp
 */

#include <iostream>
#include <vector>
#include <algorithm>
#include <numeric>
#include <iterator>

void demonstrateVector() {
    std::cout << "\n=== std::vector Basics ===\n";

    // Creating vectors
    std::vector<int> v1;                    // Empty vector
    std::vector<int> v2(5);                 // 5 elements, default-initialized
    std::vector<int> v3(5, 10);             // 5 elements, all 10
    std::vector<int> v4{1, 2, 3, 4, 5};    // Initializer list

    // Adding elements
    v1.push_back(10);
    v1.push_back(20);
    v1.push_back(30);

    std::cout << "v1 size: " << v1.size() << "\n";
    std::cout << "v1 capacity: " << v1.capacity() << "\n";

    // Accessing elements
    std::cout << "v1[0] = " << v1[0] << "\n";
    std::cout << "v1.at(1) = " << v1.at(1) << "\n";  // Bounds-checked
    std::cout << "v1.front() = " << v1.front() << "\n";
    std::cout << "v1.back() = " << v1.back() << "\n";

    // Iterating
    std::cout << "v4 contents: ";
    for (const auto& val : v4) {
        std::cout << val << " ";
    }
    std::cout << "\n";
}

void demonstrateAlgorithms() {
    std::cout << "\n=== STL Algorithms ===\n";

    std::vector<int> data{5, 2, 8, 1, 9, 3, 7};

    // Sort
    std::sort(data.begin(), data.end());
    std::cout << "Sorted: ";
    for (int x : data) std::cout << x << " ";
    std::cout << "\n";

    // Find
    auto it = std::find(data.begin(), data.end(), 8);
    if (it != data.end()) {
        std::cout << "Found 8 at position " << std::distance(data.begin(), it) << "\n";
    }

    // Count
    std::vector<int> nums{1, 2, 3, 2, 4, 2, 5};
    int count = std::count(nums.begin(), nums.end(), 2);
    std::cout << "Count of 2: " << count << "\n";

    // Transform (multiply each by 2)
    std::transform(data.begin(), data.end(), data.begin(),
                   [](int x) { return x * 2; });
    std::cout << "After doubling: ";
    for (int x : data) std::cout << x << " ";
    std::cout << "\n";

    // Accumulate (sum)
    int sum = std::accumulate(data.begin(), data.end(), 0);
    std::cout << "Sum: " << sum << "\n";

    // Min/Max elements
    auto minIt = std::min_element(data.begin(), data.end());
    auto maxIt = std::max_element(data.begin(), data.end());
    std::cout << "Min: " << *minIt << ", Max: " << *maxIt << "\n";
}

void demonstrateGPUStyle() {
    std::cout << "\n=== GPU-Style Operations ===\n";

    // Particle positions
    std::vector<float> posX{0, 1, 2, 3, 4};
    std::vector<float> posY{0, 0, 0, 0, 0};

    // Update positions (like a simple GPU kernel)
    std::transform(posX.begin(), posX.end(), posX.begin(),
                   [](float x) { return x + 0.1f; });

    std::transform(posY.begin(), posY.end(), posY.begin(),
                   [](float y) { return y - 0.05f; });

    std::cout << "Updated positions:\n";
    for (size_t i = 0; i < posX.size(); ++i) {
        std::cout << "  Particle " << i << ": (" << posX[i] << ", " << posY[i] << ")\n";
    }

    std::cout << "\nGPU CONNECTION:\n";
    std::cout << "  - thrust::device_vector is like std::vector for GPU\n";
    std::cout << "  - thrust::transform maps to GPU kernels\n";
    std::cout << "  - STL patterns translate directly to Thrust\n";
}

int main() {
    std::cout << "=== MODULE 8: STL Vector and Algorithms ===\n";

    demonstrateVector();
    demonstrateAlgorithms();
    demonstrateGPUStyle();

    std::cout << "\n=== Program Complete ===\n";
    return 0;
}

/*
 * TRY THIS:
 * 1. Use std::remove_if to filter elements
 * 2. Implement parallel sort comparison
 * 3. Use std::partition for data organization
 * 4. Create custom comparator for sorting
 * 5. Implement reduce operation for sum of squares
 *
 * GPU CONNECTION:
 * - Thrust library mirrors STL design
 * - thrust::sort, thrust::transform, thrust::reduce
 * - Seamless CPU-GPU code transition
 */
