/*
 * MODULE 8: STL
 * File: 02_map_set.cpp
 *
 * TOPIC: std::map and std::set
 *
 * COMPILE: g++ -std=c++17 -o map_set 02_map_set.cpp
 */

#include <iostream>
#include <map>
#include <unordered_map>
#include <set>
#include <string>

int main() {
    std::cout << "=== STL Map and Set ===\n\n";

    // std::map (ordered, log(n) operations)
    std::cout << "=== std::map ===\n";
    std::map<std::string, int> scores;
    scores["Alice"] = 95;
    scores["Bob"] = 87;
    scores["Charlie"] = 92;

    std::cout << "Alice's score: " << scores["Alice"] << "\n";

    // Iterate
    std::cout << "All scores:\n";
    for (const auto& [name, score] : scores) {
        std::cout << "  " << name << ": " << score << "\n";
    }

    // unordered_map (hash table, O(1) average)
    std::cout << "\n=== std::unordered_map ===\n";
    std::unordered_map<int, std::string> idToName;
    idToName[101] = "Texture1";
    idToName[102] = "Texture2";
    idToName[103] = "Mesh1";

    std::cout << "ID 102: " << idToName[102] << "\n";

    // std::set (unique, ordered)
    std::cout << "\n=== std::set ===\n";
    std::set<int> uniqueIDs{5, 2, 8, 2, 3, 5, 1};
    std::cout << "Unique IDs (sorted): ";
    for (int id : uniqueIDs) {
        std::cout << id << " ";
    }
    std::cout << "\n";

    // GPU relevance
    std::cout << "\nGPU CONNECTION:\n";
    std::cout << "  - Material/texture lookup tables\n";
    std::cout << "  - Object ID to data mapping\n";
    std::cout << "  - Unique vertex detection\n";
    std::cout << "  - Resource management\n";

    return 0;
}

/*
 * TRY THIS:
 * 1. Create texture cache using map
 * 2. Implement object pool with unordered_map
 * 3. Track unique vertices with set
 * 4. Build shader parameter map
 */
