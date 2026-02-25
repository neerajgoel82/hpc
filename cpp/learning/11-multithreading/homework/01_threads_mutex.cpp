/*
 * Homework: 01_threads_mutex.cpp
 *
 * Complete the exercises below based on the concepts from 01_threads_mutex.cpp
 * in the classwork folder.
 *
 * Instructions:
 * 1. Read the corresponding classwork file first
 * 2. Implement the solutions below
 * 3. Compile: g++ -std=c++17 -Wall -Wextra 01_threads_mutex.cpp -o homework
 * 4. Test your solutions
 */

#include <iostream>

/*
 * TRY THIS:
 * * 1. Implement producer-consumer pattern
 * 2. Add thread pool for task processing
 * 3. Use std::atomic for lock-free counter
 * 4. Implement parallel sort
 * 5. Create work-stealing scheduler
 *
 * GPU COMPARISON:
 * CPU Threads:
 *   - Dozens of threads
 *   - Heavy threads (1-2MB stack)
 *   - Complex synchronization (mutex, cv)
 *   - Context switching overhead
 *
 * GPU Threads:
 *   - Thousands/millions of threads
 *   - Lightweight threads (minimal state)
 *   - Barrier synchronization within block
 *   - No context switching (warp scheduling)
 */

int main() {
    std::cout << "Homework: 01_threads_mutex\n";
    std::cout << "Implement the exercises above\n";

    // Your code here

    return 0;
}
