/*
 * MODULE 11: Multithreading
 * File: 02_atomics.cpp
 *
 * TOPIC: std::atomic for Lock-Free Programming
 *
 * COMPILE: g++ -std=c++17 -pthread -o atomics 02_atomics.cpp
 */

#include <iostream>
#include <thread>
#include <atomic>
#include <vector>

std::atomic<int> atomicCounter{0};
std::atomic<bool> done{false};

void atomicIncrement(int iterations) {
    for (int i = 0; i < iterations; ++i) {
        atomicCounter.fetch_add(1, std::memory_order_relaxed);
    }
}

int main() {
    std::cout << "=== Atomic Operations Demo ===\n\n";

    // Atomic counter (no mutex needed!)
    std::cout << "=== Lock-Free Counter ===\n";
    std::vector<std::thread> threads;

    for (int i = 0; i < 4; ++i) {
        threads.emplace_back(atomicIncrement, 10000);
    }

    for (auto& t : threads) {
        t.join();
    }

    std::cout << "Final atomic counter: " << atomicCounter.load() << "\n";
    std::cout << "Expected: 40000\n\n";

    // Atomic flag for signaling
    std::cout << "=== Atomic Flag ===\n";
    std::thread worker([&]() {
        std::cout << "Worker: Processing...\n";
        std::this_thread::sleep_for(std::chrono::milliseconds(100));
        done.store(true);
        std::cout << "Worker: Done!\n";
    });

    while (!done.load()) {
        std::cout << "Main: Waiting...\n";
        std::this_thread::sleep_for(std::chrono::milliseconds(50));
    }

    worker.join();
    std::cout << "Main: Worker completed!\n\n";

    std::cout << "GPU CONNECTION:\n";
    std::cout << "  - GPU has atomic operations: atomicAdd, atomicCAS\n";
    std::cout << "  - Used for histograms, reductions, counters\n";
    std::cout << "  - Important for avoiding race conditions\n";
    std::cout << "  - Slower than regular ops, but thread-safe\n";

    return 0;
}

/*
 * TRY THIS:
 * 1. Implement lock-free queue with atomics
 * 2. Create spinlock using atomic flag
 * 3. Build atomic histogram
 * 4. Compare mutex vs atomic performance
 * 5. Implement atomic max/min operations
 */
