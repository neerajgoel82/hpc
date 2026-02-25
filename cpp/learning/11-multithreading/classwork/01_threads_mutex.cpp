/*
 * MODULE 11: Multithreading
 * File: 01_threads_mutex.cpp
 *
 * TOPIC: std::thread and std::mutex
 *
 * COMPILE: g++ -std=c++17 -pthread -o threads_mutex 01_threads_mutex.cpp
 */

#include <iostream>
#include <thread>
#include <mutex>
#include <vector>
#include <chrono>

std::mutex coutMutex;  // Protect std::cout

// Simple thread function
void printNumbers(int id, int count) {
    for (int i = 0; i < count; ++i) {
        std::lock_guard<std::mutex> lock(coutMutex);
        std::cout << "Thread " << id << ": " << i << "\n";
    }
}

// Shared counter example (demonstrates need for mutex)
int sharedCounter = 0;
std::mutex counterMutex;

void incrementCounter(int iterations) {
    for (int i = 0; i < iterations; ++i) {
        std::lock_guard<std::mutex> lock(counterMutex);
        ++sharedCounter;  // Protected by mutex
    }
}

// Simulate parallel particle update
struct Particle {
    float x, y, z;
    float vx, vy, vz;
};

void updateParticles(std::vector<Particle>& particles, size_t start, size_t end) {
    for (size_t i = start; i < end; ++i) {
        particles[i].x += particles[i].vx * 0.016f;
        particles[i].y += particles[i].vy * 0.016f;
        particles[i].z += particles[i].vz * 0.016f;
    }
}

int main() {
    std::cout << "=== Multithreading Demo ===\n\n";

    // Basic thread creation
    std::cout << "=== Creating Threads ===\n";
    std::thread t1(printNumbers, 1, 3);
    std::thread t2(printNumbers, 2, 3);

    t1.join();  // Wait for thread to finish
    t2.join();
    std::cout << "\n";

    // Shared counter with mutex
    std::cout << "=== Shared Counter (Protected) ===\n";
    sharedCounter = 0;
    std::vector<std::thread> threads;

    for (int i = 0; i < 4; ++i) {
        threads.emplace_back(incrementCounter, 10000);
    }

    for (auto& t : threads) {
        t.join();
    }

    std::cout << "Final counter value: " << sharedCounter << "\n";
    std::cout << "Expected: " << (4 * 10000) << "\n\n";

    // Parallel particle update
    std::cout << "=== Parallel Particle Update ===\n";
    const size_t numParticles = 1000;
    std::vector<Particle> particles(numParticles);

    // Initialize particles
    for (auto& p : particles) {
        p.x = p.y = p.z = 0;
        p.vx = p.vy = p.vz = 1.0f;
    }

    auto start = std::chrono::high_resolution_clock::now();

    // Split work across threads
    const int numThreads = 4;
    std::vector<std::thread> workers;
    size_t chunkSize = numParticles / numThreads;

    for (int i = 0; i < numThreads; ++i) {
        size_t startIdx = i * chunkSize;
        size_t endIdx = (i == numThreads - 1) ? numParticles : (i + 1) * chunkSize;
        workers.emplace_back(updateParticles, std::ref(particles), startIdx, endIdx);
    }

    for (auto& w : workers) {
        w.join();
    }

    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);

    std::cout << "Updated " << numParticles << " particles\n";
    std::cout << "Time: " << duration.count() << " microseconds\n";
    std::cout << "Using " << numThreads << " threads\n\n";

    std::cout << "GPU CONNECTION:\n";
    std::cout << "  - CPU multithreading for scene prep\n";
    std::cout << "  - GPU has thousands of threads!\n";
    std::cout << "  - Similar concepts: work distribution, synchronization\n";
    std::cout << "  - GPU: No mutex needed (different memory model)\n";

    return 0;
}

/*
 * TRY THIS:
 * 1. Implement producer-consumer pattern
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
