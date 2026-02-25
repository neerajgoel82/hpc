/*
 * MODULE 13: GPU Advanced Topics
 * File: 01_aos_vs_soa.cpp
 *
 * TOPIC: Array of Structures (AoS) vs Structure of Arrays (SoA)
 *
 * CONCEPTS:
 * - Memory layout impacts GPU performance dramatically
 * - AoS: Natural for CPU, poor for GPU (coalesced access)
 * - SoA: Better for GPU (coalesced access, vectorization)
 * - Cache efficiency and memory bandwidth
 *
 * GPU RELEVANCE:
 * - CRITICAL for GPU performance
 * - Coalesced memory access on GPU
 * - SIMD/vector operations
 * - Particle systems, physics simulations
 *
 * COMPILE: g++ -std=c++17 -O3 -o aos_vs_soa 01_aos_vs_soa.cpp
 */

#include <iostream>
#include <vector>
#include <chrono>
#include <cmath>

// ========================================
// Array of Structures (AoS)
// ========================================
struct ParticleAoS {
    float x, y, z;      // Position
    float vx, vy, vz;   // Velocity
    float mass;
    int id;
};

void updateAoS(std::vector<ParticleAoS>& particles, float dt) {
    for (auto& p : particles) {
        // Update velocity (gravity)
        p.vy -= 9.8f * dt;

        // Update position
        p.x += p.vx * dt;
        p.y += p.vy * dt;
        p.z += p.vz * dt;
    }
}

// ========================================
// Structure of Arrays (SoA)
// ========================================
struct ParticlesSoA {
    std::vector<float> x, y, z;      // Positions
    std::vector<float> vx, vy, vz;   // Velocities
    std::vector<float> mass;
    std::vector<int> id;

    void resize(size_t n) {
        x.resize(n); y.resize(n); z.resize(n);
        vx.resize(n); vy.resize(n); vz.resize(n);
        mass.resize(n);
        id.resize(n);
    }

    size_t size() const { return x.size(); }
};

void updateSoA(ParticlesSoA& particles, float dt) {
    size_t n = particles.size();

    // Update velocities (contiguous memory access)
    for (size_t i = 0; i < n; ++i) {
        particles.vy[i] -= 9.8f * dt;
    }

    // Update positions (contiguous memory access)
    for (size_t i = 0; i < n; ++i) {
        particles.x[i] += particles.vx[i] * dt;
        particles.y[i] += particles.vy[i] * dt;
        particles.z[i] += particles.vz[i] * dt;
    }
}

// Benchmark function
template <typename Func>
double benchmark(Func func, int iterations) {
    auto start = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < iterations; ++i) {
        func();
    }
    auto end = std::chrono::high_resolution_clock::now();
    return std::chrono::duration<double, std::milli>(end - start).count();
}

void demonstrateMemoryLayout() {
    std::cout << "\n=== Memory Layout Visualization ===\n\n";

    std::cout << "Array of Structures (AoS):\n";
    std::cout << "Memory: [x0 y0 z0 vx0 vy0 vz0 m0 id0][x1 y1 z1 vx1 vy1 vz1 m1 id1]...\n";
    std::cout << "        ^--- Particle 0 ---^  ^--- Particle 1 ---^\n\n";

    std::cout << "Problem on GPU:\n";
    std::cout << "  - Threads access different fields\n";
    std::cout << "  - Thread 0 reads x0, Thread 1 reads x1, but they're 32 bytes apart\n";
    std::cout << "  - Non-coalesced memory access (slow!)\n\n";

    std::cout << "Structure of Arrays (SoA):\n";
    std::cout << "Memory: [x0 x1 x2 x3 x4...] [y0 y1 y2 y3 y4...] [z0 z1 z2 z3 z4...]\n";
    std::cout << "        ^--- All X values ---^  ^--- All Y values ---^\n\n";

    std::cout << "Advantage on GPU:\n";
    std::cout << "  - Threads access adjacent memory\n";
    std::cout << "  - Thread 0 reads x0, Thread 1 reads x1 (adjacent!)\n";
    std::cout << "  - Coalesced memory access (FAST!)\n";
    std::cout << "  - Can use 128-byte cache lines efficiently\n\n";
}

void demonstratePerformance() {
    std::cout << "=== Performance Comparison ===\n\n";

    const size_t numParticles = 1000000;
    const int iterations = 100;
    const float dt = 0.016f;

    // Initialize AoS
    std::cout << "Initializing " << numParticles << " particles...\n";
    std::vector<ParticleAoS> particlesAoS(numParticles);
    for (size_t i = 0; i < numParticles; ++i) {
        particlesAoS[i] = {0, 0, 0, 1, 0, 0, 1.0f, (int)i};
    }

    // Initialize SoA
    ParticlesSoA particlesSoA;
    particlesSoA.resize(numParticles);
    for (size_t i = 0; i < numParticles; ++i) {
        particlesSoA.x[i] = 0;
        particlesSoA.y[i] = 0;
        particlesSoA.z[i] = 0;
        particlesSoA.vx[i] = 1;
        particlesSoA.vy[i] = 0;
        particlesSoA.vz[i] = 0;
        particlesSoA.mass[i] = 1.0f;
        particlesSoA.id[i] = i;
    }

    std::cout << "\nRunning benchmarks (" << iterations << " iterations)...\n\n";

    // Benchmark AoS
    double timeAoS = benchmark([&]() {
        updateAoS(particlesAoS, dt);
    }, iterations);

    // Benchmark SoA
    double timeSoA = benchmark([&]() {
        updateSoA(particlesSoA, dt);
    }, iterations);

    std::cout << "Results:\n";
    std::cout << "  AoS time: " << timeAoS << " ms\n";
    std::cout << "  SoA time: " << timeSoA << " ms\n";
    std::cout << "  Speedup: " << (timeAoS / timeSoA) << "x\n\n";

    std::cout << "Why SoA is faster (even on CPU):\n";
    std::cout << "  - Better cache utilization\n";
    std::cout << "  - Easier for compiler to vectorize (SIMD)\n";
    std::cout << "  - More predictable memory access pattern\n\n";

    std::cout << "On GPU, difference is MUCH MORE dramatic!\n";
    std::cout << "  - SoA can be 10-100x faster than AoS\n";
    std::cout << "  - Coalesced memory access is CRITICAL\n\n";
}

void demonstrateGPUPseudoCode() {
    std::cout << "=== GPU CUDA Pseudocode ===\n\n";

    std::cout << "AoS on GPU (SLOW):\n";
    std::cout << R"(
    struct Particle { float x, y, z, vx, vy, vz; };
    __global__ void updateAoS(Particle* particles, int n) {
        int i = blockIdx.x * blockDim.x + threadIdx.x;
        if (i < n) {
            particles[i].x += particles[i].vx * dt;  // Non-coalesced!
            // Thread 0 reads at offset 0
            // Thread 1 reads at offset 24 (6 floats apart)
            // Wastes memory bandwidth!
        }
    }
)" << "\n";

    std::cout << "SoA on GPU (FAST):\n";
    std::cout << R"(
    __global__ void updateSoA(float* x, float* vx, int n, float dt) {
        int i = blockIdx.x * blockDim.x + threadIdx.x;
        if (i < n) {
            x[i] += vx[i] * dt;  // Coalesced!
            // Thread 0 reads x[0]
            // Thread 1 reads x[1] (4 bytes apart)
            // All 32 threads in warp read 128 consecutive bytes
            // Single memory transaction! FAST!
        }
    }
)" << "\n";

    std::cout << "Key insight: GPU warps (32 threads) execute together.\n";
    std::cout << "Best performance when all threads access adjacent memory.\n\n";
}

int main() {
    std::cout << "=== MODULE 13: AoS vs SoA for GPU ===\n";

    demonstrateMemoryLayout();
    demonstratePerformance();
    demonstrateGPUPseudoCode();

    std::cout << "=== Key Takeaways ===\n";
    std::cout << "1. Use SoA for GPU code (CRITICAL!)\n";
    std::cout << "2. Coalesced memory access = fast GPU code\n";
    std::cout << "3. Memory layout matters more on GPU than CPU\n";
    std::cout << "4. Sometimes need to convert AoS -> SoA before GPU upload\n";
    std::cout << "5. Trade-off: SoA is less intuitive but much faster\n";

    std::cout << "\n=== Program Complete ===\n";
    return 0;
}

/*
 * TRY THIS:
 * 1. Add AoSoA (Array of Structure of Arrays) hybrid
 * 2. Measure cache misses with performance counters
 * 3. Test with different particle counts
 * 4. Implement conversion functions AoS <-> SoA
 * 5. Add SIMD intrinsics for SoA updates
 * 6. Profile with valgrind cachegrind
 *
 * REAL WORLD:
 * - Physics engines: SoA for particles
 * - Ray tracers: SoA for rays
 * - Game engines: Hybrid approaches
 * - Particle systems: Always SoA on GPU
 */
