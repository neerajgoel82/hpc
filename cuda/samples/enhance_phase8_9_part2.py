#!/usr/bin/env python3
"""
Part 2: N-body, Neural Network, Molecular Dynamics, Options, and Phase 9
"""

import os

CUDA_CHECK = '''#define CUDA_CHECK(call) \\
    do { \\
        cudaError_t err = call; \\
        if (err != cudaSuccess) { \\
            fprintf(stderr, "CUDA error at %s:%d: %s\\n", \\
                    __FILE__, __LINE__, cudaGetErrorString(err)); \\
            exit(EXIT_FAILURE); \\
        } \\
    } while(0)'''

IMPLEMENTATIONS = {
    'phase8/46_nbody_simulation.cu': '''#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>
#include <math.h>

''' + CUDA_CHECK + '''

struct Body {
    float x, y, z;     // position
    float vx, vy, vz;  // velocity
    float mass;
};

__global__ void computeForces(Body *bodies, float *fx, float *fy, float *fz, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;

    float force_x = 0.0f, force_y = 0.0f, force_z = 0.0f;
    const float G = 6.67430e-11f;  // Gravitational constant
    const float softening = 1e-9f;  // Softening factor to avoid singularities

    for (int j = 0; j < n; j++) {
        if (i != j) {
            float dx = bodies[j].x - bodies[i].x;
            float dy = bodies[j].y - bodies[i].y;
            float dz = bodies[j].z - bodies[i].z;

            float distSqr = dx*dx + dy*dy + dz*dz + softening;
            float dist = sqrtf(distSqr);
            float force = G * bodies[i].mass * bodies[j].mass / distSqr;

            force_x += force * dx / dist;
            force_y += force * dy / dist;
            force_z += force * dz / dist;
        }
    }

    fx[i] = force_x;
    fy[i] = force_y;
    fz[i] = force_z;
}

__global__ void updateBodies(Body *bodies, float *fx, float *fy, float *fz,
                              int n, float dt) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;

    // Update velocities (F = ma => a = F/m)
    float ax = fx[i] / bodies[i].mass;
    float ay = fy[i] / bodies[i].mass;
    float az = fz[i] / bodies[i].mass;

    bodies[i].vx += ax * dt;
    bodies[i].vy += ay * dt;
    bodies[i].vz += az * dt;

    // Update positions
    bodies[i].x += bodies[i].vx * dt;
    bodies[i].y += bodies[i].vy * dt;
    bodies[i].z += bodies[i].vz * dt;
}

int main() {
    printf("=== N-Body Gravitational Simulation ===\\n\\n");

    const int N = 4096;  // Number of bodies
    const float dt = 0.01f;  // Time step
    const int steps = 100;

    // Allocate host memory
    Body *h_bodies = (Body*)malloc(N * sizeof(Body));

    // Initialize with random positions and masses
    for (int i = 0; i < N; i++) {
        h_bodies[i].x = (rand() / (float)RAND_MAX - 0.5f) * 1e10f;
        h_bodies[i].y = (rand() / (float)RAND_MAX - 0.5f) * 1e10f;
        h_bodies[i].z = (rand() / (float)RAND_MAX - 0.5f) * 1e10f;
        h_bodies[i].vx = h_bodies[i].vy = h_bodies[i].vz = 0.0f;
        h_bodies[i].mass = 1e20f + rand() / (float)RAND_MAX * 1e20f;
    }

    // Allocate device memory
    Body *d_bodies;
    float *d_fx, *d_fy, *d_fz;
    CUDA_CHECK(cudaMalloc(&d_bodies, N * sizeof(Body)));
    CUDA_CHECK(cudaMalloc(&d_fx, N * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_fy, N * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_fz, N * sizeof(float)));

    CUDA_CHECK(cudaMemcpy(d_bodies, h_bodies, N * sizeof(Body),
                          cudaMemcpyHostToDevice));

    int threads = 256;
    int blocks = (N + threads - 1) / threads;

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaEventRecord(start);

    // Simulation loop
    for (int step = 0; step < steps; step++) {
        computeForces<<<blocks, threads>>>(d_bodies, d_fx, d_fy, d_fz, N);
        updateBodies<<<blocks, threads>>>(d_bodies, d_fx, d_fy, d_fz, N, dt);
    }

    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    float ms;
    cudaEventElapsedTime(&ms, start, stop);

    // Copy final positions back
    CUDA_CHECK(cudaMemcpy(h_bodies, d_bodies, N * sizeof(Body),
                          cudaMemcpyDeviceToHost));

    printf("Bodies: %d\\n", N);
    printf("Time steps: %d\\n", steps);
    printf("Total time: %.2f ms\\n", ms);
    printf("Time per step: %.2f ms\\n", ms / steps);
    printf("Interactions/sec: %.2f billion\\n",
           (long long)N * N * steps / (ms / 1000.0) / 1e9);

    printf("\\nSample final positions:\\n");
    for (int i = 0; i < 5; i++) {
        printf("  Body %d: (%.2e, %.2e, %.2e)\\n",
               i, h_bodies[i].x, h_bodies[i].y, h_bodies[i].z);
    }

    free(h_bodies);
    cudaFree(d_bodies);
    cudaFree(d_fx);
    cudaFree(d_fy);
    cudaFree(d_fz);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    return 0;
}
''',

    'phase8/47_neural_network.cu': '''#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>
#include <math.h>

''' + CUDA_CHECK + '''

__device__ float sigmoid(float x) {
    return 1.0f / (1.0f + expf(-x));
}

__device__ float sigmoid_derivative(float x) {
    float s = sigmoid(x);
    return s * (1.0f - s);
}

__global__ void forwardLayer(float *input, float *weights, float *bias,
                              float *output, int inputSize, int outputSize) {
    int neuron = blockIdx.x * blockDim.x + threadIdx.x;
    if (neuron >= outputSize) return;

    float sum = bias[neuron];
    for (int i = 0; i < inputSize; i++) {
        sum += input[i] * weights[neuron * inputSize + i];
    }
    output[neuron] = sigmoid(sum);
}

__global__ void backwardLayer(float *input, float *weights, float *delta,
                               float *prevDelta, int inputSize, int outputSize) {
    int neuron = blockIdx.x * blockDim.x + threadIdx.x;
    if (neuron >= inputSize) return;

    float error = 0.0f;
    for (int i = 0; i < outputSize; i++) {
        error += delta[i] * weights[i * inputSize + neuron];
    }
    prevDelta[neuron] = error * sigmoid_derivative(input[neuron]);
}

__global__ void updateWeights(float *weights, float *input, float *delta,
                               int inputSize, int outputSize, float learningRate) {
    int neuron = blockIdx.x * blockDim.x + threadIdx.x;
    if (neuron >= outputSize) return;

    for (int i = 0; i < inputSize; i++) {
        weights[neuron * inputSize + i] += learningRate * delta[neuron] * input[i];
    }
}

int main() {
    printf("=== Neural Network: Forward & Backward Pass ===\\n\\n");

    const int inputSize = 784;    // 28x28 image
    const int hiddenSize = 128;
    const int outputSize = 10;    // 10 digits
    const float learningRate = 0.01f;

    // Allocate device memory
    float *d_input, *d_hidden, *d_output;
    float *d_w1, *d_w2, *d_b1, *d_b2;
    float *d_delta_hidden, *d_delta_output;

    CUDA_CHECK(cudaMalloc(&d_input, inputSize * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_hidden, hiddenSize * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_output, outputSize * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_w1, inputSize * hiddenSize * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_w2, hiddenSize * outputSize * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_b1, hiddenSize * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_b2, outputSize * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_delta_hidden, hiddenSize * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_delta_output, outputSize * sizeof(float)));

    // Initialize with random weights (simplified - would use cuRAND in production)
    float *h_w1 = (float*)malloc(inputSize * hiddenSize * sizeof(float));
    float *h_w2 = (float*)malloc(hiddenSize * outputSize * sizeof(float));
    for (int i = 0; i < inputSize * hiddenSize; i++) {
        h_w1[i] = (rand() / (float)RAND_MAX - 0.5f) * 0.1f;
    }
    for (int i = 0; i < hiddenSize * outputSize; i++) {
        h_w2[i] = (rand() / (float)RAND_MAX - 0.5f) * 0.1f;
    }

    CUDA_CHECK(cudaMemcpy(d_w1, h_w1, inputSize * hiddenSize * sizeof(float),
                          cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_w2, h_w2, hiddenSize * outputSize * sizeof(float),
                          cudaMemcpyHostToDevice));

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    int threads = 256;

    // Forward pass
    cudaEventRecord(start);
    forwardLayer<<<(hiddenSize + threads - 1) / threads, threads>>>
        (d_input, d_w1, d_b1, d_hidden, inputSize, hiddenSize);
    forwardLayer<<<(outputSize + threads - 1) / threads, threads>>>
        (d_hidden, d_w2, d_b2, d_output, hiddenSize, outputSize);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    float forwardTime;
    cudaEventElapsedTime(&forwardTime, start, stop);

    // Backward pass
    cudaEventRecord(start);
    backwardLayer<<<(hiddenSize + threads - 1) / threads, threads>>>
        (d_hidden, d_w2, d_delta_output, d_delta_hidden, hiddenSize, outputSize);
    updateWeights<<<(hiddenSize + threads - 1) / threads, threads>>>
        (d_w1, d_input, d_delta_hidden, inputSize, hiddenSize, learningRate);
    updateWeights<<<(outputSize + threads - 1) / threads, threads>>>
        (d_w2, d_hidden, d_delta_output, hiddenSize, outputSize, learningRate);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    float backwardTime;
    cudaEventElapsedTime(&backwardTime, start, stop);

    printf("Network architecture: %d -> %d -> %d\\n",
           inputSize, hiddenSize, outputSize);
    printf("Forward pass:  %.3f ms\\n", forwardTime);
    printf("Backward pass: %.3f ms\\n", backwardTime);
    printf("Total:         %.3f ms\\n", forwardTime + backwardTime);
    printf("\\nParameters: %d weights\\n",
           inputSize * hiddenSize + hiddenSize * outputSize);

    free(h_w1);
    free(h_w2);
    cudaFree(d_input);
    cudaFree(d_hidden);
    cudaFree(d_output);
    cudaFree(d_w1);
    cudaFree(d_w2);
    cudaFree(d_b1);
    cudaFree(d_b2);
    cudaFree(d_delta_hidden);
    cudaFree(d_delta_output);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    return 0;
}
''',

    'phase8/48_molecular_dynamics.cu': '''#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>
#include <math.h>

''' + CUDA_CHECK + '''

struct Atom {
    float x, y, z;     // position
    float vx, vy, vz;  // velocity
    float fx, fy, fz;  // force
};

__global__ void computeLennardJonesForces(Atom *atoms, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;

    const float epsilon = 1.0f;  // Energy parameter
    const float sigma = 1.0f;    // Distance parameter
    const float cutoff = 2.5f * sigma;

    float force_x = 0.0f, force_y = 0.0f, force_z = 0.0f;

    for (int j = 0; j < n; j++) {
        if (i != j) {
            float dx = atoms[j].x - atoms[i].x;
            float dy = atoms[j].y - atoms[i].y;
            float dz = atoms[j].z - atoms[i].z;

            float r2 = dx*dx + dy*dy + dz*dz;

            if (r2 < cutoff * cutoff) {
                float r2inv = 1.0f / r2;
                float r6inv = r2inv * r2inv * r2inv;
                float sigma6 = sigma * sigma * sigma * sigma * sigma * sigma;

                // Lennard-Jones potential: V(r) = 4ε[(σ/r)^12 - (σ/r)^6]
                // Force = -dV/dr
                float force = 24.0f * epsilon * r2inv * sigma6 * r6inv *
                             (2.0f * sigma6 * r6inv - 1.0f);

                force_x += force * dx;
                force_y += force * dy;
                force_z += force * dz;
            }
        }
    }

    atoms[i].fx = force_x;
    atoms[i].fy = force_y;
    atoms[i].fz = force_z;
}

__global__ void integrateVerlet(Atom *atoms, int n, float dt) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;

    const float mass = 1.0f;

    // Velocity Verlet integration
    // v(t+dt/2) = v(t) + f(t)/(2m) * dt
    atoms[i].vx += 0.5f * atoms[i].fx / mass * dt;
    atoms[i].vy += 0.5f * atoms[i].fy / mass * dt;
    atoms[i].vz += 0.5f * atoms[i].fz / mass * dt;

    // x(t+dt) = x(t) + v(t+dt/2) * dt
    atoms[i].x += atoms[i].vx * dt;
    atoms[i].y += atoms[i].vy * dt;
    atoms[i].z += atoms[i].vz * dt;
}

int main() {
    printf("=== Molecular Dynamics: Lennard-Jones ===\\n\\n");

    const int N = 2048;  // Number of atoms
    const float dt = 0.001f;
    const int steps = 100;

    Atom *h_atoms = (Atom*)malloc(N * sizeof(Atom));

    // Initialize atoms in a grid
    int cubeSize = (int)ceil(pow(N, 1.0/3.0));
    int idx = 0;
    for (int ix = 0; ix < cubeSize && idx < N; ix++) {
        for (int iy = 0; iy < cubeSize && idx < N; iy++) {
            for (int iz = 0; iz < cubeSize && idx < N; iz++, idx++) {
                h_atoms[idx].x = ix * 1.5f;
                h_atoms[idx].y = iy * 1.5f;
                h_atoms[idx].z = iz * 1.5f;
                h_atoms[idx].vx = (rand() / (float)RAND_MAX - 0.5f) * 0.1f;
                h_atoms[idx].vy = (rand() / (float)RAND_MAX - 0.5f) * 0.1f;
                h_atoms[idx].vz = (rand() / (float)RAND_MAX - 0.5f) * 0.1f;
                h_atoms[idx].fx = h_atoms[idx].fy = h_atoms[idx].fz = 0.0f;
            }
        }
    }

    Atom *d_atoms;
    CUDA_CHECK(cudaMalloc(&d_atoms, N * sizeof(Atom)));
    CUDA_CHECK(cudaMemcpy(d_atoms, h_atoms, N * sizeof(Atom),
                          cudaMemcpyHostToDevice));

    int threads = 256;
    int blocks = (N + threads - 1) / threads;

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaEventRecord(start);

    for (int step = 0; step < steps; step++) {
        computeLennardJonesForces<<<blocks, threads>>>(d_atoms, N);
        integrateVerlet<<<blocks, threads>>>(d_atoms, N, dt);
    }

    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    float ms;
    cudaEventElapsedTime(&ms, start, stop);

    CUDA_CHECK(cudaMemcpy(h_atoms, d_atoms, N * sizeof(Atom),
                          cudaMemcpyDeviceToHost));

    printf("Atoms: %d\\n", N);
    printf("Time steps: %d\\n", steps);
    printf("Total time: %.2f ms\\n", ms);
    printf("Time per step: %.3f ms\\n", ms / steps);
    printf("Interactions/step: %d\\n", N * N);

    printf("\\nSample final positions:\\n");
    for (int i = 0; i < 5; i++) {
        printf("  Atom %d: (%.2f, %.2f, %.2f)\\n",
               i, h_atoms[i].x, h_atoms[i].y, h_atoms[i].z);
    }

    free(h_atoms);
    cudaFree(d_atoms);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    return 0;
}
''',

    'phase8/49_option_pricing.cu': '''#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>
#include <curand_kernel.h>
#include <math.h>

''' + CUDA_CHECK + '''

__global__ void monteCarloOptionPricing(float *prices, int numPaths, int numSteps,
                                         float S0, float K, float r, float sigma,
                                         float T, unsigned long seed) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= numPaths) return;

    // Initialize random number generator
    curandState state;
    curand_init(seed, idx, 0, &state);

    float dt = T / numSteps;
    float S = S0;

    // Simulate price path
    for (int step = 0; step < numSteps; step++) {
        float z = curand_normal(&state);
        S *= expf((r - 0.5f * sigma * sigma) * dt + sigma * sqrtf(dt) * z);
    }

    // Calculate payoff for European call option
    prices[idx] = fmaxf(S - K, 0.0f);
}

int main() {
    printf("=== Monte Carlo Option Pricing ===\\n\\n");

    // Option parameters
    const float S0 = 100.0f;      // Initial stock price
    const float K = 105.0f;       // Strike price
    const float r = 0.05f;        // Risk-free rate (5%)
    const float sigma = 0.2f;     // Volatility (20%)
    const float T = 1.0f;         // Time to maturity (1 year)

    const int numPaths = 1000000;  // Number of Monte Carlo paths
    const int numSteps = 252;      // Trading days in a year

    printf("European Call Option Parameters:\\n");
    printf("  Spot price (S0):    $%.2f\\n", S0);
    printf("  Strike price (K):   $%.2f\\n", K);
    printf("  Risk-free rate (r): %.1f%%\\n", r * 100);
    printf("  Volatility (σ):     %.1f%%\\n", sigma * 100);
    printf("  Time to maturity:   %.1f years\\n", T);
    printf("\\nMonte Carlo simulation:\\n");
    printf("  Paths: %d\\n", numPaths);
    printf("  Steps per path: %d\\n\\n", numSteps);

    // Allocate device memory
    float *d_prices;
    CUDA_CHECK(cudaMalloc(&d_prices, numPaths * sizeof(float)));

    int threads = 256;
    int blocks = (numPaths + threads - 1) / threads;

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaEventRecord(start);
    monteCarloOptionPricing<<<blocks, threads>>>
        (d_prices, numPaths, numSteps, S0, K, r, sigma, T, 1234ULL);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    float ms;
    cudaEventElapsedTime(&ms, start, stop);

    // Copy results back
    float *h_prices = (float*)malloc(numPaths * sizeof(float));
    CUDA_CHECK(cudaMemcpy(h_prices, d_prices, numPaths * sizeof(float),
                          cudaMemcpyDeviceToHost));

    // Calculate option price (discounted expected payoff)
    double sum = 0.0;
    for (int i = 0; i < numPaths; i++) {
        sum += h_prices[i];
    }
    float optionPrice = expf(-r * T) * (sum / numPaths);

    // Black-Scholes formula for comparison
    float d1 = (logf(S0/K) + (r + 0.5f*sigma*sigma)*T) / (sigma * sqrtf(T));
    float d2 = d1 - sigma * sqrtf(T);

    // Approximate normal CDF using error function
    auto normcdf = [](float x) {
        return 0.5f * (1.0f + erff(x / sqrtf(2.0f)));
    };

    float bsPrice = S0 * normcdf(d1) - K * expf(-r*T) * normcdf(d2);

    printf("Results:\\n");
    printf("  Monte Carlo price: $%.4f\\n", optionPrice);
    printf("  Black-Scholes price: $%.4f\\n", bsPrice);
    printf("  Difference: $%.4f (%.2f%%)\\n",
           fabs(optionPrice - bsPrice),
           fabs(optionPrice - bsPrice) / bsPrice * 100);

    printf("\\nPerformance:\\n");
    printf("  Computation time: %.2f ms\\n", ms);
    printf("  Paths/sec: %.2f million\\n", numPaths / (ms / 1000.0) / 1e6);

    free(h_prices);
    cudaFree(d_prices);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    return 0;
}
''',
}

def write_enhanced_file(filepath, content):
    """Write enhanced implementation to file."""
    try:
        full_path = f'local/{filepath}'
        with open(full_path, 'w') as f:
            f.write(content)
        print(f"✓ Enhanced: {filepath}")
        return True
    except Exception as e:
        print(f"✗ Error: {filepath} - {e}")
        return False

def main():
    print("="*70)
    print("Enhancing Phase 8 - Part 2 (N-body, Neural Net, MD, Options)")
    print("="*70)
    print()

    enhanced_count = 0

    for filepath, content in IMPLEMENTATIONS.items():
        if write_enhanced_file(filepath, content):
            enhanced_count += 1

    print(f"\\nPart 2 Complete: {enhanced_count} files enhanced")
    print()
    print("Next: Phase 9 implementations...")

if __name__ == "__main__":
    main()
