#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>
#include <math.h>

#define CUDA_CHECK(call) \
    do { \
        cudaError_t err = call; \
        if (err != cudaSuccess) { \
            fprintf(stderr, "CUDA error at %s:%d: %s\n", \
                    __FILE__, __LINE__, cudaGetErrorString(err)); \
            exit(EXIT_FAILURE); \
        } \
    } while(0)


#include <thrust/device_vector.h>
#include <thrust/host_vector.h>
#include <thrust/sort.h>
#include <thrust/reduce.h>
#include <thrust/transform.h>

struct square_functor {
    __host__ __device__
    float operator()(float x) const {
        return x * x;
    }
};

int main() {
    printf("=== Thrust Examples ===\n\n");
    const int N = 1 << 20;

    thrust::host_vector<float> h_vec(N);
    for (int i = 0; i < N; i++) h_vec[i] = (float)(rand() % 100);

    thrust::device_vector<float> d_vec = h_vec;

    printf("Thrust: C++ STL-like interface for CUDA\n\n");

    printf("1. Reduction:\n");
    float sum = thrust::reduce(d_vec.begin(), d_vec.end(), 0.0f, thrust::plus<float>());
    printf("   Sum: %.2f\n\n", sum);

    printf("2. Transform (square each element):\n");
    thrust::transform(d_vec.begin(), d_vec.end(), d_vec.begin(), square_functor());
    printf("   Transform complete\n\n");

    printf("3. Sort:\n");
    thrust::sort(d_vec.begin(), d_vec.end());
    printf("   Sort complete\n\n");

    thrust::host_vector<float> h_result = d_vec;
    printf("First 5 sorted: %.0f %.0f %.0f %.0f %.0f\n",
           h_result[0], h_result[1], h_result[2], h_result[3], h_result[4]);

    return 0;
}
