#include <stdio.h>
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

struct Sphere {
    float3 center;
    float radius;
    float3 color;
};

struct Ray {
    float3 origin;
    float3 direction;
};

__device__ float3 make_float3_device(float x, float y, float z) {
    float3 v;
    v.x = x; v.y = y; v.z = z;
    return v;
}

__device__ float dot(float3 a, float3 b) {
    return a.x * b.x + a.y * b.y + a.z * b.z;
}

__device__ float3 subtract(float3 a, float3 b) {
    return make_float3_device(a.x - b.x, a.y - b.y, a.z - b.z);
}

__device__ float3 scale(float3 v, float s) {
    return make_float3_device(v.x * s, v.y * s, v.z * s);
}

__device__ float3 add(float3 a, float3 b) {
    return make_float3_device(a.x + b.x, a.y + b.y, a.z + b.z);
}

__device__ float3 normalize(float3 v) {
    float len = sqrtf(dot(v, v));
    return scale(v, 1.0f / len);
}

__device__ bool intersectSphere(Ray ray, Sphere sphere, float *t) {
    float3 oc = subtract(ray.origin, sphere.center);
    float a = dot(ray.direction, ray.direction);
    float b = 2.0f * dot(oc, ray.direction);
    float c = dot(oc, oc) - sphere.radius * sphere.radius;
    float discriminant = b * b - 4 * a * c;

    if (discriminant < 0) return false;

    *t = (-b - sqrtf(discriminant)) / (2.0f * a);
    return *t > 0;
}

__global__ void raytraceKernel(unsigned char *image, int width, int height,
                                Sphere *spheres, int numSpheres) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x >= width || y >= height) return;

    // Setup camera
    float aspectRatio = (float)width / height;
    float3 origin = make_float3_device(0.0f, 0.0f, 0.0f);

    // Ray direction
    float u = (2.0f * x / width - 1.0f) * aspectRatio;
    float v = 1.0f - 2.0f * y / height;
    float3 direction = normalize(make_float3_device(u, v, -1.0f));

    Ray ray;
    ray.origin = origin;
    ray.direction = direction;

    // Trace ray
    float closestT = 1e10f;
    float3 color = make_float3_device(0.1f, 0.1f, 0.2f);  // background

    for (int i = 0; i < numSpheres; i++) {
        float t;
        if (intersectSphere(ray, spheres[i], &t)) {
            if (t < closestT) {
                closestT = t;

                // Simple shading: lambertian
                float3 hitPoint = add(ray.origin, scale(ray.direction, t));
                float3 normal = normalize(subtract(hitPoint, spheres[i].center));
                float3 lightDir = normalize(make_float3_device(1.0f, 1.0f, 1.0f));
                float diffuse = fmaxf(0.0f, dot(normal, lightDir));

                color = scale(spheres[i].color, 0.2f + 0.8f * diffuse);
            }
        }
    }

    // Write color (RGB)
    int idx = (y * width + x) * 3;
    image[idx + 0] = (unsigned char)(fminf(color.x, 1.0f) * 255);
    image[idx + 1] = (unsigned char)(fminf(color.y, 1.0f) * 255);
    image[idx + 2] = (unsigned char)(fminf(color.z, 1.0f) * 255);
}

int main() {
    printf("=== GPU Ray Tracer ===\n\n");

    const int width = 1920;
    const int height = 1080;
    const int imageSize = width * height * 3;  // RGB

    // Create scene with spheres
    Sphere h_spheres[3];
    h_spheres[0].center = make_float3(0.0f, 0.0f, -5.0f);
    h_spheres[0].radius = 1.0f;
    h_spheres[0].color = make_float3(1.0f, 0.3f, 0.3f);  // red

    h_spheres[1].center = make_float3(-2.0f, 0.0f, -4.0f);
    h_spheres[1].radius = 0.7f;
    h_spheres[1].color = make_float3(0.3f, 1.0f, 0.3f);  // green

    h_spheres[2].center = make_float3(2.0f, 0.0f, -4.0f);
    h_spheres[2].radius = 0.7f;
    h_spheres[2].color = make_float3(0.3f, 0.3f, 1.0f);  // blue

    // Allocate device memory
    unsigned char *d_image;
    Sphere *d_spheres;
    CUDA_CHECK(cudaMalloc(&d_image, imageSize));
    CUDA_CHECK(cudaMalloc(&d_spheres, 3 * sizeof(Sphere)));
    CUDA_CHECK(cudaMemcpy(d_spheres, h_spheres, 3 * sizeof(Sphere),
                          cudaMemcpyHostToDevice));

    dim3 threads(16, 16);
    dim3 blocks((width + 15) / 16, (height + 15) / 16);

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaEventRecord(start);
    raytraceKernel<<<blocks, threads>>>(d_image, width, height, d_spheres, 3);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    float ms;
    cudaEventElapsedTime(&ms, start, stop);

    // Copy result
    unsigned char *h_image = (unsigned char*)malloc(imageSize);
    CUDA_CHECK(cudaMemcpy(h_image, d_image, imageSize, cudaMemcpyDeviceToHost));

    printf("Image size: %dx%d\n", width, height);
    printf("Spheres in scene: 3\n");
    printf("Render time: %.2f ms\n", ms);
    printf("Ray trace rate: %.2f Mrays/sec\n",
           (width * height / 1e6) / (ms / 1000.0));
    printf("\nNote: Image data generated (would save as PPM file in production)\n");

    free(h_image);
    cudaFree(d_image);
    cudaFree(d_spheres);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    return 0;
}
