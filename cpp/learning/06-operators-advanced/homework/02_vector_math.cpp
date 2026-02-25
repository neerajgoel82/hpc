/*
 * Homework: 02_vector_math.cpp
 *
 * Complete the exercises below based on the concepts from 02_vector_math.cpp
 * in the classwork folder.
 *
 * Instructions:
 * 1. Read the corresponding classwork file first
 * 2. Implement the solutions below
 * 3. Compile: g++ -std=c++17 -Wall -Wextra 02_vector_math.cpp -o homework
 * 4. Test your solutions
 */

#include <iostream>

/*
 * TRY THIS:
 * Ray-sphere intersection using vector math
struct Ray {
    Vec3 origin;
    Vec3 direction;

    Ray(const Vec3& o, const Vec3& d) : origin(o), direction(d.normalized()) {}
    Vec3 at(float t) const { return origin + direction * t; }
};

struct Sphere {
    Vec3 center;
    float radius;

    Sphere(const Vec3& c, float r) : center(c), radius(r) {}

    bool intersect(const Ray& ray, float& t) const {
        Vec3 oc = ray.origin - center;
        float a = ray.direction.lengthSquared();
        float b = 2.0f * dot(oc, ray.direction);
        float c = oc.lengthSquared() - radius * radius;
        float discriminant = b * b - 4 * a * c;

        if (discriminant < 0) return false;

        t = (-b - std::sqrt(discriminant)) / (2.0f * a);
        return t > 0;
    }
};

int main() {
    std::cout << "=== Vector Math for GPU Programming ===\n\n";

    // Basic vector operations
    Vec3 v1(1, 0, 0);
    Vec3 v2(0, 1, 0);
    std::cout << "v1 = " << v1 << ", v2 = " << v2 << "\n";
    std::cout << "v1 + v2 = " << (v1 + v2) << "\n";
    std::cout << "v1 dot v2 = " << dot(v1, v2) << "\n";
    std::cout << "v1 cross v2 = " << cross(v1, v2) << "\n\n";

    // Array subscript operator
    Vec3 v(1, 2, 3);
    std::cout << "v = " << v << "\n";
    std::cout << "v[0] = " << v[0] << ", v[1] = " << v[1] << ", v[2] = " << v[2] << "\n";
    v[1] = 10;
    std::cout << "After v[1] = 10: v = " << v << "\n\n";

    // Reflection (like in ray tracing)
    Vec3 incident(1, -1, 0);
    Vec3 normal(0, 1, 0);
    Vec3 reflected = reflect(incident.normalized(), normal);
    std::cout << "Reflect " << incident << " across " << normal << " = " << reflected << "\n\n";

    // Ray-sphere intersection
    Ray ray(Vec3(0, 0, 0), Vec3(0, 0, -1));
    Sphere sphere(Vec3(0, 0, -5), 1.0f);
    float t;
    if (sphere.intersect(ray, t)) {
        std::cout << "Ray hits sphere at t = " << t << ", position = " << ray.at(t) << "\n";
    }

    std::cout << "\nGPU CONNECTION: This vector math is identical to GLSL/HLSL!\n";
    return 0;
}

/*
 * TRY THIS:
 * 1. Add Vec4 for homogeneous coordinates
 * 2. Implement Matrix4x4 with operator*
 * 3. Add quaternion class for rotations
 * 4. Implement plane intersection
 * 5. Add bounding box (AABB) class
 */

int main() {
    std::cout << "Homework: 02_vector_math\n";
    std::cout << "Implement the exercises above\n";

    // Your code here

    return 0;
}
