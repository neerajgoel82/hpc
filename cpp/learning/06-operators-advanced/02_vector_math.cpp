/*
 * MODULE 6: Operator Overloading
 * File: 02_vector_math.cpp
 *
 * TOPIC: Complete Vector Math Library for GPU Programming
 *
 * COMPILE: g++ -std=c++17 -o vector_math 02_vector_math.cpp
 */

#include <iostream>
#include <cmath>
#include <cassert>

class Vec3 {
public:
    float x, y, z;

    Vec3(float x_ = 0, float y_ = 0, float z_ = 0) : x(x_), y(y_), z(z_) {}

    // Arithmetic operators
    Vec3 operator+(const Vec3& v) const { return Vec3(x+v.x, y+v.y, z+v.z); }
    Vec3 operator-(const Vec3& v) const { return Vec3(x-v.x, y-v.y, z-v.z); }
    Vec3 operator*(float s) const { return Vec3(x*s, y*s, z*s); }
    Vec3 operator/(float s) const { return Vec3(x/s, y/s, z/s); }
    Vec3 operator-() const { return Vec3(-x, -y, -z); }

    // Compound assignment
    Vec3& operator+=(const Vec3& v) { x+=v.x; y+=v.y; z+=v.z; return *this; }
    Vec3& operator-=(const Vec3& v) { x-=v.x; y-=v.y; z-=v.z; return *this; }
    Vec3& operator*=(float s) { x*=s; y*=s; z*=s; return *this; }
    Vec3& operator/=(float s) { x/=s; y/=s; z/=s; return *this; }

    // Array subscript operator
    float& operator[](int i) {
        assert(i >= 0 && i < 3);
        return (i == 0) ? x : (i == 1) ? y : z;
    }

    const float& operator[](int i) const {
        assert(i >= 0 && i < 3);
        return (i == 0) ? x : (i == 1) ? y : z;
    }

    // Vector operations
    float dot(const Vec3& v) const { return x*v.x + y*v.y + z*v.z; }
    Vec3 cross(const Vec3& v) const {
        return Vec3(y*v.z - z*v.y, z*v.x - x*v.z, x*v.y - y*v.x);
    }
    float length() const { return std::sqrt(x*x + y*y + z*z); }
    float lengthSquared() const { return x*x + y*y + z*z; }
    Vec3 normalized() const { float l = length(); return (l > 0) ? (*this / l) : Vec3(0,0,0); }

    // Comparison (with epsilon for floating point)
    bool operator==(const Vec3& v) const {
        const float eps = 1e-6f;
        return std::abs(x - v.x) < eps && std::abs(y - v.y) < eps && std::abs(z - v.z) < eps;
    }
    bool operator!=(const Vec3& v) const { return !(*this == v); }

    void print() const { std::cout << "(" << x << ", " << y << ", " << z << ")"; }
};

// Non-member operators
Vec3 operator*(float s, const Vec3& v) { return v * s; }
std::ostream& operator<<(std::ostream& os, const Vec3& v) {
    return os << "(" << v.x << ", " << v.y << ", " << v.z << ")";
}

// Utility functions
float dot(const Vec3& a, const Vec3& b) { return a.dot(b); }
Vec3 cross(const Vec3& a, const Vec3& b) { return a.cross(b); }
Vec3 normalize(const Vec3& v) { return v.normalized(); }
Vec3 lerp(const Vec3& a, const Vec3& b, float t) { return a + (b - a) * t; }
Vec3 reflect(const Vec3& v, const Vec3& n) { return v - n * (2.0f * dot(v, n)); }

// TRY THIS: Ray-sphere intersection using vector math
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
