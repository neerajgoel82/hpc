/*
 * MODULE 6: Operator Overloading
 * File: 01_operator_overloading.cpp
 *
 * TOPIC: Basic Operator Overloading
 *
 * CONCEPTS:
 * - Overloading arithmetic operators (+, -, *, /)
 * - Overloading comparison operators (==, !=, <, >)
 * - Member vs non-member operators
 * - Return types and const correctness
 *
 * GPU RELEVANCE:
 * - Vector/Matrix math is fundamental to GPU programming
 * - Transform operations (translate, rotate, scale)
 * - Color manipulation (blend, multiply, add)
 * - Physics calculations (forces, velocities, positions)
 *
 * COMPILE: g++ -std=c++17 -o operator_overloading 01_operator_overloading.cpp
 */

#include <iostream>
#include <cmath>

// Example 1: 3D Vector with arithmetic operators
class Vec3 {
public:
    float x, y, z;

    // Constructor
    Vec3(float x_ = 0, float y_ = 0, float z_ = 0) : x(x_), y(y_), z(z_) {}

    // Addition operator (member function)
    Vec3 operator+(const Vec3& other) const {
        return Vec3(x + other.x, y + other.y, z + other.z);
    }

    // Subtraction operator
    Vec3 operator-(const Vec3& other) const {
        return Vec3(x - other.x, y - other.y, z - other.z);
    }

    // Scalar multiplication (Vec3 * float)
    Vec3 operator*(float scalar) const {
        return Vec3(x * scalar, y * scalar, z * scalar);
    }

    // Scalar division
    Vec3 operator/(float scalar) const {
        return Vec3(x / scalar, y / scalar, z / scalar);
    }

    // Compound assignment operators
    Vec3& operator+=(const Vec3& other) {
        x += other.x;
        y += other.y;
        z += other.z;
        return *this;
    }

    Vec3& operator-=(const Vec3& other) {
        x -= other.x;
        y -= other.y;
        z -= other.z;
        return *this;
    }

    Vec3& operator*=(float scalar) {
        x *= scalar;
        y *= scalar;
        z *= scalar;
        return *this;
    }

    // Unary minus (negation)
    Vec3 operator-() const {
        return Vec3(-x, -y, -z);
    }

    // Comparison operators
    bool operator==(const Vec3& other) const {
        return (x == other.x) && (y == other.y) && (z == other.z);
    }

    bool operator!=(const Vec3& other) const {
        return !(*this == other);
    }

    // Utility methods
    float magnitude() const {
        return std::sqrt(x*x + y*y + z*z);
    }

    Vec3 normalized() const {
        float mag = magnitude();
        return (mag > 0) ? (*this / mag) : Vec3(0, 0, 0);
    }

    float dot(const Vec3& other) const {
        return x * other.x + y * other.y + z * other.z;
    }

    Vec3 cross(const Vec3& other) const {
        return Vec3(
            y * other.z - z * other.y,
            z * other.x - x * other.z,
            x * other.y - y * other.x
        );
    }

    void print() const {
        std::cout << "(" << x << ", " << y << ", " << z << ")";
    }
};

// Non-member operator for scalar * Vec3 (float * Vec3)
// Allows: 2.0f * vec (in addition to vec * 2.0f)
Vec3 operator*(float scalar, const Vec3& vec) {
    return vec * scalar;
}

// Output stream operator (for std::cout << vec)
std::ostream& operator<<(std::ostream& os, const Vec3& vec) {
    os << "Vec3(" << vec.x << ", " << vec.y << ", " << vec.z << ")";
    return os;
}

// Example 2: Color class with operators
class Color {
public:
    float r, g, b, a;

    Color(float red = 0, float green = 0, float blue = 0, float alpha = 1.0f)
        : r(red), g(green), b(blue), a(alpha) {}

    // Color blending (addition)
    Color operator+(const Color& other) const {
        return Color(r + other.r, g + other.g, b + other.b, a + other.a);
    }

    // Color multiplication (modulation)
    Color operator*(const Color& other) const {
        return Color(r * other.r, g * other.g, b * other.b, a * other.a);
    }

    // Scalar multiplication (brightness)
    Color operator*(float scalar) const {
        return Color(r * scalar, g * scalar, b * scalar, a);
    }

    // Compound operators
    Color& operator+=(const Color& other) {
        r += other.r;
        g += other.g;
        b += other.b;
        a += other.a;
        return *this;
    }

    Color& operator*=(float scalar) {
        r *= scalar;
        g *= scalar;
        b *= scalar;
        return *this;
    }

    // Clamp color values to [0, 1]
    Color clamped() const {
        auto clamp = [](float v) { return std::max(0.0f, std::min(1.0f, v)); };
        return Color(clamp(r), clamp(g), clamp(b), clamp(a));
    }

    void print() const {
        std::cout << "Color(r=" << r << ", g=" << g << ", b=" << b << ", a=" << a << ")";
    }
};

std::ostream& operator<<(std::ostream& os, const Color& color) {
    os << "Color(" << color.r << ", " << color.g << ", " << color.b << ", " << color.a << ")";
    return os;
}

// Example 3: Complex number class
class Complex {
private:
    float real, imag;

public:
    Complex(float r = 0, float i = 0) : real(r), imag(i) {}

    // Arithmetic operators
    Complex operator+(const Complex& other) const {
        return Complex(real + other.real, imag + other.imag);
    }

    Complex operator-(const Complex& other) const {
        return Complex(real - other.real, imag - other.imag);
    }

    Complex operator*(const Complex& other) const {
        // (a + bi)(c + di) = (ac - bd) + (ad + bc)i
        return Complex(
            real * other.real - imag * other.imag,
            real * other.imag + imag * other.real
        );
    }

    Complex operator/(const Complex& other) const {
        float denom = other.real * other.real + other.imag * other.imag;
        return Complex(
            (real * other.real + imag * other.imag) / denom,
            (imag * other.real - real * other.imag) / denom
        );
    }

    float magnitude() const {
        return std::sqrt(real * real + imag * imag);
    }

    void print() const {
        std::cout << real;
        if (imag >= 0) std::cout << " + " << imag << "i";
        else std::cout << " - " << (-imag) << "i";
    }
};

// Demonstrate vector operators
void demonstrateVectorOperators() {
    std::cout << "\n=== Vector Operator Overloading ===\n";

    Vec3 v1(1, 2, 3);
    Vec3 v2(4, 5, 6);

    std::cout << "\nBasic operations:\n";
    std::cout << "v1 = " << v1 << "\n";
    std::cout << "v2 = " << v2 << "\n";

    Vec3 v3 = v1 + v2;
    std::cout << "\nv1 + v2 = " << v3 << "\n";

    Vec3 v4 = v1 - v2;
    std::cout << "v1 - v2 = " << v4 << "\n";

    Vec3 v5 = v1 * 2.0f;
    std::cout << "v1 * 2.0 = " << v5 << "\n";

    Vec3 v6 = 3.0f * v1;  // Uses non-member operator
    std::cout << "3.0 * v1 = " << v6 << "\n";

    Vec3 v7 = -v1;
    std::cout << "-v1 = " << v7 << "\n";

    std::cout << "\nCompound operations:\n";
    v1 += v2;
    std::cout << "After v1 += v2: v1 = " << v1 << "\n";

    v1 *= 0.5f;
    std::cout << "After v1 *= 0.5: v1 = " << v1 << "\n";

    std::cout << "\nVector operations:\n";
    Vec3 a(1, 0, 0);
    Vec3 b(0, 1, 0);
    std::cout << "a = " << a << "\n";
    std::cout << "b = " << b << "\n";
    std::cout << "a dot b = " << a.dot(b) << "\n";
    std::cout << "a cross b = " << a.cross(b) << "\n";
    std::cout << "magnitude(a) = " << a.magnitude() << "\n";

    Vec3 c(3, 4, 0);
    std::cout << "\nc = " << c << "\n";
    std::cout << "c.magnitude() = " << c.magnitude() << "\n";
    std::cout << "c.normalized() = " << c.normalized() << "\n";
}

// Demonstrate color operators
void demonstrateColorOperators() {
    std::cout << "\n=== Color Operator Overloading ===\n";

    Color red(1, 0, 0);
    Color blue(0, 0, 1);
    Color white(1, 1, 1);

    std::cout << "\nColors:\n";
    std::cout << "red = " << red << "\n";
    std::cout << "blue = " << blue << "\n";
    std::cout << "white = " << white << "\n";

    Color purple = red + blue;
    std::cout << "\nred + blue = " << purple << "\n";

    Color gray = white * 0.5f;
    std::cout << "white * 0.5 = " << gray << "\n";

    Color modulated = red * Color(0.5f, 0.5f, 0.5f);
    std::cout << "red * gray = " << modulated << "\n";

    std::cout << "\nColor blending example:\n";
    Color c1(0.8f, 0.3f, 0.2f);
    Color c2(0.2f, 0.5f, 0.9f);
    Color blended = c1 + c2;
    std::cout << "c1 + c2 = " << blended << " (before clamp)\n";
    std::cout << "clamped = " << blended.clamped() << "\n";
}

// Demonstrate complex numbers
void demonstrateComplexOperators() {
    std::cout << "\n=== Complex Number Operators ===\n";

    Complex c1(3, 4);
    Complex c2(1, 2);

    std::cout << "\nComplex numbers:\n";
    std::cout << "c1 = "; c1.print(); std::cout << "\n";
    std::cout << "c2 = "; c2.print(); std::cout << "\n";

    Complex sum = c1 + c2;
    std::cout << "\nc1 + c2 = "; sum.print(); std::cout << "\n";

    Complex diff = c1 - c2;
    std::cout << "c1 - c2 = "; diff.print(); std::cout << "\n";

    Complex prod = c1 * c2;
    std::cout << "c1 * c2 = "; prod.print(); std::cout << "\n";

    Complex quot = c1 / c2;
    std::cout << "c1 / c2 = "; quot.print(); std::cout << "\n";

    std::cout << "\n|c1| = " << c1.magnitude() << "\n";
}

// Physics simulation example
void demonstratePhysicsSimulation() {
    std::cout << "\n=== Physics Simulation with Operators ===\n";

    Vec3 position(0, 10, 0);
    Vec3 velocity(5, 0, 0);
    Vec3 acceleration(0, -9.8f, 0);  // Gravity
    float dt = 0.1f;

    std::cout << "\nSimulating projectile motion:\n";
    std::cout << "Initial state:\n";
    std::cout << "  position = " << position << "\n";
    std::cout << "  velocity = " << velocity << "\n";
    std::cout << "  acceleration = " << acceleration << "\n";

    for (int step = 0; step < 5; ++step) {
        // Physics update using overloaded operators
        velocity += acceleration * dt;
        position += velocity * dt;

        std::cout << "\nStep " << (step + 1) << ":\n";
        std::cout << "  position = " << position << "\n";
        std::cout << "  velocity = " << velocity << "\n";
    }
}

int main() {
    std::cout << "=== MODULE 6: Operator Overloading Demo ===\n";

    demonstrateVectorOperators();
    demonstrateColorOperators();
    demonstrateComplexOperators();
    demonstratePhysicsSimulation();

    std::cout << "\n=== Program Complete ===\n";
    return 0;
}

/*
 * KEY CONCEPTS:
 *
 * 1. OPERATOR OVERLOADING SYNTAX:
 *    ReturnType operator@(parameters) const { ... }
 *    where @ is +, -, *, /, ==, !=, etc.
 *
 * 2. MEMBER vs NON-MEMBER:
 *    Member: vec.operator+(other) -> used as: vec + other
 *    Non-member: operator*(scalar, vec) -> used as: scalar * vec
 *
 * 3. RETURN TYPES:
 *    Arithmetic (+, -, *, /): Return new object by value
 *    Compound (+=, -=, *=): Return reference to *this
 *    Comparison (==, !=): Return bool
 *
 * 4. CONST CORRECTNESS:
 *    Mark operator const if it doesn't modify object
 *    Example: Vec3 operator+(const Vec3& other) const
 *
 * TRY THIS:
 * 1. Add [] operator for Vec3 (access x, y, z by index)
 * 2. Implement Vec4 with w component for homogeneous coordinates
 * 3. Add lerp operator or function: Color::lerp(other, t)
 * 4. Implement dot product as operator* (or keep as method?)
 * 5. Add Color blending modes (multiply, screen, overlay)
 * 6. Implement 2D Point class with distance operator
 *
 * COMMON MISTAKES:
 * - Forgetting const for read-only operators
 * - Wrong return type (reference vs value)
 * - Not implementing both a+b and a+=b consistently
 * - Forgetting non-member operator for commutative operations
 *
 * GPU CONNECTION:
 * - Vector math is fundamental to all GPU programming
 * - Shader code uses vector operators extensively
 * - Transform calculations: position, rotation, scale
 * - Physics: forces, velocities, accelerations
 * - Graphics: colors, lighting, blending
 *
 * BEST PRACTICES:
 * - Implement related operators together (+, +=)
 * - Use const references for parameters
 * - Make operators const when possible
 * - Implement operator<< for debugging
 * - Consider performance (return value optimization)
 */
