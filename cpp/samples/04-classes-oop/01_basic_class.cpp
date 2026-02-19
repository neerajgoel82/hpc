/*
 * MODULE 4: Classes and Object-Oriented Programming
 * File: 01_basic_class.cpp
 *
 * TOPIC: Basic Class Structure, Constructors, and Destructors
 *
 * CONCEPTS:
 * - Classes encapsulate data (member variables) and functions (methods)
 * - Constructors initialize objects when they're created
 * - Destructors clean up when objects are destroyed
 * - Access specifiers: public, private, protected
 *
 * GPU RELEVANCE:
 * In GPU programming, classes help organize:
 * - Particle systems (position, velocity, mass per particle)
 * - Ray tracing objects (rays, spheres, materials)
 * - Mesh data (vertices, normals, texture coordinates)
 * Understanding object lifecycle is crucial for managing GPU memory.
 *
 * COMPILE: g++ -std=c++17 -o basic_class 01_basic_class.cpp
 */

#include <iostream>
#include <string>
#include <cmath>

// Example 1: Simple 3D Vector class for GPU graphics
class Vec3 {
private:
    // Private member variables - only accessible within the class
    float x, y, z;

public:
    // Default constructor - initializes to zero vector
    Vec3() : x(0.0f), y(0.0f), z(0.0f) {
        std::cout << "Vec3 default constructor called\n";
    }

    // Parameterized constructor - initializes with specific values
    Vec3(float x_val, float y_val, float z_val) : x(x_val), y(y_val), z(z_val) {
        std::cout << "Vec3 parameterized constructor called: ("
                  << x << ", " << y << ", " << z << ")\n";
    }

    // Destructor - called when object goes out of scope
    ~Vec3() {
        std::cout << "Vec3 destructor called for ("
                  << x << ", " << y << ", " << z << ")\n";
    }

    // Method to calculate magnitude (length) of vector
    float magnitude() const {  // const means this method doesn't modify the object
        return std::sqrt(x*x + y*y + z*z);
    }

    // Method to print the vector
    void print() const {
        std::cout << "Vec3(" << x << ", " << y << ", " << z << ")\n";
    }

    // Getter methods - provide read access to private data
    float getX() const { return x; }
    float getY() const { return y; }
    float getZ() const { return z; }

    // Setter methods - provide controlled write access to private data
    void setX(float val) { x = val; }
    void setY(float val) { y = val; }
    void setZ(float val) { z = val; }
};

// Example 2: Particle class for physics simulation
class Particle {
private:
    Vec3 position;
    Vec3 velocity;
    float mass;
    int id;

public:
    // Constructor with initialization list
    Particle(int particle_id, float m)
        : position(), velocity(), mass(m), id(particle_id) {
        std::cout << "Particle " << id << " created with mass " << mass << "\n";
    }

    Particle(int particle_id, float px, float py, float pz, float m)
        : position(px, py, pz), velocity(), mass(m), id(particle_id) {
        std::cout << "Particle " << id << " created at position ";
        position.print();
    }

    ~Particle() {
        std::cout << "Particle " << id << " destroyed\n";
    }

    void setVelocity(float vx, float vy, float vz) {
        velocity.setX(vx);
        velocity.setY(vy);
        velocity.setZ(vz);
    }

    void updatePosition(float dt) {
        // Simple Euler integration: position += velocity * dt
        position.setX(position.getX() + velocity.getX() * dt);
        position.setY(position.getY() + velocity.getY() * dt);
        position.setZ(position.getZ() + velocity.getZ() * dt);
    }

    void printStatus() const {
        std::cout << "Particle " << id << " (mass=" << mass << ")\n";
        std::cout << "  Position: ";
        position.print();
        std::cout << "  Velocity: ";
        velocity.print();
    }
};

// Example 3: Timer class demonstrating constructor/destructor for profiling
class ScopedTimer {
private:
    std::string name;

public:
    ScopedTimer(const std::string& timer_name) : name(timer_name) {
        std::cout << "[TIMER START] " << name << "\n";
    }

    ~ScopedTimer() {
        std::cout << "[TIMER END] " << name << "\n";
    }
};

void demonstrateScope() {
    std::cout << "\n=== Demonstrating Object Lifetime and Scope ===\n";

    {
        ScopedTimer timer("Inner scope");
        Vec3 v1(1.0f, 2.0f, 3.0f);
        std::cout << "Inside inner scope\n";
    } // v1 and timer destructors called here automatically

    std::cout << "Back in outer scope\n";
}

int main() {
    std::cout << "=== MODULE 4: Basic Classes Demo ===\n\n";

    // Example 1: Creating Vec3 objects
    std::cout << "--- Creating Vec3 objects ---\n";
    Vec3 origin;  // Calls default constructor
    Vec3 point(3.0f, 4.0f, 0.0f);  // Calls parameterized constructor

    std::cout << "\nMagnitude of point: " << point.magnitude() << "\n";

    // Example 2: Creating and using Particle objects
    std::cout << "\n--- Creating Particle objects ---\n";
    Particle p1(1, 1.5f);
    Particle p2(2, 10.0f, 0.0f, 0.0f, 2.0f);

    p2.setVelocity(1.0f, 2.0f, 0.0f);
    p2.updatePosition(0.1f);  // Simulate 0.1 second timestep

    std::cout << "\nParticle status after update:\n";
    p2.printStatus();

    // Example 3: Scope and lifetime
    demonstrateScope();

    std::cout << "\n--- Exiting main() - destructors will be called ---\n";
    return 0;

    // Destructors for origin, point, p1, p2 are automatically called here
}

/*
 * KEY CONCEPTS DEMONSTRATED:
 *
 * 1. ENCAPSULATION: Data (x, y, z) is private; access is controlled
 * 2. CONSTRUCTORS: Objects are properly initialized when created
 * 3. DESTRUCTORS: Cleanup happens automatically when objects go out of scope
 * 4. CONST CORRECTNESS: Methods marked const can't modify the object
 * 5. INITIALIZATION LISTS: Efficient way to initialize member variables
 *
 * TRY THIS:
 * 1. Add a normalize() method to Vec3 that makes the vector unit length
 * 2. Add a constructor to Particle that takes initial velocity
 * 3. Create a dotProduct() method that takes another Vec3 as parameter
 * 4. Add a static member variable to Particle to count total particles
 * 5. Create a BoundingBox class with min and max Vec3 points
 *
 * COMMON MISTAKES:
 * - Forgetting to initialize member variables in constructors
 * - Not marking methods const when they should be
 * - Trying to access private members from outside the class
 * - Not understanding when constructors/destructors are called
 *
 * GPU CONNECTION:
 * - In CUDA/GPU code, simple classes like Vec3 work on device
 * - Complex classes with dynamic memory need special handling
 * - Understanding object lifecycle prevents memory leaks on GPU
 * - Constructor/destructor patterns help with resource management
 */
