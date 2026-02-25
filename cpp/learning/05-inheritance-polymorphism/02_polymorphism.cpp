/*
 * MODULE 5: Inheritance and Polymorphism
 * File: 02_polymorphism.cpp
 *
 * TOPIC: Polymorphism - Runtime Type Selection
 *
 * CONCEPTS:
 * - Polymorphism: "many forms" - same interface, different implementations
 * - Virtual function dispatch (vtable mechanism)
 * - Base class pointers to derived objects
 * - Runtime vs compile-time binding
 * - Virtual destructors (critical!)
 *
 * GPU RELEVANCE:
 * - Rendering: Uniform interface for different object types
 * - Ray tracing: Test intersection on heterogeneous scene objects
 * - Shader management: Different shaders with same interface
 * - Resource management: Different buffer types bound uniformly
 *
 * COMPILE: g++ -std=c++17 -o polymorphism 02_polymorphism.cpp
 */

#include <iostream>
#include <string>
#include <vector>
#include <memory>
#include <cmath>

// Example 1: Classic polymorphism with Shape hierarchy
class Shape {
protected:
    std::string name;
    float x, y;

public:
    Shape(const std::string& shapeName, float posX, float posY)
        : name(shapeName), x(posX), y(posY) {
        std::cout << "  Shape '" << name << "' constructed\n";
    }

    // CRITICAL: Virtual destructor for polymorphic classes
    virtual ~Shape() {
        std::cout << "  Shape '" << name << "' destroyed\n";
    }

    // Pure virtual functions - interface contract
    virtual float area() const = 0;
    virtual float perimeter() const = 0;
    virtual void render() const = 0;

    // Regular virtual function with default implementation
    virtual void printInfo() const {
        std::cout << "Shape: " << name << " at (" << x << ", " << y << ")\n";
    }

    std::string getName() const { return name; }
    float getX() const { return x; }
    float getY() const { return y; }
};

class Circle : public Shape {
private:
    float radius;

public:
    Circle(const std::string& name, float x, float y, float r)
        : Shape(name, x, y), radius(r) {}

    ~Circle() override {
        std::cout << "  Circle '" << name << "' destroyed\n";
    }

    float area() const override {
        return 3.14159f * radius * radius;
    }

    float perimeter() const override {
        return 2.0f * 3.14159f * radius;
    }

    void render() const override {
        std::cout << "  [GPU] Rendering circle '" << name << "' with radius " << radius << "\n";
    }

    void printInfo() const override {
        Shape::printInfo();  // Call base class version
        std::cout << "  Type: Circle, Radius: " << radius << "\n";
    }
};

class Rectangle : public Shape {
private:
    float width, height;

public:
    Rectangle(const std::string& name, float x, float y, float w, float h)
        : Shape(name, x, y), width(w), height(h) {}

    ~Rectangle() override {
        std::cout << "  Rectangle '" << name << "' destroyed\n";
    }

    float area() const override {
        return width * height;
    }

    float perimeter() const override {
        return 2.0f * (width + height);
    }

    void render() const override {
        std::cout << "  [GPU] Rendering rectangle '" << name
                  << "' with size " << width << "x" << height << "\n";
    }

    void printInfo() const override {
        Shape::printInfo();
        std::cout << "  Type: Rectangle, Size: " << width << "x" << height << "\n";
    }
};

class Triangle : public Shape {
private:
    float base, height;

public:
    Triangle(const std::string& name, float x, float y, float b, float h)
        : Shape(name, x, y), base(b), height(h) {}

    ~Triangle() override {
        std::cout << "  Triangle '" << name << "' destroyed\n";
    }

    float area() const override {
        return 0.5f * base * height;
    }

    float perimeter() const override {
        // Simplified: assume right triangle
        float hypotenuse = std::sqrt(base*base + height*height);
        return base + height + hypotenuse;
    }

    void render() const override {
        std::cout << "  [GPU] Rendering triangle '" << name
                  << "' with base " << base << " and height " << height << "\n";
    }

    void printInfo() const override {
        Shape::printInfo();
        std::cout << "  Type: Triangle, Base: " << base << ", Height: " << height << "\n";
    }
};

// Example 2: Ray Tracing with polymorphism
struct Ray {
    float origin[3];
    float direction[3];
};

struct HitRecord {
    bool hit;
    float distance;
    float point[3];
    std::string objectName;
};

class Geometry {
protected:
    std::string name;

public:
    Geometry(const std::string& geoName) : name(geoName) {}
    virtual ~Geometry() {}

    virtual HitRecord intersect(const Ray& ray) const = 0;
    virtual void printType() const = 0;

    std::string getName() const { return name; }
};

class Sphere : public Geometry {
private:
    float center[3];
    float radius;

public:
    Sphere(const std::string& name, float cx, float cy, float cz, float r)
        : Geometry(name), radius(r) {
        center[0] = cx;
        center[1] = cy;
        center[2] = cz;
    }

    HitRecord intersect(const Ray& ray) const override {
        std::cout << "    Testing sphere '" << name << "' intersection\n";
        // Simplified intersection (not real math)
        HitRecord hit;
        hit.hit = true;
        hit.distance = 5.0f;
        hit.objectName = name;
        return hit;
    }

    void printType() const override {
        std::cout << "  Sphere '" << name << "' at ("
                  << center[0] << ", " << center[1] << ", " << center[2]
                  << ") r=" << radius << "\n";
    }
};

class Plane : public Geometry {
private:
    float normal[3];
    float distance;

public:
    Plane(const std::string& name, float nx, float ny, float nz, float d)
        : Geometry(name), distance(d) {
        normal[0] = nx;
        normal[1] = ny;
        normal[2] = nz;
    }

    HitRecord intersect(const Ray& ray) const override {
        std::cout << "    Testing plane '" << name << "' intersection\n";
        HitRecord hit;
        hit.hit = true;
        hit.distance = 10.0f;
        hit.objectName = name;
        return hit;
    }

    void printType() const override {
        std::cout << "  Plane '" << name << "' normal ("
                  << normal[0] << ", " << normal[1] << ", " << normal[2]
                  << ") d=" << distance << "\n";
    }
};

// Demonstrate polymorphism with base class pointers
void demonstrateShapePolymorphism() {
    std::cout << "\n=== Shape Polymorphism Demo ===\n";

    std::cout << "\nCreating shapes:\n";
    // Base class pointers to derived objects
    Shape* shapes[3];
    shapes[0] = new Circle("Circle1", 0, 0, 5);
    shapes[1] = new Rectangle("Rect1", 10, 10, 8, 6);
    shapes[2] = new Triangle("Tri1", 20, 20, 4, 3);

    // Polymorphic behavior: same interface, different implementations
    std::cout << "\nCalling virtual functions polymorphically:\n";
    for (int i = 0; i < 3; ++i) {
        std::cout << "\nShape " << i << ":\n";
        shapes[i]->printInfo();
        std::cout << "  Area: " << shapes[i]->area() << "\n";
        std::cout << "  Perimeter: " << shapes[i]->perimeter() << "\n";
        shapes[i]->render();
    }

    // Calculate total area polymorphically
    std::cout << "\nCalculating total area:\n";
    float totalArea = 0.0f;
    for (int i = 0; i < 3; ++i) {
        totalArea += shapes[i]->area();
    }
    std::cout << "Total area of all shapes: " << totalArea << "\n";

    // Cleanup: Virtual destructor ensures proper cleanup
    std::cout << "\nCleaning up (virtual destructors in action):\n";
    for (int i = 0; i < 3; ++i) {
        delete shapes[i];
    }
}

// Demonstrate with vector (more C++ style)
void demonstrateWithVector() {
    std::cout << "\n=== Polymorphism with std::vector ===\n";

    std::cout << "\nCreating shape collection:\n";
    std::vector<Shape*> scene;
    scene.push_back(new Circle("C1", 0, 0, 3));
    scene.push_back(new Circle("C2", 5, 5, 2));
    scene.push_back(new Rectangle("R1", 10, 0, 4, 6));
    scene.push_back(new Triangle("T1", 15, 15, 3, 4));

    std::cout << "\nRendering all shapes in scene:\n";
    for (Shape* shape : scene) {
        shape->render();
    }

    std::cout << "\nFinding largest shape:\n";
    Shape* largest = nullptr;
    float maxArea = 0.0f;
    for (Shape* shape : scene) {
        float area = shape->area();
        if (area > maxArea) {
            maxArea = area;
            largest = shape;
        }
    }
    if (largest) {
        std::cout << "Largest shape: " << largest->getName()
                  << " with area " << maxArea << "\n";
    }

    // Cleanup
    std::cout << "\nCleaning up scene:\n";
    for (Shape* shape : scene) {
        delete shape;
    }
}

// Demonstrate ray tracing with polymorphism
void demonstrateRayTracing() {
    std::cout << "\n=== Ray Tracing Polymorphism Demo ===\n";

    std::vector<Geometry*> scene;
    scene.push_back(new Sphere("RedSphere", 0, 0, -5, 1));
    scene.push_back(new Sphere("BlueSphere", 2, 0, -6, 0.5f));
    scene.push_back(new Plane("Ground", 0, 1, 0, -2));

    std::cout << "\nScene geometry:\n";
    for (Geometry* geo : scene) {
        geo->printType();
    }

    // Test ray intersection polymorphically
    std::cout << "\nTesting ray intersections:\n";
    Ray testRay;
    testRay.origin[0] = 0;
    testRay.origin[1] = 0;
    testRay.origin[2] = 0;
    testRay.direction[0] = 0;
    testRay.direction[1] = 0;
    testRay.direction[2] = -1;

    std::cout << "  Casting ray from origin in -Z direction\n";
    for (Geometry* geo : scene) {
        HitRecord hit = geo->intersect(testRay);
        if (hit.hit) {
            std::cout << "    HIT: " << hit.objectName
                      << " at distance " << hit.distance << "\n";
        }
    }

    // Cleanup
    for (Geometry* geo : scene) {
        delete geo;
    }
}

// Demonstrate why virtual destructor is critical
class BadBase {
public:
    int* data;
    BadBase() {
        data = new int[100];
        std::cout << "  BadBase allocated memory\n";
    }
    // NON-VIRTUAL destructor - BAD!
    ~BadBase() {
        std::cout << "  BadBase destructor (freeing memory)\n";
        delete[] data;
    }
};

class BadDerived : public BadBase {
public:
    int* moreData;
    BadDerived() {
        moreData = new int[100];
        std::cout << "  BadDerived allocated more memory\n";
    }
    ~BadDerived() {
        std::cout << "  BadDerived destructor (freeing memory)\n";
        delete[] moreData;
    }
};

void demonstrateVirtualDestructor() {
    std::cout << "\n=== Virtual Destructor Importance ===\n";

    std::cout << "\nWARNING: This demonstrates a memory leak!\n";
    std::cout << "Creating BadDerived through base pointer:\n";

    // Uncommenting this causes memory leak (BadDerived destructor not called):
    /*
    BadBase* ptr = new BadDerived();
    delete ptr;  // Only calls BadBase destructor, leaks BadDerived::moreData!
    */

    std::cout << "(Example commented out to prevent leak)\n";
    std::cout << "\nProblem: Without virtual destructor, only base destructor is called\n";
    std::cout << "Solution: Always make destructor virtual in polymorphic base classes\n";
}

int main() {
    std::cout << "=== MODULE 5: Polymorphism Demo ===\n";

    demonstrateShapePolymorphism();
    demonstrateWithVector();
    demonstrateRayTracing();
    demonstrateVirtualDestructor();

    std::cout << "\n=== Program Complete ===\n";
    return 0;
}

/*
 * KEY CONCEPTS DEMONSTRATED:
 *
 * 1. POLYMORPHISM:
 *    - Same interface (base class pointer)
 *    - Different implementations (derived class behavior)
 *    - Runtime binding (virtual function dispatch)
 *
 * 2. VIRTUAL FUNCTIONS:
 *    - Declared with 'virtual' keyword in base class
 *    - Overridden with 'override' keyword in derived class
 *    - Dispatched at runtime based on actual object type
 *
 * 3. VIRTUAL DESTRUCTOR:
 *    - CRITICAL for polymorphic classes
 *    - Ensures derived class destructor is called
 *    - Prevents memory leaks
 *
 * 4. VTABLE MECHANISM:
 *    - Each polymorphic class has a vtable (virtual function table)
 *    - Each object has a vptr (pointer to vtable)
 *    - Runtime dispatch: follow vptr -> vtable -> correct function
 *
 * 5. PURE VIRTUAL FUNCTIONS:
 *    - Syntax: virtual func() = 0;
 *    - Makes class abstract (cannot instantiate)
 *    - Defines interface contract
 *
 * TRY THIS:
 * 1. Add a new shape type (Ellipse) and test polymorphism
 * 2. Implement a draw() function that takes vector<Shape*>
 * 3. Add a Material base class with Diffuse, Metallic, Glass derived classes
 * 4. Create a Light hierarchy (PointLight, DirectionalLight, SpotLight)
 * 5. Implement a filter pipeline using polymorphism
 * 6. Uncomment the virtual destructor example to see the leak
 *
 * COMMON MISTAKES:
 * - Forgetting virtual destructor (MEMORY LEAK!)
 * - Not using 'override' keyword (typos become hidden bugs)
 * - Trying to instantiate abstract classes
 * - Slicing objects (copying derived through base)
 * - Calling virtual functions in constructors/destructors
 *
 * GPU CONNECTION:
 * - Ray tracing: Uniform intersection test on different geometry types
 * - Rendering pipeline: Different renderable objects with same interface
 * - Shader system: Different shader types (vertex, fragment, compute)
 * - Resource management: Different GPU resource types
 * - Post-processing effects: Chain different filters polymorphically
 *
 * PERFORMANCE NOTE:
 * - Virtual function call has small overhead (vtable lookup)
 * - Usually negligible compared to actual work
 * - GPU kernels avoid virtual functions (compile-time dispatch instead)
 * - CPU-side scene management uses polymorphism extensively
 *
 * REAL GPU EXAMPLE:
 * Ray tracing typically uses polymorphism on CPU for scene setup,
 * then flattens to arrays for GPU processing:
 *
 * CPU (C++):
 *   vector<Geometry*> scene;
 *   scene.push_back(new Sphere(...));
 *   scene.push_back(new Plane(...));
 *   // Use polymorphism for setup
 *
 * GPU (CUDA):
 *   // Flatten to arrays: spheres[], planes[]
 *   // Use if/switch for type dispatch (faster than vtable on GPU)
 */
