/*
 * MODULE 5: Inheritance and Polymorphism
 * File: 01_basic_inheritance.cpp
 *
 * TOPIC: Basic Inheritance and IS-A Relationships
 *
 * CONCEPTS:
 * - Inheritance: Derive new classes from existing ones
 * - Base class (parent) and derived class (child)
 * - IS-A relationship (Circle IS-A Shape)
 * - Code reuse through inheritance
 * - Protected access specifier
 *
 * GPU RELEVANCE:
 * - Shape hierarchies for ray tracing (Sphere IS-A Shape, Box IS-A Shape)
 * - Shader programs (VertexShader IS-A Shader)
 * - Buffer types (VertexBuffer IS-A Buffer, IndexBuffer IS-A Buffer)
 * - Texture types (Texture2D IS-A Texture, Texture3D IS-A Texture)
 *
 * COMPILE: g++ -std=c++17 -o basic_inheritance 01_basic_inheritance.cpp
 */

#include <iostream>
#include <string>
#include <cmath>

// Example 1: Basic Shape hierarchy for 2D graphics
class Shape {
protected:  // Accessible by derived classes
    float x, y;        // Position
    std::string color;

public:
    // Constructor
    Shape(float x_pos, float y_pos, const std::string& col)
        : x(x_pos), y(y_pos), color(col) {
        std::cout << "Shape constructor called\n";
    }

    // Destructor (should be virtual - we'll explain why in next file)
    virtual ~Shape() {
        std::cout << "Shape destructor called\n";
    }

    // Getters
    float getX() const { return x; }
    float getY() const { return y; }
    std::string getColor() const { return color; }

    // Methods that all shapes have
    void moveTo(float new_x, float new_y) {
        x = new_x;
        y = new_y;
    }

    void setColor(const std::string& col) {
        color = col;
    }

    // Virtual function (can be overridden by derived classes)
    virtual void draw() const {
        std::cout << "Drawing Shape at (" << x << ", " << y << ") with color " << color << "\n";
    }

    // Pure virtual function (MUST be overridden)
    virtual float area() const = 0;  // Makes Shape abstract
};

// Derived class: Circle IS-A Shape
class Circle : public Shape {  // public inheritance
private:
    float radius;

public:
    // Constructor: must call base class constructor
    Circle(float x_pos, float y_pos, float r, const std::string& col)
        : Shape(x_pos, y_pos, col), radius(r) {  // Initialize base class
        std::cout << "Circle constructor called\n";
    }

    ~Circle() {
        std::cout << "Circle destructor called\n";
    }

    float getRadius() const { return radius; }

    // Override virtual function
    void draw() const override {
        std::cout << "Drawing Circle at (" << x << ", " << y
                  << ") with radius " << radius
                  << " and color " << color << "\n";
    }

    // Implement pure virtual function
    float area() const override {
        return 3.14159f * radius * radius;
    }
};

// Derived class: Rectangle IS-A Shape
class Rectangle : public Shape {
private:
    float width, height;

public:
    Rectangle(float x_pos, float y_pos, float w, float h, const std::string& col)
        : Shape(x_pos, y_pos, col), width(w), height(h) {
        std::cout << "Rectangle constructor called\n";
    }

    ~Rectangle() {
        std::cout << "Rectangle destructor called\n";
    }

    float getWidth() const { return width; }
    float getHeight() const { return height; }

    void draw() const override {
        std::cout << "Drawing Rectangle at (" << x << ", " << y
                  << ") with size " << width << "x" << height
                  << " and color " << color << "\n";
    }

    float area() const override {
        return width * height;
    }
};

// Example 2: Buffer hierarchy (like GPU buffers)
class Buffer {
protected:
    void* data;
    size_t sizeBytes;
    std::string name;

public:
    Buffer(const std::string& bufferName, size_t size)
        : name(bufferName), sizeBytes(size) {
        data = malloc(size);
        std::cout << "Buffer '" << name << "' allocated " << size << " bytes\n";
    }

    virtual ~Buffer() {
        std::cout << "Buffer '" << name << "' freed\n";
        free(data);
    }

    size_t getSize() const { return sizeBytes; }
    const std::string& getName() const { return name; }

    // Virtual function for binding (like OpenGL bind)
    virtual void bind() const {
        std::cout << "Binding generic Buffer '" << name << "'\n";
    }

    virtual void printInfo() const {
        std::cout << "Buffer '" << name << "': " << sizeBytes << " bytes\n";
    }
};

// Derived class: VertexBuffer IS-A Buffer
class VertexBuffer : public Buffer {
private:
    int vertexCount;
    int vertexSize;  // bytes per vertex

public:
    VertexBuffer(const std::string& name, int numVertices, int vertexSizeBytes)
        : Buffer(name, numVertices * vertexSizeBytes),
          vertexCount(numVertices),
          vertexSize(vertexSizeBytes) {
        std::cout << "VertexBuffer created for " << vertexCount << " vertices\n";
    }

    ~VertexBuffer() {
        std::cout << "VertexBuffer destroyed\n";
    }

    void bind() const override {
        std::cout << "Binding VertexBuffer '" << name << "' (GL_ARRAY_BUFFER)\n";
    }

    void printInfo() const override {
        std::cout << "VertexBuffer '" << name << "':\n";
        std::cout << "  Vertices: " << vertexCount << "\n";
        std::cout << "  Vertex size: " << vertexSize << " bytes\n";
        std::cout << "  Total size: " << sizeBytes << " bytes\n";
    }

    int getVertexCount() const { return vertexCount; }
};

// Derived class: IndexBuffer IS-A Buffer
class IndexBuffer : public Buffer {
private:
    int indexCount;

public:
    IndexBuffer(const std::string& name, int numIndices)
        : Buffer(name, numIndices * sizeof(int)),
          indexCount(numIndices) {
        std::cout << "IndexBuffer created for " << indexCount << " indices\n";
    }

    ~IndexBuffer() {
        std::cout << "IndexBuffer destroyed\n";
    }

    void bind() const override {
        std::cout << "Binding IndexBuffer '" << name << "' (GL_ELEMENT_ARRAY_BUFFER)\n";
    }

    void printInfo() const override {
        std::cout << "IndexBuffer '" << name << "':\n";
        std::cout << "  Indices: " << indexCount << "\n";
        std::cout << "  Total size: " << sizeBytes << " bytes\n";
    }

    int getIndexCount() const { return indexCount; }
};

// Example 3: Three-level hierarchy for GPU objects
class RayTraceable {
protected:
    std::string objectName;

public:
    RayTraceable(const std::string& name) : objectName(name) {}
    virtual ~RayTraceable() {}

    virtual bool intersect(float rayOrigin[3], float rayDir[3]) const = 0;
    virtual void printType() const = 0;
};

class Sphere : public RayTraceable {
private:
    float centerX, centerY, centerZ;
    float radius;

public:
    Sphere(const std::string& name, float x, float y, float z, float r)
        : RayTraceable(name), centerX(x), centerY(y), centerZ(z), radius(r) {}

    bool intersect(float rayOrigin[3], float rayDir[3]) const override {
        std::cout << "  Testing ray intersection with sphere '" << objectName << "'\n";
        // Simplified: just return true if ray is close
        return true;
    }

    void printType() const override {
        std::cout << "Sphere '" << objectName << "' at ("
                  << centerX << ", " << centerY << ", " << centerZ
                  << ") radius " << radius << "\n";
    }
};

class Box : public RayTraceable {
private:
    float minX, minY, minZ;
    float maxX, maxY, maxZ;

public:
    Box(const std::string& name, float x1, float y1, float z1,
        float x2, float y2, float z2)
        : RayTraceable(name), minX(x1), minY(y1), minZ(z1),
          maxX(x2), maxY(y2), maxZ(z2) {}

    bool intersect(float rayOrigin[3], float rayDir[3]) const override {
        std::cout << "  Testing ray intersection with box '" << objectName << "'\n";
        return true;
    }

    void printType() const override {
        std::cout << "Box '" << objectName << "' from ("
                  << minX << ", " << minY << ", " << minZ << ") to ("
                  << maxX << ", " << maxY << ", " << maxZ << ")\n";
    }
};

// Demonstrate shape hierarchy
void demonstrateShapes() {
    std::cout << "\n=== Shape Hierarchy Demo ===\n";

    // Cannot create Shape directly (abstract class)
    // Shape s(0, 0, "red");  // ERROR!

    Circle circle(10, 20, 5, "red");
    Rectangle rect(30, 40, 15, 10, "blue");

    circle.draw();
    rect.draw();

    std::cout << "\nCircle area: " << circle.area() << "\n";
    std::cout << "Rectangle area: " << rect.area() << "\n";

    // Moving shapes (inherited method)
    std::cout << "\nMoving shapes...\n";
    circle.moveTo(100, 200);
    circle.draw();

    std::cout << "\n";
}

// Demonstrate buffer hierarchy
void demonstrateBuffers() {
    std::cout << "\n=== Buffer Hierarchy Demo ===\n";

    VertexBuffer vbo("vertices", 1000, 32);  // 1000 vertices, 32 bytes each
    IndexBuffer ibo("indices", 3000);         // 3000 indices

    vbo.printInfo();
    ibo.printInfo();

    std::cout << "\nBinding buffers:\n";
    vbo.bind();
    ibo.bind();

    std::cout << "\n";
}

// Demonstrate ray tracing hierarchy
void demonstrateRayTracing() {
    std::cout << "\n=== Ray Tracing Hierarchy Demo ===\n";

    Sphere sphere("RedSphere", 0, 0, -5, 1);
    Box box("BlueBox", -2, -2, -10, 2, 2, -6);

    std::cout << "Scene objects:\n";
    sphere.printType();
    box.printType();

    std::cout << "\nTesting ray intersections:\n";
    float rayOrigin[3] = {0, 0, 0};
    float rayDir[3] = {0, 0, -1};

    sphere.intersect(rayOrigin, rayDir);
    box.intersect(rayOrigin, rayDir);

    std::cout << "\n";
}

// Demonstrate constructor/destructor order
void demonstrateLifecycle() {
    std::cout << "\n=== Constructor/Destructor Order Demo ===\n";
    std::cout << "Creating Circle:\n";

    {
        Circle c(0, 0, 1, "green");
        std::cout << "\nCircle in use...\n";
    }  // Destructors called in reverse order of construction

    std::cout << "\nCircle destroyed\n";
}

int main() {
    std::cout << "=== MODULE 5: Basic Inheritance Demo ===\n";

    demonstrateShapes();
    demonstrateBuffers();
    demonstrateRayTracing();
    demonstrateLifecycle();

    std::cout << "\n=== Program Complete ===\n";
    return 0;
}

/*
 * KEY CONCEPTS DEMONSTRATED:
 *
 * 1. INHERITANCE SYNTAX:
 *    class Derived : public Base { ... };
 *    - public inheritance: IS-A relationship
 *    - Derived inherits all members from Base
 *
 * 2. ACCESS SPECIFIERS:
 *    - private: Only accessible within the class
 *    - protected: Accessible within class and derived classes
 *    - public: Accessible everywhere
 *
 * 3. VIRTUAL FUNCTIONS:
 *    - virtual: Can be overridden by derived classes
 *    - = 0: Pure virtual (must be overridden, makes class abstract)
 *    - override: Explicitly marks function as overriding base
 *
 * 4. CONSTRUCTOR/DESTRUCTOR ORDER:
 *    - Construction: Base first, then Derived
 *    - Destruction: Derived first, then Base (reverse order)
 *
 * 5. ABSTRACT CLASSES:
 *    - Has at least one pure virtual function
 *    - Cannot instantiate directly
 *    - Used as interfaces/base classes
 *
 * TRY THIS:
 * 1. Add a Triangle class derived from Shape
 * 2. Create a TextureBuffer derived from Buffer
 * 3. Add a perimeter() virtual method to Shape hierarchy
 * 4. Implement a Plane class derived from RayTraceable
 * 5. Add a Material base class with Diffuse and Metallic derived classes
 * 6. Create a three-level hierarchy: Object -> Renderable -> Mesh
 *
 * COMMON MISTAKES:
 * - Forgetting to call base class constructor
 * - Not using virtual destructors (memory leaks!)
 * - Confusing IS-A with HAS-A (use composition instead)
 * - Making everything public in derived class
 *
 * GPU CONNECTION:
 * - Ray tracing: Different shape types all derive from Intersectable
 * - OpenGL/Vulkan: Different buffer types (Vertex, Index, Uniform)
 * - Shaders: VertexShader, FragmentShader derive from Shader
 * - Textures: Texture2D, Texture3D, TextureCube derive from Texture
 * - Inheritance helps organize GPU resource hierarchies
 *
 * WHEN TO USE INHERITANCE:
 * - Use inheritance for IS-A relationships
 * - Use composition for HAS-A relationships
 * - "Prefer composition over inheritance" (but use inheritance when appropriate)
 * - Good for: Type hierarchies, polymorphism, interface definitions
 */
