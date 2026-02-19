/*
 * MODULE 4: Classes and Object-Oriented Programming
 * File: 04_composition.cpp
 *
 * TOPIC: Object Composition and Aggregation
 *
 * CONCEPTS:
 * - Composition: "has-a" relationship (car HAS-A engine)
 * - Aggregation: Looser "has-a" (university HAS students)
 * - Composition: Contained object's lifetime tied to container
 * - Building complex objects from simpler parts
 *
 * GPU RELEVANCE:
 * - Scene graphs: Node HAS Transform, Material, Mesh
 * - Particle systems: ParticleSystem HAS position buffer, velocity buffer
 * - Ray tracers: Scene HAS objects, camera, lights
 * - Render pipelines: Pipeline HAS shaders, textures, buffers
 *
 * COMPILE: g++ -std=c++17 -o composition 04_composition.cpp
 */

#include <iostream>
#include <string>
#include <vector>
#include <cmath>

// Basic building blocks
class Vec3 {
public:
    float x, y, z;

    Vec3(float x_ = 0, float y_ = 0, float z_ = 0) : x(x_), y(y_), z(z_) {}

    float magnitude() const {
        return std::sqrt(x*x + y*y + z*z);
    }

    void print() const {
        std::cout << "(" << x << ", " << y << ", " << z << ")";
    }

    Vec3 operator+(const Vec3& other) const {
        return Vec3(x + other.x, y + other.y, z + other.z);
    }

    Vec3 operator*(float scalar) const {
        return Vec3(x * scalar, y * scalar, z * scalar);
    }
};

// Example 1: Composition in Graphics - Transform class
class Transform {
private:
    Vec3 position;
    Vec3 rotation;  // Euler angles in degrees
    Vec3 scale;

public:
    Transform()
        : position(0, 0, 0), rotation(0, 0, 0), scale(1, 1, 1) {}

    Transform(const Vec3& pos, const Vec3& rot, const Vec3& scl)
        : position(pos), rotation(rot), scale(scl) {}

    // Accessors
    const Vec3& getPosition() const { return position; }
    const Vec3& getRotation() const { return rotation; }
    const Vec3& getScale() const { return scale; }

    void setPosition(const Vec3& pos) { position = pos; }
    void setRotation(const Vec3& rot) { rotation = rot; }
    void setScale(const Vec3& scl) { scale = scl; }

    void translate(const Vec3& delta) {
        position = position + delta;
    }

    void print() const {
        std::cout << "Transform:\n";
        std::cout << "  Position: ";
        position.print();
        std::cout << "\n  Rotation: ";
        rotation.print();
        std::cout << "\n  Scale: ";
        scale.print();
        std::cout << "\n";
    }
};

// Example 2: Composition - Material class
class Material {
private:
    Vec3 albedoColor;
    float roughness;
    float metallic;

public:
    Material()
        : albedoColor(1, 1, 1), roughness(0.5f), metallic(0.0f) {}

    Material(const Vec3& color, float rough, float metal)
        : albedoColor(color), roughness(rough), metallic(metal) {}

    void print() const {
        std::cout << "Material:\n";
        std::cout << "  Albedo: ";
        albedoColor.print();
        std::cout << "\n  Roughness: " << roughness;
        std::cout << "\n  Metallic: " << metallic << "\n";
    }
};

// Example 3: Composition - Mesh Data
class MeshData {
private:
    std::vector<Vec3> vertices;
    std::vector<int> indices;

public:
    MeshData() {}

    void addVertex(const Vec3& v) {
        vertices.push_back(v);
    }

    void addTriangle(int i0, int i1, int i2) {
        indices.push_back(i0);
        indices.push_back(i1);
        indices.push_back(i2);
    }

    size_t getVertexCount() const { return vertices.size(); }
    size_t getTriangleCount() const { return indices.size() / 3; }

    void print() const {
        std::cout << "MeshData:\n";
        std::cout << "  Vertices: " << vertices.size() << "\n";
        std::cout << "  Triangles: " << getTriangleCount() << "\n";
    }
};

// Example 4: Complex composition - RenderObject has multiple components
class RenderObject {
private:
    std::string name;
    Transform transform;     // Composition: RenderObject HAS-A Transform
    Material material;       // Composition: RenderObject HAS-A Material
    MeshData mesh;          // Composition: RenderObject HAS-A MeshData
    bool isVisible;

public:
    RenderObject(const std::string& objName)
        : name(objName), transform(), material(), mesh(), isVisible(true) {
        std::cout << "RenderObject '" << name << "' created\n";
    }

    // Initialize with components
    RenderObject(const std::string& objName,
                 const Transform& trans,
                 const Material& mat)
        : name(objName), transform(trans), material(mat), mesh(), isVisible(true) {
        std::cout << "RenderObject '" << name << "' created with components\n";
    }

    ~RenderObject() {
        std::cout << "RenderObject '" << name << "' destroyed\n";
    }

    // Access to components
    Transform& getTransform() { return transform; }
    const Transform& getTransform() const { return transform; }

    Material& getMaterial() { return material; }
    const Material& getMaterial() const { return material; }

    MeshData& getMesh() { return mesh; }
    const MeshData& getMesh() const { return mesh; }

    void setVisible(bool visible) { isVisible = visible; }
    bool getVisible() const { return isVisible; }

    const std::string& getName() const { return name; }

    // High-level operations
    void moveTo(const Vec3& position) {
        transform.setPosition(position);
    }

    void moveBy(const Vec3& delta) {
        transform.translate(delta);
    }

    void print() const {
        std::cout << "\n=== RenderObject: " << name << " ===\n";
        std::cout << "Visible: " << (isVisible ? "Yes" : "No") << "\n";
        transform.print();
        material.print();
        mesh.print();
    }
};

// Example 5: Scene - Aggregation (Scene HAS RenderObjects)
class Scene {
private:
    std::string name;
    std::vector<RenderObject*> objects;  // Aggregation: pointers to objects
    // Note: Using pointers for aggregation (loose ownership)
    // In real code, might use smart pointers

public:
    Scene(const std::string& sceneName) : name(sceneName) {
        std::cout << "Scene '" << name << "' created\n";
    }

    ~Scene() {
        std::cout << "Scene '" << name << "' destroyed\n";
        // Note: Not deleting objects (aggregation, not composition)
        // Caller manages object lifetime
    }

    void addObject(RenderObject* obj) {
        objects.push_back(obj);
        std::cout << "  Added '" << obj->getName() << "' to scene '" << name << "'\n";
    }

    void removeObject(RenderObject* obj) {
        auto it = std::find(objects.begin(), objects.end(), obj);
        if (it != objects.end()) {
            std::cout << "  Removed '" << (*it)->getName() << "' from scene '" << name << "'\n";
            objects.erase(it);
        }
    }

    size_t getObjectCount() const {
        return objects.size();
    }

    void printSummary() const {
        std::cout << "\n=== Scene: " << name << " ===\n";
        std::cout << "Objects: " << objects.size() << "\n";
        for (const auto* obj : objects) {
            std::cout << "  - " << obj->getName()
                      << " (visible: " << (obj->getVisible() ? "yes" : "no") << ")\n";
        }
    }

    void printDetailed() const {
        std::cout << "\n=== Scene: " << name << " (Detailed) ===\n";
        for (const auto* obj : objects) {
            obj->print();
        }
    }
};

// Example 6: Particle System with composition
class ParticleSystem {
private:
    std::vector<Vec3> positions;
    std::vector<Vec3> velocities;
    std::vector<float> lifetimes;
    Vec3 emitterPosition;
    int maxParticles;
    int activeParticles;

public:
    ParticleSystem(const Vec3& emitterPos, int maxCount)
        : emitterPosition(emitterPos), maxParticles(maxCount), activeParticles(0) {
        positions.reserve(maxParticles);
        velocities.reserve(maxParticles);
        lifetimes.reserve(maxParticles);
        std::cout << "ParticleSystem created at ";
        emitterPosition.print();
        std::cout << "\n";
    }

    void emit(int count) {
        for (int i = 0; i < count && activeParticles < maxParticles; ++i) {
            positions.push_back(emitterPosition);
            velocities.push_back(Vec3(
                (rand() % 200 - 100) / 100.0f,
                (rand() % 200) / 100.0f,
                (rand() % 200 - 100) / 100.0f
            ));
            lifetimes.push_back(2.0f);  // 2 second lifetime
            activeParticles++;
        }
    }

    void update(float deltaTime) {
        // Simple physics update
        for (int i = 0; i < activeParticles; ++i) {
            positions[i] = positions[i] + velocities[i] * deltaTime;
            velocities[i] = velocities[i] + Vec3(0, -9.8f, 0) * deltaTime;  // gravity
            lifetimes[i] -= deltaTime;
        }

        // Remove dead particles (simplified)
        // In real code, would compact arrays
    }

    void print() const {
        std::cout << "ParticleSystem: " << activeParticles << "/" << maxParticles << " active\n";
        std::cout << "  Emitter position: ";
        emitterPosition.print();
        std::cout << "\n";
    }
};

// Demonstrate composition
void demonstrateComposition() {
    std::cout << "\n=== Demonstrating Composition ===\n";

    // Create a render object with composed components
    RenderObject sphere("Sphere");

    // Setup transform
    sphere.getTransform().setPosition(Vec3(5, 2, 3));
    sphere.getTransform().setScale(Vec3(2, 2, 2));

    // Setup material
    sphere.getMaterial() = Material(Vec3(1, 0, 0), 0.3f, 0.8f);  // Red, smooth, metallic

    // Setup mesh
    sphere.getMesh().addVertex(Vec3(0, 1, 0));
    sphere.getMesh().addVertex(Vec3(-1, 0, 0));
    sphere.getMesh().addVertex(Vec3(1, 0, 0));
    sphere.getMesh().addTriangle(0, 1, 2);

    sphere.print();

    // The sphere's components are automatically destroyed when sphere is destroyed
}

// Demonstrate aggregation
void demonstrateAggregation() {
    std::cout << "\n=== Demonstrating Aggregation ===\n";

    // Create objects
    RenderObject cube("Cube");
    RenderObject pyramid("Pyramid");
    RenderObject plane("Plane");

    cube.moveTo(Vec3(0, 0, 0));
    pyramid.moveTo(Vec3(5, 0, 0));
    plane.moveTo(Vec3(0, -1, 0));

    // Create scene and add objects (aggregation)
    Scene mainScene("Main Scene");
    mainScene.addObject(&cube);
    mainScene.addObject(&pyramid);
    mainScene.addObject(&plane);

    mainScene.printSummary();

    // Remove one object
    mainScene.removeObject(&pyramid);
    mainScene.printSummary();

    // Objects still exist even after being removed from scene
    std::cout << "\nPyramid still exists after removal from scene:\n";
    pyramid.print();

    // Scene destroyed here, but objects continue to exist
}

// Demonstrate particle system
void demonstrateParticleSystem() {
    std::cout << "\n=== Demonstrating Particle System ===\n";

    ParticleSystem particles(Vec3(0, 10, 0), 100);

    particles.emit(10);
    particles.print();

    std::cout << "\nUpdating particles...\n";
    particles.update(0.016f);  // ~60 FPS timestep
    particles.update(0.016f);
    particles.update(0.016f);

    particles.print();
}

int main() {
    std::cout << "=== MODULE 4: Composition and Aggregation Demo ===\n";

    demonstrateComposition();
    demonstrateAggregation();
    demonstrateParticleSystem();

    std::cout << "\n=== Program Complete ===\n";
    return 0;
}

/*
 * KEY CONCEPTS DEMONSTRATED:
 *
 * 1. COMPOSITION vs AGGREGATION:
 *    - Composition: Strong "has-a", contained object destroyed with container
 *    - Aggregation: Weak "has-a", contained object can exist independently
 *
 * 2. COMPOSITION BENEFITS:
 *    - Build complex objects from simple parts
 *    - Better encapsulation than inheritance
 *    - More flexible (can change components at runtime)
 *    - Automatic lifecycle management
 *
 * 3. DESIGN PRINCIPLE:
 *    - "Favor composition over inheritance"
 *    - Use inheritance for IS-A relationships
 *    - Use composition for HAS-A relationships
 *
 * 4. EXAMPLES:
 *    - RenderObject HAS Transform, Material, Mesh (composition)
 *    - Scene HAS RenderObjects (aggregation)
 *    - ParticleSystem HAS position/velocity arrays (composition)
 *
 * TRY THIS:
 * 1. Add a Camera class that HAS Transform and projection parameters
 * 2. Create a Light class (position, color, intensity)
 * 3. Add lights to Scene using aggregation
 * 4. Implement a GameObject that can have multiple "Components"
 * 5. Create a MaterialLibrary that manages multiple materials
 * 6. Add a BoundingBox component for collision detection
 *
 * COMMON MISTAKES:
 * - Using inheritance when composition is better
 * - Confusing composition with aggregation
 * - Not considering object ownership and lifetime
 * - Over-complicated hierarchies
 *
 * GPU CONNECTION:
 * - GPU scene graphs use composition extensively
 * - RenderObject pattern common in graphics engines
 * - Particle systems: positions/velocities as separate buffers on GPU
 * - Material systems: compose textures, shaders, parameters
 * - Ray tracing: Scene contains geometries, materials, lights
 *
 * REAL GPU EXAMPLE:
 * In CUDA/OpenGL, you might have:
 * - Mesh class HAS vertex buffer (GPU memory)
 * - Mesh class HAS index buffer (GPU memory)
 * - Scene HAS array of mesh pointers
 * - Each component manages its own GPU resources using RAII
 */
