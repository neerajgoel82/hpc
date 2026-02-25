/*
 * MODULE 9: Modern C++ Features
 * File: 01_smart_pointers.cpp
 *
 * TOPIC: Smart Pointers (unique_ptr, shared_ptr, weak_ptr)
 *
 * CONCEPTS:
 * - Automatic memory management
 * - RAII for pointers
 * - Ownership semantics
 * - unique_ptr: Exclusive ownership
 * - shared_ptr: Shared ownership with reference counting
 * - weak_ptr: Non-owning reference
 *
 * GPU RELEVANCE:
 * - Managing GPU resources (textures, buffers, shaders)
 * - Scene graph node management
 * - Texture/model loading and caching
 * - Preventing resource leaks in complex applications
 *
 * COMPILE: g++ -std=c++17 -o smart_pointers 01_smart_pointers.cpp
 */

#include <iostream>
#include <memory>
#include <vector>
#include <string>

// Example class to track lifetime
class Texture {
private:
    std::string name;
    int width, height;

public:
    Texture(const std::string& texName, int w, int h)
        : name(texName), width(w), height(h) {
        std::cout << "  [CREATE] Texture '" << name << "' (" << width << "x" << height << ")\n";
    }

    ~Texture() {
        std::cout << "  [DESTROY] Texture '" << name << "'\n";
    }

    void bind() const {
        std::cout << "    [GPU] Binding texture '" << name << "'\n";
    }

    std::string getName() const { return name; }
};

// Example: unique_ptr (exclusive ownership)
void demonstrateUniquePtr() {
    std::cout << "\n=== std::unique_ptr (Exclusive Ownership) ===\n";

    {
        std::cout << "Creating unique_ptr:\n";
        std::unique_ptr<Texture> tex1 = std::make_unique<Texture>("Diffuse", 1024, 1024);

        tex1->bind();

        // Cannot copy unique_ptr (exclusive ownership)
        // std::unique_ptr<Texture> tex2 = tex1;  // ERROR!

        // But can move ownership
        std::unique_ptr<Texture> tex2 = std::move(tex1);
        std::cout << "After move, tex1 is: " << (tex1 ? "valid" : "null") << "\n";
        std::cout << "tex2 is: " << (tex2 ? "valid" : "null") << "\n";

        tex2->bind();

        std::cout << "Leaving scope...\n";
    }  // Texture automatically destroyed here

    std::cout << "Texture cleaned up automatically!\n";
}

// Example: shared_ptr (shared ownership with reference counting)
void demonstrateSharedPtr() {
    std::cout << "\n=== std::shared_ptr (Shared Ownership) ===\n";

    {
        std::cout << "Creating shared_ptr:\n";
        std::shared_ptr<Texture> tex1 = std::make_shared<Texture>("Normal", 512, 512);
        std::cout << "tex1 ref count: " << tex1.use_count() << "\n";

        {
            // Can copy shared_ptr (reference counting)
            std::shared_ptr<Texture> tex2 = tex1;
            std::cout << "After copy, ref count: " << tex1.use_count() << "\n";

            tex2->bind();

            std::cout << "Inner scope ending...\n";
        }  // tex2 destroyed, but texture still alive

        std::cout << "After inner scope, ref count: " << tex1.use_count() << "\n";
        tex1->bind();

        std::cout << "Outer scope ending...\n";
    }  // tex1 destroyed, ref count reaches 0, texture destroyed

    std::cout << "Texture cleaned up when last reference gone!\n";
}

// Example: weak_ptr (non-owning reference)
class Material {
public:
    std::string name;
    std::shared_ptr<Texture> albedoTexture;
    std::weak_ptr<Texture> normalTexture;  // Optional, doesn't own

    Material(const std::string& matName) : name(matName) {}

    void setNormalTexture(std::shared_ptr<Texture> tex) {
        normalTexture = tex;
    }

    void render() {
        std::cout << "  Rendering material '" << name << "':\n";

        if (albedoTexture) {
            albedoTexture->bind();
        }

        // Check if weak_ptr is still valid
        if (auto tex = normalTexture.lock()) {
            tex->bind();
        } else {
            std::cout << "    Normal texture no longer available\n";
        }
    }
};

void demonstrateWeakPtr() {
    std::cout << "\n=== std::weak_ptr (Non-Owning Reference) ===\n";

    Material mat("GoldMaterial");

    {
        auto albedo = std::make_shared<Texture>("Gold_Albedo", 1024, 1024);
        auto normal = std::make_shared<Texture>("Gold_Normal", 1024, 1024);

        mat.albedoTexture = albedo;  // shared_ptr (owns)
        mat.setNormalTexture(normal);  // weak_ptr (doesn't own)

        std::cout << "\nRendering with all textures:\n";
        mat.render();

        std::cout << "\nNormal texture ref count: " << normal.use_count() << "\n";

        std::cout << "\nLeaving scope (normal texture will be destroyed)...\n";
    }  // normal goes out of scope and is destroyed

    std::cout << "\nRendering after normal texture destroyed:\n";
    mat.render();  // weak_ptr detects texture is gone
}

// Example: Smart pointers in containers
void demonstrateSmartPtrContainers() {
    std::cout << "\n=== Smart Pointers in Containers ===\n";

    std::vector<std::unique_ptr<Texture>> textures;

    std::cout << "Loading textures:\n";
    textures.push_back(std::make_unique<Texture>("Tex1", 512, 512));
    textures.push_back(std::make_unique<Texture>("Tex2", 1024, 1024));
    textures.push_back(std::make_unique<Texture>("Tex3", 256, 256));

    std::cout << "\nUsing textures:\n";
    for (const auto& tex : textures) {
        tex->bind();
    }

    std::cout << "\nClearing container...\n";
    textures.clear();  // All textures automatically destroyed

    std::cout << "All resources cleaned up!\n";
}

// Example: Factory function returning unique_ptr
std::unique_ptr<Texture> loadTexture(const std::string& name, int size) {
    std::cout << "Loading texture from disk...\n";
    return std::make_unique<Texture>(name, size, size);
}

void demonstrateFactory() {
    std::cout << "\n=== Factory Pattern with unique_ptr ===\n";

    auto tex = loadTexture("Skybox", 2048);
    tex->bind();

    // Ownership clearly transferred from factory to caller
}

// Example: Custom deleter for GPU resources
class GPUBuffer {
private:
    void* devicePtr;
    size_t bytes;

public:
    GPUBuffer(size_t size) : bytes(size) {
        devicePtr = malloc(size);  // Simulates cudaMalloc
        std::cout << "  [GPU] Allocated " << bytes << " bytes\n";
    }

    ~GPUBuffer() {
        std::cout << "  [GPU] Freeing " << bytes << " bytes\n";
        free(devicePtr);  // Simulates cudaFree
    }

    void* getPtr() const { return devicePtr; }
};

void demonstrateCustomDeleter() {
    std::cout << "\n=== Smart Pointers with GPU Resources ===\n";

    {
        auto buffer = std::make_unique<GPUBuffer>(1024 * 1024);
        std::cout << "Using buffer at " << buffer->getPtr() << "\n";
    }  // Automatically calls GPUBuffer destructor

    std::cout << "GPU memory automatically freed!\n";
}

int main() {
    std::cout << "=== MODULE 9: Smart Pointers Demo ===\n";

    demonstrateUniquePtr();
    demonstrateSharedPtr();
    demonstrateWeakPtr();
    demonstrateSmartPtrContainers();
    demonstrateFactory();
    demonstrateCustomDeleter();

    std::cout << "\n=== Program Complete ===\n";
    return 0;
}

/*
 * KEY CONCEPTS:
 *
 * 1. unique_ptr:
 *    - Exclusive ownership (only one owner)
 *    - Cannot be copied, only moved
 *    - Zero overhead compared to raw pointer
 *    - Use case: Clear single ownership
 *
 * 2. shared_ptr:
 *    - Shared ownership (multiple owners)
 *    - Reference counting
 *    - Small overhead (ref count storage)
 *    - Use case: Shared resources, caches
 *
 * 3. weak_ptr:
 *    - Non-owning reference
 *    - Doesn't affect ref count
 *    - Must convert to shared_ptr to use
 *    - Use case: Break circular references, optional dependencies
 *
 * 4. make_unique / make_shared:
 *    - Preferred way to create smart pointers
 *    - Exception safe
 *    - More efficient (shared_ptr)
 *
 * TRY THIS:
 * 1. Create texture cache using shared_ptr
 * 2. Implement scene graph with weak_ptr for parent references
 * 3. Add custom deleter for CUDA memory
 * 4. Build resource manager with smart pointers
 * 5. Implement observer pattern with weak_ptr
 *
 * GPU CONNECTION:
 * - Texture management: shared_ptr for cache
 * - Mesh data: unique_ptr for exclusive ownership
 * - Scene graph: weak_ptr for parent pointers
 * - GPU buffer RAII: unique_ptr with custom deleter
 * - Reference counting prevents memory leaks
 *
 * BEST PRACTICES:
 * - Default to unique_ptr
 * - Use shared_ptr when sharing is needed
 * - Use weak_ptr to break cycles
 * - Never manually delete smart pointer contents
 * - Prefer make_unique/make_shared
 */
