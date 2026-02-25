/*
 * MODULE 5: Inheritance and Polymorphism
 * File: 04_abstract_interfaces.cpp
 *
 * TOPIC: Abstract Classes and Interface Design
 *
 * CONCEPTS:
 * - Abstract classes (at least one pure virtual function)
 * - Interface classes (all pure virtual functions)
 * - Programming to interfaces, not implementations
 * - Dependency inversion principle
 * - Multiple inheritance for interfaces
 *
 * GPU RELEVANCE:
 * - Texture interface for different texture types
 * - Buffer interface for vertex/index/uniform buffers
 * - Shader interface for vertex/fragment/compute shaders
 * - Renderer interface for different rendering backends (OpenGL/Vulkan/DX)
 *
 * COMPILE: g++ -std=c++17 -o abstract_interfaces 04_abstract_interfaces.cpp
 */

#include <iostream>
#include <string>
#include <vector>
#include <memory>

// Example 1: Pure interface (all functions pure virtual)
class ITexture {
public:
    virtual ~ITexture() = default;

    // Pure virtual interface - no implementation
    virtual void bind(int slot) const = 0;
    virtual void unbind() const = 0;
    virtual int getWidth() const = 0;
    virtual int getHeight() const = 0;
    virtual void* getData() const = 0;
};

// Concrete implementations
class Texture2D : public ITexture {
private:
    int width, height;
    void* data;
    std::string name;

public:
    Texture2D(const std::string& texName, int w, int h)
        : name(texName), width(w), height(h) {
        data = malloc(w * h * 4);  // RGBA
        std::cout << "  Texture2D '" << name << "' created (" << w << "x" << h << ")\n";
    }

    ~Texture2D() override {
        std::cout << "  Texture2D '" << name << "' destroyed\n";
        free(data);
    }

    void bind(int slot) const override {
        std::cout << "    [GPU] Binding Texture2D '" << name << "' to slot " << slot << "\n";
    }

    void unbind() const override {
        std::cout << "    [GPU] Unbinding Texture2D '" << name << "'\n";
    }

    int getWidth() const override { return width; }
    int getHeight() const override { return height; }
    void* getData() const override { return data; }
};

class TextureCube : public ITexture {
private:
    int faceSize;
    void* faces[6];
    std::string name;

public:
    TextureCube(const std::string& texName, int size)
        : name(texName), faceSize(size) {
        for (int i = 0; i < 6; ++i) {
            faces[i] = malloc(size * size * 4);
        }
        std::cout << "  TextureCube '" << name << "' created (" << size << "x" << size << " per face)\n";
    }

    ~TextureCube() override {
        std::cout << "  TextureCube '" << name << "' destroyed\n";
        for (int i = 0; i < 6; ++i) {
            free(faces[i]);
        }
    }

    void bind(int slot) const override {
        std::cout << "    [GPU] Binding TextureCube '" << name << "' to slot " << slot << "\n";
    }

    void unbind() const override {
        std::cout << "    [GPU] Unbinding TextureCube '" << name << "'\n";
    }

    int getWidth() const override { return faceSize; }
    int getHeight() const override { return faceSize; }
    void* getData() const override { return faces[0]; }  // Return first face
};

// Example 2: Abstract base class (mix of pure virtual and implemented functions)
class Shader {
protected:
    std::string name;
    bool compiled;

public:
    Shader(const std::string& shaderName)
        : name(shaderName), compiled(false) {}

    virtual ~Shader() = default;

    // Implemented in base class
    const std::string& getName() const { return name; }
    bool isCompiled() const { return compiled; }

    // Pure virtual - must be implemented by derived classes
    virtual void compile() = 0;
    virtual void bind() const = 0;
    virtual void unbind() const = 0;
    virtual std::string getType() const = 0;
};

class VertexShader : public Shader {
public:
    VertexShader(const std::string& name) : Shader(name) {}

    void compile() override {
        std::cout << "  Compiling vertex shader '" << name << "'\n";
        compiled = true;
    }

    void bind() const override {
        std::cout << "  [GPU] Binding vertex shader '" << name << "'\n";
    }

    void unbind() const override {
        std::cout << "  [GPU] Unbinding vertex shader '" << name << "'\n";
    }

    std::string getType() const override { return "VertexShader"; }
};

class FragmentShader : public Shader {
public:
    FragmentShader(const std::string& name) : Shader(name) {}

    void compile() override {
        std::cout << "  Compiling fragment shader '" << name << "'\n";
        compiled = true;
    }

    void bind() const override {
        std::cout << "  [GPU] Binding fragment shader '" << name << "'\n";
    }

    void unbind() const override {
        std::cout << "  [GPU] Unbinding fragment shader '" << name << "'\n";
    }

    std::string getType() const override { return "FragmentShader"; }
};

// Example 3: Multiple inheritance with interfaces
class ISerializable {
public:
    virtual ~ISerializable() = default;
    virtual std::string serialize() const = 0;
    virtual void deserialize(const std::string& data) = 0;
};

class ILoggable {
public:
    virtual ~ILoggable() = default;
    virtual void log() const = 0;
};

// Class implementing multiple interfaces
class Material : public ISerializable, public ILoggable {
private:
    std::string name;
    float roughness;
    float metallic;

public:
    Material(const std::string& matName, float r, float m)
        : name(matName), roughness(r), metallic(m) {}

    // ISerializable interface
    std::string serialize() const override {
        return name + "," + std::to_string(roughness) + "," + std::to_string(metallic);
    }

    void deserialize(const std::string& data) override {
        std::cout << "  Deserializing material from: " << data << "\n";
        // Simplified: actual parsing would go here
    }

    // ILoggable interface
    void log() const override {
        std::cout << "  Material '" << name << "': roughness=" << roughness
                  << ", metallic=" << metallic << "\n";
    }

    // Material-specific methods
    std::string getName() const { return name; }
    float getRoughness() const { return roughness; }
    float getMetallic() const { return metallic; }
};

// Example 4: Renderer interface (strategy pattern)
class IRenderer {
public:
    virtual ~IRenderer() = default;

    virtual void initialize() = 0;
    virtual void shutdown() = 0;
    virtual void beginFrame() = 0;
    virtual void endFrame() = 0;
    virtual void drawTriangles(int count) = 0;
    virtual std::string getAPIName() const = 0;
};

class OpenGLRenderer : public IRenderer {
public:
    void initialize() override {
        std::cout << "  [OpenGL] Initializing renderer\n";
    }

    void shutdown() override {
        std::cout << "  [OpenGL] Shutting down renderer\n";
    }

    void beginFrame() override {
        std::cout << "  [OpenGL] Begin frame (glClear)\n";
    }

    void endFrame() override {
        std::cout << "  [OpenGL] End frame (SwapBuffers)\n";
    }

    void drawTriangles(int count) override {
        std::cout << "  [OpenGL] glDrawArrays(GL_TRIANGLES, 0, " << count << ")\n";
    }

    std::string getAPIName() const override { return "OpenGL"; }
};

class VulkanRenderer : public IRenderer {
public:
    void initialize() override {
        std::cout << "  [Vulkan] Initializing renderer (creating instance, device)\n";
    }

    void shutdown() override {
        std::cout << "  [Vulkan] Shutting down renderer (destroying resources)\n";
    }

    void beginFrame() override {
        std::cout << "  [Vulkan] Begin frame (acquire swapchain image)\n";
    }

    void endFrame() override {
        std::cout << "  [Vulkan] End frame (present queue)\n";
    }

    void drawTriangles(int count) override {
        std::cout << "  [Vulkan] vkCmdDraw(commandBuffer, " << count << ", ...)\n";
    }

    std::string getAPIName() const override { return "Vulkan"; }
};

// Application that depends on abstract interface, not concrete implementation
class RenderApplication {
private:
    IRenderer* renderer;

public:
    RenderApplication(IRenderer* r) : renderer(r) {
        std::cout << "RenderApplication created with " << renderer->getAPIName() << " backend\n";
    }

    void run() {
        std::cout << "\nRunning application:\n";
        renderer->initialize();
        renderer->beginFrame();
        renderer->drawTriangles(100);
        renderer->endFrame();
        renderer->shutdown();
    }
};

// Demonstrate texture interface
void demonstrateTextureInterface() {
    std::cout << "\n=== Texture Interface Demo ===\n";

    std::vector<ITexture*> textures;
    textures.push_back(new Texture2D("Diffuse", 1024, 1024));
    textures.push_back(new Texture2D("Normal", 512, 512));
    textures.push_back(new TextureCube("Skybox", 512));

    std::cout << "\nBinding all textures:\n";
    for (size_t i = 0; i < textures.size(); ++i) {
        textures[i]->bind(i);
    }

    std::cout << "\nTexture info:\n";
    for (ITexture* tex : textures) {
        std::cout << "  Size: " << tex->getWidth() << "x" << tex->getHeight() << "\n";
    }

    std::cout << "\nUnbinding all textures:\n";
    for (ITexture* tex : textures) {
        tex->unbind();
    }

    // Cleanup
    for (ITexture* tex : textures) {
        delete tex;
    }
}

// Demonstrate shader interface
void demonstrateShaderInterface() {
    std::cout << "\n=== Shader Interface Demo ===\n";

    std::vector<Shader*> shaders;
    shaders.push_back(new VertexShader("BasicVert"));
    shaders.push_back(new FragmentShader("BasicFrag"));
    shaders.push_back(new VertexShader("SkinnedVert"));

    std::cout << "\nCompiling all shaders:\n";
    for (Shader* shader : shaders) {
        shader->compile();
    }

    std::cout << "\nShader info:\n";
    for (Shader* shader : shaders) {
        std::cout << "  " << shader->getType() << " '" << shader->getName()
                  << "' - " << (shader->isCompiled() ? "compiled" : "not compiled") << "\n";
    }

    // Cleanup
    for (Shader* shader : shaders) {
        delete shader;
    }
}

// Demonstrate multiple interface implementation
void demonstrateMultipleInterfaces() {
    std::cout << "\n=== Multiple Interface Implementation Demo ===\n";

    Material mat("Gold", 0.2f, 0.9f);

    std::cout << "\nUsing as ILoggable:\n";
    ILoggable* loggable = &mat;
    loggable->log();

    std::cout << "\nUsing as ISerializable:\n";
    ISerializable* serializable = &mat;
    std::string data = serializable->serialize();
    std::cout << "  Serialized data: " << data << "\n";

    Material mat2("Silver", 0.0f, 0.0f);
    mat2.deserialize(data);
}

// Demonstrate renderer abstraction
void demonstrateRendererInterface() {
    std::cout << "\n=== Renderer Interface Demo ===\n";

    std::cout << "\nRunning with OpenGL backend:\n";
    OpenGLRenderer glRenderer;
    RenderApplication app1(&glRenderer);
    app1.run();

    std::cout << "\n" << std::string(50, '-') << "\n";
    std::cout << "\nRunning with Vulkan backend:\n";
    VulkanRenderer vkRenderer;
    RenderApplication app2(&vkRenderer);
    app2.run();

    std::cout << "\nNote: Same application code works with different backends!\n";
    std::cout << "This is the power of programming to interfaces.\n";
}

int main() {
    std::cout << "=== MODULE 5: Abstract Classes and Interfaces Demo ===\n";

    demonstrateTextureInterface();
    demonstrateShaderInterface();
    demonstrateMultipleInterfaces();
    demonstrateRendererInterface();

    std::cout << "\n=== Program Complete ===\n";
    return 0;
}

/*
 * KEY CONCEPTS DEMONSTRATED:
 *
 * 1. ABSTRACT CLASS:
 *    - Has at least one pure virtual function
 *    - Cannot be instantiated
 *    - Defines interface contract
 *
 * 2. PURE INTERFACE:
 *    - All functions are pure virtual
 *    - Often prefixed with 'I' (ITexture, IRenderer)
 *    - Like Java interfaces or C# interfaces
 *
 * 3. PROGRAMMING TO INTERFACES:
 *    - Depend on abstractions, not concrete implementations
 *    - Makes code flexible and testable
 *    - Follows Dependency Inversion Principle
 *
 * 4. MULTIPLE INTERFACE INHERITANCE:
 *    - Class can implement multiple interfaces
 *    - C++'s way of achieving multiple inheritance safely
 *    - Common pattern: ISerializable, ILoggable, ICloneable
 *
 * 5. STRATEGY PATTERN:
 *    - Different implementations of same interface
 *    - Switch implementations at runtime
 *    - Example: Different renderer backends
 *
 * TRY THIS:
 * 1. Create IBuffer interface with VertexBuffer, IndexBuffer, UniformBuffer
 * 2. Implement IMesh interface with StaticMesh and DynamicMesh
 * 3. Add IUpdatable interface for game objects
 * 4. Create IPostProcessor interface for rendering effects
 * 5. Implement IResourceLoader interface for different file formats
 * 6. Design ICamera interface with PerspectiveCamera and OrthographicCamera
 *
 * COMMON MISTAKES:
 * - Forgetting virtual destructor in interface
 * - Making interface too specific (not abstract enough)
 * - Diamond problem with multiple inheritance (avoid with interfaces)
 * - Overusing interfaces where simple inheritance would suffice
 *
 * GPU CONNECTION:
 * - Modern graphics APIs have abstract interfaces
 * - Renderer abstraction: Switch between OpenGL/Vulkan/DirectX
 * - Texture types: 2D, 3D, Cube, Array - all share ITexture interface
 * - Buffer types: Vertex, Index, Uniform - all share IBuffer interface
 * - Shader types: Vertex, Fragment, Compute - all share IShader interface
 *
 * DESIGN PRINCIPLES:
 *
 * 1. DEPENDENCY INVERSION:
 *    - High-level modules shouldn't depend on low-level modules
 *    - Both should depend on abstractions (interfaces)
 *
 * 2. INTERFACE SEGREGATION:
 *    - Many specific interfaces better than one general interface
 *    - Don't force classes to implement unused methods
 *
 * 3. OPEN/CLOSED PRINCIPLE:
 *    - Open for extension (new implementations)
 *    - Closed for modification (interface stays stable)
 *
 * REAL GPU EXAMPLE:
 *
 * Many graphics engines use interface patterns:
 *
 * class IRenderDevice {
 *     virtual void drawIndexed(int count) = 0;
 *     virtual void setShader(IShader* shader) = 0;
 *     virtual void setTexture(int slot, ITexture* tex) = 0;
 * };
 *
 * class D3D11Device : public IRenderDevice { ... };
 * class OpenGLDevice : public IRenderDevice { ... };
 * class VulkanDevice : public IRenderDevice { ... };
 *
 * Then game code uses IRenderDevice*, works with any backend!
 */
