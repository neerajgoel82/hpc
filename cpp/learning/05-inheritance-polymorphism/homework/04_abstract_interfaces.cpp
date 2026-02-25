/*
 * Homework: 04_abstract_interfaces.cpp
 *
 * Complete the exercises below based on the concepts from 04_abstract_interfaces.cpp
 * in the classwork folder.
 *
 * Instructions:
 * 1. Read the corresponding classwork file first
 * 2. Implement the solutions below
 * 3. Compile: g++ -std=c++17 -Wall -Wextra 04_abstract_interfaces.cpp -o homework
 * 4. Test your solutions
 */

#include <iostream>

/*
 * TRY THIS:
 * * 1. Create IBuffer interface with VertexBuffer, IndexBuffer, UniformBuffer
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

int main() {
    std::cout << "Homework: 04_abstract_interfaces\n";
    std::cout << "Implement the exercises above\n";

    // Your code here

    return 0;
}
