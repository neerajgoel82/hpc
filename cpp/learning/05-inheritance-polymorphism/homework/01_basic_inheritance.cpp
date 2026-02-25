/*
 * Homework: 01_basic_inheritance.cpp
 *
 * Complete the exercises below based on the concepts from 01_basic_inheritance.cpp
 * in the classwork folder.
 *
 * Instructions:
 * 1. Read the corresponding classwork file first
 * 2. Implement the solutions below
 * 3. Compile: g++ -std=c++17 -Wall -Wextra 01_basic_inheritance.cpp -o homework
 * 4. Test your solutions
 */

#include <iostream>

/*
 * TRY THIS:
 * * 1. Add a Triangle class derived from Shape
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

int main() {
    std::cout << "Homework: 01_basic_inheritance\n";
    std::cout << "Implement the exercises above\n";

    // Your code here

    return 0;
}
