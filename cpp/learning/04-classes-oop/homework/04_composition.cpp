/*
 * Homework: 04_composition.cpp
 *
 * Complete the exercises below based on the concepts from 04_composition.cpp
 * in the classwork folder.
 *
 * Instructions:
 * 1. Read the corresponding classwork file first
 * 2. Implement the solutions below
 * 3. Compile: g++ -std=c++17 -Wall -Wextra 04_composition.cpp -o homework
 * 4. Test your solutions
 */

#include <iostream>

/*
 * TRY THIS:
 * * 1. Add a Camera class that HAS Transform and projection parameters
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

int main() {
    std::cout << "Homework: 04_composition\n";
    std::cout << "Implement the exercises above\n";

    // Your code here

    return 0;
}
