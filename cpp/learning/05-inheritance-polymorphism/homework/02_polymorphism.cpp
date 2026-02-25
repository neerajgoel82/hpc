/*
 * Homework: 02_polymorphism.cpp
 *
 * Complete the exercises below based on the concepts from 02_polymorphism.cpp
 * in the classwork folder.
 *
 * Instructions:
 * 1. Read the corresponding classwork file first
 * 2. Implement the solutions below
 * 3. Compile: g++ -std=c++17 -Wall -Wextra 02_polymorphism.cpp -o homework
 * 4. Test your solutions
 */

#include <iostream>

/*
 * TRY THIS:
 * * 1. Add a new shape type (Ellipse) and test polymorphism
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

int main() {
    std::cout << "Homework: 02_polymorphism\n";
    std::cout << "Implement the exercises above\n";

    // Your code here

    return 0;
}
