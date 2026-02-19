/*
 * MODULE 5: Inheritance and Polymorphism
 * File: 03_virtual_functions.cpp
 *
 * TOPIC: Virtual Functions Deep Dive
 *
 * CONCEPTS:
 * - Virtual function mechanics (vtable, vptr)
 * - Early binding vs late binding
 * - Function overriding vs overloading
 * - final keyword
 * - Pure virtual vs virtual with implementation
 *
 * GPU RELEVANCE:
 * - Understanding virtual function cost for CPU-side code
 * - When to use polymorphism vs templates for GPU code
 * - Design patterns for heterogeneous GPU scene data
 * - Interface design for shader/material systems
 *
 * COMPILE: g++ -std=c++17 -o virtual_functions 03_virtual_functions.cpp
 */

#include <iostream>
#include <string>
#include <vector>
#include <chrono>

// Example 1: Virtual vs Non-Virtual comparison
class Base {
public:
    // Non-virtual function (early binding, compile-time)
    void nonVirtualFunc() const {
        std::cout << "  Base::nonVirtualFunc()\n";
    }

    // Virtual function (late binding, runtime)
    virtual void virtualFunc() const {
        std::cout << "  Base::virtualFunc()\n";
    }

    // Pure virtual (must override)
    virtual void pureVirtualFunc() const = 0;

    // Virtual with implementation (can override or use default)
    virtual void optionalOverride() const {
        std::cout << "  Base::optionalOverride() - default implementation\n";
    }

    virtual ~Base() = default;
};

class Derived : public Base {
public:
    // Overrides virtual function
    void virtualFunc() const override {
        std::cout << "  Derived::virtualFunc()\n";
    }

    // Implements pure virtual
    void pureVirtualFunc() const override {
        std::cout << "  Derived::pureVirtualFunc()\n";
    }

    // Uses default implementation (doesn't override optionalOverride)

    // This hides (not overrides) non-virtual function
    void nonVirtualFunc() const {
        std::cout << "  Derived::nonVirtualFunc()\n";
    }
};

void demonstrateBindingDifference() {
    std::cout << "\n=== Early vs Late Binding ===\n";

    Derived d;
    Base* basePtr = &d;

    std::cout << "\nDirect call on Derived object:\n";
    d.nonVirtualFunc();    // Calls Derived version
    d.virtualFunc();       // Calls Derived version

    std::cout << "\nCall through Base pointer:\n";
    basePtr->nonVirtualFunc();    // Early binding -> Base version
    basePtr->virtualFunc();       // Late binding -> Derived version
    basePtr->pureVirtualFunc();   // Late binding -> Derived version
    basePtr->optionalOverride();  // Late binding -> Base version (not overridden)

    std::cout << "\nKey difference:\n";
    std::cout << "  Non-virtual: Resolved at compile-time based on pointer type\n";
    std::cout << "  Virtual: Resolved at runtime based on actual object type\n";
}

// Example 2: The 'final' keyword
class BaseClass {
public:
    virtual void overridableFunc() {
        std::cout << "  BaseClass::overridableFunc()\n";
    }

    virtual void finalFunc() final {  // Cannot be overridden
        std::cout << "  BaseClass::finalFunc() - cannot override\n";
    }

    virtual ~BaseClass() = default;
};

class DerivedClass : public BaseClass {
public:
    void overridableFunc() override {
        std::cout << "  DerivedClass::overridableFunc()\n";
    }

    // ERROR: Cannot override final function
    // void finalFunc() override { }
};

// Final class - cannot be derived from
class FinalClass final : public BaseClass {
public:
    void overridableFunc() override {
        std::cout << "  FinalClass::overridableFunc()\n";
    }
};

// ERROR: Cannot derive from final class
// class CannotDeriveThis : public FinalClass { };

void demonstrateFinal() {
    std::cout << "\n=== The 'final' Keyword ===\n";

    BaseClass* obj1 = new DerivedClass();
    BaseClass* obj2 = new FinalClass();

    std::cout << "\nCalling overridable function:\n";
    obj1->overridableFunc();  // DerivedClass version
    obj2->overridableFunc();  // FinalClass version

    std::cout << "\nCalling final function:\n";
    obj1->finalFunc();  // BaseClass version (cannot be overridden)
    obj2->finalFunc();  // BaseClass version (cannot be overridden)

    delete obj1;
    delete obj2;

    std::cout << "\n'final' benefits:\n";
    std::cout << "  - Compiler can optimize (devirtualization)\n";
    std::cout << "  - Prevents unintended overriding\n";
    std::cout << "  - Documents design intent\n";
}

// Example 3: Virtual function call in constructor (DANGEROUS!)
class Parent {
public:
    Parent() {
        std::cout << "  Parent constructor\n";
        std::cout << "    Calling virtual function from constructor:\n";
        virtualFunc();  // Always calls Parent version!
    }

    virtual void virtualFunc() const {
        std::cout << "      Parent::virtualFunc()\n";
    }

    virtual ~Parent() {
        std::cout << "  Parent destructor\n";
        std::cout << "    Calling virtual function from destructor:\n";
        virtualFunc();  // Always calls Parent version!
    }
};

class Child : public Parent {
private:
    int* data;

public:
    Child() : Parent() {
        data = new int(42);
        std::cout << "  Child constructor (data initialized)\n";
    }

    void virtualFunc() const override {
        std::cout << "      Child::virtualFunc() - accessing data: " << *data << "\n";
    }

    ~Child() override {
        std::cout << "  Child destructor\n";
        delete data;
        data = nullptr;
    }
};

void demonstrateConstructorVirtual() {
    std::cout << "\n=== Virtual Functions in Constructor/Destructor ===\n";

    std::cout << "\nCreating Child object:\n";
    {
        Child c;
        std::cout << "\nCalling virtual function on fully constructed object:\n";
        c.virtualFunc();  // Now calls Child version
    }
    std::cout << "\n";

    std::cout << "IMPORTANT:\n";
    std::cout << "  - During construction: Only parent's virtual functions work\n";
    std::cout << "  - During destruction: Only parent's virtual functions work\n";
    std::cout << "  - Reason: Child not fully constructed/already destroyed\n";
    std::cout << "  - This prevents accessing uninitialized/freed child data\n";
}

// Example 4: Covariant return types
class Animal {
public:
    virtual Animal* clone() const {
        std::cout << "  Cloning Animal\n";
        return new Animal(*this);
    }

    virtual void makeSound() const {
        std::cout << "  Generic animal sound\n";
    }

    virtual ~Animal() = default;
};

class Dog : public Animal {
public:
    // Covariant return type: Can return Dog* instead of Animal*
    Dog* clone() const override {
        std::cout << "  Cloning Dog\n";
        return new Dog(*this);
    }

    void makeSound() const override {
        std::cout << "  Woof!\n";
    }
};

void demonstrateCovariantReturn() {
    std::cout << "\n=== Covariant Return Types ===\n";

    Dog dog;
    Animal* animalPtr = &dog;

    std::cout << "\nCloning through base pointer:\n";
    Animal* cloned = animalPtr->clone();  // Returns Dog*, converts to Animal*

    std::cout << "\nOriginal:\n";
    dog.makeSound();

    std::cout << "Clone:\n";
    cloned->makeSound();  // Polymorphic call -> Dog version

    delete cloned;
}

// Example 5: Multiple inheritance and virtual functions
class Renderable {
public:
    virtual void render() const {
        std::cout << "    Rendering...\n";
    }
    virtual ~Renderable() = default;
};

class Updateable {
public:
    virtual void update(float dt) {
        std::cout << "    Updating with dt=" << dt << "\n";
    }
    virtual ~Updateable() = default;
};

class GameObject : public Renderable, public Updateable {
private:
    std::string name;

public:
    GameObject(const std::string& n) : name(n) {}

    void render() const override {
        std::cout << "    Rendering GameObject '" << name << "'\n";
    }

    void update(float dt) override {
        std::cout << "    Updating GameObject '" << name << "' (dt=" << dt << ")\n";
    }
};

void demonstrateMultipleInheritance() {
    std::cout << "\n=== Multiple Inheritance with Virtual Functions ===\n";

    GameObject obj("Player");

    std::cout << "\nDirect calls:\n";
    obj.render();
    obj.update(0.016f);

    std::cout << "\nPolymorphic calls through different base pointers:\n";
    Renderable* renderPtr = &obj;
    Updateable* updatePtr = &obj;

    renderPtr->render();
    updatePtr->update(0.016f);
}

// Example 6: Performance consideration (simplified micro-benchmark)
class NonVirtualClass {
public:
    int compute(int x) const {
        return x * 2 + 1;
    }
};

class VirtualClass {
public:
    virtual int compute(int x) const {
        return x * 2 + 1;
    }
    virtual ~VirtualClass() = default;
};

void demonstratePerformance() {
    std::cout << "\n=== Performance Consideration ===\n";

    const int ITERATIONS = 10000000;

    // Non-virtual function calls
    NonVirtualClass nvc;
    auto start1 = std::chrono::high_resolution_clock::now();
    long long sum1 = 0;
    for (int i = 0; i < ITERATIONS; ++i) {
        sum1 += nvc.compute(i);
    }
    auto end1 = std::chrono::high_resolution_clock::now();
    auto duration1 = std::chrono::duration_cast<std::chrono::milliseconds>(end1 - start1);

    // Virtual function calls
    VirtualClass vc;
    VirtualClass* ptr = &vc;
    auto start2 = std::chrono::high_resolution_clock::now();
    long long sum2 = 0;
    for (int i = 0; i < ITERATIONS; ++i) {
        sum2 += ptr->compute(i);
    }
    auto end2 = std::chrono::high_resolution_clock::now();
    auto duration2 = std::chrono::duration_cast<std::chrono::milliseconds>(end2 - start2);

    std::cout << "\nNon-virtual calls: " << duration1.count() << " ms (sum=" << sum1 << ")\n";
    std::cout << "Virtual calls: " << duration2.count() << " ms (sum=" << sum2 << ")\n";
    std::cout << "\nNote: Overhead is usually small and negligible compared to actual work\n";
    std::cout << "GPU kernels avoid virtual functions entirely (use templates instead)\n";
}

int main() {
    std::cout << "=== MODULE 5: Virtual Functions Deep Dive ===\n";

    demonstrateBindingDifference();
    demonstrateFinal();
    demonstrateConstructorVirtual();
    demonstrateCovariantReturn();
    demonstrateMultipleInheritance();
    demonstratePerformance();

    std::cout << "\n=== Program Complete ===\n";
    return 0;
}

/*
 * KEY CONCEPTS DEMONSTRATED:
 *
 * 1. EARLY vs LATE BINDING:
 *    - Non-virtual: Early binding (compile-time, based on pointer type)
 *    - Virtual: Late binding (runtime, based on object type)
 *
 * 2. VIRTUAL FUNCTION MECHANISM:
 *    - Each polymorphic class has vtable (array of function pointers)
 *    - Each object has vptr (pointer to class's vtable)
 *    - Virtual call: object->vptr->vtable[index]->function
 *
 * 3. THE 'final' KEYWORD:
 *    - Prevents function overriding
 *    - Prevents class inheritance
 *    - Enables compiler optimizations
 *
 * 4. CONSTRUCTOR/DESTRUCTOR RULES:
 *    - Virtual functions called in constructor/destructor
 *      only call the current class's version
 *    - Prevents accessing uninitialized/destroyed data
 *
 * 5. COVARIANT RETURN TYPES:
 *    - Override can return derived class pointer
 *    - Must be pointer or reference
 *    - Useful for clone() patterns
 *
 * TRY THIS:
 * 1. Add another level of inheritance and trace virtual calls
 * 2. Implement a visitor pattern using virtual functions
 * 3. Create a command pattern with virtual execute() method
 * 4. Test performance with deeper inheritance hierarchies
 * 5. Implement a factory pattern returning polymorphic objects
 * 6. Create an abstract Shader class with concrete implementations
 *
 * COMMON MISTAKES:
 * - Calling virtual functions in constructors (doesn't work as expected)
 * - Forgetting 'override' keyword (typo becomes hidden bug)
 * - Not understanding early vs late binding
 * - Excessive use of virtual functions where templates would be better
 *
 * GPU CONNECTION:
 * - CPU-side scene management uses virtual functions
 * - GPU kernels DON'T use virtual functions (no vtable support)
 * - GPU alternatives:
 *   * Templates for compile-time polymorphism
 *   * Switch statements on type enum
 *   * Separate arrays per type
 * - Virtual function overhead negligible for CPU prep work
 *
 * PERFORMANCE NOTES:
 * - Virtual call overhead: ~1-3 nanoseconds
 * - Usually negligible compared to actual work
 * - Can prevent some compiler optimizations (inlining)
 * - 'final' keyword can enable devirtualization
 * - Profile before optimizing!
 *
 * WHEN TO USE VIRTUAL:
 * - Polymorphic behavior needed at runtime
 * - Type not known at compile-time
 * - Clean interface for heterogeneous collections
 *
 * WHEN NOT TO USE VIRTUAL:
 * - Performance-critical tight loops (use templates)
 * - GPU device code (not supported)
 * - Type known at compile-time (use templates)
 */
