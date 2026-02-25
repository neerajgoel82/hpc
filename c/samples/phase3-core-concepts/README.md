# Phase 3: Core Concepts

**Duration**: Weeks 7-10 (4 weeks)

**Goal**: Master pointers, dynamic memory, and data structures - the heart of C programming.

---

## What You'll Learn

- Pointer fundamentals and operations
- Pointer arithmetic
- Relationship between pointers and arrays
- Pass-by-reference with pointers
- Dynamic memory allocation (malloc, free)
- Memory management best practices
- Structures (user-defined types)
- Enums and unions

---

## ⚠️ Important Note

This is the **most critical phase** in C programming. Pointers and dynamic memory are:
- Essential for C programming
- Required for understanding HPC and GPU programming
- Common source of bugs if not understood well
- Foundation for data structures and algorithms

**Take your time here!** Master these concepts before moving forward.

---

## Programs in This Phase

### 01. Pointers Basics (`01_pointers_basics.c`)
**Concept**: Pointer fundamentals

Understanding pointers from the ground up:
- What is a pointer? (memory address)
- Pointer declaration: `int *ptr`
- Address-of operator: `&variable`
- Dereference operator: `*ptr`
- Pointer initialization
- NULL pointers
- Pointer arithmetic basics

**Key Takeaway**: A pointer is just a variable that stores a memory address.

**Compile & Run**:
```bash
gcc -std=c11 -Wall -Wextra 01_pointers_basics.c -o pointers_basics
./pointers_basics
```

---

### 02. Pointers and Arrays (`02_pointers_arrays.c`)
**Concept**: The pointer-array duality

Deep dive into pointers and arrays:
- Array name as pointer to first element
- Pointer arithmetic (`ptr++`, `ptr + i`)
- Array indexing with pointers: `*(ptr + i)` == `ptr[i]`
- Passing arrays to functions (really passing pointers)
- String manipulation with pointers
- Pointer to pointer (`**ptr`)

**Key Takeaway**: In C, arrays and pointers are intimately related.

---

### 03. Pointers and Functions (`03_pointers_functions.c`)
**Concept**: Pass-by-reference, function pointers

Using pointers with functions:
- Pass-by-reference (modify variables in caller)
- Returning pointers from functions
- Array parameters are pointers
- Function pointers (pointers to functions!)
- Callback functions
- Common patterns

**Key Takeaway**: Pointers enable powerful function designs.

---

### 04. Structures (`04_structures.c`)
**Concept**: User-defined composite types

Creating custom data types:
- Structure definition with `struct`
- Member access: `.` and `->`
- Structure initialization
- Nested structures
- Arrays of structures
- Passing structures to functions
- `typedef` for cleaner syntax
- Practical examples

**Key Takeaway**: Structures let you group related data together.

---

### 05. Dynamic Memory (`05_dynamic_memory.c`)
**Concept**: Heap allocation, malloc/free

Mastering dynamic memory:
- Stack vs heap memory
- `malloc()` - allocate memory
- `calloc()` - allocate and zero
- `realloc()` - resize memory
- `free()` - release memory
- Memory leaks and how to avoid them
- Valgrind for leak detection
- Best practices

**Key Takeaway**: Dynamic memory gives flexibility but requires careful management.

⚠️ **Critical**: Every `malloc` needs a matching `free`!

---

### 06. Enums and Unions (`06_enums_unions.c`)
**Concept**: Enumerations and variant types

Additional type tools:
- Enumerations (`enum`) for named constants
- When to use enums
- Unions for memory-efficient variants
- Size and alignment
- Practical use cases
- Comparison with structures

**Key Takeaway**: Enums improve code readability; unions save memory.

---

## Quick Start

### Build All Programs
```bash
make
```

### Build Specific Program
```bash
make pointers_basics
# or
gcc -std=c11 -Wall -Wextra 01_pointers_basics.c -o pointers_basics
```

### Run a Program
```bash
./pointers_basics
```

### Check for Memory Leaks
```bash
make memcheck
# Runs valgrind on dynamic_memory program
```

### Build and Test All
```bash
make test
```

### Clean Up
```bash
make clean
```

---

## Compilation Guide

### Standard Compilation
```bash
gcc -std=c11 -Wall -Wextra program.c -o program
```

### With Debugging Symbols (Recommended for this phase!)
```bash
gcc -std=c11 -Wall -Wextra -g program.c -o program
```

### Check for Memory Leaks with Valgrind
```bash
valgrind --leak-check=full ./program
```

---

## Learning Checklist

After completing Phase 3, you should be able to:

- [ ] Explain what a pointer is and how it works
- [ ] Use pointer operators (`&`, `*`) correctly
- [ ] Perform pointer arithmetic
- [ ] Understand the relationship between pointers and arrays
- [ ] Pass pointers to functions for pass-by-reference
- [ ] Allocate memory dynamically with `malloc`
- [ ] Free memory properly with `free`
- [ ] Avoid memory leaks
- [ ] Define and use structures
- [ ] Access structure members with `.` and `->`
- [ ] Use enums for named constants
- [ ] Understand when to use unions
- [ ] Debug pointer-related errors
- [ ] Use valgrind to check for memory issues

---

## Common Mistakes to Avoid

### Pointers
1. **Uninitialized pointers** - Always initialize: `int *ptr = NULL;`
2. **Dereferencing NULL** - Check before dereferencing!
3. **Dangling pointers** - Pointer to freed/invalid memory
4. **Lost addresses** - Saving the original pointer before arithmetic

### Dynamic Memory
1. **Memory leaks** - Forgetting to `free()`
2. **Double free** - Freeing the same memory twice
3. **Use after free** - Using memory after freeing it
4. **Not checking malloc return** - `malloc` can fail!

### Structures
1. **Mixing `.` and `->`** - Use `.` for struct, `->` for struct pointer
2. **Copying large structures** - Pass by pointer for efficiency
3. **Padding and alignment** - Structure size may be larger than sum of members

---

## Debugging Tips

### Using GDB with Pointers
```bash
gcc -g program.c -o program
gdb ./program

# GDB commands:
(gdb) break main
(gdb) run
(gdb) print ptr          # Print pointer value (address)
(gdb) print *ptr         # Print value pointed to
(gdb) x/4xw ptr          # Examine 4 words at address
```

### Valgrind for Memory Issues
```bash
# Check for memory leaks
valgrind --leak-check=full ./program

# Check for invalid memory access
valgrind --track-origins=yes ./program
```

---

## Practice Exercises

### Pointers
1. **Swap Function**: Use pointers to swap two integers
2. **String Length**: Calculate string length using pointers (no `strlen`)
3. **Array Reversal**: Reverse array in-place using two pointers

### Dynamic Memory
1. **Dynamic Array**: Read N integers, store in dynamically allocated array
2. **String Concatenation**: Concatenate two strings into new allocated memory
3. **Resizable Array**: Implement growing array with `realloc`

### Structures
1. **Student Database**: Array of student structures with search/sort
2. **Linked List Node**: Define structure for linked list
3. **Complex Numbers**: Structure with real/imaginary parts and operations

---

## Key Concepts for HPC

### Why Pointers Matter
- **GPU Programming**: CUDA uses pointers extensively
- **Memory Management**: Control over memory layout and access patterns
- **Performance**: Direct memory access without copying
- **Data Structures**: Essential for efficient algorithms

### Memory Hierarchy
```
Stack (automatic, fast, limited)
  ↓
Heap (dynamic, larger, slower)
  ↓
CPU Cache (very fast)
  ↓
RAM (main memory)
  ↓
GPU Memory (separate address space)
```

### Performance Considerations
- Stack allocation is faster than heap
- Cache-friendly: contiguous memory access
- Minimize indirection: fewer pointer dereferences
- Consider alignment for SIMD/vectorization

---

## Memory Management Best Practices

### Pattern: Allocate and Check
```c
int *array = (int*)malloc(n * sizeof(int));
if (array == NULL) {
    fprintf(stderr, "Memory allocation failed\n");
    return 1;
}
// Use array...
free(array);
array = NULL;  // Prevent use-after-free
```

### Pattern: Free in Reverse Order
```c
// Allocate
ptr1 = malloc(...);
ptr2 = malloc(...);

// Free in reverse
free(ptr2);
free(ptr1);
```

### Pattern: Always Initialize Pointers
```c
int *ptr = NULL;  // Good
// vs
int *ptr;  // Bad - contains garbage
```

---

## Visualization

### Pointer Basics
```
Variable:  x = 42
Address:   0x1000

Pointer:   int *ptr = &x;
           ptr = 0x1000  (contains address of x)
           *ptr = 42     (value at that address)
```

### Array and Pointer
```
Array:     int arr[3] = {10, 20, 30};

Memory:    [10][20][30]
Address:   0x1000  0x1004  0x1008

Pointer:   int *p = arr;
           p points to arr[0] (0x1000)
           p + 1 points to arr[1] (0x1004)
           *(p + 2) == 30
```

---

## Next Steps

Once comfortable with Phase 3:
- **Move to Phase 4**: File I/O and advanced data structures
- **Practice extensively**: Pointers require practice!
- **Build projects**: Implement basic data structures
- **Use debugging tools**: Get comfortable with gdb and valgrind

---

## Mini-Project Ideas

1. **Dynamic String Library**
   - Functions for string operations with malloc/free
   - Proper memory management
   - String concatenation, splitting, searching

2. **Contact Manager**
   - Array of person structures (dynamically allocated)
   - Add, delete, search, sort contacts
   - Save/load from file

3. **Memory Allocator Simulator**
   - Simulate malloc/free behavior
   - Track allocations
   - Detect leaks

---

## Resources

- [Pointer Basics Tutorial](https://en.cppreference.com/w/c/language/pointer)
- [Dynamic Memory Management](https://en.cppreference.com/w/c/memory)
- [GDB Tutorial](https://www.gnu.org/software/gdb/documentation/)
- [Valgrind Quick Start](https://valgrind.org/docs/manual/quick-start.html)
- Main curriculum: [../../README.md](../../README.md)

---

## Tips for Success

- **Draw diagrams** - Visualize memory and pointers
- **Use debugger** - Step through and inspect memory
- **Check with valgrind** - Always verify no leaks
- **Start simple** - Master basics before complex scenarios
- **Practice daily** - Pointers need repetition to internalize
- **Test edge cases** - NULL pointers, empty arrays, etc.
- **Read compiler warnings** - They catch many pointer errors

---

## Warning Signs You Need More Practice

- Confused about when to use `&` vs `*`
- Not sure when `.` vs `->` for structures
- Forgetting to free memory
- Getting segmentation faults often
- Not understanding pointer arithmetic

If any of these apply, **spend more time in this phase!**

---

**Time Estimate**: 4 weeks at 7-10 hours/week

**Prerequisite**: Phases 1-2 complete

**Next Phase**: [Phase 4: Advanced Topics](../phase4-advanced/)

**Previous Phase**: [Phase 2: Building Blocks](../phase2-building-blocks/)

---

**Remember**: Mastering pointers and memory management is the key to becoming a proficient C programmer. Take your time and practice extensively!
