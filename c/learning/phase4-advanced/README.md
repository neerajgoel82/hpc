# Phase 4: Advanced Topics

**Duration**: Weeks 11-14 (4 weeks)

**Goal**: Master file I/O, preprocessor, and fundamental data structures.

---

## What You'll Learn

- File input/output operations
- Text and binary files
- Preprocessor directives and macros
- Conditional compilation
- Data structures: linked lists, stacks, queues
- Abstract data types (ADTs)
- Dynamic data structure implementation

---

## Programs in This Phase

### 01. File I/O (`01_file_io.c`)
**Concept**: Reading and writing files

Mastering file operations:
- Opening files with `fopen()`
- Reading: `fscanf()`, `fgets()`, `fread()`
- Writing: `fprintf()`, `fputs()`, `fwrite()`
- Binary vs text mode
- Error handling with file operations
- Closing files with `fclose()`
- File positioning: `fseek()`, `ftell()`
- Common file I/O patterns

**Key Takeaway**: File I/O is essential for data processing and persistence.

**Compile & Run**:
```bash
gcc -std=c11 -Wall -Wextra 01_file_io.c -o file_io
./file_io
```

---

### 02. Preprocessor (`02_preprocessor.c`)
**Concept**: Macros, conditional compilation

Understanding the C preprocessor:
- `#define` for constants and macros
- `#include` directives
- Conditional compilation: `#ifdef`, `#ifndef`, `#if`
- Macro functions (and their pitfalls)
- `#pragma` directives
- Predefined macros: `__FILE__`, `__LINE__`, `__DATE__`
- Header guards
- When to use macros vs functions

**Key Takeaway**: Preprocessor runs before compilation and enables code generation.

---

### 03. Linked List (`03_linked_list.c`)
**Concept**: Dynamic linear data structure

Implementing singly linked lists:
- Node structure with data and next pointer
- Operations: insert, delete, search
- Traversing the list
- Memory management for nodes
- Advantages over arrays
- Time complexity of operations
- Common patterns and algorithms

**Key Takeaway**: Foundation for understanding dynamic data structures.

---

### 04. Stack (`04_stack.c`)
**Concept**: LIFO (Last-In-First-Out) data structure

Implementing a stack:
- Array-based implementation
- Linked list-based implementation
- Operations: push, pop, peek, isEmpty
- Stack overflow/underflow handling
- Applications: expression evaluation, undo/redo
- Real-world examples

**Key Takeaway**: Essential for algorithms and recursion understanding.

---

### 05. Queue (`05_queue.c`)
**Concept**: FIFO (First-In-First-Out) data structure

Implementing a queue:
- Circular array implementation
- Linked list implementation
- Operations: enqueue, dequeue, front, isEmpty
- Circular buffer concept
- Applications: scheduling, breadth-first search
- Comparison with stack

**Key Takeaway**: Fundamental for understanding algorithms and system design.

---

## Quick Start

### Build All Programs
```bash
make
```

### Build Specific Program
```bash
make linked_list
# or
gcc -std=c11 -Wall -Wextra 03_linked_list.c -o linked_list
```

### Run a Program
```bash
./linked_list
```

### Check for Memory Leaks (Critical!)
```bash
make memcheck
# Checks linked_list, stack, and queue for leaks
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

### With Debugging (Recommended for data structures!)
```bash
gcc -std=c11 -Wall -Wextra -g program.c -o program
```

### Check for Memory Leaks
```bash
valgrind --leak-check=full --show-leak-kinds=all ./program
```

---

## Learning Checklist

After completing Phase 4, you should be able to:

- [ ] Read from and write to text files
- [ ] Work with binary files
- [ ] Handle file errors properly
- [ ] Use preprocessor directives effectively
- [ ] Write and debug macros
- [ ] Implement a linked list from scratch
- [ ] Implement a stack (array and linked-list based)
- [ ] Implement a queue (array and linked-list based)
- [ ] Choose appropriate data structure for a problem
- [ ] Analyze time/space complexity of operations
- [ ] Properly manage memory in dynamic structures
- [ ] Debug data structure implementations

---

## Common Mistakes to Avoid

### File I/O
1. **Not checking fopen return** - Can return NULL!
2. **Forgetting to close files** - Resource leaks
3. **Wrong file mode** - Read vs write, text vs binary
4. **Not handling EOF properly** - Check return values

### Data Structures
1. **Memory leaks in nodes** - Must free all nodes!
2. **Not checking malloc** - Allocation can fail
3. **Losing list head** - Save pointer before traversal
4. **Off-by-one errors** - Especially in array-based structures
5. **Not handling empty structure** - Check before operations
6. **Circular references** - Can cause infinite loops

### Preprocessor
1. **Macro side effects** - `#define SQUARE(x) x*x` is wrong!
2. **Missing parentheses** - Always parenthesize macro arguments
3. **Macros with semicolons** - Can cause syntax errors

---

## Practice Exercises

### File I/O
1. **CSV Parser**: Read CSV file and parse into structures
2. **Binary File Manager**: Write/read structures to binary file
3. **Log File Analyzer**: Process log files and extract statistics

### Data Structures
1. **Doubly Linked List**: Implement with prev and next pointers
2. **Sorted List**: Maintain list in sorted order
3. **Stack Calculator**: Implement calculator using stack
4. **Priority Queue**: Queue where items have priorities
5. **Circular Buffer**: Implement efficient ring buffer

---

## Key Concepts for HPC

### File I/O Performance
- **Buffered vs unbuffered I/O** - Buffering improves performance
- **Sequential vs random access** - Sequential is faster
- **Binary files** - More efficient than text for large data
- **Memory-mapped files** - Advanced technique for large files

### Data Structure Choice
```
Operation          | Array | Linked List
-------------------|-------|------------
Access by index    | O(1)  | O(n)
Insert at front    | O(n)  | O(1)
Insert at end      | O(1)  | O(1) with tail pointer
Delete             | O(n)  | O(1) if pointer known
Memory overhead    | Low   | High (pointers)
Cache performance  | Good  | Poor (fragmented)
```

### Why This Matters
- Understanding data structure performance critical for HPC
- Cache-friendly data layouts improve GPU performance
- File I/O patterns affect throughput in data-intensive apps

---

## Debugging Data Structures

### Visualizing Linked Lists
```c
void printList(Node *head) {
    printf("List: ");
    while (head != NULL) {
        printf("[%d]->", head->data);
        head = head->next;
    }
    printf("NULL\n");
}
```

### Using GDB
```bash
gdb ./linked_list
(gdb) break insertNode
(gdb) run
(gdb) print *head           # Examine head node
(gdb) print head->next      # Follow pointer
(gdb) call printList(head)  # Call your print function
```

### Memory Debugging with Valgrind
```bash
# Detailed leak report
valgrind --leak-check=full --show-leak-kinds=all ./linked_list

# Track where allocated memory came from
valgrind --leak-check=full --track-origins=yes ./linked_list
```

---

## Abstract Data Types (ADTs)

### Stack ADT Interface
```c
typedef struct Stack Stack;

Stack* createStack(int capacity);
void push(Stack *s, int item);
int pop(Stack *s);
int peek(Stack *s);
int isEmpty(Stack *s);
void destroyStack(Stack *s);
```

### Benefits of ADT Approach
- **Encapsulation**: Hide implementation details
- **Flexibility**: Change implementation without affecting users
- **Testability**: Clear interface for testing
- **Maintainability**: Easier to understand and modify

---

## Performance Considerations

### Array vs Linked Implementation

**Array-based**:
- ✅ Cache-friendly (contiguous memory)
- ✅ No pointer overhead
- ✅ Better for small, fixed-size structures
- ❌ Fixed capacity (or expensive realloc)
- ❌ Wasted space if not full

**Linked List-based**:
- ✅ Dynamic size, no wasted space
- ✅ Easy insert/delete
- ✅ No capacity limit
- ❌ Poor cache performance
- ❌ Pointer overhead (extra memory per node)
- ❌ Dynamic allocation overhead

---

## File I/O Best Practices

### Always Check Returns
```c
FILE *fp = fopen("file.txt", "r");
if (fp == NULL) {
    perror("Error opening file");
    return 1;
}
```

### Use fgets for Line Reading
```c
char buffer[256];
while (fgets(buffer, sizeof(buffer), fp) != NULL) {
    // Process line
}
```

### Remember to Close
```c
fclose(fp);
fp = NULL;  // Prevent use after close
```

### Binary I/O for Structures
```c
// Write
fwrite(&student, sizeof(Student), 1, fp);

// Read
fread(&student, sizeof(Student), 1, fp);
```

---

## Next Steps

Once comfortable with Phase 4:
- **Move to Phase 5**: Real-world projects
- **Practice extensively**: Implement variations of each structure
- **Build complex structures**: Trees, graphs, hash tables
- **Use in projects**: Apply data structures to solve problems

---

## Mini-Project Ideas

1. **Text Editor Buffer**
   - Use linked list or gap buffer
   - Support insert, delete, undo operations
   - Save/load files

2. **Task Scheduler Simulator**
   - Multiple queues for different priority tasks
   - Round-robin scheduling
   - Statistics tracking

3. **Expression Evaluator**
   - Infix to postfix conversion using stack
   - Evaluate postfix expressions
   - Handle parentheses and operators

4. **File Database**
   - Store records in binary file
   - Index for fast lookup
   - Support CRUD operations

---

## Resources

- [File I/O in C](https://en.cppreference.com/w/c/io)
- [Preprocessor Reference](https://en.cppreference.com/w/c/preprocessor)
- [Data Structures and Algorithms](https://www.geeksforgeeks.org/data-structures/)
- [Valgrind User Manual](https://valgrind.org/docs/manual/manual.html)
- Main curriculum: [../../README.md](../../README.md)

---

## Tips for Success

- **Draw diagrams** - Visualize list/stack/queue operations
- **Test edge cases** - Empty, single element, full capacity
- **Use valgrind religiously** - No memory leaks allowed!
- **Implement unit tests** - Test each operation separately
- **Start with array-based** - Easier before linked versions
- **Trace through code** - Walk through with sample data
- **Compare complexities** - Understand when to use each structure

---

## Common Interview Questions

1. Implement a stack using two queues
2. Reverse a linked list
3. Detect cycle in linked list
4. Check for balanced parentheses using stack
5. Implement queue using two stacks
6. Find middle element of linked list in one pass

**Practice these!** They're fundamental to interviews.

---

**Time Estimate**: 4 weeks at 7-10 hours/week

**Prerequisite**: Phase 3 complete (especially pointers and dynamic memory)

**Next Phase**: [Phase 5: Real-World Projects](../phase5-projects/)

**Previous Phase**: [Phase 3: Core Concepts](../phase3-core-concepts/)

---

**Important**: Phase 4 builds directly on Phase 3. Make sure you're comfortable with pointers and dynamic memory before starting!
