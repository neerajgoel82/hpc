# Module 8: STL and Standard Library

## Overview
Master the Standard Template Library (STL) - C++'s powerful collection of containers, algorithms, and utilities. Essential for CPU-side data preparation before GPU processing.

## Topics Covered

### Sequential Containers
- **vector** - Dynamic array (most important!)
- **array** - Fixed-size array (C++11)
- **list** - Doubly-linked list
- **deque** - Double-ended queue
- **forward_list** - Singly-linked list

### Associative Containers
- **map** - Ordered key-value pairs
- **set** - Ordered unique elements
- **multimap** - Allows duplicate keys
- **multiset** - Allows duplicate elements

### Unordered Containers (C++11)
- **unordered_map** - Hash table
- **unordered_set** - Hash set
- **unordered_multimap** - Hash table with duplicates
- **unordered_multiset** - Hash set with duplicates

### Container Adapters
- **stack** - LIFO structure
- **queue** - FIFO structure
- **priority_queue** - Heap-based priority queue

### Utility Types
- **pair** - Two values
- **tuple** - Multiple values (C++11)
- **optional** - Maybe has value (C++17)
- **variant** - Type-safe union (C++17)
- **any** - Type-safe void* (C++17)

### Iterators
- Input, output, forward, bidirectional, random access
- Iterator operations
- `begin()`, `end()`, `rbegin()`, `rend()`
- Iterator invalidation
- Const iterators

### STL Algorithms
- **Sorting**: `sort`, `stable_sort`, `partial_sort`
- **Searching**: `find`, `binary_search`, `lower_bound`
- **Transforming**: `transform`, `for_each`
- **Numeric**: `accumulate`, `inner_product`
- **Copying**: `copy`, `copy_if`, `move`
- **Removing**: `remove`, `remove_if`, `unique`
- **Min/Max**: `min_element`, `max_element`

### Strings and Streams
- **string** operations
- **stringstream** - String as stream
- **istringstream** - Input from string
- **ostringstream** - Output to string

### File Streams
- **ifstream** - Read files
- **ofstream** - Write files
- **fstream** - Read and write
- Binary vs text mode
- File positioning
- Error handling

### Iterators and Algorithms
- Predicates and functors
- Lambda with algorithms
- Range-based algorithms (C++20 ranges)

## Why This Matters for GPU

### CPU-Side Data Preparation
```cpp
std::vector<float> host_data(1000000);
// Prepare data on CPU
std::sort(host_data.begin(), host_data.end());

// Transfer to GPU
cudaMemcpy(device_data, host_data.data(),
           host_data.size() * sizeof(float),
           cudaMemcpyHostToDevice);
```

### Common Patterns
- **vector**: Store data before GPU transfer
- **map**: Configuration parameters
- **unordered_map**: Fast lookups on CPU
- **algorithms**: Process results from GPU
- **File I/O**: Load data for GPU processing

### GPU Equivalents
- Thrust library provides STL-like interface for GPU
- Many STL algorithms have GPU counterparts
- Understanding STL helps learn Thrust

## Coming Soon

Extensive examples covering:
- Each container with practical use cases
- Iterator usage patterns
- Common algorithms
- File I/O for GPU data
- Performance characteristics
- When to use which container

## Estimated Time
20-25 hours

## Prerequisites
Complete Modules 1-7 first.

**Note**: STL is vast - focus on vector, map, and common algorithms first!