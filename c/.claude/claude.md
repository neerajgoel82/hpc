# C Programming - Claude Instructions

## Curriculum Structure

The C samples are organized by phases:
- **phase1-foundations**: Basic syntax, data types, operators, control flow
- **phase2-building-blocks**: Functions, arrays, strings, basic I/O
- **phase3-core-concepts**: Pointers, memory management, structures
- **phase4-advanced**: File I/O, dynamic memory, advanced data structures
- **phase5-projects**: Complete applications combining multiple concepts

## Compilation Standards

### Required Flags
```bash
gcc -std=c11 -Wall -Wextra -O2 file.c -o output
```

- Always use C11 standard (`-std=c11`)
- Enable all warnings (`-Wall -Wextra`)
- No warnings tolerated - fix them all
- Use `-O2` for performance samples

### Build System
- Makefile exists in `learning/` directory
- Don't create new build systems unless required
- Test with `make clean && make` before committing

## Coding Style

### Naming Conventions
- Functions: `snake_case` (e.g., `calculate_sum`)
- Variables: `snake_case` (e.g., `user_count`)
- Constants: `UPPER_CASE` (e.g., `MAX_SIZE`)
- Macros: `UPPER_CASE` (e.g., `DEBUG_MODE`)

### Code Organization
- One main() per file for samples
- Include guards for headers: `#ifndef FILE_H` / `#define FILE_H`
- Order: includes, defines, typedefs, function declarations, main, function definitions

### Formatting
- 4 spaces for indentation (no tabs)
- Opening brace on same line for functions
- Consistent spacing around operators

## Phase-Specific Guidelines

### Phase 1-2: Foundations and Building Blocks
- Keep extremely simple
- Minimal or no error handling
- Focus on demonstrating the concept clearly
- Single concept per file
- Don't add comments explaining obvious code

### Phase 3: Core Concepts
- Introduce proper error checking
- Add comments for complex pointer operations
- Demonstrate memory management clearly

### Phase 4-5: Advanced and Projects
- Full error handling required
- Check all return values
- Free all allocated memory
- Add comprehensive comments
- Consider edge cases

## File Naming
- Descriptive names: `03_dynamic_arrays.c` not `test3.c`
- Use phase prefix: `phase1_01_hello.c` if needed
- Lowercase with underscores

## Common Patterns

### Error Handling
```c
// Phase 1-2: Simple or none
result = function();

// Phase 3+: Check returns
if (function() != 0) {
    fprintf(stderr, "Error message\n");
    return 1;
}

// Phase 4+: Full error handling
ptr = malloc(size);
if (ptr == NULL) {
    perror("malloc failed");
    return EXIT_FAILURE;
}
```

### Memory Management
- Always pair malloc/calloc with free
- Set pointers to NULL after freeing
- Use valgrind to check for leaks in advanced phases

## What NOT to Do

- Don't use C++ features (use .c not .cpp)
- Don't add CMake (we use Make)
- Don't use VLAs (variable length arrays) unless specifically teaching them
- Don't add platform-specific code without guards
- Don't use deprecated functions (gets, etc.)
- Don't add unnecessary abstractions in learning samples

## When Adding New Samples
1. Place in appropriate phase directory
2. Follow existing file naming patterns
3. Match complexity level of the phase
4. Test compilation with gcc
5. Verify it runs correctly
6. Update phase README if needed

## Testing
- Compile with warnings: `gcc -Wall -Wextra -Werror`
- Run the program and verify output
- For advanced phases, run valgrind: `valgrind --leak-check=full ./program`

## Documentation
- Samples should be mostly self-explanatory
- Add comments for non-obvious algorithms
- Don't over-comment simple code
- Header comments: brief description, author optional
