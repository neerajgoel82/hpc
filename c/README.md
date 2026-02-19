# C Programming

Learn C programming from fundamentals to advanced systems programming.

## Structure

```
c/
├── samples/     # Learning samples organized by phases
├── projects/    # Complete C applications
└── notebooks/   # C-related notebooks (if needed)
```

## Curriculum

The samples are organized into progressive learning phases:

### Phase 1: Foundations
- Basic syntax and structure
- Data types and variables
- Operators and expressions
- Control flow (if, switch, loops)

### Phase 2: Building Blocks
- Functions and scope
- Arrays and strings
- Basic input/output
- Program organization

### Phase 3: Core Concepts
- Pointers and addresses
- Dynamic memory allocation
- Structures and unions
- Typedef and enums

### Phase 4: Advanced
- File I/O
- Command-line arguments
- Advanced data structures
- Preprocessor directives

### Phase 5: Projects
- Complete applications
- Multi-file programs
- Real-world problems

## Getting Started

### Prerequisites
```bash
# Install GCC (if not already installed)
# macOS
brew install gcc

# Ubuntu/Debian
sudo apt-get install build-essential

# Check version (should be 11+)
gcc --version
```

### Compiling Samples

```bash
# Navigate to samples directory
cd samples/

# Compile a single file
gcc -std=c11 -Wall -Wextra file.c -o output

# Using Makefile (if available)
make
make clean
```

### Compilation Standards

All C code uses:
- **Standard**: C11 (`-std=c11`)
- **Warnings**: `-Wall -Wextra`
- **Optimization**: `-O2` (for performance samples)

## Learning Path

1. **Start with phase1-foundations**
   - Work through samples sequentially
   - Compile and run each example
   - Experiment with modifications

2. **Progress through phases**
   - Don't skip ahead
   - Each phase builds on previous concepts
   - Practice is key

3. **Build projects**
   - Apply learned concepts
   - Combine multiple techniques
   - Solve real problems

## Code Style

- **Naming**: `snake_case` for functions and variables
- **Constants**: `UPPER_CASE`
- **Indentation**: 4 spaces (no tabs)
- **Comments**: Explain complex logic, not obvious code

## Resources

- **[samples/CURRICULUM_OVERVIEW.md](samples/CURRICULUM_OVERVIEW.md)** - Complete curriculum guide with detailed learning objectives
- **[samples/GETTING_STARTED.md](samples/GETTING_STARTED.md)** - Detailed setup and compilation instructions
- **samples/Makefile** - Build system for all phases
- Samples include inline documentation explaining key concepts
- Check phase-specific README files in each phase directory

## Using the Makefile

The samples directory includes a comprehensive Makefile:

```bash
cd samples/

# Build specific phase
make phase1
make phase2
make phase3
make phase4
make phase5

# Build everything
make all

# Clean compiled files
make clean

# Test all samples (compile and verify)
make test
```

## Tips

- Always compile with `-Wall -Wextra` to catch warnings
- Test your code with different inputs
- Use `valgrind` to check for memory leaks in advanced phases
- Read compiler error messages carefully
- Start with phase1 and progress sequentially

## Sample Count

- **Phase 1 (Foundations)**: 8+ samples covering basics
- **Phase 2 (Building Blocks)**: 10+ samples on functions and arrays
- **Phase 3 (Core Concepts)**: 8+ samples on pointers and memory
- **Phase 4 (Advanced)**: 6+ samples on file I/O and data structures
- **Phase 5 (Projects)**: Multi-file projects

## Next Steps

- Explore [C++ samples](../cpp/) to learn object-oriented programming
- Check [Python samples](../python/) for high-level programming
- Try [CUDA samples](../cuda/) for GPU programming
- Compare C implementations with other languages for performance insights
