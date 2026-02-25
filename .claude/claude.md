# HPC Learning Monorepo - Claude Instructions

## Purpose
This is a learning repository for High Performance Computing containing samples and projects across C, C++, CUDA, and Python.

## Repository Structure

### Language Directories
- `c/` - C programming samples and projects
- `cpp/` - C++ programming samples and projects
- `cuda/` - CUDA GPU programming samples and projects
- `python/` - Python programming samples and projects

### Within Each Language Directory
- `learning/` - Small, focused examples demonstrating specific concepts (C, C++)
- `samples/` - Small, focused examples demonstrating specific concepts (CUDA, Python)
- `projects/` - Larger, complete implementations combining multiple concepts
- `notebooks/` - Jupyter notebooks (when needed)
- `.claude/claude.md` - Language-specific instructions

### Shared Resources
- `docs/` - Cross-language documentation and learning paths
- `shared/datasets/` - Test data used across languages
- `shared/utils/` - Cross-language utilities
- `scripts/` - Build and test automation scripts

## Samples vs Projects

**Samples** are:
- Single-file or very simple examples
- Focused on one specific concept
- Part of a learning curriculum/phases
- Quick to compile and run
- Examples: hello_world.c, pointers.c, vector_add.cu

**Projects** are:
- Multi-file applications
- Combine multiple concepts
- More complete implementations
- May include build systems, tests, documentation
- Examples: ray tracer, sorting algorithms, neural network

## General Guidelines

### When Adding New Code
1. Determine if it's a sample or project
2. Place in appropriate language directory
3. Follow the language-specific conventions (see language .claude/claude.md)
4. Keep samples simple and focused
5. Don't over-engineer learning code

### Compilation Standards
- **C**: C11 standard (`-std=c11 -Wall -Wextra`)
- **C++**: C++17 standard (`-std=c++17 -Wall -Wextra`)
- **CUDA**: Use nvcc with appropriate compute capability
- **Python**: Python 3.9+ with type hints when appropriate

### Code Quality by Context
- **Early phase samples**: Keep simple, minimal error handling, focus on the concept
- **Advanced samples**: Add proper error handling and edge cases
- **Projects**: Production-quality code with full error handling, tests, documentation

### What NOT to Do
- Don't add features beyond what's requested
- Don't refactor existing code unless explicitly asked
- Don't add "professional" overhead to beginner samples
- Don't mix languages in the same file/directory
- Don't duplicate shared resources (use shared/)

### Performance Focus
This is an HPC repository - when writing code:
- Consider memory access patterns and cache locality
- Think about parallelization opportunities
- Document performance characteristics
- Compare implementations across languages when relevant

### Documentation
- Update relevant README.md when adding significant content
- Keep documentation close to the code it describes
- Link between related implementations in different languages

### Testing
- Verify code compiles before committing
- Use existing build systems (Makefiles, scripts)
- Run language-specific test scripts when available

## Cross-Language Patterns

When implementing the same algorithm in multiple languages:
- Keep the core logic similar for comparison
- Document performance differences
- Place in respective language directories
- Consider adding a comparison notebook

## Learning Path Awareness
Each language has a curriculum/learning path. When adding samples:
- Understand where they fit in the progression
- Maintain difficulty progression
- Update curriculum docs if needed
