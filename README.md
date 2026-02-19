# HPC Learning Monorepo

A comprehensive learning repository for High Performance Computing, containing samples, projects, and notebooks across multiple programming languages.

## Repository Structure

### Language Directories

- **[c/](c/)** - C programming samples and projects
- **[cpp/](cpp/)** - C++ programming samples and projects
- **[cuda/](cuda/)** - CUDA GPU programming samples and projects
- **[python/](python/)** - Python programming samples and projects

Each language directory contains:
- `samples/` - Small, focused examples organized by learning phases/modules
- `projects/` - Larger, complete applications
- `notebooks/` - Jupyter notebooks (when applicable)
- `README.md` - Language-specific documentation

### Supporting Directories

- **[docs/](docs/)** - Cross-language documentation and learning paths
- **[shared/](shared/)** - Shared resources across languages
  - `datasets/` - Test data and benchmarks
  - `utils/` - Cross-language utilities
- **[scripts/](scripts/)** - Build and test automation

## Quick Start

### Prerequisites

- **C**: GCC 11+ with C11 support
- **C++**: G++ 11+ with C++17 support
- **CUDA**: NVIDIA CUDA Toolkit 11.0+
- **Python**: Python 3.9+

### Setup

```bash
# Run setup script to check dependencies
./scripts/setup.sh

# Or manually check individual tools
gcc --version
g++ --version
nvcc --version
python3 --version
```

## Learning Paths

### Beginner Path
Start here if you're new to HPC:
1. C basics ([c/samples/phase1-foundations](c/samples/))
2. Python fundamentals ([python/samples/phase1-foundations](python/samples/))
3. C++ basics ([cpp/samples/01-basics](cpp/samples/))

### Performance Computing Path
Focus on optimization and parallelization:
1. C advanced concepts
2. C++ modern features and STL
3. CUDA GPU programming
4. Python with NumPy/Numba

### Project-Based Learning
Build complete applications in [projects/](projects/) directories of each language.

## Samples vs Projects

**Samples** are small, focused examples demonstrating specific concepts:
- Single concept per file
- Quick to compile and run
- Part of a structured curriculum
- Example: `vector_add.cu`, `hello_world.c`

**Projects** are larger applications combining multiple concepts:
- Multi-file implementations
- Real-world applications
- May include build systems, tests
- Example: Ray tracer, sorting benchmarks

## Development Workflow

```bash
# Navigate to a language directory
cd c/samples/

# Build and run samples
make
./output

# Run tests
make test

# Clean build artifacts
make clean
```

## Documentation

- Each language has its own [README.md](c/README.md) with getting started guides
- See [docs/](docs/) for cross-language documentation and learning resources
- Check `.claude/claude.md` files for coding conventions and guidelines

## Contributing

This is a personal learning repository, but suggestions are welcome:
1. Follow the existing structure (samples vs projects)
2. Match the complexity level of the target phase/module
3. Test that code compiles and runs
4. Update relevant documentation

## Resources

- [Getting Started Guide](docs/getting-started.md)
- [Learning Paths](docs/learning-paths.md)
- [Coding Conventions](docs/conventions.md)

## License

MIT License - see individual directories for language-specific licensing
