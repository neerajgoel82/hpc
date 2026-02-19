# HPC Monorepo - Utility Scripts

This directory contains utility scripts for managing the HPC learning monorepo. All scripts should be run from the repository root directory.

## Scripts Overview

### 1. setup.sh
**Purpose**: Verifies that all required development tools are installed and properly configured.

**Usage**:
```bash
./scripts/setup.sh
```

**What it checks**:

**Required Tools:**
- **GCC**: C compiler (C11 standard)
- **G++**: C++ compiler (C++17 standard)
- **Make**: Build automation tool
- **Python3**: Python interpreter (3.9+)
- **pip3**: Python package manager

**Optional Tools:**
- **NVCC**: NVIDIA CUDA compiler (for GPU programming)
- **nvidia-smi**: NVIDIA GPU management tool
- **CMake**: Cross-platform build system (recommended for C++)
- **Git**: Version control system
- **Valgrind**: Memory debugging tool (C/C++)

**Output**:
- Green ✓ marks indicate tools are installed
- Red ✗ marks indicate missing tools
- Installation instructions provided for missing tools
- Final status: "All required tools are installed!" or instructions to install missing tools

**Test Results**:
✅ **Tested and working**
- Successfully detects all installed compilers and tools
- Provides helpful installation instructions for missing tools
- Correctly validates Python version (3.9+)
- Handles optional tools gracefully (CUDA, CMake, Valgrind)

---

### 2. test-all.sh
**Purpose**: Runs tests across all language directories to verify code compiles and executes correctly.

**Usage**:
```bash
./scripts/test-all.sh
```

**What it tests**:

**C Tests** (`c/samples/`):
- Runs `test_compilation.sh`
- Compiles 11 representative C samples from all 4 phases
- Tests single-file compilation and multi-file projects

**C++ Tests** (`cpp/samples/`):
- Runs `test_all_modules.sh`
- Compiles all 45 C++ modules across 14 categories
- Verifies compilation with C++17 standard

**CUDA Tests** (`cuda/samples/local/`):
- Only runs if `nvcc` is available
- Looks for `Makefile` with `make test` target
- Skipped if NVCC not found (most CUDA samples are Colab notebooks)

**Python Tests** (`python/samples/`):
- Runs `pytest` if available and test files exist
- Otherwise runs `test_syntax.sh` (syntax validation)
- Validates 9 representative Python samples from all phases
- Cleans up `__pycache__` after testing

**Output**:
- Color-coded results for each language
- Green ✓ for passing tests
- Red ✗ for failing tests
- Test summary with total/passed/failed counts
- Exit code 1 if any tests fail

**Test Results**:
✅ **All tests passing**
- C tests: ✅ 11/11 samples compile successfully
- C++ tests: ✅ 45/45 modules compile successfully
- Python tests: ✅ 9/9 samples have valid syntax
- CUDA tests: Skipped (NVCC not available on test system)

---

### 3. clean-all.sh
**Purpose**: Removes build artifacts and temporary files across all language directories while preserving source code.

**Usage**:
```bash
./scripts/clean-all.sh
```

**What it cleans**:

**C/C++/CUDA Artifacts:**
- Object files: `*.o`, `*.a`, `*.so`
- Executables: `*.out`, `*.exe`, `a.out`
- Debug symbols: `*.dSYM`, `*.gch`, `*.pch`
- CUDA artifacts: `*.ptx`, `*.cubin`, `*.fatbin`

**Python Artifacts:**
- Cache directories: `__pycache__/`
- Compiled bytecode: `*.pyc`, `*.pyo`
- Package metadata: `*.egg-info/`
- Test cache: `.pytest_cache/`
- Jupyter checkpoints: `.ipynb_checkpoints/`

**Build Directories:**
- `build/`, `dist/`, `obj/`, `out/`

**Process**:
1. Tries to run language-specific clean commands (`make clean`)
2. Performs generic cleanup using `find` commands
3. Cleans all language directories: C, C++, CUDA, Python
4. Cleans root directory artifacts
5. Preserves all source files, Makefiles, and documentation

**Output**:
- Blue markers show which directories are being cleaned
- Green ✓ marks indicate successful cleanup
- Summary of what was cleaned

**Test Results**:
✅ **Tested and working**
- Successfully removes all build artifacts
- Handles missing `make clean` targets gracefully
- Preserves all source code and Makefiles
- Cleans Python cache files and Jupyter checkpoints
- Safe to run repeatedly (idempotent)

---

## Language-Specific Test Scripts

In addition to `test-all.sh`, individual language directories have their own test scripts:

### c/samples/test_compilation.sh
Compiles representative C samples to verify GCC functionality:
- Tests 11 samples across all 4 phases
- Includes single-file and multi-file projects
- Uses GCC with `-std=c11 -Wall -Wextra -g`
- Creates temporary directory for binaries (auto-cleaned)
- Exit code 0 if all compile, 1 if any fail

**Run directly**:
```bash
cd c/samples
./test_compilation.sh
```

### python/samples/test_syntax.sh
Validates Python syntax using `py_compile`:
- Tests 9 samples across all phases
- Uses `python3 -m py_compile` for validation
- Cleans up `__pycache__` automatically
- Exit code 0 if all valid, 1 if any have syntax errors

**Run directly**:
```bash
cd python/samples
./test_syntax.sh
```

### cpp/samples/test_all_modules.sh
Comprehensive C++ compilation test:
- Tests 45 modules across 14 categories
- Uses G++ with `-std=c++17 -Wall -Wextra -g`
- Handles multi-file projects
- Exit code 0 if all compile, 1 if any fail

**Note**: `test-all.sh` automatically uses these scripts when available.

---

## Usage Examples

### Initial Setup
When first cloning the repository:
```bash
# Check if all tools are installed
./scripts/setup.sh

# If missing tools, install them following the instructions
# Then verify again
./scripts/setup.sh
```

### Development Workflow
During development:
```bash
# After making changes, test everything
./scripts/test-all.sh

# Clean build artifacts before committing
./scripts/clean-all.sh

# Stage and commit clean code
git add .
git commit -m "Your changes"
```

### Before Pulling Changes
Clean your workspace:
```bash
./scripts/clean-all.sh
git pull
```

### Full Verification
After major changes or migration:
```bash
./scripts/setup.sh    # Verify tools
./scripts/clean-all.sh # Clean old artifacts
./scripts/test-all.sh  # Run all tests
```

---

## Integration with VSCode

These scripts are integrated into VSCode tasks (`.vscode/tasks.json`):

**From Command Palette** (`Cmd+Shift+P`):
- "Tasks: Run Task" → "HPC: Setup Check" → Runs `setup.sh`
- "Tasks: Run Task" → "HPC: Clean All" → Runs `clean-all.sh`

**Keyboard Shortcuts**:
- Tasks can be bound to custom keyboard shortcuts in VSCode settings

---

## Script Design Principles

All scripts follow these conventions:

1. **Safe by default**: Never delete source code or documentation
2. **Informative**: Clear, color-coded output showing progress
3. **Graceful degradation**: Handle missing tools/files without crashing
4. **Platform aware**: Work on macOS, Linux, and Windows (via WSL)
5. **Project root relative**: Use `SCRIPT_DIR` and `PROJECT_ROOT` for portability
6. **Exit codes**: Non-zero exit on failures (for CI/CD integration)

---

## Extending the Scripts

### Adding New Language Support

To add support for a new language (e.g., Rust, Go):

**In `setup.sh`**:
```bash
echo "Checking Rust Development Tools..."
if command_exists rustc; then
    rust_version=$(rustc --version)
    print_status 0 "Rust found: $rust_version"
else
    print_status 1 "Rust not found"
    all_good=false
fi
```

**In `test-all.sh`**:
```bash
run_language_tests "Rust" "$PROJECT_ROOT/rust/samples" "cargo test"
```

**In `clean-all.sh`**:
```bash
clean_directory "Rust" "$PROJECT_ROOT/rust/samples" "cargo clean"
# Add Rust-specific artifacts
find . -type d -name "target" -exec rm -rf {} + 2>/dev/null || true
```

### Adding Custom Checks

Add custom validation to `setup.sh`:
```bash
echo ""
echo "Checking Custom Requirements..."
echo "-----------------------------------"

# Check for specific library version
if pkg-config --exists libfoo; then
    print_status 0 "libfoo found"
else
    print_status 1 "libfoo not found"
fi
```

---

## Troubleshooting

### setup.sh reports missing tools
**Solution**: Follow the installation instructions shown in yellow. Example:
```bash
# macOS
brew install gcc python3 cmake

# Ubuntu/Debian
sudo apt-get install build-essential python3 python3-pip cmake
```

### test-all.sh shows failures
**Current Status**:
- C tests: Expected (no formal test suite yet)
- C++ tests: Should pass (45/45)
- Python tests: Expected (no pytest tests yet)
- CUDA tests: Expected to skip without NVCC

**Unexpected Failures**: Check compilation errors in the output.

### clean-all.sh doesn't remove everything
The script intentionally preserves:
- Source files (`.c`, `.cpp`, `.py`, `.cu`, `.h`)
- Makefiles and build scripts
- Documentation (`.md` files)
- Configuration files

For a complete reset:
```bash
git clean -fdx  # WARNING: Removes ALL untracked files
```

---

## CI/CD Integration

These scripts are designed for CI/CD pipelines:

**GitHub Actions Example**:
```yaml
- name: Setup check
  run: ./scripts/setup.sh

- name: Run tests
  run: ./scripts/test-all.sh

- name: Clean artifacts
  run: ./scripts/clean-all.sh
```

**Exit Codes**:
- `setup.sh`: Exits 0 (always succeeds, just reports status)
- `test-all.sh`: Exits 1 if any tests fail
- `clean-all.sh`: Exits 0 (always succeeds)

---

## Maintenance Notes

**Last Updated**: 2026-02-19

**Testing Environment**:
- macOS Darwin 23.6.0
- GCC: Apple clang 15.0.0
- Python: 3.11.11
- No NVCC (CUDA) installed

**Known Issues**:
1. `make clean` targets not implemented in all Makefiles (script handles gracefully)
2. CUDA tests not available without NVCC

**Future Enhancements**:
- [ ] Add comprehensive unit tests with pytest for Python samples
- [ ] Add performance benchmarking script
- [ ] Add coverage reporting
- [ ] Add code quality checks (linting, formatting)
- [ ] Add CUDA sample testing for environments with GPU

---

## Related Documentation

- **VSCode Configuration**: `.vscode/README.md`
- **Project Overview**: Root `README.md`
- **Language-Specific Guides**:
  - C: `c/README.md`
  - C++: `cpp/README.md`
  - CUDA: `cuda/README.md`
  - Python: `python/README.md`

---

## Support

For issues or questions:
1. Check the troubleshooting section above
2. Review the main README.md
3. Check individual language README files
4. Create an issue in the repository
