# VSCode Configuration for HPC Monorepo

This directory contains consolidated VSCode configurations for working with C, C++, Python, and CUDA in the HPC learning monorepo.

## Files

### settings.json
Workspace settings including:
- **C/C++ Configuration**: C11 and C++17 standards, compiler paths, IntelliSense
- **Python Configuration**: Virtual environment, linting (flake8), formatting (black)
- **Editor Settings**: Formatting on save, tab size, rulers
- **File Associations**: Language modes for different file types
- **Code Runner**: Quick execution commands for C, C++, and Python
- **File Exclusions**: Hide build artifacts and cache files

### tasks.json
Build and run tasks for all languages:

**C Tasks**:
- Build C (current file)
- Build and Run C
- Build All C Phases

**C++ Tasks**:
- Build C++ (current file) - Default build task
- Build and Run C++
- Build C++ (optimized with -O2)
- Test All C++ Modules

**Python Tasks**:
- Run Python - Default test task
- Check Python Syntax

**Utility Tasks**:
- HPC: Setup Check - Run dependency verification
- HPC: Clean All - Clean all build artifacts
- Clean Binaries (current dir) - Clean current directory only

### launch.json
Debug configurations:

**C Debugging**:
- Debug C - Full debugging with LLDB

**C++ Debugging**:
- Debug C++ - Full debugging with LLDB
- Run C++ - Run without stopping at entry

**Python Debugging**:
- Debug Python - Debug with integrated terminal
- Run Python - Run without debugging

### extensions.json
Recommended VSCode extensions:
- **C/C++**: ms-vscode.cpptools, cmake-tools
- **Python**: ms-python.python, pylance, black-formatter
- **CUDA**: nvidia.nsight-vscode-edition
- **Jupyter**: ms-toolsai.jupyter
- **Utilities**: code-runner, gitlens, markdown tools

## Usage

### Building and Running

**Keyboard Shortcuts**:
- `Cmd+Shift+B` (macOS) or `Ctrl+Shift+B` (Linux/Windows) - Build (default task)
- `Cmd+Shift+P` → "Tasks: Run Task" - See all available tasks
- `F5` - Start debugging (uses launch.json)

**Command Palette Tasks**:
1. Open Command Palette: `Cmd+Shift+P`
2. Type "Tasks: Run Task"
3. Select from:
   - Build C/C++/Python
   - Test tasks
   - Clean tasks
   - HPC utility tasks

### Code Runner (if extension installed)

With the Code Runner extension, you can:
- Click the ▶ button in the top-right corner
- Use `Ctrl+Alt+N` to run current file
- Automatically uses correct compiler/interpreter

### Language-Specific Notes

**C Files**:
- Compiles with: `gcc -std=c11 -Wall -Wextra -g`
- Build task: "Build C (current file)"
- Or use Makefile: `cd c/samples && make all`

**C++ Files**:
- Compiles with: `g++ -std=c++17 -Wall -Wextra -g`
- Build task: "Build C++ (current file)" (default)
- Test script: "Test All C++ Modules"

**Python Files**:
- Runs with: `python3`
- Test task: "Run Python" (default)
- Syntax check: "Check Python Syntax"

## Customization

To customize for your environment:

1. **Compiler Path**: Edit `C_Cpp.default.compilerPath` in settings.json
2. **Python Interpreter**: Edit `python.defaultInterpreterPath` in settings.json
3. **Add Tasks**: Add new tasks to tasks.json
4. **Debug Configurations**: Add new configurations to launch.json

## Per-Language Configs

Individual language directories may have their own `.vscode` folders for language-specific settings:
- `c/samples/.vscode/` - C-specific settings
- `cpp/samples/.vscode/` - C++-specific settings

The root `.vscode` folder provides monorepo-wide defaults that work across all languages.
