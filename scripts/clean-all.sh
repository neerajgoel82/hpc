#!/bin/bash

# HPC Monorepo Clean Script
# Cleans build artifacts across all language directories

set -e

echo "==================================="
echo "HPC Monorepo - Clean All"
echo "==================================="
echo ""

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Get script directory and project root
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

# Function to clean a directory
clean_directory() {
    local lang=$1
    local lang_dir=$2
    local clean_command=$3

    echo -e "${BLUE}Cleaning $lang...${NC}"

    if [ ! -d "$lang_dir" ]; then
        echo -e "${YELLOW}  Directory not found: $lang_dir${NC}"
        echo ""
        return
    fi

    cd "$lang_dir"

    # Run custom clean command if provided and exists
    if [ -n "$clean_command" ]; then
        if [ -f "Makefile" ] || command -v "${clean_command%% *}" >/dev/null 2>&1; then
            echo "  Running: $clean_command"
            eval "$clean_command" 2>/dev/null || echo -e "${YELLOW}  Clean command not available yet${NC}"
        fi
    fi

    # Generic cleanup
    echo "  Removing common build artifacts..."

    # C/C++/CUDA artifacts
    find . -type f \( -name "*.o" -o -name "*.out" -o -name "*.exe" -o -name "*.so" -o -name "*.a" \) -delete 2>/dev/null || true
    find . -type f \( -name "*.dSYM" -o -name "*.gch" -o -name "*.pch" \) -delete 2>/dev/null || true
    find . -type f \( -name "*.ptx" -o -name "*.cubin" -o -name "*.fatbin" \) -delete 2>/dev/null || true
    find . -type f -name "a.out" -delete 2>/dev/null || true

    # Python artifacts
    find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
    find . -type f -name "*.pyc" -delete 2>/dev/null || true
    find . -type f -name "*.pyo" -delete 2>/dev/null || true
    find . -type d -name "*.egg-info" -exec rm -rf {} + 2>/dev/null || true
    find . -type d -name ".pytest_cache" -exec rm -rf {} + 2>/dev/null || true
    find . -type d -name ".ipynb_checkpoints" -exec rm -rf {} + 2>/dev/null || true

    # Build directories
    find . -type d -name "build" -not -path "*/node_modules/*" -exec rm -rf {} + 2>/dev/null || true
    find . -type d -name "dist" -exec rm -rf {} + 2>/dev/null || true

    echo -e "${GREEN}✓ $lang cleaned${NC}"
    echo ""

    cd "$PROJECT_ROOT"
}

# Clean C
clean_directory "C" "$PROJECT_ROOT/c/samples" "make clean"

# Clean C++
clean_directory "C++" "$PROJECT_ROOT/cpp/samples" "make clean"

# Clean CUDA
clean_directory "CUDA" "$PROJECT_ROOT/cuda/samples" "make clean"

# Clean Python
clean_directory "Python" "$PROJECT_ROOT/python/samples" ""

# Clean root level artifacts
echo -e "${BLUE}Cleaning root directory...${NC}"
cd "$PROJECT_ROOT"

# Remove any stray build artifacts at root
rm -rf build/ dist/ *.egg-info/ 2>/dev/null || true
find . -maxdepth 1 -type f \( -name "*.o" -o -name "*.out" -o -name "*.pyc" \) -delete 2>/dev/null || true

echo -e "${GREEN}✓ Root cleaned${NC}"
echo ""

# Summary
echo "==================================="
echo -e "${GREEN}All build artifacts cleaned!${NC}"
echo "==================================="
echo ""
echo "Cleaned:"
echo "  • C/C++/CUDA object files (*.o, *.out, *.exe)"
echo "  • Python cache (__pycache__, *.pyc)"
echo "  • Build directories (build/, dist/)"
echo "  • Jupyter checkpoints"
echo ""
echo "Source files and Makefiles preserved."
