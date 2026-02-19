#!/bin/bash

# HPC Monorepo Test Script
# Runs tests across all language directories

set -e

echo "==================================="
echo "HPC Monorepo - Test All"
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

# Track results
total_tests=0
passed_tests=0
failed_tests=0

# Function to run tests for a language
run_language_tests() {
    local lang=$1
    local lang_dir=$2
    local test_command=$3

    echo -e "${BLUE}Testing $lang...${NC}"
    echo "-----------------------------------"

    if [ ! -d "$lang_dir" ]; then
        echo -e "${YELLOW}  Directory not found: $lang_dir${NC}"
        echo ""
        return
    fi

    cd "$lang_dir"

    if [ -z "$test_command" ]; then
        echo -e "${YELLOW}  No tests configured yet${NC}"
        echo ""
        cd "$PROJECT_ROOT"
        return
    fi

    # Check if test command/file exists
    if [ ! -f "${test_command%% *}" ] && ! command -v "${test_command%% *}" >/dev/null 2>&1; then
        echo -e "${YELLOW}  Test command not found: $test_command${NC}"
        echo ""
        cd "$PROJECT_ROOT"
        return
    fi

    total_tests=$((total_tests + 1))

    if eval "$test_command"; then
        echo -e "${GREEN}✓ $lang tests passed${NC}"
        passed_tests=$((passed_tests + 1))
    else
        echo -e "${RED}✗ $lang tests failed${NC}"
        failed_tests=$((failed_tests + 1))
    fi

    echo ""
    cd "$PROJECT_ROOT"
}

# Test C samples
if command -v gcc >/dev/null 2>&1; then
    run_language_tests "C" "$PROJECT_ROOT/c/samples" "./test_compilation.sh"
else
    echo -e "${YELLOW}Skipping C tests (gcc not found)${NC}"
    echo ""
fi

# Test C++ samples
if command -v g++ >/dev/null 2>&1; then
    run_language_tests "C++" "$PROJECT_ROOT/cpp/samples" "./test_all_modules.sh"
else
    echo -e "${YELLOW}Skipping C++ tests (g++ not found)${NC}"
    echo ""
fi

# Test CUDA samples (if available)
if command -v nvcc >/dev/null 2>&1; then
    run_language_tests "CUDA" "$PROJECT_ROOT/cuda/samples/local" "make test"
else
    echo -e "${YELLOW}Skipping CUDA tests (nvcc not found)${NC}"
    echo ""
fi

# Test Python samples
if command -v python3 >/dev/null 2>&1; then
    # Check if pytest is available and if test files exist
    if python3 -c "import pytest" 2>/dev/null && find "$PROJECT_ROOT/python/samples" -name "test_*.py" -o -name "*_test.py" | grep -q .; then
        run_language_tests "Python (pytest)" "$PROJECT_ROOT/python/samples" "pytest"
    else
        # Run syntax checks instead
        run_language_tests "Python" "$PROJECT_ROOT/python/samples" "./test_syntax.sh"
    fi
else
    echo -e "${YELLOW}Skipping Python tests (python3 not found)${NC}"
    echo ""
fi

# Print summary
echo "==================================="
echo "Test Summary"
echo "==================================="
if [ $total_tests -eq 0 ]; then
    echo -e "${YELLOW}No tests were run${NC}"
    echo "Note: Tests will be available after migrating content from existing repositories"
else
    echo -e "Total: $total_tests"
    echo -e "${GREEN}Passed: $passed_tests${NC}"
    if [ $failed_tests -gt 0 ]; then
        echo -e "${RED}Failed: $failed_tests${NC}"
    else
        echo -e "Failed: $failed_tests"
    fi

    echo ""
    if [ $failed_tests -eq 0 ]; then
        echo -e "${GREEN}All tests passed!${NC}"
    else
        echo -e "${RED}Some tests failed${NC}"
        exit 1
    fi
fi
echo "==================================="
