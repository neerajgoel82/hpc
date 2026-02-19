#!/bin/bash

# C Samples - Compilation Test Script
# Tests that representative C samples compile successfully

set -e

echo "======================================"
echo "   C Samples - Compilation Test"
echo "======================================"
echo ""

# Colors
GREEN='\033[0;32m'
RED='\033[0;31m'
NC='\033[0m'

# Get script directory
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
cd "$SCRIPT_DIR"

# Track results
total=0
passed=0
failed=0

# Temporary directory for compiled binaries
TEMP_DIR=$(mktemp -d)
trap "rm -rf $TEMP_DIR" EXIT

# Function to test compile a C file
test_compile() {
    local file=$1
    local output_name=$(basename "$file" .c)

    total=$((total + 1))

    if gcc -std=c11 -Wall -Wextra -g "$file" -o "$TEMP_DIR/$output_name" 2>/dev/null; then
        echo -e "${GREEN}✓${NC} $file"
        passed=$((passed + 1))
    else
        echo -e "${RED}✗${NC} $file"
        failed=$((failed + 1))
    fi
}

# Test representative samples from each phase
echo "Phase 1: Foundations"
test_compile "phase1-foundations/01_hello_world.c"
test_compile "phase1-foundations/02_data_types.c"
test_compile "phase1-foundations/03_operators.c"

echo ""
echo "Phase 2: Building Blocks"
test_compile "phase2-building-blocks/01_functions.c"
test_compile "phase2-building-blocks/04_arrays.c"
test_compile "phase2-building-blocks/06_strings.c"

echo ""
echo "Phase 3: Core Concepts"
test_compile "phase3-core-concepts/01_pointers_basics.c"
test_compile "phase3-core-concepts/04_structures.c"

echo ""
echo "Phase 4: Advanced"
test_compile "phase4-advanced/02_preprocessor.c"
test_compile "phase4-advanced/03_linked_list.c"

# Special case: header files example (needs multiple files)
echo ""
echo "Phase 4: Header Files (multi-file)"
if [ -d "phase4-advanced/06_header_files" ]; then
    cd "phase4-advanced/06_header_files"
    if make clean > /dev/null 2>&1 && make > /dev/null 2>&1; then
        echo -e "${GREEN}✓${NC} phase4-advanced/06_header_files (multi-file project)"
        passed=$((passed + 1))
        make clean > /dev/null 2>&1
    else
        echo -e "${RED}✗${NC} phase4-advanced/06_header_files (multi-file project)"
        failed=$((failed + 1))
    fi
    total=$((total + 1))
    cd "$SCRIPT_DIR"
fi

echo ""
echo "======================================"
echo "   Results"
echo "======================================"
echo "Passed: $passed"
echo "Failed: $failed"
echo "Total:  $total"
echo ""

if [ $failed -eq 0 ]; then
    echo "🎉 All C samples compile successfully!"
    exit 0
else
    echo "❌ Some C samples failed to compile"
    exit 1
fi
