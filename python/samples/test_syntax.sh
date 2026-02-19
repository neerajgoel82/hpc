#!/bin/bash

# Python Samples - Syntax Check Script
# Tests that representative Python samples have valid syntax

set -e

echo "======================================"
echo "   Python Samples - Syntax Check"
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

# Function to test Python syntax
test_syntax() {
    local file=$1

    total=$((total + 1))

    if python3 -m py_compile "$file" 2>/dev/null; then
        echo -e "${GREEN}✓${NC} $file"
        passed=$((passed + 1))
    else
        echo -e "${RED}✗${NC} $file"
        failed=$((failed + 1))
    fi
}

# Test representative samples from each phase
echo "Phase 1: Foundations"
test_syntax "phase1-foundations/01_hello_world.py"
test_syntax "phase1-foundations/02_variables_types.py"
test_syntax "phase1-foundations/03_operators.py"

echo ""
echo "Phase 2: Intermediate"
test_syntax "phase2-intermediate/16_functions_basics.py"
test_syntax "phase2-intermediate/22_exceptions.py"
test_syntax "phase2-intermediate/26_json_files.py"

echo ""
echo "Phase 3: OOP"
test_syntax "phase3-oop/32_classes_objects.py"
test_syntax "phase3-oop/37_inheritance.py"

echo ""
echo "Phase 4: Advanced"
test_syntax "phase4-advanced/43_generators.py"

echo ""
echo "Phase 5: Projects"
# Check if any project files exist
project_files=$(find phase5-projects -name "*.py" 2>/dev/null | head -1)
if [ -n "$project_files" ]; then
    for file in phase5-projects/*.py; do
        if [ -f "$file" ]; then
            test_syntax "$file"
        fi
    done
fi

# Clean up __pycache__
find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
find . -name "*.pyc" -delete 2>/dev/null || true

echo ""
echo "======================================"
echo "   Results"
echo "======================================"
echo "Passed: $passed"
echo "Failed: $failed"
echo "Total:  $total"
echo ""

if [ $failed -eq 0 ]; then
    echo "🎉 All Python samples have valid syntax!"
    exit 0
else
    echo "❌ Some Python samples have syntax errors"
    exit 1
fi
