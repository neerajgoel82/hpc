#!/bin/bash
# Comprehensive compilation test for all modules

cd "$(dirname "$0")"
FAILED=0
PASSED=0

echo "======================================"
echo "   C++ to GPU - Compilation Test"
echo "======================================"
echo ""

test_file() {
    local file=$1
    local flags=$2
    local name=$(basename "$file" .cpp)
    
    if g++ -std=c++17 $flags "$file" -o "/tmp/${name}_test" 2>/dev/null; then
        echo "✓ $file"
        rm -f "/tmp/${name}_test"
        ((PASSED++))
    else
        echo "✗ $file FAILED"
        ((FAILED++))
    fi
}

echo "Module 1: C++ Fundamentals"
for f in 01-basics/*.cpp; do test_file "$f" ""; done
echo ""

echo "Module 2: Functions and Program Structure"
test_file "02-functions-structure/01_function_parameters.cpp" ""
test_file "02-functions-structure/02_function_overloading.cpp" ""
test_file "02-functions-structure/03_inline_functions.cpp" ""
test_file "02-functions-structure/05_preprocessor.cpp" ""
test_file "02-functions-structure/06_file_io.cpp" ""
# Multi-file project
if g++ -std=c++17 02-functions-structure/04_header_example_main.cpp 02-functions-structure/math_utils.cpp -o /tmp/header_test 2>/dev/null; then
    echo "✓ 02-functions-structure/04_header_example_main.cpp (with math_utils.cpp)"
    rm -f /tmp/header_test
    ((PASSED++))
else
    echo "✗ 02-functions-structure/04_header_example_main.cpp FAILED"
    ((FAILED++))
fi
echo ""

echo "Module 3: Pointers and Memory"
for f in 03-pointers-memory/*.cpp; do test_file "$f" ""; done
echo ""

echo "Module 4: Classes and OOP"
for f in 04-classes-oop/*.cpp; do test_file "$f" ""; done
echo ""

echo "Module 5: Inheritance and Polymorphism"
for f in 05-inheritance-polymorphism/*.cpp; do test_file "$f" ""; done
echo ""

echo "Module 6: Operator Overloading"
for f in 06-operators-advanced/*.cpp; do test_file "$f" ""; done
echo ""

echo "Module 7: Templates"
for f in 07-templates/*.cpp; do test_file "$f" ""; done
echo ""

echo "Module 8: STL"
for f in 08-stl/*.cpp; do test_file "$f" ""; done
echo ""

echo "Module 9: Modern C++"
for f in 09-modern-cpp/*.cpp; do test_file "$f" ""; done
echo ""

echo "Module 10: Exceptions"
for f in 10-exceptions/*.cpp; do test_file "$f" ""; done
echo ""

echo "Module 11: Multithreading"
for f in 11-multithreading/*.cpp; do test_file "$f" "-pthread"; done
echo ""

echo "Module 13: GPU Advanced"
for f in 13-gpu-advanced/*.cpp; do test_file "$f" "-O2"; done
echo ""

echo "Module 14: GPU Preparation"
for f in 14-gpu-prep/*.cpp; do test_file "$f" "-pthread"; done
echo ""

echo "======================================"
echo "   Results"
echo "======================================"
echo "Passed: $PASSED"
echo "Failed: $FAILED"
echo "Total:  $((PASSED + FAILED))"
echo ""

if [ $FAILED -eq 0 ]; then
    echo "🎉 All modules compile successfully!"
    exit 0
else
    echo "❌ Some files failed to compile"
    exit 1
fi
