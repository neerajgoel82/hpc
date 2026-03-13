#!/bin/bash

echo "==================================="
echo "Python Setup Diagnostics"
echo "==================================="
echo ""

echo "1. Current directory:"
pwd
echo ""

echo "2. Virtual environment:"
if [ -f .venv/bin/python ]; then
    echo "   ✓ Virtual environment exists"
    .venv/bin/python --version
else
    echo "   ✗ Virtual environment NOT found at .venv/bin/python"
fi
echo ""

echo "3. PYTHON variable in Makefile:"
VENV_DIR=.venv
PYTHON=$(if [ -f $VENV_DIR/bin/python ]; then echo $VENV_DIR/bin/python; else echo python3; fi)
echo "   PYTHON=$PYTHON"
echo ""

echo "4. Test Python execution:"
if [ -f .venv/bin/python ]; then
    .venv/bin/python -c "import sys; print(f'   Python: {sys.version}')"
else
    python3 -c "import sys; print(f'   Python: {sys.version}')"
fi
echo ""

echo "5. Phase directories:"
for phase in 01-foundations 02-intermediate 03-oop 04-advanced 05-datascience 06-pytorch; do
    if [ -d "$phase" ]; then
        count=$(find "$phase" -name "*.py" -type f 2>/dev/null | wc -l | tr -d ' ')
        echo "   ✓ $phase ($count Python files)"
    else
        echo "   ✗ $phase (NOT FOUND)"
    fi
done
echo ""

echo "6. Dependencies for Phase 5/6:"
if [ -f .venv/bin/python ]; then
    echo "   Checking numpy:"
    .venv/bin/python -c "import numpy; print(f'   ✓ numpy {numpy.__version__}')" 2>/dev/null || echo "   ✗ numpy not installed"

    echo "   Checking torch:"
    .venv/bin/python -c "import torch; print(f'   ✓ torch {torch.__version__}')" 2>/dev/null || echo "   ✗ torch not installed"
fi
echo ""

echo "7. Test 'make test' detection:"
echo "   Running simplified version..."
PYTHON=$(.venv/bin/python --version > /dev/null 2>&1 && echo .venv/bin/python || echo python3)
total=0
for phase in 01-foundations 02-intermediate; do
    if [ -d "$phase/classwork" ]; then
        for file in $(find "$phase/classwork" -name "*.py" -type f 2>/dev/null | head -3); do
            total=$((total + 1))
            echo "   Found: $file"
        done
    fi
done
echo "   Total files found in test: $total"
echo ""

echo "==================================="
echo "Run this script on the other machine:"
echo "bash diagnose.sh"
echo "==================================="
