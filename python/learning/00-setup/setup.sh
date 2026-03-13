#!/bin/bash

# Python Learning Environment Setup Script
# Creates a virtual environment and installs required packages

set -e  # Exit on error

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LEARNING_DIR="$(dirname "$SCRIPT_DIR")"
VENV_DIR="$LEARNING_DIR/.venv"

echo "======================================"
echo "Python Learning Environment Setup"
echo "======================================"
echo ""

# Check Python version
echo "1. Checking Python installation..."
if ! command -v python3 &> /dev/null; then
    echo "Error: python3 not found. Please install Python 3.8 or higher."
    exit 1
fi

PYTHON_VERSION=$(python3 --version | awk '{print $2}')
echo "   Found Python $PYTHON_VERSION"

# Check if version is at least 3.8
MAJOR=$(echo $PYTHON_VERSION | cut -d. -f1)
MINOR=$(echo $PYTHON_VERSION | cut -d. -f2)

if [ "$MAJOR" -lt 3 ] || ([ "$MAJOR" -eq 3 ] && [ "$MINOR" -lt 8 ]); then
    echo "   Warning: Python 3.8+ recommended, you have $PYTHON_VERSION"
fi
echo ""

# Create virtual environment
echo "2. Creating virtual environment..."
if [ -d "$VENV_DIR" ]; then
    echo "   Virtual environment already exists at: $VENV_DIR"
    read -p "   Remove and recreate? (y/N): " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        echo "   Removing existing virtual environment..."
        rm -rf "$VENV_DIR"
    else
        echo "   Keeping existing virtual environment."
        echo "   To reinstall packages, activate the venv and run: pip install -r 00-setup/requirements.txt"
        exit 0
    fi
fi

echo "   Creating virtual environment at: $VENV_DIR"
python3 -m venv "$VENV_DIR"
echo "   ✓ Virtual environment created"
echo ""

# Activate virtual environment
echo "3. Activating virtual environment..."
source "$VENV_DIR/bin/activate"
echo "   ✓ Virtual environment activated"
echo ""

# Upgrade pip
echo "4. Upgrading pip..."
pip install --quiet --upgrade pip
echo "   ✓ pip upgraded to $(pip --version | awk '{print $2}')"
echo ""

# Install requirements
echo "5. Installing requirements..."
if [ -f "$SCRIPT_DIR/requirements.txt" ]; then
    pip install --quiet -r "$SCRIPT_DIR/requirements.txt"
    echo "   ✓ Requirements installed"
else
    echo "   Warning: requirements.txt not found"
fi
echo ""

# Summary
echo "======================================"
echo "Setup Complete!"
echo "======================================"
echo ""
echo "To activate the virtual environment:"
echo "  source .venv/bin/activate"
echo ""
echo "To deactivate when done:"
echo "  deactivate"
echo ""
echo "To test all Python files:"
echo "  make test"
echo ""
echo "Optional phase-specific setups:"
echo "  make install-phase5     # Data Science packages"
echo "  make install-phase6-cpu # PyTorch (CPU)"
echo "  make install-phase6-gpu # PyTorch (GPU)"
echo ""
