# Python Learning Environment Setup

This directory contains setup scripts and configuration for the Python learning environment.

## Quick Start

From the `python/learning` directory:

```bash
make setup
```

This will:
1. Check your Python installation (requires Python 3.8+)
2. Create a virtual environment at `.venv/`
3. Upgrade pip to the latest version
4. Install basic requirements

## Manual Setup

If you prefer to run the setup script directly:

```bash
cd python/learning
./00-setup/setup.sh
```

## What Gets Installed

### Basic Setup (Phase 1-4)
- **IPython**: Enhanced interactive Python shell

Phase 1-4 samples use only Python's standard library, so no additional packages are required.

### Optional: Phase 5 (Data Science)
Install separately when needed:
```bash
make install-phase5
```

This installs:
- NumPy - Numerical computing
- Pandas - Data manipulation
- Matplotlib - Plotting
- SciPy - Scientific computing
- Scikit-learn - Machine learning
- Seaborn - Statistical visualization
- Jupyter - Interactive notebooks

### Optional: Phase 6 (PyTorch)
Install separately when needed:
```bash
make install-phase6-cpu  # For CPU-only
# OR
make install-phase6-gpu  # For GPU support
```

This installs PyTorch and related libraries.

## Virtual Environment

The setup creates a virtual environment at `python/learning/.venv/`

### Activating the Environment

**On macOS/Linux:**
```bash
source .venv/bin/activate
```

**On Windows:**
```bash
.venv\Scripts\activate
```

### Deactivating
```bash
deactivate
```

## Testing Your Setup

After setup, test that all Python files have valid syntax:

```bash
make test
```

This checks all Python files across all phases for syntax errors.

## Requirements File

The `requirements.txt` file contains:
- Base requirements for Phase 1-4
- Commented-out optional requirements for Phase 5-6

You can install additional packages as needed:
```bash
source .venv/bin/activate
pip install <package-name>
```

## Troubleshooting

### Python Version Issues
If you need a specific Python version:
```bash
python3.11 -m venv .venv  # Use Python 3.11
```

### Virtual Environment Already Exists
The setup script will ask if you want to recreate it. To force recreation:
```bash
rm -rf .venv
make setup
```

### Permission Errors
Make sure the setup script is executable:
```bash
chmod +x 00-setup/setup.sh
```

## Python Version Requirements

- **Minimum**: Python 3.8
- **Recommended**: Python 3.10+
- **Tested with**: Python 3.11

## Platform-Specific Notes

### macOS
Python 3 should be available. If not:
```bash
brew install python@3.11
```

### Linux (Ubuntu/Debian)
```bash
sudo apt update
sudo apt install python3 python3-venv python3-pip
```

### Windows
Download Python from [python.org](https://www.python.org/downloads/) and ensure "Add Python to PATH" is checked during installation.

## Next Steps

After setup:
1. Run `make test` to verify all files
2. Start with Phase 1: `cd 01-foundations`
3. Read the phase README and follow along
4. Work through classwork, then homework

Happy learning!
