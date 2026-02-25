# Homework – Exercise Stubs

This folder contains **homework exercise stubs** that mirror the classwork programs. Implement each file based on the corresponding one in `../classwork/`.

## Workflow

1. Read the classwork file (e.g. `../classwork/01_hello_world.py`).
2. Edit the matching homework file (e.g. `01_hello_world.py`) and implement the exercises.
3. Run and test from this directory or from the phase directory.

## Run programs

From this directory:

```bash
make run       # Run all .py in this directory
make run-all   # Run all .py (including subdirs, if any)
make list      # List Python files here
make clean     # Remove __pycache__ and .pyc
make help      # Show Makefile targets
```

From the phase directory:

```bash
make homework       # Run all homework (phase Makefile)
make test-homework  # Same
```

## Run a single file

```bash
python3 01_hello_world.py
# etc.
```

Each stub runs as-is (e.g. prints a reminder). Replace with your solution.
