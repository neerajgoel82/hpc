# Homework – Exercise Stubs

This folder contains **homework exercise stubs** that mirror the classwork programs. Implement each one based on the corresponding file in `../classwork/`.

## Workflow

1. Read the classwork file (e.g. `../classwork/01_hello_world.cu`).
2. Edit the matching homework file (e.g. `01_hello_world.cu`) and implement the exercises.
3. Build and test from this directory or from the phase directory.

## Build and run

From this directory:

```bash
make          # Build all homework programs
make test     # Build and run all homework programs
make clean    # Remove executables
```

From the phase directory:

```bash
make homework       # Build homework programs
make test-homework  # Build and run homework
make both           # Build both classwork and homework
```

## Running a single program

```bash
./01_hello_world
# etc.
```

Each stub compiles and runs as-is (prints a reminder message). Replace the stub implementation with your solution.
