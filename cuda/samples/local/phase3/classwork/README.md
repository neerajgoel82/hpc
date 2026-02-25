# Classwork – Reference Programs

This folder contains the **reference / in-class** CUDA programs for this phase. Use them as the source of truth while implementing the matching exercises in `../homework/`.

## Build and run

From this directory:

```bash
make          # Build all programs
make test     # Build and run all programs
make clean    # Remove executables
```

From the phase directory:

```bash
make classwork       # Build classwork programs
make test            # Build and run classwork
```

## Running a single program

```bash
./01_hello_world
./02_device_query
# etc.
```

## Files

Each `.cu` file compiles to an executable with the same base name (e.g. `01_hello_world.cu` → `01_hello_world`). See the phase `README.md` for a list of topics.
