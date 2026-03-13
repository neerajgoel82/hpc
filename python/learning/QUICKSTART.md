# Quick Start Guide

## New Machine Setup

### Step 1: Basic Setup (Required)

```bash
make setup
```

This creates the virtual environment and installs basic requirements.

**Tests Phase 1-4 only:**
```bash
make test
```

At this point, Phase 5 and 6 files will be skipped with warnings like:
```
⚠ Warning: Phase 5 (Data Science) dependencies not installed
⚠ Warning: Phase 6 (PyTorch) dependencies not installed
```

This is **normal** - continue to Step 2 to install them.

---

### Step 2: Install Phase 5 Dependencies (Optional)

```bash
make install-phase5
```

Installs: numpy, pandas, matplotlib, scikit-learn, etc.

**Now tests Phase 1-5:**
```bash
make test
```

---

### Step 3: Install Phase 6 Dependencies (Optional)

**For CPU machines:**
```bash
make install-phase6-cpu
```

**For GPU machines with CUDA:**
```bash
make install-phase6-gpu
make check-cuda  # Verify GPU is detected
```

**Now tests all phases:**
```bash
make test
```

---

## Complete Setup (All Phases)

If you want everything at once:

```bash
# Step 1: Basic
make setup

# Step 2: Phase 5
make install-phase5

# Step 3: Phase 6
make install-phase6-cpu   # or install-phase6-gpu

# Test everything
make test
```

---

## Troubleshooting

### "make test does nothing"

**Diagnosis:**
```bash
bash diagnose.sh
```

**Common causes:**
1. Not in the correct directory (`python/learning/`)
2. Virtual environment not created (`make setup` first)
3. Dependencies not installed (see warnings in `make test`)

### "Phase 5/6 files all skipped"

This means dependencies aren't installed:

```bash
# Check what's missing
.venv/bin/python -c "import numpy"     # Phase 5
.venv/bin/python -c "import torch"     # Phase 6

# Install missing dependencies
make install-phase5      # For Phase 5
make install-phase6-cpu  # For Phase 6
```

### "make test fails on GPU machine"

If you installed CPU version but have GPU:

```bash
# Reinstall with GPU support
pip uninstall torch torchvision torchaudio
make install-phase6-gpu
```

---

## What Each Step Does

| Step | Command | Installs | Tests |
|------|---------|----------|-------|
| 1 | `make setup` | IPython | Phase 1-4 |
| 2 | `make install-phase5` | numpy, pandas, scikit-learn | Phase 1-5 |
| 3 | `make install-phase6-cpu` | PyTorch (CPU) | Phase 1-6 |
| 3 | `make install-phase6-gpu` | PyTorch (GPU+CUDA) | Phase 1-6 |

---

## Summary

- ✅ `make setup` is required first
- ✅ Phase 5/6 need separate installation commands
- ✅ `make test` will skip phases without dependencies (with warnings)
- ✅ Install phases as you need them
- ✅ GPU setup auto-detects CUDA version

**Most common workflow:**
```bash
make setup && make install-phase5 && make install-phase6-cpu && make test
```
