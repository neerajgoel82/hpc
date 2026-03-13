.PHONY: help clean classwork homework both test test-homework list install-cpu install-gpu install check-cuda

VENV_DIR := ../.venv
PYTHON := $(shell if [ -f $(VENV_DIR)/bin/python ]; then echo $(VENV_DIR)/bin/python; else echo python3; fi)
PIP := $(shell if [ -f $(VENV_DIR)/bin/pip ]; then echo $(VENV_DIR)/bin/pip; else echo pip3; fi)

# Directories
CLASSWORK_DIR := classwork
HOMEWORK_DIR := homework

# Find Python files
CLASSWORK_SAMPLES := $(wildcard $(CLASSWORK_DIR)/[0-9]*.py)
CLASSWORK_PROJECTS := $(wildcard $(CLASSWORK_DIR)/project_*.py)
HOMEWORK_SAMPLES := $(wildcard $(HOMEWORK_DIR)/[0-9]*.py)
HOMEWORK_PROJECTS := $(wildcard $(HOMEWORK_DIR)/project_*.py)

help:
	@echo "Phase 6: PyTorch Deep Learning - Makefile Commands"
	@echo "================================================"
	@echo "make help         - Show this help message"
	@echo "make install-cpu  - Install Phase 6 dependencies (CPU-only)"
	@echo "make install-gpu  - Install Phase 6 dependencies (with GPU support)"
	@echo "make check-cuda   - Check CUDA availability and GPU info"
	@echo "make classwork    - Run all classwork programs"
	@echo "make homework     - Run all homework programs"
	@echo "make both         - Run both classwork and homework"
	@echo "make test         - Test classwork programs"
	@echo "make test-homework- Test homework programs"
	@echo "make clean        - Remove Python cache files"
	@echo "make list         - List all files"
	@echo ""
	@echo "From classwork/: python3 01_hello_world.py"
	@echo "From homework/:  python3 01_hello_world.py"

install-cpu:
	@echo "Installing Phase 6: PyTorch (CPU-only) dependencies..."
	@echo "======================================================="
	@if [ ! -f "$(VENV_DIR)/bin/python" ]; then \
		echo "Error: Virtual environment not found at $(VENV_DIR)"; \
		echo "Run 'make setup' from the parent directory first"; \
		exit 1; \
	fi
	@if [ ! -f "requirements-base.txt" ]; then \
		echo "Error: requirements-base.txt not found"; \
		exit 1; \
	fi
	@echo "Installing PyTorch CPU version..."
	@echo "(Downloading ~200MB)"
	@$(PIP) install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
	@echo ""
	@echo "Installing other dependencies..."
	@$(PIP) install -q -r requirements-base.txt
	@echo ""
	@echo "✓ Phase 6 (CPU) dependencies installed successfully!"
	@echo ""
	@echo "Verifying installation:"
	@$(PYTHON) -c 'import torch; print(f"  PyTorch version: {torch.__version__}"); print(f"  CUDA available: {torch.cuda.is_available()}")'

install-gpu:
	@echo "Installing Phase 6: PyTorch (GPU) dependencies..."
	@echo "==================================================="
	@if [ ! -f "$(VENV_DIR)/bin/python" ]; then \
		echo "Error: Virtual environment not found at $(VENV_DIR)"; \
		echo "Run 'make setup' from the parent directory first"; \
		exit 1; \
	fi
	@if [ ! -f "requirements-base.txt" ]; then \
		echo "Error: requirements-base.txt not found"; \
		exit 1; \
	fi
	@echo ""
	@echo "Step 1: Detecting system CUDA version..."
	@echo "----------------------------------------"
	@if command -v nvidia-smi >/dev/null 2>&1; then \
		echo "  GPU detected:"; \
		nvidia-smi --query-gpu=name,driver_version --format=csv,noheader | head -1 | awk -F',' '{print "    " $$1 " (Driver: " $$2 ")"}'; \
	else \
		echo "  ⚠ nvidia-smi not found - GPU drivers may not be installed"; \
	fi
	@if command -v nvcc >/dev/null 2>&1; then \
		echo "  System CUDA: $$(nvcc --version | grep -oP 'release \K[0-9.]+')"; \
	else \
		echo "  ⚠ nvcc not found - CUDA toolkit not in PATH"; \
		echo "    (This is OK - PyTorch includes its own CUDA libraries)"; \
	fi
	@echo ""
	@echo "Step 2: Installing PyTorch with CUDA support..."
	@echo "----------------------------------------"
	@echo "  PyTorch will auto-detect and use the appropriate CUDA version"
	@echo "  Compatible with CUDA 11.8 and 12.1+"
	@echo "  Downloading ~2-3GB (includes CUDA runtime libraries)"
	@echo ""
	@$(PIP) install torch torchvision torchaudio
	@echo ""
	@echo "Step 3: Installing other dependencies..."
	@echo "----------------------------------------"
	@$(PIP) install -q -r requirements-base.txt
	@echo ""
	@echo "Step 4: Verifying installation..."
	@echo "----------------------------------------"
	@$(PYTHON) -c "import torch; \
print(f'  PyTorch version: {torch.__version__}'); \
print(f'  CUDA available: {torch.cuda.is_available()}'); \
print(f'  CUDA version (PyTorch): {torch.version.cuda if torch.cuda.is_available() else \"N/A\"}'); \
print(f'  cuDNN version: {torch.backends.cudnn.version() if torch.cuda.is_available() else \"N/A\"}'); \
print(f'  Number of GPUs: {torch.cuda.device_count() if torch.cuda.is_available() else 0}'); \
[print(f'  GPU {i}: {torch.cuda.get_device_name(i)} ({torch.cuda.get_device_properties(i).total_memory / 1024**3:.1f} GB)') for i in range(torch.cuda.device_count())] if torch.cuda.is_available() else None"
	@echo ""
	@if ! $(PYTHON) -c 'import torch; exit(0 if torch.cuda.is_available() else 1)' 2>/dev/null; then \
		echo "⚠ WARNING: PyTorch installed but CUDA not available!"; \
		echo ""; \
		echo "Possible reasons:"; \
		echo "  1. No NVIDIA GPU present in this machine"; \
		echo "  2. NVIDIA drivers not installed (run: nvidia-smi)"; \
		echo "  3. Driver version too old for PyTorch CUDA version"; \
		echo ""; \
		echo "Solutions:"; \
		echo "  • Update NVIDIA drivers from https://www.nvidia.com/drivers"; \
		echo "  • Or install CPU version: make install-cpu"; \
		echo ""; \
		echo "Note: Scripts will still work but will use CPU instead of GPU."; \
		exit 1; \
	else \
		echo "========================================"; \
		echo "✓ SUCCESS: CUDA is properly configured!"; \
		echo "========================================"; \
		echo "All PyTorch scripts will automatically use GPU acceleration."; \
	fi

install: install-cpu

check-cuda:
	@echo "============================================================"
	@echo "CUDA Availability Check"
	@echo "============================================================"
	@echo ""
	@echo "1. System CUDA Information:"
	@echo "----------------------------------------"
	@if command -v nvidia-smi >/dev/null 2>&1; then \
		nvidia-smi --query-gpu=name,memory.total,driver_version,compute_cap --format=csv,noheader,nounits | \
		awk -F',' '{print "   GPU: " $$1 "\n   Memory: " $$2 " MB\n   Driver: " $$3 "\n   Compute Capability: " $$4}'; \
	else \
		echo "   nvidia-smi not found"; \
		echo "   NVIDIA drivers may not be installed"; \
	fi
	@echo ""
	@if command -v nvcc >/dev/null 2>&1; then \
		echo "   CUDA Toolkit: $$(nvcc --version | grep -oP 'release \K[0-9.]+')"; \
	else \
		echo "   CUDA Toolkit: Not found (nvcc not in PATH)"; \
	fi
	@echo ""
	@echo "2. PyTorch CUDA Information:"
	@echo "----------------------------------------"
	@if [ -f "$(VENV_DIR)/bin/python" ]; then \
		$(PYTHON) -c "import torch; \
print(f'   PyTorch version: {torch.__version__}'); \
print(f'   CUDA available: {torch.cuda.is_available()}'); \
print(f'   CUDA version (PyTorch): {torch.version.cuda if torch.cuda.is_available() else \"N/A\"}'); \
print(f'   cuDNN enabled: {torch.backends.cudnn.enabled if torch.cuda.is_available() else False}'); \
print(f'   cuDNN version: {torch.backends.cudnn.version() if torch.cuda.is_available() else \"N/A\"}'); \
print(f'   Number of GPUs: {torch.cuda.device_count()}'); \
[print(f'   GPU {i}: {torch.cuda.get_device_name(i)} ({torch.cuda.get_device_properties(i).total_memory / 1024**3:.1f} GB)') for i in range(torch.cuda.device_count())]" 2>/dev/null || echo "   PyTorch not installed or import error"; \
	else \
		echo "   Virtual environment not found"; \
		echo "   Run 'make setup' first"; \
	fi
	@echo ""
	@echo "3. Recommendations:"
	@echo "----------------------------------------"
	@if ! $(PYTHON) -c 'import torch; exit(0 if torch.cuda.is_available() else 1)' 2>/dev/null; then \
		echo "   ⚠ CUDA not available in PyTorch"; \
		echo "   • For CPU-only: This is fine, scripts will use CPU"; \
		echo "   • For GPU: Install NVIDIA drivers and CUDA toolkit"; \
		echo "   • Then run: make install-gpu"; \
	else \
		echo "   ✓ CUDA is properly configured!"; \
		echo "   • All scripts will automatically use GPU"; \
		echo "   • Run 'python classwork/01-pytorch-basics/05_gpu_acceleration.py' to test"; \
	fi
	@echo ""

clean:
	@echo "Cleaning Python cache files..."
	find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -delete
	find . -type f -name "*.pyo" -delete
	@echo "Clean complete!"

classwork:
	@echo "Running classwork programs..."
	@echo "=============================="
	@for file in $(sort $(CLASSWORK_SAMPLES)); do \
		echo ""; \
		echo "Running $$file..."; \
		$(PYTHON) $$file || exit 1; \
	done
	@echo "Classwork complete!"

homework:
	@echo "Running homework programs..."
	@echo "=============================="
	@for file in $(sort $(HOMEWORK_SAMPLES)); do \
		echo ""; \
		echo "Running $$file..."; \
		$(PYTHON) $$file || exit 1; \
	done
	@echo "Homework complete!"

both: classwork homework

test: classwork
	@echo "Classwork tests passed!"

test-homework: homework
	@echo "Homework tests passed!"

list:
	@echo "Classwork files:"
	@for file in $(sort $(CLASSWORK_SAMPLES)); do echo "  - $$file"; done
	@if [ -n "$(CLASSWORK_PROJECTS)" ]; then \
		echo ""; \
		echo "Classwork projects:"; \
		for file in $(sort $(CLASSWORK_PROJECTS)); do echo "  - $$file"; done; \
	fi
	@echo ""
	@echo "Homework files:"
	@for file in $(sort $(HOMEWORK_SAMPLES)); do echo "  - $$file"; done
	@if [ -n "$(HOMEWORK_PROJECTS)" ]; then \
		echo ""; \
		echo "Homework projects:"; \
		for file in $(sort $(HOMEWORK_PROJECTS)); do echo "  - $$file"; done; \
	fi
