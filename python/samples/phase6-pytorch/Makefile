.PHONY: help install install-cpu install-gpu clean run-module test venv check-deps check-cuda list

PYTHON := python3
VENV := venv
VENV_BIN := $(VENV)/bin
PIP := $(VENV_BIN)/pip
PYTHON_VENV := $(VENV_BIN)/python

help:
	@echo "Phase 6: PyTorch & Deep Learning - Makefile Commands"
	@echo "====================================================="
	@echo "Setup:"
	@echo "  make venv         - Create virtual environment"
	@echo "  make install-cpu  - Install PyTorch (CPU version) + dependencies"
	@echo "  make install-gpu  - Install PyTorch (GPU/CUDA 11.8) + dependencies"
	@echo "  make install      - Alias for install-cpu"
	@echo "  make check-deps   - Check installed dependencies"
	@echo "  make check-cuda   - Check CUDA availability"
	@echo ""
	@echo "Running:"
	@echo "  make run-module MODULE=01-pytorch-basics  - Run all files in a module"
	@echo "  make run-starter                          - Run installation checker"
	@echo "  make run-projects                         - Run all project files"
	@echo ""
	@echo "Utilities:"
	@echo "  make list         - List all modules and files"
	@echo "  make clean        - Remove cache files"
	@echo "  make clean-all    - Remove cache and virtual environment"
	@echo "  make test         - Run installation checker as test"
	@echo ""
	@echo "Examples:"
	@echo "  make install-cpu"
	@echo "  make check-cuda"
	@echo "  make run-module MODULE=01-pytorch-basics"

venv:
	@if [ ! -d "$(VENV)" ]; then \
		echo "Creating virtual environment..."; \
		$(PYTHON) -m venv $(VENV); \
		echo "Virtual environment created!"; \
	else \
		echo "Virtual environment already exists."; \
	fi

install-cpu: venv
	@echo "Installing PyTorch (CPU version) and dependencies..."
	$(PIP) install --upgrade pip
	$(PIP) install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
	$(PIP) install -r requirements.txt
	@echo ""
	@echo "Installation complete!"
	@echo "Activate with: source $(VENV_BIN)/activate"
	@echo "Run 'make check-cuda' to verify installation"

install-gpu: venv
	@echo "Installing PyTorch (GPU/CUDA 11.8) and dependencies..."
	$(PIP) install --upgrade pip
	$(PIP) install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
	$(PIP) install -r requirements.txt
	@echo ""
	@echo "Installation complete!"
	@echo "Activate with: source $(VENV_BIN)/activate"
	@echo "Run 'make check-cuda' to verify GPU support"

install: install-cpu

check-deps: venv
	@echo "Checking installed packages..."
	@$(PIP) list | grep -E "(torch|torchvision|tensorboard|lightning|timm|transformers)" || echo "Some packages may be missing. Run 'make install-cpu' or 'make install-gpu'"

check-cuda: venv
	@echo "Checking PyTorch and CUDA availability..."
	@echo "=========================================="
	@$(PYTHON_VENV) -c "import torch; print(f'PyTorch version: {torch.__version__}'); print(f'CUDA available: {torch.cuda.is_available()}'); print(f'CUDA version: {torch.version.cuda if torch.cuda.is_available() else \"N/A\"}'); print(f'Number of GPUs: {torch.cuda.device_count()}' if torch.cuda.is_available() else 'Using CPU')"

run-starter: venv
	@echo "Running installation checker..."
	@echo "==============================="
	@$(PYTHON_VENV) 01-pytorch-basics/01_installation_setup.py

run-module: venv
	@if [ -z "$(MODULE)" ]; then \
		echo "Error: MODULE not specified"; \
		echo "Usage: make run-module MODULE=01-pytorch-basics"; \
		exit 1; \
	fi
	@if [ ! -d "$(MODULE)" ]; then \
		echo "Error: Module '$(MODULE)' not found"; \
		echo "Available modules:"; \
		echo "  01-pytorch-basics, 02-tensors-operations, 03-autograd-backprop"; \
		echo "  04-neural-networks, 05-training-optimization, 06-cnn-image-processing"; \
		echo "  07-rnn-sequences, 08-transfer-learning, 09-custom-models, 10-projects"; \
		exit 1; \
	fi
	@echo "Running all files in $(MODULE)..."
	@echo "=================================="
	@for file in $(MODULE)/*.py; do \
		echo ""; \
		echo "Running $$file..."; \
		echo "----------------------------------------"; \
		$(PYTHON_VENV) $$file || true; \
		echo ""; \
	done
	@echo "Module $(MODULE) completed!"

run-projects: venv
	@echo "Running project files..."
	@echo "========================"
	@for file in 10-projects/*.py; do \
		echo ""; \
		echo "Running $$file..."; \
		echo "----------------------------------------"; \
		$(PYTHON_VENV) $$file || true; \
		echo ""; \
	done

list:
	@echo "Phase 6 Modules:"
	@echo "================"
	@for dir in 0*-*/ 1*-*/; do \
		echo ""; \
		echo "$$dir"; \
		ls $$dir*.py 2>/dev/null | sed 's/^/  - /' || true; \
	done

clean:
	@echo "Cleaning Python cache files..."
	find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -delete
	find . -type f -name "*.pyo" -delete
	find . -type f -name "*.pyd" -delete
	find . -type f -name ".DS_Store" -delete
	find . -type d -name ".ipynb_checkpoints" -exec rm -rf {} + 2>/dev/null || true
	@echo "Clean complete!"

clean-all: clean
	@echo "Removing virtual environment..."
	rm -rf $(VENV)
	@echo "Full clean complete!"

test: venv
	@echo "Running tests (installation checker)..."
	@$(PYTHON_VENV) 01-pytorch-basics/01_installation_setup.py > /dev/null && echo "✓ PyTorch installation check passed"
	@echo "All tests passed!"
