.PHONY: help install clean run-module test venv check-deps list

PYTHON := python3
VENV := venv
VENV_BIN := $(VENV)/bin
PIP := $(VENV_BIN)/pip
PYTHON_VENV := $(VENV_BIN)/python

help:
	@echo "Phase 5: Data Science - Makefile Commands"
	@echo "=========================================="
	@echo "Setup:"
	@echo "  make venv         - Create virtual environment"
	@echo "  make install      - Install dependencies (creates venv if needed)"
	@echo "  make check-deps   - Check installed dependencies"
	@echo ""
	@echo "Running:"
	@echo "  make run-module MODULE=01-numpy    - Run all files in a module"
	@echo "  make run-starter                   - Run starter files (NumPy, Pandas)"
	@echo "  make run-projects                  - Run all project files"
	@echo ""
	@echo "Utilities:"
	@echo "  make list         - List all modules and files"
	@echo "  make clean        - Remove cache files"
	@echo "  make clean-all    - Remove cache and virtual environment"
	@echo "  make test         - Run starter files as test"
	@echo ""
	@echo "Examples:"
	@echo "  make install"
	@echo "  make run-module MODULE=01-numpy"
	@echo "  make run-starter"

venv:
	@if [ ! -d "$(VENV)" ]; then \
		echo "Creating virtual environment..."; \
		$(PYTHON) -m venv $(VENV); \
		echo "Virtual environment created!"; \
	else \
		echo "Virtual environment already exists."; \
	fi

install: venv
	@echo "Installing dependencies..."
	$(PIP) install --upgrade pip
	$(PIP) install -r requirements.txt
	@echo ""
	@echo "Installation complete!"
	@echo "Activate with: source $(VENV_BIN)/activate"

check-deps: venv
	@echo "Checking installed packages..."
	@$(PIP) list | grep -E "(numpy|pandas|matplotlib|seaborn|scikit-learn|scipy|statsmodels|plotly|jupyter)" || echo "Some packages may be missing. Run 'make install'"

run-starter: venv
	@echo "Running starter files..."
	@echo "========================"
	@echo ""
	@echo "Running NumPy basics..."
	@$(PYTHON_VENV) 01-numpy/01_arrays_basics.py
	@echo ""
	@echo "Running Pandas basics..."
	@$(PYTHON_VENV) 02-pandas/01_series_dataframes.py
	@echo ""
	@echo "Starter files completed!"

run-module: venv
	@if [ -z "$(MODULE)" ]; then \
		echo "Error: MODULE not specified"; \
		echo "Usage: make run-module MODULE=01-numpy"; \
		exit 1; \
	fi
	@if [ ! -d "$(MODULE)" ]; then \
		echo "Error: Module '$(MODULE)' not found"; \
		echo "Available modules: 01-numpy, 02-pandas, 03-visualization, 04-data-cleaning, 05-statistical-analysis, 06-ml-basics, 07-projects"; \
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
	@for file in 07-projects/*.py; do \
		echo ""; \
		echo "Running $$file..."; \
		echo "----------------------------------------"; \
		$(PYTHON_VENV) $$file || true; \
		echo ""; \
	done

list:
	@echo "Phase 5 Modules:"
	@echo "================"
	@for dir in 0*-*/; do \
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
	@echo "Running tests (starter files)..."
	@$(PYTHON_VENV) 01-numpy/01_arrays_basics.py > /dev/null && echo "✓ NumPy starter passed"
	@$(PYTHON_VENV) 02-pandas/01_series_dataframes.py > /dev/null && echo "✓ Pandas starter passed"
	@echo "All tests passed!"
