# Classwork - run Python programs in this directory
.PHONY: help run run-all list clean

PYTHON := python3

# Current directory only (flat layout)
run:
	@echo "Running classwork programs (this directory)..."
	@for f in $$(find . -maxdepth 1 -name '*.py' -type f | sort); do \
		echo ""; echo ">>> $$f"; $(PYTHON) $$f || true; \
	done

# Recursive (for phases with subdirs)
run-all:
	@echo "Running all classwork programs (this directory and subdirs)..."
	@for f in $$(find . -name '*.py' -type f ! -path './__pycache__/*' | sort); do \
		echo ""; echo ">>> $$f"; $(PYTHON) $$f || true; \
	done

list:
	@echo "Classwork Python files:"
	@find . -maxdepth 1 -name '*.py' -type f | sort | sed 's|^\./||' | sed 's/^/  /'

list-all:
	@echo "Classwork Python files (including subdirs):"
	@find . -name '*.py' -type f ! -path './__pycache__/*' | sort | sed 's|^\./||' | sed 's/^/  /'

clean:
	@find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
	@find . -type f -name "*.pyc" -delete 2>/dev/null || true
	@echo "Clean complete."

help:
	@echo "Classwork - Makefile"
	@echo "  make run      - Run all .py in this directory"
	@echo "  make run-all  - Run all .py (including subdirs)"
	@echo "  make list     - List .py files here"
	@echo "  make list-all - List all .py (including subdirs)"
	@echo "  make clean    - Remove __pycache__ and .pyc"
	@echo "  make help     - This message"
	@echo ""
	@echo "Run a single file: python3 01_hello_world.py"
