.PHONY: help clean run-all test

PYTHON := python3
SAMPLES := $(wildcard [0-9]*.py)
PROJECTS := $(wildcard project_*.py)

help:
	@echo "Phase 1: Python Foundations - Makefile Commands"
	@echo "================================================"
	@echo "make help        - Show this help message"
	@echo "make clean       - Remove Python cache files"
	@echo "make run-all     - Run all sample files in sequence"
	@echo "make run-projects - Run all project files"
	@echo "make list        - List all sample files"
	@echo ""
	@echo "Run individual files with: python3 01_hello_world.py"

clean:
	@echo "Cleaning Python cache files..."
	find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -delete
	find . -type f -name "*.pyo" -delete
	find . -type f -name "*.pyd" -delete
	find . -type f -name ".DS_Store" -delete
	@echo "Clean complete!"

run-all:
	@echo "Running all sample files in sequence..."
	@echo "========================================"
	@for file in $(sort $(SAMPLES)); do \
		echo ""; \
		echo "Running $$file..."; \
		echo "----------------------------------------"; \
		$(PYTHON) $$file || exit 1; \
		echo ""; \
	done
	@echo "All samples completed!"

run-projects:
	@echo "Running all project files..."
	@echo "============================"
	@for file in $(sort $(PROJECTS)); do \
		echo ""; \
		echo "Running $$file..."; \
		echo "----------------------------------------"; \
		$(PYTHON) $$file || exit 1; \
		echo ""; \
	done

list:
	@echo "Sample files:"
	@for file in $(sort $(SAMPLES)); do echo "  - $$file"; done
	@echo ""
	@echo "Project files:"
	@for file in $(sort $(PROJECTS)); do echo "  - $$file"; done

test: run-all
	@echo "All tests passed!"
