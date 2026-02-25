.PHONY: help clean run-all test

PYTHON := python3
SAMPLES := $(wildcard [0-9]*.py)

help:
	@echo "Phase 4: Python Advanced - Makefile Commands"
	@echo "============================================"
	@echo "make help        - Show this help message"
	@echo "make clean       - Remove Python cache files"
	@echo "make run-all     - Run all sample files in sequence"
	@echo "make list        - List all sample files"
	@echo ""
	@echo "Run individual files with: python3 43_generators.py"

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

list:
	@echo "Sample files:"
	@for file in $(sort $(SAMPLES)); do echo "  - $$file"; done

test: run-all
	@echo "All tests passed!"
