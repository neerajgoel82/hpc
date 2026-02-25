.PHONY: help clean classwork homework both test test-homework list

PYTHON := python3

# Directories
CLASSWORK_DIR := classwork
HOMEWORK_DIR := homework

# Find Python files
CLASSWORK_SAMPLES := $(wildcard $(CLASSWORK_DIR)/[0-9]*.py)
CLASSWORK_PROJECTS := $(wildcard $(CLASSWORK_DIR)/project_*.py)
HOMEWORK_SAMPLES := $(wildcard $(HOMEWORK_DIR)/[0-9]*.py)
HOMEWORK_PROJECTS := $(wildcard $(HOMEWORK_DIR)/project_*.py)

help:
	@echo "Phase 5: Data Science - Makefile Commands"
	@echo "================================================"
	@echo "make help         - Show this help message"
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
