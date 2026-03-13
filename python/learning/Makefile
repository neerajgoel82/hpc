.PHONY: help clean clean-all setup install-phase5 install-phase6 test test-syntax test-all list-all check-cuda clean-outputs

VENV_DIR := .venv
PYTHON := $(shell if [ -f $(VENV_DIR)/bin/python ]; then echo $(VENV_DIR)/bin/python; else echo python3; fi)
PHASES := 01-foundations 02-intermediate 03-oop 04-advanced 05-datascience 06-pytorch

help:
	@echo "Python Samples - Master Makefile"
	@echo "================================="
	@echo ""
	@echo "Overview Commands:"
	@echo "  make help           - Show this help message"
	@echo "  make setup          - Create virtual environment and install requirements"
	@echo "  make test           - Execute all Python files to verify they run"
	@echo "  make test-syntax    - Quick syntax check without execution"
	@echo "  make test-all       - Run tests in all phases"
	@echo "  make classwork-all  - Run all classwork in all phases"
	@echo "  make homework-all   - Run all homework in all phases"
	@echo "  make both-all       - Run both classwork and homework"
	@echo "  make list-all       - List all phases and their contents"
	@echo "  make clean-all      - Clean all phases (cache files only)"
	@echo "  make clean-outputs  - Remove generated files (models, images, data)"
	@echo "  make deep-clean     - Clean all phases including virtual environments"
	@echo ""
	@echo "Structure:"
	@echo "  Each phase contains:"
	@echo "    classwork/  - Learning examples with concepts"
	@echo "    homework/   - Exercise files for practice"
	@echo ""
	@echo "Phase-Specific Setup:"
	@echo "  make install-phase5     - Setup Phase 5 (Data Science) with dependencies"
	@echo "  make install-phase6-cpu - Setup Phase 6 (PyTorch CPU)"
	@echo "  make install-phase6-gpu - Setup Phase 6 (PyTorch GPU with CUDA)"
	@echo "  make check-cuda         - Check CUDA availability for GPU training"
	@echo ""
	@echo "Phase-Specific Commands:"
	@echo "  make phase1         - Run Phase 1 help"
	@echo "  make phase2         - Run Phase 2 help"
	@echo "  make phase3         - Run Phase 3 help"
	@echo "  make phase4         - Run Phase 4 help"
	@echo "  make phase5         - Run Phase 5 help"
	@echo "  make phase6         - Run Phase 6 help"
	@echo ""
	@echo "Quick Start Examples:"
	@echo "  make install-phase5"
	@echo "  cd 05-datascience && make run-starter"
	@echo ""
	@echo "  make install-phase6-gpu"
	@echo "  cd 06-pytorch && make check-cuda"
	@echo ""
	@echo "Available Phases:"
	@echo "  1. 01-foundations     - Python basics"
	@echo "  2. 02-intermediate    - Functions, file I/O, exceptions"
	@echo "  3. 03-oop             - Object-oriented programming"
	@echo "  4. 04-advanced        - Advanced Python features"
	@echo "  5. 05-datascience     - NumPy, Pandas, ML (6-8 weeks)"
	@echo "  6. 06-pytorch         - Deep Learning with PyTorch (8-10 weeks)"
	@echo ""
	@echo "For detailed phase commands, cd into the phase directory and run 'make help'"

list-all:
	@echo "Python Learning Phases"
	@echo "======================"
	@echo ""
	@for phase in $(PHASES); do \
		if [ -d "$$phase" ]; then \
			echo "$$phase:"; \
			cd $$phase && $(MAKE) --no-print-directory list 2>/dev/null || ls *.py 2>/dev/null | sed 's/^/  - /' || echo "  (no files)"; \
			cd ..; \
			echo ""; \
		fi; \
	done

setup:
	@echo "Setting up Python learning environment..."
	@echo "=========================================="
	@if [ ! -f "00-setup/setup.sh" ]; then \
		echo "Error: setup script not found at 00-setup/setup.sh"; \
		exit 1; \
	fi
	@./00-setup/setup.sh
	@echo ""
	@echo "Setup complete! Virtual environment created at .venv/"
	@echo "To use it: source .venv/bin/activate"

clean:
	@echo "Cleaning all phases (cache files)..."
	@echo "====================================="
	@for phase in $(PHASES); do \
		if [ -d "$$phase" ]; then \
			echo "Cleaning $$phase..."; \
			cd $$phase && $(MAKE) --no-print-directory clean || true; \
			cd ..; \
		fi; \
	done
	@echo ""
	@echo "All phases cleaned!"

clean-all: clean

deep-clean:
	@echo "Deep cleaning all phases (including venv)..."
	@echo "============================================="
	@for phase in $(PHASES); do \
		if [ -d "$$phase" ]; then \
			echo "Deep cleaning $$phase..."; \
			cd $$phase && $(MAKE) --no-print-directory clean-all 2>/dev/null || $(MAKE) --no-print-directory clean || true; \
			cd ..; \
		fi; \
	done
	@echo ""
	@echo "Deep clean complete!"

install-phase5:
	@echo "Setting up Phase 5: Data Science"
	@echo "================================="
	@if [ ! -d "05-datascience" ]; then \
		echo "Error: 05-datascience directory not found"; \
		exit 1; \
	fi
	@cd 05-datascience && $(MAKE) --no-print-directory install
	@echo ""
	@echo "Phase 5 setup complete!"
	@echo "Try: cd 05-datascience && make run-starter"

install-phase6-cpu:
	@echo "Setting up Phase 6: PyTorch (CPU)"
	@echo "=================================="
	@if [ ! -d "06-pytorch" ]; then \
		echo "Error: 06-pytorch directory not found"; \
		exit 1; \
	fi
	@cd 06-pytorch && $(MAKE) --no-print-directory install-cpu
	@echo ""
	@echo "Phase 6 (CPU) setup complete!"
	@echo "Try: cd 06-pytorch && make run-starter"

install-phase6-gpu:
	@echo "Setting up Phase 6: PyTorch (GPU)"
	@echo "=================================="
	@if [ ! -d "06-pytorch" ]; then \
		echo "Error: 06-pytorch directory not found"; \
		exit 1; \
	fi
	@cd 06-pytorch && $(MAKE) --no-print-directory install-gpu
	@echo ""
	@echo "Phase 6 (GPU) setup complete!"
	@echo "Try: cd 06-pytorch && make check-cuda"

check-cuda:
	@echo "Checking CUDA availability..."
	@echo "=============================="
	@if [ ! -d "06-pytorch" ]; then \
		echo "Error: 06-pytorch directory not found"; \
		exit 1; \
	fi
	@cd 06-pytorch && $(MAKE) --no-print-directory check-cuda

clean-outputs:
	@echo "Cleaning generated output files..."
	@echo "==================================="
	@echo ""
	@echo "Removing:"
	@echo "  - Image files (*.png, *.jpg, *.jpeg, *.gif)"
	@echo "  - Model files (*.pth, *.pt, *.ckpt)"
	@echo "  - Pickle files (*.pkl)"
	@echo "  - Directories: data/, runs/, checkpoints/, outputs/"
	@echo ""
	@read -p "Continue? [y/N] " -n 1 -r; \
	echo; \
	if [[ $$REPLY =~ ^[Yy]$$ ]]; then \
		echo "Cleaning..."; \
		find . -maxdepth 1 -name "*.png" -delete; \
		find . -maxdepth 1 -name "*.jpg" -delete; \
		find . -maxdepth 1 -name "*.jpeg" -delete; \
		find . -maxdepth 1 -name "*.gif" -delete; \
		find . -maxdepth 1 -name "*.pth" -delete; \
		find . -maxdepth 1 -name "*.pt" -delete; \
		find . -maxdepth 1 -name "*.ckpt" -delete; \
		find . -maxdepth 1 -name "*.pkl" -delete; \
		rm -rf data/ runs/ checkpoints/ outputs/ checkpoints_demo/; \
		echo ""; \
		echo "✓ Output files cleaned!"; \
		echo ""; \
		echo "Note: Scripts will auto-download datasets when needed"; \
	else \
		echo "Cancelled."; \
	fi

test-all:
	@echo "Running tests in all phases..."
	@echo "=============================="
	@echo ""
	@for phase in $(PHASES); do \
		if [ -d "$$phase" ]; then \
			echo "Testing $$phase..."; \
			cd $$phase && $(MAKE) --no-print-directory test 2>/dev/null || echo "  (no tests or test failed)"; \
			cd ..; \
			echo ""; \
		fi; \
	done
	@echo "All tests completed!"

test:
	@echo "Executing all Python files..."
	@echo "======================================"
	@echo ""
	@echo "Note: Files requiring input or missing dependencies may fail"
	@echo ""
	@failed=0; \
	total=0; \
	skipped=0; \
	timeout_cmd=""; \
	if command -v timeout >/dev/null 2>&1; then \
		timeout_cmd="timeout 5"; \
	elif command -v gtimeout >/dev/null 2>&1; then \
		timeout_cmd="gtimeout 5"; \
	fi; \
	for phase in $(PHASES); do \
		if [ -d "$$phase" ]; then \
			echo "Testing $$phase..."; \
			if [ -d "$$phase/classwork" ]; then \
				for file in $$(find "$$phase/classwork" -name "*.py" -type f 2>/dev/null); do \
					total=$$((total + 1)); \
					if [ -n "$$timeout_cmd" ]; then \
						if $$timeout_cmd $(PYTHON) "$$file" </dev/null >/dev/null 2>&1; then \
							echo "  ✓ $$file"; \
						elif [ $$? -eq 124 ] || [ $$? -eq 143 ]; then \
							echo "  ⏱ $$file (timeout)"; \
							skipped=$$((skipped + 1)); \
						else \
							echo "  ✗ $$file (failed)"; \
							failed=$$((failed + 1)); \
						fi; \
					else \
						if $(PYTHON) -c "import sys; sys.stdin = open('/dev/null'); exec(open('$$file').read())" >/dev/null 2>&1; then \
							echo "  ✓ $$file"; \
						else \
							echo "  ⚠ $$file (needs input/deps?)"; \
							skipped=$$((skipped + 1)); \
						fi; \
					fi; \
				done; \
			fi; \
			if [ -d "$$phase/homework" ]; then \
				for file in $$(find "$$phase/homework" -name "*.py" -type f 2>/dev/null); do \
					total=$$((total + 1)); \
					if [ -n "$$timeout_cmd" ]; then \
						if $$timeout_cmd $(PYTHON) "$$file" </dev/null >/dev/null 2>&1; then \
							echo "  ✓ $$file"; \
						elif [ $$? -eq 124 ] || [ $$? -eq 143 ]; then \
							echo "  ⏱ $$file (timeout)"; \
							skipped=$$((skipped + 1)); \
						else \
							echo "  ✗ $$file (failed)"; \
							failed=$$((failed + 1)); \
						fi; \
					else \
						if $(PYTHON) -c "import sys; sys.stdin = open('/dev/null'); exec(open('$$file').read())" >/dev/null 2>&1; then \
							echo "  ✓ $$file"; \
						else \
							echo "  ⚠ $$file (needs input/deps?)"; \
							skipped=$$((skipped + 1)); \
						fi; \
					fi; \
				done; \
			fi; \
			echo ""; \
		fi; \
	done; \
	echo "======================================"; \
	echo "Total files tested: $$total"; \
	echo "Passed: $$((total - failed - skipped))"; \
	echo "Skipped/Warnings: $$skipped"; \
	echo "Failed: $$failed"; \
	if [ $$failed -gt 0 ]; then \
		echo ""; \
		echo "⚠ Some files failed to execute!"; \
		echo "Run individual files to see error details."; \
		exit 1; \
	else \
		echo ""; \
		echo "✓ All testable Python files executed!"; \
		if [ $$skipped -gt 0 ]; then \
			echo "  ($$skipped files skipped - may need input or dependencies)"; \
		fi; \
	fi

test-syntax:
	@echo "Checking Python syntax for all files..."
	@echo "======================================"
	@echo ""
	@failed=0; \
	total=0; \
	for phase in $(PHASES); do \
		if [ -d "$$phase" ]; then \
			echo "Checking $$phase..."; \
			if [ -d "$$phase/classwork" ]; then \
				for file in $$(find "$$phase/classwork" -name "*.py" -type f 2>/dev/null); do \
					total=$$((total + 1)); \
					if $(PYTHON) -m py_compile "$$file" 2>/dev/null; then \
						echo "  ✓ $$file"; \
					else \
						echo "  ✗ $$file - syntax error"; \
						failed=$$((failed + 1)); \
					fi; \
				done; \
			fi; \
			if [ -d "$$phase/homework" ]; then \
				for file in $$(find "$$phase/homework" -name "*.py" -type f 2>/dev/null); do \
					total=$$((total + 1)); \
					if $(PYTHON) -m py_compile "$$file" 2>/dev/null; then \
						echo "  ✓ $$file"; \
					else \
						echo "  ✗ $$file - syntax error"; \
						failed=$$((failed + 1)); \
					fi; \
				done; \
			fi; \
			echo ""; \
		fi; \
	done; \
	echo "======================================"; \
	echo "Total files checked: $$total"; \
	echo "Passed: $$((total - failed))"; \
	echo "Failed: $$failed"; \
	if [ $$failed -gt 0 ]; then \
		echo ""; \
		echo "Some files have syntax errors!"; \
		exit 1; \
	else \
		echo ""; \
		echo "All Python files have valid syntax! ✓"; \
	fi

phase1:
	@cd 01-foundations && $(MAKE) --no-print-directory help

phase2:
	@cd 02-intermediate && $(MAKE) --no-print-directory help

phase3:
	@cd 03-oop && $(MAKE) --no-print-directory help

phase4:
	@cd 04-advanced && $(MAKE) --no-print-directory help

phase5:
	@cd 05-datascience && $(MAKE) --no-print-directory help

phase6:
	@cd 06-pytorch && $(MAKE) --no-print-directory help

# Statistics target
stats:
	@echo "Python Samples Statistics"
	@echo "========================="
	@echo ""
	@total_files=0; \
	total_phases=0; \
	for phase in $(PHASES); do \
		if [ -d "$$phase" ]; then \
			total_phases=$$((total_phases + 1)); \
			count=$$(find $$phase -name "*.py" -type f | wc -l | tr -d ' '); \
			total_files=$$((total_files + count)); \
			printf "%-25s %3d files\n" "$$phase:" "$$count"; \
		fi; \
	done; \
	echo ""; \
	echo "Total phases: $$total_phases"; \
	echo "Total Python files: $$total_files"

# Quick reference
quick-ref:
	@echo "Quick Reference Guide"
	@echo "===================="
	@echo ""
	@echo "FIRST TIME SETUP:"
	@echo "  For Phase 1-4: No setup needed, just run files"
	@echo "  For Phase 5:   make install-phase5"
	@echo "  For Phase 6:   make install-phase6-cpu (or -gpu)"
	@echo ""
	@echo "RUNNING SAMPLES:"
	@echo "  Phase 1-4:  cd 01-foundations && python3 01_hello_world.py"
	@echo "  Phase 5:    cd 05-datascience && make run-module MODULE=01-numpy"
	@echo "  Phase 6:    cd 06-pytorch && make run-module MODULE=01-pytorch-basics"
	@echo ""
	@echo "USEFUL COMMANDS:"
	@echo "  make test        - Execute all Python files"
	@echo "  make test-syntax - Quick syntax check (no execution)"
	@echo "  make test-all    - Run tests in all phases"
	@echo "  make stats       - Show file counts per phase"
	@echo "  make list-all    - List all files in all phases"
	@echo "  make clean-all   - Clean cache files"
	@echo ""
	@echo "For phase-specific help: cd <phase-dir> && make help"
