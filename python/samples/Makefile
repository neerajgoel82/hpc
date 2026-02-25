.PHONY: help clean clean-all install-phase5 install-phase6 test-all list-all

PYTHON := python3
PHASES := phase1-foundations phase2-intermediate phase3-oop phase4-advanced phase5-datascience phase6-pytorch

help:
	@echo "Python Samples - Master Makefile"
	@echo "================================="
	@echo ""
	@echo "Overview Commands:"
	@echo "  make help           - Show this help message"
	@echo "  make list-all       - List all phases and their contents"
	@echo "  make test-all       - Run tests in all phases"
	@echo "  make clean-all      - Clean all phases (cache files only)"
	@echo "  make deep-clean     - Clean all phases including virtual environments"
	@echo ""
	@echo "Phase-Specific Setup:"
	@echo "  make install-phase5 - Setup Phase 5 (Data Science) with dependencies"
	@echo "  make install-phase6-cpu - Setup Phase 6 (PyTorch CPU)"
	@echo "  make install-phase6-gpu - Setup Phase 6 (PyTorch GPU)"
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
	@echo "  cd phase5-datascience && make run-starter"
	@echo ""
	@echo "  make install-phase6-gpu"
	@echo "  cd phase6-pytorch && make check-cuda"
	@echo ""
	@echo "Available Phases:"
	@echo "  1. phase1-foundations     - Python basics"
	@echo "  2. phase2-intermediate    - Functions, file I/O, exceptions"
	@echo "  3. phase3-oop             - Object-oriented programming"
	@echo "  4. phase4-advanced        - Advanced Python features"
	@echo "  5. phase5-datascience     - NumPy, Pandas, ML (6-8 weeks)"
	@echo "  6. phase6-pytorch         - Deep Learning with PyTorch (8-10 weeks)"
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
	@if [ ! -d "phase5-datascience" ]; then \
		echo "Error: phase5-datascience directory not found"; \
		exit 1; \
	fi
	@cd phase5-datascience && $(MAKE) --no-print-directory install
	@echo ""
	@echo "Phase 5 setup complete!"
	@echo "Try: cd phase5-datascience && make run-starter"

install-phase6-cpu:
	@echo "Setting up Phase 6: PyTorch (CPU)"
	@echo "=================================="
	@if [ ! -d "phase6-pytorch" ]; then \
		echo "Error: phase6-pytorch directory not found"; \
		exit 1; \
	fi
	@cd phase6-pytorch && $(MAKE) --no-print-directory install-cpu
	@echo ""
	@echo "Phase 6 (CPU) setup complete!"
	@echo "Try: cd phase6-pytorch && make run-starter"

install-phase6-gpu:
	@echo "Setting up Phase 6: PyTorch (GPU)"
	@echo "=================================="
	@if [ ! -d "phase6-pytorch" ]; then \
		echo "Error: phase6-pytorch directory not found"; \
		exit 1; \
	fi
	@cd phase6-pytorch && $(MAKE) --no-print-directory install-gpu
	@echo ""
	@echo "Phase 6 (GPU) setup complete!"
	@echo "Try: cd phase6-pytorch && make check-cuda"

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

phase1:
	@cd phase1-foundations && $(MAKE) --no-print-directory help

phase2:
	@cd phase2-intermediate && $(MAKE) --no-print-directory help

phase3:
	@cd phase3-oop && $(MAKE) --no-print-directory help

phase4:
	@cd phase4-advanced && $(MAKE) --no-print-directory help

phase5:
	@cd phase5-datascience && $(MAKE) --no-print-directory help

phase6:
	@cd phase6-pytorch && $(MAKE) --no-print-directory help

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
	@echo "  Phase 1-4:  cd phase1-foundations && python3 01_hello_world.py"
	@echo "  Phase 5:    cd phase5-datascience && make run-module MODULE=01-numpy"
	@echo "  Phase 6:    cd phase6-pytorch && make run-module MODULE=01-pytorch-basics"
	@echo ""
	@echo "USEFUL COMMANDS:"
	@echo "  make stats       - Show file counts per phase"
	@echo "  make list-all    - List all files in all phases"
	@echo "  make test-all    - Run tests in all phases"
	@echo "  make clean-all   - Clean cache files"
	@echo ""
	@echo "For phase-specific help: cd <phase-dir> && make help"
