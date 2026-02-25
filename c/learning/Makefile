# C Samples - Root Makefile
# Builds all samples across all phases

# Phases
PHASES = phase1-foundations phase2-building-blocks phase3-core-concepts phase4-advanced phase5-projects

# Default target: build all phases
all:
	@echo "=== Building All C Samples ==="
	@for phase in $(PHASES); do \
		echo "\n--- Building $$phase ---"; \
		$(MAKE) -C $$phase all || exit 1; \
	done
	@echo "\n=== Build Complete! ==="

# Build specific phase
phase1:
	@echo "=== Building Phase 1: Foundations ==="
	$(MAKE) -C phase1-foundations

phase2:
	@echo "=== Building Phase 2: Building Blocks ==="
	$(MAKE) -C phase2-building-blocks

phase3:
	@echo "=== Building Phase 3: Core Concepts ==="
	$(MAKE) -C phase3-core-concepts

phase4:
	@echo "=== Building Phase 4: Advanced Topics ==="
	$(MAKE) -C phase4-advanced

phase5:
	@echo "=== Building Phase 5: Real-World Projects ==="
	$(MAKE) -C phase5-projects

# Clean all phases
clean:
	@echo "=== Cleaning All C Samples ==="
	@for phase in $(PHASES); do \
		echo "Cleaning $$phase..."; \
		$(MAKE) -C $$phase clean; \
	done
	-find . -name '*.dSYM' -type d -exec rm -rf {} +
	@echo "=== Clean Complete! ==="

# Test all phases
test:
	@echo "=== Testing All C Samples ==="
	@for phase in $(PHASES); do \
		echo "\n=== Testing $$phase ==="; \
		$(MAKE) -C $$phase test || echo "$$phase tests failed"; \
	done
	@echo "\n=== All Tests Complete! ==="

# Memory check on programs with dynamic memory
memcheck:
	@echo "=== Memory Check on All Phases ==="
	@echo "\n--- Phase 3: Core Concepts ---"
	@$(MAKE) -C phase3-core-concepts memcheck || true
	@echo "\n--- Phase 4: Advanced Topics ---"
	@$(MAKE) -C phase4-advanced memcheck || true
	@echo "\n--- Phase 5: Projects ---"
	@$(MAKE) -C phase5-projects memcheck || true
	@echo "\n=== Memory Check Complete! ==="

# Count lines of code
stats:
	@echo "=== C Samples Statistics ==="
	@echo "\nTotal .c files:"
	@find . -name "*.c" | wc -l
	@echo "\nTotal lines of C code:"
	@find . -name "*.c" -exec wc -l {} + | tail -1
	@echo "\nPrograms by phase:"
	@for phase in $(PHASES); do \
		count=$$(find $$phase -name "*.c" 2>/dev/null | wc -l); \
		echo "  $$phase: $$count programs"; \
	done

# List all programs
list:
	@echo "=== All C Sample Programs ==="
	@for phase in $(PHASES); do \
		echo "\n$$phase:"; \
		find $$phase -name "*.c" -exec basename {} \; 2>/dev/null | sort; \
	done

# Help target
help:
	@echo "C Samples - Root Makefile"
	@echo ""
	@echo "Available targets:"
	@echo "  make              - Build all samples in all phases"
	@echo "  make all          - Build all samples in all phases"
	@echo "  make phase1       - Build only Phase 1 samples"
	@echo "  make phase2       - Build only Phase 2 samples"
	@echo "  make phase3       - Build only Phase 3 samples"
	@echo "  make phase4       - Build only Phase 4 samples"
	@echo "  make phase5       - Build only Phase 5 samples"
	@echo "  make test         - Build and test all phases"
	@echo "  make memcheck     - Run valgrind on phases with dynamic memory"
	@echo "  make clean        - Remove all executables from all phases"
	@echo "  make stats        - Show code statistics"
	@echo "  make list         - List all programs"
	@echo "  make help         - Show this help message"
	@echo ""
	@echo "Phases:"
	@echo "  Phase 1: Foundations (Weeks 1-3)"
	@echo "  Phase 2: Building Blocks (Weeks 4-6)"
	@echo "  Phase 3: Core Concepts (Weeks 7-10)"
	@echo "  Phase 4: Advanced Topics (Weeks 11-14)"
	@echo "  Phase 5: Real-World Projects (Weeks 15-16+)"
	@echo ""
	@echo "For phase-specific help, run: make -C <phase> help"
	@echo "Example: make -C phase1-foundations help"

.PHONY: all phase1 phase2 phase3 phase4 phase5 clean test memcheck stats list help
