# C Samples - Root Makefile
# Builds all samples across all phases

# Phases
PHASES = phase1-foundations phase2-building-blocks phase3-core-concepts phase4-advanced phase5-projects

# Default target: build all classwork
all:
	@echo "=== Building All C Classwork ==="
	@for phase in $(PHASES); do \
		echo "\n--- Building $$phase classwork ---"; \
		$(MAKE) -C $$phase classwork || exit 1; \
	done
	@echo "\n=== Build Complete! ==="

# Build all homework
homework:
	@echo "=== Building All C Homework ==="
	@for phase in $(PHASES); do \
		echo "\n--- Building $$phase homework ---"; \
		$(MAKE) -C $$phase homework || exit 1; \
	done
	@echo "\n=== Homework Build Complete! ==="

# Build both classwork and homework
both:
	@echo "=== Building All Classwork and Homework ==="
	@for phase in $(PHASES); do \
		echo "\n--- Building $$phase ---"; \
		$(MAKE) -C $$phase both || exit 1; \
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
	@echo "C Learning - Root Makefile"
	@echo ""
	@echo "Available targets:"
	@echo "  make              - Build all classwork in all phases"
	@echo "  make all          - Build all classwork in all phases"
	@echo "  make homework     - Build all homework in all phases"
	@echo "  make both         - Build both classwork and homework"
	@echo "  make phase1       - Build Phase 1 classwork"
	@echo "  make phase2       - Build Phase 2 classwork"
	@echo "  make phase3       - Build Phase 3 classwork"
	@echo "  make phase4       - Build Phase 4 classwork"
	@echo "  make phase5       - Build Phase 5 classwork"
	@echo "  make test         - Build and test all classwork"
	@echo "  make memcheck     - Run valgrind on phases with dynamic memory"
	@echo "  make clean        - Remove all executables from all phases"
	@echo "  make stats        - Show code statistics"
	@echo "  make list         - List all programs"
	@echo "  make help         - Show this help message"
	@echo ""
	@echo "Structure:"
	@echo "  Each phase contains:"
	@echo "    classwork/  - Learning examples with concepts explained"
	@echo "    homework/   - Exercise files for practice"
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

.PHONY: all homework both phase1 phase2 phase3 phase4 phase5 clean test memcheck stats list help
