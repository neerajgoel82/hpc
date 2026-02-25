# C++ Samples - Root Makefile
# Builds all samples across all modules

# Modules (ordered by learning progression)
MODULES = 01-basics \
          02-functions-structure \
          03-pointers-memory \
          04-classes-oop \
          05-inheritance-polymorphism \
          06-operators-advanced \
          07-templates \
          08-stl \
          09-modern-cpp \
          10-exceptions \
          11-multithreading \
          12-build-debug \
          13-gpu-advanced \
          14-gpu-prep

# Default target: build all classwork
all:
	@echo "=== Building All C++ Classwork ==="
	@for module in $(MODULES); do \
		echo "\n--- Building Module $$module classwork ---"; \
		$(MAKE) -C $$module classwork || exit 1; \
	done
	@echo "\n=== Build Complete! ==="

# Build all homework
homework:
	@echo "=== Building All C++ Homework ==="
	@for module in $(MODULES); do \
		echo "\n--- Building Module $$module homework ---"; \
		$(MAKE) -C $$module homework || exit 1; \
	done
	@echo "\n=== Homework Build Complete! ==="

# Build both classwork and homework
both:
	@echo "=== Building All Classwork and Homework ==="
	@for module in $(MODULES); do \
		echo "\n--- Building Module $$module ---"; \
		$(MAKE) -C $$module both || exit 1; \
	done
	@echo "\n=== Build Complete! ==="

# Build specific module
module1:
	@echo "=== Building Module 01: C++ Basics ==="
	$(MAKE) -C 01-basics

module2:
	@echo "=== Building Module 02: Functions and Structure ==="
	$(MAKE) -C 02-functions-structure

module3:
	@echo "=== Building Module 03: Pointers and Memory ==="
	$(MAKE) -C 03-pointers-memory

module4:
	@echo "=== Building Module 04: Classes and OOP ==="
	$(MAKE) -C 04-classes-oop

module5:
	@echo "=== Building Module 05: Inheritance and Polymorphism ==="
	$(MAKE) -C 05-inheritance-polymorphism

module6:
	@echo "=== Building Module 06: Operator Overloading ==="
	$(MAKE) -C 06-operators-advanced

module7:
	@echo "=== Building Module 07: Templates ==="
	$(MAKE) -C 07-templates

module8:
	@echo "=== Building Module 08: STL ==="
	$(MAKE) -C 08-stl

module9:
	@echo "=== Building Module 09: Modern C++ ==="
	$(MAKE) -C 09-modern-cpp

module10:
	@echo "=== Building Module 10: Exception Handling ==="
	$(MAKE) -C 10-exceptions

module11:
	@echo "=== Building Module 11: Multithreading ==="
	$(MAKE) -C 11-multithreading

module12:
	@echo "=== Building Module 12: Build Systems and Debugging ==="
	$(MAKE) -C 12-build-debug

module13:
	@echo "=== Building Module 13: Advanced GPU Topics ==="
	$(MAKE) -C 13-gpu-advanced

module14:
	@echo "=== Building Module 14: GPU Preparation ==="
	$(MAKE) -C 14-gpu-prep

# Build by category
fundamentals: module1 module2 module3
	@echo "=== Fundamentals modules built ==="

oop: module4 module5 module6
	@echo "=== OOP modules built ==="

advanced: module7 module8 module9 module10
	@echo "=== Advanced modules built ==="

parallel: module11 module12 module13 module14
	@echo "=== Parallel/GPU modules built ==="

# Clean all modules
clean:
	@echo "=== Cleaning All C++ Samples ==="
	@for module in $(MODULES); do \
		echo "Cleaning $$module..."; \
		$(MAKE) -C $$module clean 2>/dev/null || true; \
	done
	-find . -name '*.dSYM' -type d -exec rm -rf {} +
	@echo "=== Clean Complete! ==="

# Test all modules (build classwork and run each program)
test:
	@echo "=== Testing All C++ Samples ==="
	@for module in $(MODULES); do \
		echo "\n=== Testing $$module ==="; \
		$(MAKE) -C $$module test || exit 1; \
	done
	@echo "\n=== All Tests Complete! ==="

# Run all executables (be careful - might have interactive ones)
run:
	@echo "=== Running All C++ Samples ==="
	@echo "Warning: This will run all executables. Some may be interactive."
	@read -p "Press Enter to continue or Ctrl-C to cancel..."
	@for module in $(MODULES); do \
		echo "\n=== Running $$module ==="; \
		$(MAKE) -C $$module run 2>/dev/null || echo "No run target for $$module"; \
	done

# Count lines of code
stats:
	@echo "=== C++ Samples Statistics ==="
	@echo "\nTotal .cpp files:"
	@find . -name "*.cpp" -not -path "./build/*" | wc -l
	@echo "\nTotal .h/.hpp files:"
	@find . \( -name "*.h" -o -name "*.hpp" \) -not -path "./build/*" | wc -l
	@echo "\nTotal lines of C++ code:"
	@find . \( -name "*.cpp" -o -name "*.h" -o -name "*.hpp" \) -not -path "./build/*" -exec wc -l {} + | tail -1
	@echo "\nPrograms by module:"
	@for module in $(MODULES); do \
		count=$$(find $$module -name "*.cpp" 2>/dev/null | wc -l); \
		echo "  $$module: $$count programs"; \
	done
	@echo "\nTotal modules: $$(echo $(MODULES) | wc -w)"

# List all programs
list:
	@echo "=== All C++ Sample Programs ==="
	@for module in $(MODULES); do \
		echo "\n$$module:"; \
		find $$module -name "*.cpp" -exec basename {} \; 2>/dev/null | sort; \
	done

# Check compilation of all modules
check:
	@echo "=== Checking Compilation of All Modules ==="
	@failed=0; \
	for module in $(MODULES); do \
		echo -n "Checking $$module... "; \
		if $(MAKE) -C $$module all > /dev/null 2>&1; then \
			echo "✓ OK"; \
		else \
			echo "✗ FAILED"; \
			failed=$$((failed + 1)); \
		fi; \
	done; \
	echo "\n=== Check Complete ==="; \
	if [ $$failed -eq 0 ]; then \
		echo "All modules compile successfully!"; \
	else \
		echo "$$failed module(s) failed to compile"; \
		exit 1; \
	fi

# Help target
help:
	@echo "C++ Samples - Root Makefile"
	@echo ""
	@echo "Available targets:"
	@echo "  make              - Build all classwork in all modules"
	@echo "  make all          - Build all classwork in all modules"
	@echo "  make homework     - Build all homework in all modules"
	@echo "  make both         - Build both classwork and homework"
	@echo "  make module<N>    - Build specific module (e.g., make module1)"
	@echo "  make fundamentals - Build Modules 1-3 (basics)"
	@echo "  make oop          - Build Modules 4-6 (object-oriented)"
	@echo "  make advanced     - Build Modules 7-10 (advanced features)"
	@echo "  make parallel     - Build Modules 11-14 (parallel/GPU prep)"
	@echo "  make test         - Run tests for all classwork"
	@echo "  make check        - Check if all modules compile"
	@echo "  make clean        - Remove all executables from all modules"
	@echo "  make stats        - Show code statistics"
	@echo "  make list         - List all programs"
	@echo "  make help         - Show this help message"
	@echo ""
	@echo "Structure:"
	@echo "  Each module contains:"
	@echo "    classwork/  - Learning examples with concepts explained"
	@echo "    homework/   - Exercise files for practice"
	@echo ""
	@echo "Modules:"
	@echo "  Module 01: C++ Fundamentals"
	@echo "  Module 02: Functions and Program Structure"
	@echo "  Module 03: Pointers and Memory Management"
	@echo "  Module 04: Classes and OOP Fundamentals"
	@echo "  Module 05: Inheritance and Polymorphism"
	@echo "  Module 06: Operator Overloading"
	@echo "  Module 07: Templates and Generic Programming"
	@echo "  Module 08: STL and Standard Library"
	@echo "  Module 09: Modern C++ (C++11/14/17/20)"
	@echo "  Module 10: Exception Handling"
	@echo "  Module 11: Multithreading and Concurrency"
	@echo "  Module 12: Build Systems, Debugging & Tools"
	@echo "  Module 13: Advanced Topics for GPU"
	@echo "  Module 14: GPU Programming Preparation"
	@echo ""
	@echo "For module-specific help, run: make -C <module> help"
	@echo "Example: make -C 01-basics help"
	@echo ""
	@echo "Learning Path:"
	@echo "  Fundamentals (Modules 1-3):  4-6 weeks"
	@echo "  OOP (Modules 4-6):          4-5 weeks"
	@echo "  Advanced (Modules 7-10):    5-7 weeks"
	@echo "  Parallel (Modules 11-14):   5-6 weeks"
	@echo "  Total:                      18-24 weeks"

.PHONY: all homework both module1 module2 module3 module4 module5 module6 module7 module8 \
        module9 module10 module11 module12 module13 module14 \
        fundamentals oop advanced parallel clean test run stats list check help
