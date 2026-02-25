# Module 09: Modern C++ Features - Makefile
# Compiles all C++ programs in this module

CXX = g++
CXXFLAGS = -std=c++17 -Wall -Wextra -O2 -g
LDFLAGS =

# Source files
SRCS = $(wildcard *.cpp)
# Executables (remove .cpp extension)
PROGS = $(SRCS:.cpp=)

# Default target: build all programs
all: $(PROGS)

# Rule to build each program
%: %.cpp
	$(CXX) $(CXXFLAGS) $< -o $@ $(LDFLAGS)

# Clean up executables
clean:
	rm -f $(PROGS)
	-rm -rf *.dSYM

# Run all programs
run: all
	@echo "=== Running Module 09 Programs ==="
	@for prog in $(PROGS); do \
		echo "\n--- Running $$prog ---"; \
		./$$prog || echo "$$prog failed"; \
	done

# Test target (same as run for this module)
test: run

# Help target
help:
	@echo "Module 09: Modern C++ Features - Makefile"
	@echo ""
	@echo "Available targets:"
	@echo "  make           - Build all programs"
	@echo "  make all       - Build all programs"
	@echo "  make <program> - Build specific program (without extension)"
	@echo "  make run       - Build and run all programs"
	@echo "  make test      - Build and test all programs"
	@echo "  make clean     - Remove all executables"
	@echo "  make help      - Show this help message"
	@echo ""
	@echo "Programs in this module:"
	@echo "  $(PROGS)" | tr ' ' '\n' | grep -v "^$$"

.PHONY: all clean run test help
