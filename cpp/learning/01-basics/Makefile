# Module 01: C++ Basics - Makefile
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

# Individual program targets for convenience
01_hello_world: 01_hello_world.cpp
	$(CXX) $(CXXFLAGS) $< -o $@ $(LDFLAGS)

02_types_and_variables: 02_types_and_variables.cpp
	$(CXX) $(CXXFLAGS) $< -o $@ $(LDFLAGS)

03_functions: 03_functions.cpp
	$(CXX) $(CXXFLAGS) $< -o $@ $(LDFLAGS)

04_control_flow: 04_control_flow.cpp
	$(CXX) $(CXXFLAGS) $< -o $@ $(LDFLAGS)

05_strings: 05_strings.cpp
	$(CXX) $(CXXFLAGS) $< -o $@ $(LDFLAGS)

06_input: 06_input.cpp
	$(CXX) $(CXXFLAGS) $< -o $@ $(LDFLAGS)

07_enums: 07_enums.cpp
	$(CXX) $(CXXFLAGS) $< -o $@ $(LDFLAGS)

08_const_constexpr: 08_const_constexpr.cpp
	$(CXX) $(CXXFLAGS) $< -o $@ $(LDFLAGS)

09_namespaces: 09_namespaces.cpp
	$(CXX) $(CXXFLAGS) $< -o $@ $(LDFLAGS)

# Clean up executables
clean:
	rm -f $(PROGS)
	-rm -rf *.dSYM

# Run all programs
run: all
	@echo "=== Running Module 01 Programs ==="
	@for prog in $(PROGS); do \
		echo "\n--- Running $$prog ---"; \
		./$$prog || echo "$$prog failed"; \
	done

# Test target (same as run for this module)
test: run

# Help target
help:
	@echo "Module 01: C++ Basics - Makefile"
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
	@echo "Examples:"
	@echo "  make 01_hello_world    - Build just hello world"
	@echo "  make                   - Build everything"
	@echo ""
	@echo "Programs in this module:"
	@echo "  $(PROGS)" | tr ' ' '\n' | grep -v "^$$"

.PHONY: all clean run test help
