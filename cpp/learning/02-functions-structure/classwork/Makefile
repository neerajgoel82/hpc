# Classwork Makefile
# Builds all classwork programs in this directory

CXX = g++
CXXFLAGS = -std=c++17 -Wall -Wextra -O2 -g
LDFLAGS =

# Source files
SRCS = $(wildcard *.cpp)
# Exclude implementation files (not standalone programs)
IMPL_FILES = math_utils.cpp
MAIN_SRCS = $(filter-out $(IMPL_FILES), $(SRCS))
# Executables (remove .cpp extension)
PROGS = $(MAIN_SRCS:.cpp=)

# Default target: build all programs
all: $(PROGS)
	@echo "All classwork programs built successfully"

# Special rule for multi-file programs
04_header_example_main: 04_header_example_main.cpp math_utils.cpp
	$(CXX) $(CXXFLAGS) 04_header_example_main.cpp math_utils.cpp -o $@ $(LDFLAGS)

# Rule to build each program
%: %.cpp
	$(CXX) $(CXXFLAGS) $< -o $@ $(LDFLAGS)

# Clean up executables
clean:
	rm -f $(PROGS)
	-rm -rf *.dSYM

# Run all programs (use with caution - some may be interactive)
run:
	@for prog in $(PROGS); do \
		echo "\n=== Running $$prog ==="; \
		./$$prog; \
	done

# List all programs
list:
	@echo "Classwork programs:"
	@echo "$(PROGS)" | tr ' ' '\n'

help:
	@echo "Classwork Makefile"
	@echo ""
	@echo "Available targets:"
	@echo "  make       - Build all classwork programs"
	@echo "  make all   - Build all classwork programs"
	@echo "  make clean - Remove all executables"
	@echo "  make run   - Run all programs"
	@echo "  make list  - List all programs"
	@echo "  make help  - Show this help"
	@echo ""
	@echo "To build a specific program:"
	@echo "  make 01_hello_world"

.PHONY: all clean run list help
