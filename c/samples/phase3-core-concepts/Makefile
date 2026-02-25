# Phase 3: Core Concepts - Makefile
# Compiles all C programs in this phase

CC = gcc
CFLAGS = -std=c11 -Wall -Wextra -O2 -g
LDFLAGS = -lm

# Source files
SRCS = $(wildcard *.c)
# Executables (remove .c extension)
PROGS = $(SRCS:.c=)

# Default target: build all programs
all: $(PROGS)

# Rule to build each program
%: %.c
	$(CC) $(CFLAGS) $< -o $@ $(LDFLAGS)

# Individual program targets for convenience
pointers_basics: 01_pointers_basics.c
	$(CC) $(CFLAGS) $< -o $@ $(LDFLAGS)

pointers_arrays: 02_pointers_arrays.c
	$(CC) $(CFLAGS) $< -o $@ $(LDFLAGS)

pointers_functions: 03_pointers_functions.c
	$(CC) $(CFLAGS) $< -o $@ $(LDFLAGS)

structures: 04_structures.c
	$(CC) $(CFLAGS) $< -o $@ $(LDFLAGS)

dynamic_memory: 05_dynamic_memory.c
	$(CC) $(CFLAGS) $< -o $@ $(LDFLAGS)

enums_unions: 06_enums_unions.c
	$(CC) $(CFLAGS) $< -o $@ $(LDFLAGS)

# Clean up executables
clean:
	rm -f $(PROGS)

# Test target - build and run all programs
test: all
	@echo "=== Testing Phase 3 Programs ==="
	@for prog in $(PROGS); do \
		echo "\n--- Running $$prog ---"; \
		./$$prog || echo "$$prog failed"; \
	done

# Memory check with valgrind (if available)
memcheck: dynamic_memory
	@echo "=== Memory Check with Valgrind ==="
	@if command -v valgrind >/dev/null 2>&1; then \
		valgrind --leak-check=full ./dynamic_memory; \
	else \
		echo "Valgrind not installed. Install with: brew install valgrind (macOS) or apt install valgrind (Linux)"; \
	fi

# Help target
help:
	@echo "Phase 3: Core Concepts - Makefile"
	@echo ""
	@echo "Available targets:"
	@echo "  make           - Build all programs"
	@echo "  make all       - Build all programs"
	@echo "  make <program> - Build specific program (e.g., make pointers_basics)"
	@echo "  make test      - Build and run all programs"
	@echo "  make memcheck  - Check for memory leaks (requires valgrind)"
	@echo "  make clean     - Remove all executables"
	@echo "  make help      - Show this help message"
	@echo ""
	@echo "Programs:"
	@echo "  $(PROGS)"

.PHONY: all clean test memcheck help
