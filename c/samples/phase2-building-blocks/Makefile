# Phase 2: Building Blocks - Makefile
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
functions: 01_functions.c
	$(CC) $(CFLAGS) $< -o $@ $(LDFLAGS)

function_scope: 02_function_scope.c
	$(CC) $(CFLAGS) $< -o $@ $(LDFLAGS)

recursion: 03_recursion.c
	$(CC) $(CFLAGS) $< -o $@ $(LDFLAGS)

arrays: 04_arrays.c
	$(CC) $(CFLAGS) $< -o $@ $(LDFLAGS)

multidimensional_arrays: 05_multidimensional_arrays.c
	$(CC) $(CFLAGS) $< -o $@ $(LDFLAGS)

strings: 06_strings.c
	$(CC) $(CFLAGS) $< -o $@ $(LDFLAGS)

command_line_args: 07_command_line_args.c
	$(CC) $(CFLAGS) $< -o $@ $(LDFLAGS)

# Clean up executables
clean:
	rm -f $(PROGS)

# Test target - build and run all programs
test: all
	@echo "=== Testing Phase 2 Programs ==="
	@for prog in $(PROGS); do \
		echo "\n--- Running $$prog ---"; \
		./$$prog || echo "$$prog failed"; \
	done
	@echo "\n--- Testing command line args ---"
	./07_command_line_args arg1 arg2 arg3

# Help target
help:
	@echo "Phase 2: Building Blocks - Makefile"
	@echo ""
	@echo "Available targets:"
	@echo "  make           - Build all programs"
	@echo "  make all       - Build all programs"
	@echo "  make <program> - Build specific program (e.g., make functions)"
	@echo "  make test      - Build and run all programs"
	@echo "  make clean     - Remove all executables"
	@echo "  make help      - Show this help message"
	@echo ""
	@echo "Programs:"
	@echo "  $(PROGS)"

.PHONY: all clean test help
