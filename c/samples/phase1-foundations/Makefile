# Phase 1: Foundations - Makefile
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
hello_world: 01_hello_world.c
	$(CC) $(CFLAGS) $< -o $@ $(LDFLAGS)

data_types: 02_data_types.c
	$(CC) $(CFLAGS) $< -o $@ $(LDFLAGS)

operators: 03_operators.c
	$(CC) $(CFLAGS) $< -o $@ $(LDFLAGS)

input_output: 04_input_output.c
	$(CC) $(CFLAGS) $< -o $@ $(LDFLAGS)

if_else: 05_if_else.c
	$(CC) $(CFLAGS) $< -o $@ $(LDFLAGS)

switch: 06_switch.c
	$(CC) $(CFLAGS) $< -o $@ $(LDFLAGS)

loops: 07_loops.c
	$(CC) $(CFLAGS) $< -o $@ $(LDFLAGS)

bitwise_operators: 08_bitwise_operators.c
	$(CC) $(CFLAGS) $< -o $@ $(LDFLAGS)

# Clean up executables
clean:
	rm -f $(PROGS)
	-rm -rf *.dSYM

# Test target - build and run all programs
test: all
	@echo "=== Testing Phase 1 Programs ==="
	@for prog in $(PROGS); do \
		echo "\n--- Running $$prog ---"; \
		./$$prog || echo "$$prog failed"; \
	done

# Help target
help:
	@echo "Phase 1: Foundations - Makefile"
	@echo ""
	@echo "Available targets:"
	@echo "  make           - Build all programs"
	@echo "  make all       - Build all programs"
	@echo "  make <program> - Build specific program (e.g., make hello_world)"
	@echo "  make test      - Build and run all programs"
	@echo "  make clean     - Remove all executables"
	@echo "  make help      - Show this help message"
	@echo ""
	@echo "Programs:"
	@echo "  $(PROGS)"

.PHONY: all clean test help
