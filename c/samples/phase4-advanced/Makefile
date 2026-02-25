# Phase 4: Advanced Topics - Makefile
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
file_io: 01_file_io.c
	$(CC) $(CFLAGS) $< -o $@ $(LDFLAGS)

preprocessor: 02_preprocessor.c
	$(CC) $(CFLAGS) $< -o $@ $(LDFLAGS)

linked_list: 03_linked_list.c
	$(CC) $(CFLAGS) $< -o $@ $(LDFLAGS)

stack: 04_stack.c
	$(CC) $(CFLAGS) $< -o $@ $(LDFLAGS)

queue: 05_queue.c
	$(CC) $(CFLAGS) $< -o $@ $(LDFLAGS)

# Clean up executables and test files
clean:
	rm -f $(PROGS)
	rm -f test_output.txt output.txt
	-rm -rf *.dSYM

# Test target - build and run all programs
test: all
	@echo "=== Testing Phase 4 Programs ==="
	@for prog in $(PROGS); do \
		echo "\n--- Running $$prog ---"; \
		./$$prog || echo "$$prog failed"; \
	done

# Memory check with valgrind (critical for data structures)
memcheck: linked_list stack queue
	@echo "=== Memory Check with Valgrind ==="
	@if command -v valgrind >/dev/null 2>&1; then \
		echo "\n--- Checking linked_list ---"; \
		valgrind --leak-check=full ./linked_list; \
		echo "\n--- Checking stack ---"; \
		valgrind --leak-check=full ./stack; \
		echo "\n--- Checking queue ---"; \
		valgrind --leak-check=full ./queue; \
	else \
		echo "Valgrind not installed. Install with: brew install valgrind (macOS) or apt install valgrind (Linux)"; \
	fi

# Help target
help:
	@echo "Phase 4: Advanced Topics - Makefile"
	@echo ""
	@echo "Available targets:"
	@echo "  make           - Build all programs"
	@echo "  make all       - Build all programs"
	@echo "  make <program> - Build specific program (e.g., make linked_list)"
	@echo "  make test      - Build and run all programs"
	@echo "  make memcheck  - Check for memory leaks in data structures"
	@echo "  make clean     - Remove all executables and test files"
	@echo "  make help      - Show this help message"
	@echo ""
	@echo "Programs:"
	@echo "  $(PROGS)"

.PHONY: all clean test memcheck help
