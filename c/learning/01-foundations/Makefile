# Phase 1: Foundations - Makefile
# Compiles C programs in classwork and homework folders

CC = gcc
CFLAGS = -std=c11 -Wall -Wextra -O2 -g
LDFLAGS = -lm

# Directories
CLASSWORK_DIR = classwork
HOMEWORK_DIR = homework

# Source files
CLASSWORK_SRCS = $(wildcard $(CLASSWORK_DIR)/*.c)
HOMEWORK_SRCS = $(wildcard $(HOMEWORK_DIR)/*.c)

# Executables (placed in respective directories)
CLASSWORK_PROGS = $(CLASSWORK_SRCS:.c=)
HOMEWORK_PROGS = $(HOMEWORK_SRCS:.c=)

# Default target: build all classwork programs
all: classwork

# Build classwork programs
classwork: $(CLASSWORK_PROGS)
	@echo "Phase 1 classwork programs built successfully"

# Build homework programs
homework: $(HOMEWORK_PROGS)
	@echo "Phase 1 homework programs built successfully"

# Build both classwork and homework
both: classwork homework

# Rule to build classwork programs
$(CLASSWORK_DIR)/%: $(CLASSWORK_DIR)/%.c
	$(CC) $(CFLAGS) $< -o $@ $(LDFLAGS)

# Rule to build homework programs
$(HOMEWORK_DIR)/%: $(HOMEWORK_DIR)/%.c
	$(CC) $(CFLAGS) $< -o $@ $(LDFLAGS)

# Clean up executables
clean:
	rm -f $(CLASSWORK_PROGS) $(HOMEWORK_PROGS)
	-rm -rf $(CLASSWORK_DIR)/*.dSYM $(HOMEWORK_DIR)/*.dSYM

# Test classwork programs
test: classwork
	@echo "=== Testing Phase 1 Classwork Programs ==="
	@for prog in $(CLASSWORK_PROGS); do \
		echo "\n--- Running $$prog ---"; \
		./$$prog || echo "$$prog failed"; \
	done

# Test homework programs
test-homework: homework
	@echo "=== Testing Phase 1 Homework Programs ==="
	@for prog in $(HOMEWORK_PROGS); do \
		echo "\n--- Running $$prog ---"; \
		./$$prog || echo "$$prog failed"; \
	done

# Help target
help:
	@echo "Phase 1: Foundations - Makefile"
	@echo ""
	@echo "Available targets:"
	@echo "  make              - Build all classwork programs"
	@echo "  make classwork    - Build all classwork programs"
	@echo "  make homework     - Build all homework programs"
	@echo "  make both         - Build both classwork and homework"
	@echo "  make test         - Build and run classwork programs"
	@echo "  make test-homework- Build and run homework programs"
	@echo "  make clean        - Remove all executables"
	@echo "  make help         - Show this help message"
	@echo ""
	@echo "Classwork programs: $(notdir $(CLASSWORK_PROGS))"
	@echo "Homework programs: $(notdir $(HOMEWORK_PROGS))"

.PHONY: all classwork homework both clean test test-homework help
