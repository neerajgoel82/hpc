# Phase 2: Building Blocks - Makefile
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

# Executables
CLASSWORK_PROGS = $(CLASSWORK_SRCS:.c=)
HOMEWORK_PROGS = $(HOMEWORK_SRCS:.c=)

# Default target: build all classwork programs
all: classwork

classwork: $(CLASSWORK_PROGS)
	@echo "Phase 2 classwork programs built successfully"

homework: $(HOMEWORK_PROGS)
	@echo "Phase 2 homework programs built successfully"

both: classwork homework

$(CLASSWORK_DIR)/%: $(CLASSWORK_DIR)/%.c
	$(CC) $(CFLAGS) $< -o $@ $(LDFLAGS)

$(HOMEWORK_DIR)/%: $(HOMEWORK_DIR)/%.c
	$(CC) $(CFLAGS) $< -o $@ $(LDFLAGS)

clean:
	rm -f $(CLASSWORK_PROGS) $(HOMEWORK_PROGS)
	-rm -rf $(CLASSWORK_DIR)/*.dSYM $(HOMEWORK_DIR)/*.dSYM

test: classwork
	@echo "=== Testing Phase 2 Classwork Programs ==="
	@for prog in $(CLASSWORK_PROGS); do \
		echo "\n--- Running $$prog ---"; \
		./$$prog || echo "$$prog failed"; \
	done

test-homework: homework
	@echo "=== Testing Phase 2 Homework Programs ==="
	@for prog in $(HOMEWORK_PROGS); do \
		echo "\n--- Running $$prog ---"; \
		./$$prog || echo "$$prog failed"; \
	done

help:
	@echo "Phase 2: Building Blocks - Makefile"
	@echo ""
	@echo "Available targets:"
	@echo "  make              - Build all classwork programs"
	@echo "  make classwork    - Build all classwork programs"
	@echo "  make homework     - Build all homework programs"
	@echo "  make both         - Build both classwork and homework"
	@echo "  make test         - Build and run classwork programs"
	@echo "  make test-homework- Build and run homework programs"
	@echo "  make clean        - Remove all executables"

.PHONY: all classwork homework both clean test test-homework help
