# Phase 3: Core Concepts - Makefile
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

# Default target
all: classwork

classwork: $(CLASSWORK_PROGS)
	@echo "Phase 3 classwork programs built successfully"

homework: $(HOMEWORK_PROGS)
	@echo "Phase 3 homework programs built successfully"

both: classwork homework

$(CLASSWORK_DIR)/%: $(CLASSWORK_DIR)/%.c
	$(CC) $(CFLAGS) $< -o $@ $(LDFLAGS)

$(HOMEWORK_DIR)/%: $(HOMEWORK_DIR)/%.c
	$(CC) $(CFLAGS) $< -o $@ $(LDFLAGS)

clean:
	rm -f $(CLASSWORK_PROGS) $(HOMEWORK_PROGS)
	-rm -rf $(CLASSWORK_DIR)/*.dSYM $(HOMEWORK_DIR)/*.dSYM

test: classwork
	@echo "=== Testing Phase 3 Classwork Programs ==="
	@for prog in $(CLASSWORK_PROGS); do \
		echo "\n--- Running $$prog ---"; \
		./$$prog || echo "$$prog failed"; \
	done

memcheck: classwork
	@echo "=== Memory Check Phase 3 ==="
	@for prog in $(CLASSWORK_PROGS); do \
		echo "\n--- Checking $$prog ---"; \
		valgrind --leak-check=full ./$$prog 2>&1 | grep -E "ERROR SUMMARY|lost"; \
	done

.PHONY: all classwork homework both clean test memcheck help
