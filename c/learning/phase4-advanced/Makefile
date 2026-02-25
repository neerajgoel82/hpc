# Phase 4: Advanced Topics - Makefile
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
all: classwork header_files

classwork: $(CLASSWORK_PROGS)
	@echo "Phase 4 classwork programs built successfully"

homework: $(HOMEWORK_PROGS)
	@echo "Phase 4 homework programs built successfully"

header_files:
	@echo "Building header files example..."
	@$(MAKE) -C $(CLASSWORK_DIR)/06_header_files all 2>/dev/null || true

both: classwork homework header_files

$(CLASSWORK_DIR)/%: $(CLASSWORK_DIR)/%.c
	$(CC) $(CFLAGS) $< -o $@ $(LDFLAGS)

$(HOMEWORK_DIR)/%: $(HOMEWORK_DIR)/%.c
	$(CC) $(CFLAGS) $< -o $@ $(LDFLAGS)

clean:
	rm -f $(CLASSWORK_PROGS) $(HOMEWORK_PROGS)
	-rm -rf $(CLASSWORK_DIR)/*.dSYM $(HOMEWORK_DIR)/*.dSYM
	@$(MAKE) -C $(CLASSWORK_DIR)/06_header_files clean 2>/dev/null || true

test: classwork
	@echo "=== Testing Phase 4 Classwork Programs ==="
	@for prog in $(CLASSWORK_PROGS); do \
		echo "\n--- Running $$prog ---"; \
		./$$prog || echo "$$prog failed"; \
	done

memcheck: classwork
	@echo "=== Memory Check Phase 4 ==="
	@for prog in $(CLASSWORK_PROGS); do \
		echo "\n--- Checking $$prog ---"; \
		valgrind --leak-check=full ./$$prog 2>&1 | grep -E "ERROR SUMMARY|lost"; \
	done

.PHONY: all classwork homework both header_files clean test memcheck help
