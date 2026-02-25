# Phase 5: Real-World Projects - Makefile
# Compiles C programs in classwork folder

CC = gcc
CFLAGS = -std=c11 -Wall -Wextra -O2 -g
LDFLAGS = -lm

# Directories
CLASSWORK_DIR = classwork

# Source files
CLASSWORK_SRCS = $(wildcard $(CLASSWORK_DIR)/*.c)

# Executables
CLASSWORK_PROGS = $(CLASSWORK_SRCS:.c=)

# Default target
all: classwork

classwork: $(CLASSWORK_PROGS)
	@echo "Phase 5 projects built successfully"

$(CLASSWORK_DIR)/%: $(CLASSWORK_DIR)/%.c
	$(CC) $(CFLAGS) $< -o $@ $(LDFLAGS)

clean:
	rm -f $(CLASSWORK_PROGS)
	-rm -rf $(CLASSWORK_DIR)/*.dSYM

test: classwork
	@echo "=== Phase 5 projects are interactive ==="
	@echo "Run them individually to test"

.PHONY: all classwork clean test help
