# Phase 5: Real-World Projects - Makefile
# Compiles C programs in classwork folder

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
	@echo "Phase 5 projects built successfully"

homework: $(HOMEWORK_PROGS)
	@echo "Phase 5 homework built successfully"

both: classwork homework

$(CLASSWORK_DIR)/%: $(CLASSWORK_DIR)/%.c
	$(CC) $(CFLAGS) $< -o $@ $(LDFLAGS)

$(HOMEWORK_DIR)/%: $(HOMEWORK_DIR)/%.c
	$(CC) $(CFLAGS) $< -o $@ $(LDFLAGS)

clean:
	rm -f $(CLASSWORK_PROGS) $(HOMEWORK_PROGS)
	-rm -rf $(CLASSWORK_DIR)/*.dSYM $(HOMEWORK_DIR)/*.dSYM

test: classwork
	@echo "=== Phase 5 projects are interactive ==="
	@echo "Run them individually to test"

help:
	@echo "Phase 5: Real-World Projects - Makefile"
	@echo ""
	@echo "Available targets:"
	@echo "  make              - Build all classwork projects"
	@echo "  make classwork    - Build all classwork projects"
	@echo "  make homework     - Build all homework projects"
	@echo "  make both         - Build both classwork and homework"
	@echo "  make clean        - Remove all executables"
	@echo "  make test         - Info about testing projects"
	@echo "  make help         - Show this help message"

.PHONY: all classwork homework both clean test help
