# Homework Makefile
# Builds all homework programs in this directory

CC = gcc
CFLAGS = -std=c11 -Wall -Wextra -O2 -g
LDFLAGS = -lm

# Source files
SRCS = $(wildcard *.c)
# Executables (remove .c extension)
PROGS = $(SRCS:.c=)

# Default target: build all programs
all: $(PROGS)
	@echo "All homework programs built successfully"

# Rule to build each program
%: %.c
	$(CC) $(CFLAGS) $< -o $@ $(LDFLAGS)

# Clean up executables
clean:
	rm -f $(PROGS)
	-rm -rf *.dSYM

# Run all programs (use with caution - some may be interactive)
run:
	@for prog in $(PROGS); do \
		echo "\n=== Running $$prog ==="; \
		./$$prog; \
	done

# List all programs
list:
	@echo "Homework programs:"
	@echo "$(PROGS)" | tr ' ' '\n'

help:
	@echo "Homework Makefile"
	@echo ""
	@echo "Available targets:"
	@echo "  make       - Build all homework programs"
	@echo "  make all   - Build all homework programs"
	@echo "  make clean - Remove all executables"
	@echo "  make run   - Run all programs"
	@echo "  make list  - List all programs"
	@echo "  make help  - Show this help"
	@echo ""
	@echo "To build a specific program:"
	@echo "  make 01_hello_world"

.PHONY: all clean run list help
