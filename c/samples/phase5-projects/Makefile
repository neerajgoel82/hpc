# Phase 5: Real-World Projects - Makefile
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

# Individual program targets
contact_manager: 01_contact_manager.c
	$(CC) $(CFLAGS) $< -o $@ $(LDFLAGS)

text_analyzer: 02_text_analyzer.c
	$(CC) $(CFLAGS) $< -o $@ $(LDFLAGS)

# Clean up executables and data files
clean:
	rm -f $(PROGS)
	rm -f contacts.dat *.txt.bak
	-rm -rf *.dSYM

# Test target - build and run all programs
test: all
	@echo "=== Testing Phase 5 Projects ==="
	@echo "\n--- Contact Manager ---"
	@echo "Build successful. Run interactively: ./contact_manager"
	@echo "\n--- Text Analyzer ---"
	@if [ -f sample.txt ]; then \
		./text_analyzer sample.txt; \
	else \
		echo "Create a sample.txt file to test text analyzer"; \
		echo "Example: echo 'This is a test file' > sample.txt"; \
	fi

# Memory check for projects
memcheck: contact_manager text_analyzer
	@echo "=== Memory Check with Valgrind ==="
	@if command -v valgrind >/dev/null 2>&1; then \
		echo "\n--- Checking contact_manager ---"; \
		echo "Note: Run interactively for full test"; \
		if [ -f sample.txt ]; then \
			echo "\n--- Checking text_analyzer ---"; \
			valgrind --leak-check=full ./text_analyzer sample.txt; \
		fi \
	else \
		echo "Valgrind not installed. Install with: brew install valgrind (macOS) or apt install valgrind (Linux)"; \
	fi

# Help target
help:
	@echo "Phase 5: Real-World Projects - Makefile"
	@echo ""
	@echo "Available targets:"
	@echo "  make              - Build all projects"
	@echo "  make all          - Build all projects"
	@echo "  make <project>    - Build specific project (e.g., make contact_manager)"
	@echo "  make test         - Build and test projects"
	@echo "  make memcheck     - Check for memory leaks"
	@echo "  make clean        - Remove all executables and data files"
	@echo "  make help         - Show this help message"
	@echo ""
	@echo "Projects:"
	@echo "  $(PROGS)"
	@echo ""
	@echo "Usage Examples:"
	@echo "  ./contact_manager           - Interactive contact management"
	@echo "  ./text_analyzer file.txt    - Analyze text file"

.PHONY: all clean test memcheck help
