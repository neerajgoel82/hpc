# Makefile for mathlib project
# Demonstrates multi-file compilation

CC = gcc
CFLAGS = -Wall -Wextra -g -std=c11
TARGET = math_program

# Source files
SRCS = main.c mathlib.c

# Object files (replace .c with .o)
OBJS = $(SRCS:.c=.o)

# Default target - build the program
all: $(TARGET)
	@echo "Build complete! Run with: ./$(TARGET)"

# Link object files to create executable
$(TARGET): $(OBJS)
	@echo "Linking..."
	$(CC) $(OBJS) -o $(TARGET)

# Compile source files to object files
# $< is the source file, $@ is the target file
%.o: %.c mathlib.h
	@echo "Compiling $<..."
	$(CC) $(CFLAGS) -c $< -o $@

# Run the program
run: $(TARGET)
	@echo "Running $(TARGET)..."
	@./$(TARGET)

# Clean up compiled files
clean:
	@echo "Cleaning up..."
	rm -f $(OBJS) $(TARGET)
	@echo "Clean complete!"

# Phony targets (not actual files)
.PHONY: all run clean

# Dependencies
# These tell make which files depend on which headers
main.o: main.c mathlib.h
mathlib.o: mathlib.c mathlib.h
