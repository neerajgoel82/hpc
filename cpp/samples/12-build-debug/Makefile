# MODULE 12: Build Systems
# File: Makefile
#
# TOPIC: Makefile for Building C++ Projects
#
# USAGE:
#   make          - Build all targets
#   make clean    - Remove build artifacts
#   make run      - Build and run
#   make debug    - Build with debug symbols

# Compiler and flags
CXX = g++
CXXFLAGS = -std=c++17 -Wall -Wextra
DEBUGFLAGS = -g -O0
RELEASEFLAGS = -O3 -DNDEBUG

# Directories
SRCDIR = src
OBJDIR = obj
BINDIR = bin

# Source files
SOURCES = $(wildcard $(SRCDIR)/*.cpp)
OBJECTS = $(SOURCES:$(SRCDIR)/%.cpp=$(OBJDIR)/%.o)

# Target executable
TARGET = $(BINDIR)/myapp

# Default target
all: $(TARGET)

# Link
$(TARGET): $(OBJECTS) | $(BINDIR)
	$(CXX) $(CXXFLAGS) $(OBJECTS) -o $@
	@echo "Build complete: $(TARGET)"

# Compile
$(OBJDIR)/%.o: $(SRCDIR)/%.cpp | $(OBJDIR)
	$(CXX) $(CXXFLAGS) $(RELEASEFLAGS) -c $< -o $@

# Debug build
debug: CXXFLAGS += $(DEBUGFLAGS)
debug: clean $(TARGET)

# Create directories
$(OBJDIR):
	mkdir -p $(OBJDIR)

$(BINDIR):
	mkdir -p $(BINDIR)

# Run
run: $(TARGET)
	./$(TARGET)

# Clean
clean:
	rm -rf $(OBJDIR) $(BINDIR)
	@echo "Clean complete"

# Phony targets
.PHONY: all debug run clean

# GPU example targets
gpu: gpu_vector_add gpu_matrix_mult

gpu_vector_add:
	nvcc -o $(BINDIR)/vector_add $(SRCDIR)/vector_add.cu

gpu_matrix_mult:
	nvcc -o $(BINDIR)/matrix_mult $(SRCDIR)/matrix_mult.cu

# TRY THIS:
# 1. Add dependency tracking
# 2. Add separate debug/release directories
# 3. Add library linking (-lGL -lGLU for OpenGL)
# 4. Add CUDA compilation rules
# 5. Add install target
