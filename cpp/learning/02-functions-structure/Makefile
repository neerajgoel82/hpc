# Module 02 - Makefile
# Builds C++ programs in classwork and homework folders

CXX = g++
CXXFLAGS = -std=c++17 -Wall -Wextra -O2 -g
LDFLAGS =

# Directories
CLASSWORK_DIR = classwork
HOMEWORK_DIR = homework

# Source files
CLASSWORK_SRCS = $(wildcard $(CLASSWORK_DIR)/*.cpp)
HOMEWORK_SRCS = $(wildcard $(HOMEWORK_DIR)/*.cpp)

# Exclude implementation files (not standalone programs)
CLASSWORK_IMPL_FILES = $(CLASSWORK_DIR)/math_utils.cpp
CLASSWORK_MAIN_SRCS = $(filter-out $(CLASSWORK_IMPL_FILES), $(CLASSWORK_SRCS))

# Executables
CLASSWORK_PROGS = $(CLASSWORK_MAIN_SRCS:.cpp=)
HOMEWORK_PROGS = $(HOMEWORK_SRCS:.cpp=)

# Default target: build classwork
all: classwork

classwork: $(CLASSWORK_PROGS)
	@echo "Module 02 classwork programs built"

homework: $(HOMEWORK_PROGS)
	@echo "Module 02 homework programs built"

both: classwork homework

# Special rule for multi-file programs
$(CLASSWORK_DIR)/04_header_example_main: $(CLASSWORK_DIR)/04_header_example_main.cpp $(CLASSWORK_DIR)/math_utils.cpp
	$(CXX) $(CXXFLAGS) $(CLASSWORK_DIR)/04_header_example_main.cpp $(CLASSWORK_DIR)/math_utils.cpp -o $@ $(LDFLAGS)

# General rule for single-file programs
$(CLASSWORK_DIR)/%: $(CLASSWORK_DIR)/%.cpp
	$(CXX) $(CXXFLAGS) $< -o $@ $(LDFLAGS)

$(HOMEWORK_DIR)/%: $(HOMEWORK_DIR)/%.cpp
	$(CXX) $(CXXFLAGS) $< -o $@ $(LDFLAGS)

clean:
	rm -f $(CLASSWORK_PROGS) $(HOMEWORK_PROGS)
	-rm -rf $(CLASSWORK_DIR)/*.dSYM $(HOMEWORK_DIR)/*.dSYM

test: classwork
	@echo "=== Testing Module 02 Classwork ==="
	@for prog in $(CLASSWORK_PROGS); do \
		echo "\n--- Running $$prog ---"; \
		./$$prog || echo "$$prog failed"; \
	done

test-homework: homework
	@echo "=== Testing Module 02 Homework ==="
	@for prog in $(HOMEWORK_PROGS); do \
		echo "\n--- Running $$prog ---"; \
		./$$prog || echo "$$prog failed"; \
	done

help:
	@echo "Module 02 - Makefile"
	@echo ""
	@echo "Available targets:"
	@echo "  make              - Build classwork"
	@echo "  make classwork    - Build classwork"
	@echo "  make homework     - Build homework"
	@echo "  make both         - Build both"
	@echo "  make test         - Test classwork"
	@echo "  make test-homework- Test homework"
	@echo "  make clean        - Remove executables"

.PHONY: all classwork homework both clean test test-homework help
