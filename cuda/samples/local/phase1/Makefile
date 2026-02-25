# Makefile for Phase CUDA programs
NVCC = nvcc
NVCC_FLAGS = -arch=sm_70 -O2
TARGETS = 01_hello_world 02_vector_add 03_matrix_add

all: $(TARGETS)

%: %.cu
	$(NVCC) $(NVCC_FLAGS) $< -o $@

clean:
	rm -f $(TARGETS)

test: all
	@for target in $(TARGETS); do \
		echo "Running $$target..."; \
		./$$target || exit 1; \
		echo ""; \
	done

.PHONY: all clean test
