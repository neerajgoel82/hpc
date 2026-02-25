# Makefile for Phase 9
NVCC = nvcc
NVCC_FLAGS = -arch=sm_70 -O2
TARGETS = 01_cuda_graphs

all: $(TARGETS)

%: %.cu
	$(NVCC) $(NVCC_FLAGS) $< -o $@

clean:
	rm -f $(TARGETS)

test: all
	@for target in $(TARGETS); do \
		echo "=== Running $$target ===" ; \
		./$$target || exit 1; \
		echo ""; \
	done

.PHONY: all clean test
