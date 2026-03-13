# Homework - build all .cu in this directory
NVCC = nvcc
NVCC_FLAGS = -arch=sm_70 -O2
# Dynamic parallelism requires relocatable device code
NVCC_FLAGS_DP = -arch=sm_70 -O2 -rdc=true -lcudadevrt

SRCS = $(wildcard *.cu)
PROGS = $(SRCS:.cu=)

all: $(PROGS)
	@echo "Homework programs built successfully"

# Special rule for dynamic parallelism
50_dynamic_parallelism: 50_dynamic_parallelism.cu
	$(NVCC) $(NVCC_FLAGS_DP) $< -o $@

# Special rule for cooperative groups advanced (needs relocatable device code)
60_cooperative_groups_advanced: 60_cooperative_groups_advanced.cu
	$(NVCC) $(NVCC_FLAGS_DP) $< -o $@

%: %.cu
	$(NVCC) $(NVCC_FLAGS) $< -o $@

clean:
	rm -f $(PROGS)
	-rm -rf *.dSYM

test: all
	@for prog in $(PROGS); do \
		echo "=== Running $$prog ==="; \
		./$$prog || exit 1; \
		echo ""; \
	done

.PHONY: all clean test
