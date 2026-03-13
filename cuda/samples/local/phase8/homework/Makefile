# Homework - build all .cu in this directory
NVCC = nvcc
NVCC_FLAGS = -arch=sm_70 -O2
LIBS = -lcufft -lcusparse -lcurand

SRCS = $(wildcard *.cu)
# Exclude cuTENSOR sample if library not installed
SRCS := $(filter-out 61_cutensor_basics.cu, $(SRCS))
PROGS = $(SRCS:.cu=)

all: $(PROGS)
	@echo "Homework programs built successfully"

# Special rule for cuTENSOR (only if library installed)
61_cutensor_basics: 61_cutensor_basics.cu
	$(NVCC) $(NVCC_FLAGS) $< -o $@ $(LIBS) -lcutensor

%: %.cu
	$(NVCC) $(NVCC_FLAGS) $< -o $@ $(LIBS)

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
