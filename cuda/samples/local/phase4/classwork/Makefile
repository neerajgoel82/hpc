# Classwork - build all .cu in this directory
NVCC = nvcc
NVCC_FLAGS = -arch=sm_70 -O2

SRCS = $(wildcard *.cu)
PROGS = $(SRCS:.cu=)

all: $(PROGS)
	@echo "Classwork programs built successfully"

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
