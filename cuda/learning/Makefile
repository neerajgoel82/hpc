# Root Makefile - CUDA samples (classwork / homework per phase)
PHASES = 01-foundations 02-memory-management 03-optimization 04-advanced-memory 05-advanced-algorithms 06-streams-concurrency 07-performance-engineering 08-applications 09-modern-cuda

all:
	@for phase in $(PHASES); do \
		echo "Building $$phase..."; \
		$(MAKE) -C $$phase || exit 1; \
	done

# Build both classwork and homework in every phase
both:
	@for phase in $(PHASES); do \
		echo "Building $$phase (classwork + homework)..."; \
		$(MAKE) -C $$phase both || exit 1; \
	done

clean:
	@for phase in $(PHASES); do \
		$(MAKE) -C $$phase clean; \
	done

test:
	@for phase in $(PHASES); do \
		echo "\n=== Testing $$phase (classwork) ==="; \
		$(MAKE) -C $$phase test || exit 1; \
	done

test-homework:
	@for phase in $(PHASES); do \
		echo "\n=== Testing $$phase (homework) ==="; \
		$(MAKE) -C $$phase test-homework || exit 1; \
	done

.PHONY: all both clean test test-homework
