# Root Makefile - CUDA samples (classwork / homework per phase)
PHASES = phase1 phase2 phase3 phase4 phase5 phase6 phase7 phase8 phase9

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
