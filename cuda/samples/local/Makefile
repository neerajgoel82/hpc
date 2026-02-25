# Root Makefile
PHASES = phase1 phase2 phase3 phase4 phase5 phase6 phase7 phase8 phase9

all:
	@for phase in $(PHASES); do \
		echo "Building $$phase..."; \
		$(MAKE) -C $$phase || exit 1; \
	done

clean:
	@for phase in $(PHASES); do \
		$(MAKE) -C $$phase clean; \
	done

test:
	@for phase in $(PHASES); do \
		echo "\n=== Testing $$phase ==="; \
		$(MAKE) -C $$phase test || exit 1; \
	done

.PHONY: all clean test
