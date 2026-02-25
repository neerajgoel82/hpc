# Phase Makefile - delegates to classwork/ and homework/
CLASSWORK_DIR = classwork
HOMEWORK_DIR = homework

all: classwork

classwork:
	$(MAKE) -C $(CLASSWORK_DIR)

homework:
	$(MAKE) -C $(HOMEWORK_DIR)

both: classwork homework

clean:
	$(MAKE) -C $(CLASSWORK_DIR) clean
	$(MAKE) -C $(HOMEWORK_DIR) clean

test: classwork
	$(MAKE) -C $(CLASSWORK_DIR) test

test-homework: homework
	$(MAKE) -C $(HOMEWORK_DIR) test

.PHONY: all classwork homework both clean test test-homework
