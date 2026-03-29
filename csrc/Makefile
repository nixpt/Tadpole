CC = cc
CFLAGS = -O3 -march=native -Wall
LDFLAGS = -lm

# Auto-detect ARM NEON
UNAME_M := $(shell uname -m)
ifeq ($(UNAME_M),aarch64)
    CFLAGS += -mfpu=neon-fp-armv8
endif
ifeq ($(UNAME_M),armv7l)
    CFLAGS += -mfpu=neon-vfpv4
endif

.PHONY: all clean

all: goby

goby: goby.c
	$(CC) $(CFLAGS) -o $@ $< $(LDFLAGS)
	@echo "Built: ./goby"
	@echo "Usage: ./goby goby.bin -p 'Hello' | -i | -b"

# Debug build (no optimization, address sanitizer)
debug: goby.c
	$(CC) -g -O0 -fsanitize=address -o goby_debug $< $(LDFLAGS)

clean:
	rm -f goby goby_debug
