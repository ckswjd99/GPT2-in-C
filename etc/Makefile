SOURCE := $(wildcard *.c)
TARGETS = $(SOURCE:.c=)

AVX2=1
NEON=0

DEBUG=0

CC=gcc

CFLAGS=-mavx
LDFLAGS=
OPTS=-march=native -funroll-loops

ifeq ($(AVX2), 1)
OPTS+=-mavx2 -mfma -DAVX2
endif

ifeq ($(NEON), 1)
OPTS+=-DNEON
endif

all: $(TARGETS)

$(TARGETS): $(SOURCE)
	$(CC) $(CFLAGS) $(OPTS) $@.c -o $@.out $(LDFLAGS)

clean:
	rm $(addsuffix ".out", $(TARGETS))