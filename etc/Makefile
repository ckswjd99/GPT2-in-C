SOURCE := $(wildcard *.c)
TARGETS = $(SOURCE:.c=)

AVX2=0
NEON=1

DEBUG=1

CC=gcc -g

CFLAGS=
LDFLAGS=-lopenblas
OPTS=-march=native -funroll-loops

ifeq ($(AVX2), 1)
OPTS+=-mavx -mavx2 -mfma -DAVX2
endif

ifeq ($(NEON), 1)
OPTS+=-DNEON
endif

all: $(TARGETS)

$(TARGETS): $(SOURCE)
	$(CC) $(CFLAGS) $(OPTS) $@.c -o $@.out $(LDFLAGS)

clean:
	rm $(addsuffix .out, $(TARGETS))