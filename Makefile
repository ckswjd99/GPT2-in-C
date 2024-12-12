CC=gcc
OBJDIR=./obj/
COMMON=-Iinclude/
BLASLIB=openblas
DEBUG=0
OPTS=
SOURCES=$(wildcard src/*.c)
OBJS=$(patsubst src/%.c, $(OBJDIR)%.o, $(SOURCES))
LDFLAGS=-l$(BLASLIB) -lm

APPLE=$(shell uname -a | grep -q "Darwin" && echo 1 || echo 0)
ifeq ($(APPLE), 1)
	COMMON+=-I/opt/homebrew/opt/openblas/include
	LDFLAGS+=-L/opt/homebrew/opt/openblas/lib
endif

ifeq ($(DEBUG), 1)
	CC+=-g
	OPTS+=-DDEBUG
endif

DEPS = $(wildcard src/*.h) $(wildcard include/*.h) Makefile

make: $(OBJS)
	$(CC) $(COMMON) $(OPTS) main.c $^ -o main.out $(LDFLAGS)

$(OBJS): $(OBJDIR)%.o: src/%.c $(DEPS) $(OBJDIR)
	$(CC) $(COMMON) $(OPTS) -c $< -o $@ $(LDFLAGS)

$(OBJDIR):
	mkdir obj

clean:
	rm -rf $(OBJDIR) main.out