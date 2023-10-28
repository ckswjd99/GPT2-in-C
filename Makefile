CC=gcc
OBJDIR=./obj/
COMMON=-Iinclude/
BLASLIB=openblas
DEBUG=0
OPTS=
SOURCES=$(wildcard src/*.c)
OBJS=$(patsubst src/%.c, $(OBJDIR)%.o, $(SOURCES))

ifeq ($(DEBUG), 1)
	CC+=-g
	OPTS+=-DDEBUG
endif

DEPS = $(wildcard src/*.h) $(wildcard include/*.h) Makefile

make: $(OBJS)
	$(CC) $(COMMON) $(OPTS) main.c $^ -o main.out -l$(BLASLIB) -lm

$(OBJS): $(OBJDIR)%.o: src/%.c $(DEPS) $(OBJDIR)
	$(CC) $(COMMON) $(OPTS) -c $< -o $@ -l$(BLASLIB) -lm

$(OBJDIR):
	mkdir obj

clean:
	rm -rf $(OBJDIR) main.out