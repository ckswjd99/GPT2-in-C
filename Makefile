CC=gcc
OBJDIR=./obj/
COMMON=-Iinclude/
BLASLIB=openblas
DEBUG=1
OPTS=

ifeq ($(DEBUG), 1)
	CC+=-g
	OPTS+=-DDEBUG
endif

DEPS = $(wildcard src/*.h) $(wildcard include/*.h) Makefile

make: $(OBJDIR)gpt2.o
	$(CC) $(COMMON) $(OPTS) main.c $^ -o main.out -l$(BLASLIB) -lm

$(OBJDIR)gpt2.o: $(OBJDIR)
	$(CC) $(COMMON) $(OPTS) src/gpt2.c -c -o $(OBJDIR)gpt2.o -l$(BLASLIB) -lm

$(OBJDIR):
	mkdir obj

clean:
	rm -rf $(OBJDIR) main.out