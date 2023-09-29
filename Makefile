CC = gcc -g
OBJDIR = ./obj/
COMMON= -Iinclude/
BLASLIB = openblas

DEPS = $(wildcard src/*.h) $(wildcard include/*.h) Makefile

make: $(OBJDIR)gpt2.o
	$(CC) $(COMMON) main.c $^ -o gpt2_mine.out -l$(BLASLIB) -lm

$(OBJDIR)gpt2.o: $(OBJDIR)
	$(CC) $(COMMON) src/gpt2.c -c -o $(OBJDIR)gpt2.o -l$(BLASLIB) -lm

$(OBJDIR):
	mkdir obj

clean:
	rm -rf $(OBJDIR) gpt2_mine.out