CC = gcc
CFLAGS = -O2 -std=c11 -Wall -lm
TARGET = chebyshev_solver

all: $(TARGET)

$(TARGET): chebyshev_solver.c
	$(CC) $(CFLAGS) -o $(TARGET) chebyshev_solver.c

run: $(TARGET)
	./$(TARGET)

clean:
	rm -f $(TARGET)