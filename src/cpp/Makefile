CC=g++
CFLAGS=-Wall -Wextra -pedantic -std=c++14
LDFLAGS=

SRCS=main.cpp kdtree.cpp point.cpp
OBJS=$(SRCS:.cpp=.o)
DEPS=kdtree.h point.h
TARGET=my_program

all: $(TARGET)

$(TARGET): $(OBJS)
	$(CC) $(LDFLAGS) $^ -o $@

%.o: %.cpp $(DEPS)
	$(CC) $(CFLAGS) -c $< -o $@

clean:
	rm -f $(OBJS) $(TARGET)
