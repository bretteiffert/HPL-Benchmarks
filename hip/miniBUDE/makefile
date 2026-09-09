CC = hipcc

FLAGS=-std=c++17 -O3 -march=native --offload-arch=gfx942

SRCS = main.cpp
HEADERS = bude.h fasten.hpp metrics.h
TARGET = bude-hip

all: $(TARGET)

$(TARGET): $(SRCS) $(HEADERS)
	$(CC) $(FLAGS) -o $@ $(SRCS)

clean:
	rm -f $(TARGET)