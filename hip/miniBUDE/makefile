SYSTEM_ARCH := $(shell amd-smi static --asic --json 2>/dev/null | grep -oE 'gfx[0-9a-f]+' | head -n 1)

CC = hipcc

FLAGS=-std=c++17 -O3 -march=native --offload-arch=$(SYSTEM_ARCH)

SRCS = main.cpp
HEADERS = bude.h fasten.hpp metrics.h
TARGET = bude-hip

all: $(TARGET)

$(TARGET): $(SRCS) $(HEADERS)
	$(CC) $(FLAGS) -o $@ $(SRCS)

clean:
	rm -f $(TARGET)