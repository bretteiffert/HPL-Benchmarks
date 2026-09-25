SYSTEM_ARCH := $(shell amd-smi static --asic --json 2>/dev/null | grep -oE 'gfx[0-9a-f]+' | head -n 1)

CC = hipcc
FLAGS=-O3 -std=c++17 -march=native --offload-arch=$(SYSTEM_ARCH)

SRCS = main.cpp HIPStream.cpp
HEADERS = Stream.h HIPStream.h
TARGET = babelstream_hip

all: $(TARGET)

$(TARGET): $(SRCS) $(HEADERS)
	$(CC) $(FLAGS) -o $@ $(SRCS)

clean:
	rm -f $(TARGET)
