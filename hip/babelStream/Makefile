# Builds the HIP BabelStream executable for gfx942 with hipcc. The clean target removes the resulting babelstream_hip binary.

CC = hipcc
FLAGS=-O3 -std=c++17 -march=native --offload-arch=gfx942

SRCS = main.cpp HIPStream.cpp
HEADERS = Stream.h HIPStream.h
TARGET = babelstream_hip

all: $(TARGET)

$(TARGET): $(SRCS) $(HEADERS)
	$(CC) $(FLAGS) -o $@ $(SRCS)

clean:
	rm -f $(TARGET)
