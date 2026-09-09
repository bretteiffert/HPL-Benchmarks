CC = nvcc -x cu

# Builds the CUDA miniBUDE executable for sm_90 with optimization and C++17 enabled. No fast-math option is present in the active flags, and the clean target removes the bude-cuda binary.

FLAGS=-std=c++17 -forward-unknown-to-host-compiler -O3 -extended-lambda -arch=sm_90

SRCS = main.cpp
HEADERS = bude.h fasten.hpp metrics.h
TARGET = bude-cuda

all: $(TARGET)

$(TARGET): $(SRCS) $(HEADERS)
	$(CC) $(FLAGS) -o $@ $(SRCS)

clean:
	rm -f $(TARGET)
