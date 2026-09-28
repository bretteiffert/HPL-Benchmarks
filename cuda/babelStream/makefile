SYSTEM_ARCH := $(shell nvidia-smi --query-gpu=compute_cap --format=csv,noheader | head -n 1 | tr -d '.')

CC = nvcc
FLAGS=-forward-unknown-to-host-compiler -arch=sm_$(SYSTEM_ARCH)  -DNDEBUG -O3 -march=native -std=c++11

SRCS = main.cpp CUDAStream.cu
HEADERS = Stream.h CUDAStream.h
TARGET = babelstream_cuda

all: $(TARGET)

$(TARGET): $(SRCS) $(HEADERS)
	$(CC) $(FLAGS) -o $@ $(SRCS)

clean:
	rm -f $(TARGET)
