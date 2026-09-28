SYSTEM_ARCH := $(shell nvidia-smi --query-gpu=compute_cap --format=csv,noheader | head -n 1 | tr -d '.')

CC = nvcc
FLAGS = -O3 -arch=sm_$(SYSTEM_ARCH)

SRCS = hartree-fock.cu
TARGET = hartree-fock-cuda

all: $(TARGET)

$(TARGET): $(SRCS)
	$(CC) $(FLAGS) -o $@ $(SRCS)

clean:
	rm -f $(TARGET)

