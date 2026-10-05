SYSTEM_ARCH := $(shell amd-smi static --asic --json 2>/dev/null | grep -oE 'gfx[0-9a-f]+' | head -n 1)

CC = hipcc
FLAGS = -O3 -march=native --offload-arch=$(SYSTEM_ARCH)

SRCS = hartree-fock.cc
TARGET = hartree-fock-hip

all: $(TARGET)

$(TARGET): $(SRCS)
	$(CC) $(FLAGS) -o $@ $(SRCS)

clean:
	rm -f $(TARGET)

