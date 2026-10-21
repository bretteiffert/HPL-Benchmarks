# Hartree-Fock - CUDA
#
#   make                     build for the GPU in this machine
#   make ARCH=90             build for sm_90 without querying a GPU

NVCC ?= nvcc
# Host compiler pinned so an exported CC/CXX such as nvc does not leak in.
HOST_CXX ?= g++

ARCH ?= $(shell nvidia-smi --query-gpu=compute_cap --format=csv,noheader 2>/dev/null | head -n 1 | tr -d '.')

ENERGY_DIR ?= ../../dependencies
ENERGY_FLAGS = -I$(ENERGY_DIR) -L$(ENERGY_DIR) -Xlinker -rpath -Xlinker '$$ORIGIN/../../dependencies' -lkernelenergy

FLAGS = -ccbin $(HOST_CXX) -O3 -arch=sm_$(ARCH) -std=c++17 $(EXTRA_FLAGS)

SRCS = main.cpp cli.cpp deck.cpp report.cpp energy.cpp backend.cpp hf.cu
HEADERS = backend.h cli.h deck.h energy.h hf.h kernels.cuh report.h $(ENERGY_DIR)/kernel_energy_capi.h
TARGET = hartree-fock-cuda

all: $(TARGET)

$(TARGET): $(SRCS) $(HEADERS)
	@if [ -z "$(ARCH)" ]; then \
	  echo "error: could not detect the GPU architecture with nvidia-smi; pass it, e.g. make ARCH=90" >&2; \
	  exit 1; \
	fi
	$(NVCC) $(FLAGS) -o $@ $(SRCS) $(ENERGY_FLAGS) -ldl

clean:
	rm -f $(TARGET)

.PHONY: all clean
