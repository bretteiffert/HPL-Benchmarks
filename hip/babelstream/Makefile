# BabelStream - HIP
#
#   make                     build for the GPU in this machine
#   make ARCH=gfx90a         build for gfx90a without querying a GPU
#   make EXTRA_FLAGS=-DFMAD=0
#

HIP_PATH ?= /opt/rocm
HIPCC ?= $(HIP_PATH)/bin/hipcc

# First GPU target reported by amdgpu-arch, falling back to rocminfo.
ARCH ?= $(shell ( $(HIP_PATH)/llvm/bin/amdgpu-arch 2>/dev/null || amdgpu-arch 2>/dev/null || \
                  rocminfo 2>/dev/null | grep -o -m 1 'gfx[0-9a-f]*' ) | head -n 1)

ENERGY_DIR ?= ../../dependencies
ENERGY_FLAGS = -I$(ENERGY_DIR) -L$(ENERGY_DIR) -Wl,-rpath,'$$ORIGIN/../../dependencies' -lkernelenergy

FLAGS = --offload-arch=$(ARCH) -DNDEBUG -O3 -march=native -std=c++17 $(EXTRA_FLAGS)

SRCS = main.cpp cli.cpp report.cpp energy.cpp backend.cpp stream.hip
HEADERS = backend.h cli.h energy.h kernels.hip.h report.h stream.h $(ENERGY_DIR)/kernel_energy_capi.h
TARGET = babelstream_hip

all: $(TARGET)

$(TARGET): $(SRCS) $(HEADERS)
	@if [ -z "$(ARCH)" ]; then \
	  echo "error: could not detect the GPU architecture with amdgpu-arch or rocminfo; pass it, e.g. make ARCH=gfx90a" >&2; \
	  exit 1; \
	fi
	$(HIPCC) $(FLAGS) -o $@ $(SRCS) $(ENERGY_FLAGS) -ldl

clean:
	rm -f $(TARGET)

.PHONY: all clean
