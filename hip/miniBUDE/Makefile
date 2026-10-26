# miniBUDE - HIP
#
#   make                          build for the GPU in this machine
#   make ARCH=gfx90a              build for gfx90a without querying a GPU
#   make EXTRA_FLAGS=-DUSE_SHARED forcefield in shared memory
#                                 (also -DNO_RESTRICT, -DMANAGED, -DPAGEFAULT)

HIP_PATH ?= /opt/rocm
HIPCC ?= $(HIP_PATH)/bin/hipcc

# First GPU target reported by amdgpu-arch, falling back to rocminfo.
ARCH ?= $(shell ( $(HIP_PATH)/llvm/bin/amdgpu-arch 2>/dev/null || amdgpu-arch 2>/dev/null || \
                  rocminfo 2>/dev/null | grep -o -m 1 'gfx[0-9a-f]*' ) | head -n 1)

ENERGY_DIR ?= ../../dependencies
ENERGY_FLAGS = -I$(ENERGY_DIR) -L$(ENERGY_DIR) -Wl,-rpath,'$$ORIGIN/../../dependencies' -lkernelenergy

FLAGS = --offload-arch=$(ARCH) -std=c++17 -O3 $(EXTRA_FLAGS)

SRCS = main.cpp cli.cpp deck.cpp report.cpp energy.cpp backend.cpp bude.hip
HEADERS = backend.h bude.h cli.h deck.h energy.h kernels.hip.h metrics.h report.h $(ENERGY_DIR)/kernel_energy_capi.h
TARGET = bude-hip

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
