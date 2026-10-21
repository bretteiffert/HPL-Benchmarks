##############################################################################
#Copyright (c) 2022 Advanced Micro Devices, Inc. (AMD)
#
#Permission is hereby granted, free of charge, to any person obtaining a copy
#of this software and associated documentation files (the "Software"), to deal
#in the Software without restriction, including without limitation the rights
#to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
#copies of the Software, and to permit persons to whom the Software is
#furnished to do so, subject to the following conditions:
#
#The above copyright notice and this permission notice shall be included in all
#copies or substantial portions of the Software.
#
#THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
#IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
#FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
#AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
#LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
#OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
#SOFTWARE.
###############################################################################
# Seven-point stencil - HIP
#
#   make                     build for the GPU in this machine
#   make ARCH=gfx90a         build for gfx90a without querying a GPU

HIP_PATH ?= /opt/rocm
HIPCC ?= $(HIP_PATH)/bin/hipcc

# First GPU target reported by amdgpu-arch, falling back to rocminfo.
ARCH ?= $(shell ( $(HIP_PATH)/llvm/bin/amdgpu-arch 2>/dev/null || amdgpu-arch 2>/dev/null || \
                  rocminfo 2>/dev/null | grep -o -m 1 'gfx[0-9a-f]*' ) | head -n 1)

ENERGY_DIR ?= ../../dependencies
ENERGY_FLAGS = -I$(ENERGY_DIR) -L$(ENERGY_DIR) -Wl,-rpath,'$$ORIGIN/../../dependencies' -lkernelenergy

FLAGS = --offload-arch=$(ARCH) -O3 -std=c++17 $(EXTRA_FLAGS)

SRCS = main.cpp cli.cpp report.cpp energy.cpp backend.cpp stencil.hip
HEADERS = backend.h cli.h energy.h kernels.hip.h report.h stencil.h $(ENERGY_DIR)/kernel_energy_capi.h
TARGET = laplacian_kernel_hip

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
