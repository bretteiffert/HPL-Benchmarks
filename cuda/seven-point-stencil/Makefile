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
# Seven-point stencil - CUDA
#
#   make                     build for the GPU in this machine
#   make ARCH=90             build for sm_90 without querying a GPU

NVCC ?= nvcc
# Host compiler pinned so an exported CC/CXX such as nvc does not leak in.
HOST_CXX ?= g++

ARCH ?= $(shell nvidia-smi --query-gpu=compute_cap --format=csv,noheader 2>/dev/null | head -n 1 | tr -d '.')

ENERGY_DIR ?= ../../dependencies
ENERGY_FLAGS = -I$(ENERGY_DIR) -L$(ENERGY_DIR) -Xlinker -rpath -Xlinker '$$ORIGIN/../../dependencies' -lkernelenergy

FLAGS = -ccbin $(HOST_CXX) -O3 -std=c++17 -arch=sm_$(ARCH) $(EXTRA_FLAGS)

SRCS = main.cpp cli.cpp report.cpp energy.cpp backend.cpp stencil.cu
HEADERS = backend.h cli.h energy.h kernels.cuh report.h stencil.h $(ENERGY_DIR)/kernel_energy_capi.h
TARGET = laplacian_kernel

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
