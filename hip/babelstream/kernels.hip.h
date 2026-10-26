// Copyright (c) 2015-16 Tom Deakin, Simon McIntosh-Smith,
// University of Bristol HPC
//
// For full license terms please see the LICENSE file distributed with this
// source code

// BabelStream device code only. The kernel bodies are unchanged from the
// original BabelStream kernels except one forced edit for
// --half: dot_kernel's "tb_sum[local_i] = {};" is now "= T{};". Both
// value-initialise the slot to zero, but __half has several assignment
// operators, so "= {}" does not compile for it. The other addition is a
// __half overload of mad_op.
//
// FMAD selects __fma*_rn or a separate multiply and add. hipcc (clang)
// contracts a * b + c into an FMA by default for HIP (-ffp-contract=fast-honor-pragmas),
// so FMAD=0 normally compiles to the same instruction and the switch is
// effectively a no-op.

#pragma once

#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>

#include "stream.h"   // startA, startB, startC, startScalar

// Shared-memory slots in dot_kernel. The launch block size must be a power
// of two no larger than this (validated on the host before launch).
#define TBSIZE 256

#ifndef FMAD
#define FMAD 1
#endif

__device__ __forceinline__ float mad_op(float a, float b, float c)
{
#if FMAD
  return __fmaf_rn(a, b, c);
#else
  return a * b + c;
#endif
}

__device__ __forceinline__ double mad_op(double a, double b, double c)
{
#if FMAD
  return __fma_rn(a, b, c);
#else
  return a * b + c;
#endif
}

__device__ __forceinline__ __half mad_op(__half a, __half b, __half c)
{
#if FMAD
  return __hfma(a, b, c);
#else
  return a * b + c;
#endif
}

template <typename T>
__global__ void init_kernel(T * a, T * b, T * c, T initA, T initB, T initC)
{
  const int i = blockDim.x * blockIdx.x + threadIdx.x;
  a[i] = initA;
  b[i] = initB;
  c[i] = initC;
}

template <typename T>
__global__ void copy_kernel(const T * a, T * c)
{
  const int i = blockDim.x * blockIdx.x + threadIdx.x;
  c[i] = a[i];
}

template <typename T>
__global__ void mul_kernel(T * b, const T * c)
{
  const T scalar = startScalar;
  const int i = blockDim.x * blockIdx.x + threadIdx.x;
  b[i] = scalar * c[i];
}

template <typename T>
__global__ void add_kernel(const T * a, const T * b, T * c)
{
  const int i = blockDim.x * blockIdx.x + threadIdx.x;
  c[i] = a[i] + b[i];
}

template <typename T>
__global__ void triad_kernel(T * a, const T * b, const T * c)
{
  const T scalar = startScalar;
  const int i = blockDim.x * blockIdx.x + threadIdx.x;
  a[i] = mad_op(scalar, c[i], b[i]);
}

template <class T>
__global__ void dot_kernel(const T * a, const T * b, T * sum, int array_size)
{
  __shared__ T tb_sum[TBSIZE];

  int i = blockDim.x * blockIdx.x + threadIdx.x;
  const size_t local_i = threadIdx.x;

  tb_sum[local_i] = T{};   // was '= {}', ambiguous for __half (see header comment)
  for (; i < array_size; i += blockDim.x*gridDim.x)
    tb_sum[local_i] = mad_op(a[i], b[i], tb_sum[local_i]);

  for (int offset = blockDim.x / 2; offset > 0; offset /= 2)
  {
    __syncthreads();
    if (local_i < offset)
    {
      tb_sum[local_i] += tb_sum[local_i+offset];
    }
  }

  if (local_i == 0)
    sum[blockIdx.x] = tb_sum[local_i];
}
