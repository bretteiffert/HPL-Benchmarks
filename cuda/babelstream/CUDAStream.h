// Copyright (c) 2015-16 Tom Deakin, Simon McIntosh-Smith,
// University of Bristol HPC
//
// For full license terms please see the LICENSE file distributed with this
// source code

// Declares the CUDA BabelStream implementation, its device buffers, reduction storage, and event-based timing state. FMAD selects explicit fused or separate multiply-add expressions, while warmup uses scratch buffers so one-off launch costs are excluded without touching the timed arrays.

#pragma once

#include <iostream>
#include <stdexcept>
#include <sstream>

#include <cuda_runtime.h>

#include "Stream.h"

#define IMPLEMENTATION_STRING "CUDA"

#define TBSIZE 1024

#ifndef FMAD
#define FMAD 1
#endif

template <class T>
class CUDAStream : public Stream<T>
{
  protected:
    int array_size;

    T *sums;

    T *d_a;
    T *d_b;
    T *d_c;
    T *d_sum;

    int dot_num_blocks;

    cudaEvent_t start_ev;
    cudaEvent_t stop_ev;

    double last_kernel_time;

    void record_start();
    void record_stop();

    void warmup();

  public:

    CUDAStream(const int, const int);
    ~CUDAStream();

    virtual void copy() override;
    virtual void add() override;
    virtual void mul() override;
    virtual void triad() override;
    virtual T dot() override;

    virtual double getTimeTaken() const override;

    virtual void init_arrays(T initA, T initB, T initC) override;
    virtual void read_arrays(std::vector<T>& a, std::vector<T>& b, std::vector<T>& c) override;

};
