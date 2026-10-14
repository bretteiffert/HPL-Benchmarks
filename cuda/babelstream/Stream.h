// Copyright (c) 2015-16 Tom Deakin, Simon McIntosh-Smith,
// University of Bristol HPC
//
// For full license terms please see the LICENSE file distributed with this
// source code

// Defines the blocking BabelStream kernel interface, initial array values, and device-query helpers. Implementations expose the most recent device-measured kernel duration in seconds and provide host/device array transfers.

#pragma once

#include <vector>
#include <string>

#define startA (0.1)
#define startB (0.2)
#define startC (0.0)
#define startScalar (0.4)

template <class T>
class Stream
{
  public:

    virtual ~Stream(){}

    virtual void copy() = 0;
    virtual void mul() = 0;
    virtual void add() = 0;
    virtual void triad() = 0;
    virtual T dot() = 0;

    virtual double getTimeTaken() const = 0;

    virtual void init_arrays(T initA, T initB, T initC) = 0;
    virtual void read_arrays(std::vector<T>& a, std::vector<T>& b, std::vector<T>& c) = 0;

};


void listDevices(void);
std::string getDeviceName(const int);
std::string getDeviceDriver(const int);
