// Copyright (c) 2015-16 Tom Deakin, Simon McIntosh-Smith,
// University of Bristol HPC
//
// For full license terms please see the LICENSE file distributed with this
// source code

// Implements HIP BabelStream kernels with explicit FMAD control, blocking HIP-event timing, and a partial-sum dot reduction whose measured region includes device-to-host transfer but not the final host sum. Construction validates capacity and launch geometry, then warms every kernel on separate scratch storage before timed initialization.

#include "hip/hip_runtime.h"
#include "HIPStream.h"

void check_error(void)
{
  hipError_t err = hipGetLastError();
  if (err != hipSuccess)
  {
    std::cerr << "Error: " << hipGetErrorString(err) << std::endl;
    exit(err);
  }
}

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

template <class T>
CUDAStream<T>::CUDAStream(const int ARRAY_SIZE, const int device_index)
{

  if (ARRAY_SIZE % TBSIZE != 0)
  {
    std::stringstream ss;
    ss << "Array size must be a multiple of " << TBSIZE;
    throw std::runtime_error(ss.str());
  }

  int count;
  hipGetDeviceCount(&count);
  check_error();
  if (device_index >= count)
    throw std::runtime_error("Invalid device index");
  hipSetDevice(device_index);
  check_error();

  std::cout << "Using CUDA device " << getDeviceName(device_index) << std::endl;
  std::cout << "Driver: " << getDeviceDriver(device_index) << std::endl;
  array_size = ARRAY_SIZE;


  hipDeviceProp_t props;
  hipGetDeviceProperties(&props, device_index);
  check_error();
  dot_num_blocks = props.multiProcessorCount * 4;

  sums = (T*)malloc(sizeof(T) * dot_num_blocks);

  size_t array_bytes = sizeof(T);
  array_bytes *= ARRAY_SIZE;
  size_t total_bytes = array_bytes * 4;
  std::cout << "Reduction kernel config: " << dot_num_blocks << " groups of (fixed) size " << TBSIZE << std::endl;

  if (props.totalGlobalMem < total_bytes)
    throw std::runtime_error("Device does not have enough memory for all 3 buffers");

  hipMalloc(&d_a, array_bytes);
  check_error();
  hipMalloc(&d_b, array_bytes);
  check_error();
  hipMalloc(&d_c, array_bytes);
  check_error();
  hipMalloc(&d_sum, dot_num_blocks*sizeof(T));
  check_error();

  hipEventCreate(&start_ev);
  check_error();
  hipEventCreate(&stop_ev);
  check_error();

  last_kernel_time = 0.0;

  warmup();
}


template <class T>
CUDAStream<T>::~CUDAStream()
{
  free(sums);

  hipFree(d_a);
  check_error();
  hipFree(d_b);
  check_error();
  hipFree(d_c);
  check_error();
  hipFree(d_sum);
  check_error();

  hipEventDestroy(start_ev);
  check_error();
  hipEventDestroy(stop_ev);
  check_error();
}


template <class T>
void CUDAStream<T>::record_start()
{
  hipEventRecord(start_ev);
  check_error();
}

template <class T>
void CUDAStream<T>::record_stop()
{
  hipEventRecord(stop_ev);
  check_error();

  hipEventSynchronize(stop_ev);
  check_error();

  float elapsed_ms = 0.0f;
  hipEventElapsedTime(&elapsed_ms, start_ev, stop_ev);
  check_error();

  last_kernel_time = static_cast<double>(elapsed_ms) * 1.0E-3;
}

template <class T>
double CUDAStream<T>::getTimeTaken() const
{
  return last_kernel_time;
}


template <typename T>
__global__ void init_kernel(T * a, T * b, T * c, T initA, T initB, T initC)
{
  const int i = blockDim.x * blockIdx.x + threadIdx.x;
  a[i] = initA;
  b[i] = initB;
  c[i] = initC;
}

template <class T>
void CUDAStream<T>::init_arrays(T initA, T initB, T initC)
{
  record_start();
  init_kernel<<<array_size/TBSIZE, TBSIZE>>>(d_a, d_b, d_c, initA, initB, initC);
  check_error();
  record_stop();
}

template <class T>
void CUDAStream<T>::read_arrays(std::vector<T>& a, std::vector<T>& b, std::vector<T>& c)
{
  hipMemcpy(a.data(), d_a, a.size()*sizeof(T), hipMemcpyDeviceToHost);
  check_error();
  hipMemcpy(b.data(), d_b, b.size()*sizeof(T), hipMemcpyDeviceToHost);
  check_error();
  hipMemcpy(c.data(), d_c, c.size()*sizeof(T), hipMemcpyDeviceToHost);
  check_error();
}


template <typename T>
__global__ void copy_kernel(const T * a, T * c)
{
  const int i = blockDim.x * blockIdx.x + threadIdx.x;
  c[i] = a[i];
}

template <class T>
void CUDAStream<T>::copy()
{
  record_start();
  copy_kernel<<<array_size/TBSIZE, TBSIZE>>>(d_a, d_c);
  check_error();
  record_stop();
}

template <typename T>
__global__ void mul_kernel(T * b, const T * c)
{
  const T scalar = startScalar;
  const int i = blockDim.x * blockIdx.x + threadIdx.x;
  b[i] = scalar * c[i];
}

template <class T>
void CUDAStream<T>::mul()
{
  record_start();
  mul_kernel<<<array_size/TBSIZE, TBSIZE>>>(d_b, d_c);
  check_error();
  record_stop();
}

template <typename T>
__global__ void add_kernel(const T * a, const T * b, T * c)
{
  const int i = blockDim.x * blockIdx.x + threadIdx.x;
  c[i] = a[i] + b[i];
}

template <class T>
void CUDAStream<T>::add()
{
  record_start();
  add_kernel<<<array_size/TBSIZE, TBSIZE>>>(d_a, d_b, d_c);
  check_error();
  record_stop();
}

template <typename T>
__global__ void triad_kernel(T * a, const T * b, const T * c)
{
  const T scalar = startScalar;
  const int i = blockDim.x * blockIdx.x + threadIdx.x;
  a[i] = mad_op(scalar, c[i], b[i]);
}

template <class T>
void CUDAStream<T>::triad()
{
  record_start();
  triad_kernel<<<array_size/TBSIZE, TBSIZE>>>(d_a, d_b, d_c);
  check_error();
  record_stop();
}

template <class T>
__global__ void dot_kernel(const T * a, const T * b, T * sum, int array_size)
{
  __shared__ T tb_sum[TBSIZE];

  int i = blockDim.x * blockIdx.x + threadIdx.x;
  const size_t local_i = threadIdx.x;

  tb_sum[local_i] = {};
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

template <class T>
T CUDAStream<T>::dot()
{
  record_start();
  dot_kernel<<<dot_num_blocks, TBSIZE>>>(d_a, d_b, d_sum, array_size);
  check_error();

  hipMemcpy(sums, d_sum, dot_num_blocks*sizeof(T), hipMemcpyDeviceToHost);
  check_error();

  record_stop();

  T sum = 0.0;
  for (int i = 0; i < dot_num_blocks; i++)
  {
    sum += sums[i];
  }

  return sum;
}

template <class T>
void CUDAStream<T>::warmup()
{
  const int warmup_size = TBSIZE;

  T *w_a, *w_b, *w_c, *w_sum;
  hipMalloc(&w_a, sizeof(T) * warmup_size);
  check_error();
  hipMalloc(&w_b, sizeof(T) * warmup_size);
  check_error();
  hipMalloc(&w_c, sizeof(T) * warmup_size);
  check_error();
  hipMalloc(&w_sum, sizeof(T) * dot_num_blocks);
  check_error();

  init_kernel<<<1, TBSIZE>>>(w_a, w_b, w_c, (T)startA, (T)startB, (T)startC);
  check_error();
  copy_kernel<<<1, TBSIZE>>>(w_a, w_c);
  check_error();
  mul_kernel<<<1, TBSIZE>>>(w_b, w_c);
  check_error();
  add_kernel<<<1, TBSIZE>>>(w_a, w_b, w_c);
  check_error();
  triad_kernel<<<1, TBSIZE>>>(w_a, w_b, w_c);
  check_error();

  dot_kernel<<<dot_num_blocks, TBSIZE>>>(w_a, w_b, w_sum, warmup_size);
  check_error();
  hipMemcpy(sums, w_sum, dot_num_blocks*sizeof(T), hipMemcpyDeviceToHost);
  check_error();

  hipDeviceSynchronize();
  check_error();

  hipFree(w_a);
  check_error();
  hipFree(w_b);
  check_error();
  hipFree(w_c);
  check_error();
  hipFree(w_sum);
  check_error();
}

void listDevices(void)
{
  int count;
  hipGetDeviceCount(&count);
  check_error();

  if (count == 0)
  {
    std::cerr << "No devices found." << std::endl;
  }
  else
  {
    std::cout << std::endl;
    std::cout << "Devices:" << std::endl;
    for (int i = 0; i < count; i++)
    {
      std::cout << i << ": " << getDeviceName(i) << std::endl;
    }
    std::cout << std::endl;
  }
}


std::string getDeviceName(const int device)
{
  hipDeviceProp_t props;
  hipGetDeviceProperties(&props, device);
  check_error();
  return std::string(props.name);
}


std::string getDeviceDriver(const int device)
{
  hipSetDevice(device);
  check_error();
  int driver;
  hipDriverGetVersion(&driver);
  check_error();
  return std::to_string(driver);
}

template class CUDAStream<float>;
template class CUDAStream<double>;
