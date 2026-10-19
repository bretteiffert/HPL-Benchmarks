// Seven-point stencil device code. laplacian_kernel is the original
// kernel.cu kernel, unchanged. test_function_kernel (the quadratic input
// field, from laplacian.cu) has one forced edit for --half: it computes in
// SetupType<T> and stores (T)(...). SetupType<T> is T for float and double,
// so their arithmetic is exactly the original; __half has no int * __half
// operator, so for float16 the field is computed in float and rounded once.

#pragma once

#include <cstddef>

#include <cuda_runtime.h>
#include <cuda_fp16.h>

template <typename T> struct SetupType { using type = T; };
template <> struct SetupType<__half> { using type = float; };

template <typename T>
__global__ void laplacian_kernel(T * f, const T * u, int nx, int ny, int nz, T invhx2, T invhy2, T invhz2, T invhxyz2) {

    int i = threadIdx.x + blockIdx.x * blockDim.x;
    int j = threadIdx.y + blockIdx.y * blockDim.y;
    int k = threadIdx.z + blockIdx.z * blockDim.z;

    if (i == 0 || i >= nx - 1 ||
        j == 0 || j >= ny - 1 ||
        k == 0 || k >= nz - 1)
        return;

    const int slice = nx * ny;
    size_t pos = i + nx * j + slice * k;

    f[pos] = u[pos] * invhxyz2
           + (u[pos - 1]     + u[pos + 1]) * invhx2
           + (u[pos - nx]    + u[pos + nx]) * invhy2
           + (u[pos - slice] + u[pos + slice]) * invhz2;
}

template <typename T>
__global__ void test_function_kernel(T *u, int nx, int ny, int nz, T hx, T hy, T hz) {

    int i = threadIdx.x + blockIdx.x * blockDim.x;
    int j = threadIdx.y + blockIdx.y * blockDim.y;
    int k = threadIdx.z + blockIdx.z * blockDim.z;

    if (i >= nx ||
        j >= ny ||
        k >= nz)
        return;

    size_t pos = i + nx * (j +  ny * k);

    using C = typename SetupType<T>::type;   // T, or float for __half
    C c = 0.5;
    C x = i*(C)hx;
    C y = j*(C)hy;
    C z = k*(C)hz;
    C Lx = nx*(C)hx;
    C Ly = ny*(C)hy;
    C Lz = nz*(C)hz;
    u[pos] = (T)(c * x * (x - Lx) + c * y * (y - Ly) + c * z * (z - Lz));
}
