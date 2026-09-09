// Benchmarks a HIP seven-point finite-difference Laplacian on an equidistant 3D grid, based on Claire Winogrodzki's JACC-repro stencil. A quadratic test field has analytical interior Laplacian 3, enabling max, mean, and relative L2 correctness metrics. HIP events measure each kernel launch after synchronization; the preceding synchronization orders work but does not flush caches.

#include "hip/hip_runtime.h"
#include <iostream>
#include <math.h>
#include <cmath>
#include <limits>
#include <stdio.h>
#include <assert.h>
#include <vector>
#include <string>
#include "kernel.hpp"

using precision = double;
using namespace std;

char precision_str[] = "double";

const int TBSize = 1024;
const int L = 1024;
const int NUM_ITER = 100;
bool csv_output = false;

#define CUDA_CHECK(stat)                                           \
{                                                                 \
    if(stat != hipSuccess)                                        \
    {                                                             \
        std::cerr << "CUDA error: " << hipGetErrorString(stat) <<  \
        " in file " << __FILE__ << ":" << __LINE__ << std::endl;  \
        exit(-1);                                                 \
    }                                                             \
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

    T c = 0.5;
    T x = i*hx;
    T y = j*hy;
    T z = k*hz;
    T Lx = nx*hx;
    T Ly = ny*hy;
    T Lz = nz*hz;
    u[pos] = c * x * (x - Lx) + c * y * (y - Ly) + c * z * (z - Lz);
}

template <typename T>
void test_function(T *d_f, int nx, int ny, int nz, T hx, T hy, T hz) {

    dim3 block(TBSize, 1);
    dim3 grid((nx - 1) / block.x + 1, ny, nz);

    test_function_kernel<<<grid, block>>>(d_f, nx, ny, nz, hx, hy, hz);
    CUDA_CHECK( hipGetLastError() );
}

struct CorrectnessResult {
    double max_abs_err;
    double l2_rel_err;
    double mean_abs_err;
    size_t num_checked;
};

template <typename T>
CorrectnessResult check_correctness(T *d_f, size_t nx, size_t ny, size_t nz) {

    CorrectnessResult res{0.0, 0.0, 0.0, 0};

    size_t n = nx * ny * nz;
    std::vector<T> h_f(n);
    CUDA_CHECK( hipMemcpy(h_f.data(), d_f, n * sizeof(T), hipMemcpyDeviceToHost) );

    const double expected = 3.0;
    double sum_abs = 0.0;
    double sum_sq_err = 0.0;
    double sum_sq_exact = 0.0;
    double max_err = 0.0;
    size_t count = 0;

    for (size_t k = 1; k + 1 < nz; ++k) {
        for (size_t j = 1; j + 1 < ny; ++j) {
            for (size_t i = 1; i + 1 < nx; ++i) {
                size_t pos = i + nx * (j + ny * k);
                double val = static_cast<double>(h_f[pos]);
                double err = std::fabs(val - expected);
                sum_abs += err;
                sum_sq_err += err * err;
                sum_sq_exact += expected * expected;
                if (err > max_err) max_err = err;
                ++count;
            }
        }
    }

    res.num_checked = count;
    if (count > 0) {
        res.max_abs_err = max_err;
        res.mean_abs_err = sum_abs / count;
        res.l2_rel_err = (sum_sq_exact > 0.0) ? std::sqrt(sum_sq_err / sum_sq_exact) : std::sqrt(sum_sq_err);
    }

    return res;
}

int main(int argc, char **argv){

    int BLK_X = TBSize;
    int BLK_Y = 1;
    int BLK_Z = 1;

    size_t nx = L, ny = L, nz = L;

    if (argc > 1) nx = atoi(argv[1]);
    if (argc > 2) ny = atoi(argv[2]);
    if (argc > 3) nz = atoi(argv[3]);
    if (argc > 4) BLK_X = atoi(argv[4]);
    if (argc > 5) BLK_Y = atoi(argv[5]);
    if (argc > 6) BLK_Z = atoi(argv[6]);
    if (argc > 7 && string(argv[7]) == "--csv") csv_output = true;

    size_t theoretical_fetch_size = (nx * ny * nz - 8 - 4 * (nx - 2) - 4 * (ny - 2) - 4 * (nz - 2) ) * sizeof(precision);
    size_t theoretical_write_size = ((nx - 2) * (ny - 2) * (nz - 2)) * sizeof(precision);
    size_t datasize = theoretical_fetch_size + theoretical_write_size;


    int device;
    CUDA_CHECK( hipGetDevice(&device) );
    hipDeviceProp_t props;
    CUDA_CHECK( hipGetDeviceProperties(&props, device) );

    if (csv_output) {
        cout << "backend,GPU,precision,L,blk_x,blk_y,blk_z,BW_GBs" << endl;
    }
    else {
        cout << props.name << endl;
        cout << "Precision: " << precision_str << endl;
        cout << "nx,ny,nz = " << nx << ", " << ny << ", " << nz << endl;
        cout << "block sizes = " << BLK_X << ", " << BLK_Y << ", " << BLK_Z << endl;
        cout << "Theoretical fetch size (GB): " << theoretical_fetch_size * 1e-9 << endl;
        cout << "Theoretical write size (GB): " << theoretical_write_size * 1e-9 << endl;
    }

    size_t numbytes = nx * ny * nz * sizeof(precision);

    precision *d_u, *d_f;

    CUDA_CHECK( hipMalloc((void**)&d_u, numbytes) );
    CUDA_CHECK( hipMalloc((void**)&d_f, numbytes) );

    precision hx = 1.0 / (nx - 1), hy = 1.0 / (ny - 1), hz = 1.0 / (nz - 1);

    test_function(d_u, nx, ny, nz, hx, hy, hz);

    laplacian(d_f, d_u, nx, ny, nz, BLK_X, BLK_Y, BLK_Z, hx, hy, hz);
    CUDA_CHECK( hipGetLastError() );
    CUDA_CHECK( hipDeviceSynchronize() );

    CorrectnessResult corr = check_correctness<precision>(d_f, nx, ny, nz);

    if (csv_output) {
        std::cerr << "correctness,precision,max_abs_err,l2_rel_err,mean_abs_err,points_checked" << std::endl;
        std::cerr << "correctness," << precision_str << ","
                   << corr.max_abs_err << "," << corr.l2_rel_err << ","
                   << corr.mean_abs_err << "," << corr.num_checked << std::endl;
    } else {
        cout << "--- Correctness metrics (precision = " << precision_str << ") ---" << endl;
        cout << "Interior points checked : " << corr.num_checked << endl;
        cout << "Max abs error           : " << corr.max_abs_err << endl;
        cout << "Mean abs error          : " << corr.mean_abs_err << endl;
        cout << "Relative L2 error       : " << corr.l2_rel_err << endl << endl;
    }

    float total_elapsed = 0;
    float elapsed;
    hipEvent_t start, stop;
    CUDA_CHECK( hipEventCreate(&start) );
    CUDA_CHECK( hipEventCreate(&stop)  );

    for (int iter = 0; iter < NUM_ITER; ++iter) {
        CUDA_CHECK( hipDeviceSynchronize()                     );
        CUDA_CHECK( hipEventRecord(start)                      );
        laplacian(d_f, d_u, nx, ny, nz, BLK_X, BLK_Y, BLK_Z, hx, hy, hz);
        CUDA_CHECK( hipGetLastError()                          );
        CUDA_CHECK( hipEventRecord(stop)                       );
        CUDA_CHECK( hipEventSynchronize(stop)                  );
        CUDA_CHECK( hipEventElapsedTime(&elapsed, start, stop) );
        if (csv_output) {
            printf("CUDA,%s,%s,%zu,%d,%d,%d,%g\n", props.name, precision_str,
            nx, BLK_X, BLK_Y, BLK_Z, datasize / elapsed / 1e6);
        }
        total_elapsed += elapsed;
    }

    if (!csv_output) {
        printf("Laplacian kernel took: %g ms, effective memory bandwidth: %g GB/s \n\n",
                total_elapsed / NUM_ITER,
                datasize * NUM_ITER / total_elapsed / 1e6
                );
    }

    CUDA_CHECK( hipFree(d_f) );
    CUDA_CHECK( hipFree(d_u) );

    return 0;
}
