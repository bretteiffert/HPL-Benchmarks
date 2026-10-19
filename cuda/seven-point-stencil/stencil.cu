// Seven-point stencil on CUDA: the host half of the original laplacian.cu and
// kernel.cu (test-field launch, laplacian() wrapper, event-timed launches,
// correctness metrics). The correctness loop is the original one, but it
// reads f back one z-slice at a time instead of copying the whole grid.
// Nothing here prints; failures throw.

#include "stencil.h"
#include "kernels.cuh"

#include <cmath>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

void api_check(cudaError_t err, const char* what)
{
  if (err != cudaSuccess)
    throw std::runtime_error(std::string("CUDA ") + what + ": " + cudaGetErrorString(err));
}

// Host conversions. For float and double these are plain casts; __half goes
// through float because host __half arithmetic is not guaranteed.
template <typename T> struct Host
{
  static T from_double(double x) { return (T)x; }
  static double to_double(T x) { return static_cast<double>(x); }
};
template <> struct Host<__half>
{
  static __half from_double(double x) { return __float2half((float)x); }
  static double to_double(__half x) { return static_cast<double>(__half2float(x)); }
};

template <typename T>
class CUDAStencil : public Stencil
{
  public:
    CUDAStencil(const Dims& n, const Dims& b);
    ~CUDAStencil() override;

    void first_run() override;
    double timed_run() override;
    Correctness check() const override;

  private:
    int nx, ny, nz;
    int BLK_X, BLK_Y, BLK_Z;
    T hx, hy, hz;
    T *d_u = nullptr, *d_f = nullptr;
    cudaEvent_t start = nullptr, stop = nullptr;

    void test_function();
    void laplacian();
    void release();
};

template <typename T>
CUDAStencil<T>::CUDAStencil(const Dims& n, const Dims& b)
  : nx(int(n.x)), ny(int(n.y)), nz(int(n.z)), BLK_X(int(b.x)), BLK_Y(int(b.y)), BLK_Z(int(b.z))
{
    // precision hx = 1.0 / (nx - 1), ... as in the original main()
    hx = Host<T>::from_double(1.0 / (nx - 1));
    hy = Host<T>::from_double(1.0 / (ny - 1));
    hz = Host<T>::from_double(1.0 / (nz - 1));

    const size_t numbytes = n.count() * sizeof(T);
    try
    {
        api_check( cudaMalloc((void**)&d_u, numbytes), "cudaMalloc (u)" );
        api_check( cudaMalloc((void**)&d_f, numbytes), "cudaMalloc (f)" );
        api_check( cudaEventCreate(&start), "cudaEventCreate" );
        api_check( cudaEventCreate(&stop), "cudaEventCreate" );
    }
    catch (...)
    {
        release();
        throw;
    }
}

template <typename T>
CUDAStencil<T>::~CUDAStencil()
{
    release();
}

template <typename T>
void CUDAStencil<T>::release()
{
    // Best effort: never throw from cleanup.
    if (d_u) (void)cudaFree(d_u);
    if (d_f) (void)cudaFree(d_f);
    if (start) (void)cudaEventDestroy(start);
    if (stop) (void)cudaEventDestroy(stop);
    d_u = d_f = nullptr;
    start = stop = nullptr;
}

// The original test_function(): fixed TBSize-wide blocks over x, one block
// row per (y, z).
template <typename T>
void CUDAStencil<T>::test_function()
{
    dim3 block(kSetupBlock, 1);
    dim3 grid((nx - 1) / block.x + 1, ny, nz);

    test_function_kernel<<<grid, block>>>(d_u, nx, ny, nz, hx, hy, hz);
    api_check( cudaGetLastError(), "test_function_kernel launch" );
}

// The original laplacian() wrapper. The coefficients are computed in
// SetupType<T> (T itself for float and double, float for __half).
template <typename T>
void CUDAStencil<T>::laplacian()
{
    using C = typename SetupType<T>::type;
    dim3 block(BLK_X, BLK_Y, BLK_Z);
    dim3 grid((nx - 1) / block.x + 1, (ny - 1) / block.y + 1, (nz - 1) / block.z + 1);
    C invhx2 = (C)1./(C)hx/(C)hx;
    C invhy2 = (C)1./(C)hy/(C)hy;
    C invhz2 = (C)1./(C)hz/(C)hz;
    C invhxyz2 = -2. * (invhx2 + invhy2 + invhz2);

    laplacian_kernel<<<grid, block>>>(d_f, d_u, nx, ny, nz, (T)invhx2, (T)invhy2, (T)invhz2, (T)invhxyz2);
    api_check( cudaGetLastError(), "laplacian_kernel launch" );
}

template <typename T>
void CUDAStencil<T>::first_run()
{
    test_function();
    laplacian();
    api_check( cudaDeviceSynchronize(), "cudaDeviceSynchronize (first run)" );
}

template <typename T>
double CUDAStencil<T>::timed_run()
{
    float elapsed = 0.0f;
    api_check( cudaDeviceSynchronize(), "cudaDeviceSynchronize" );
    api_check( cudaEventRecord(start), "cudaEventRecord" );
    laplacian();
    api_check( cudaEventRecord(stop), "cudaEventRecord" );
    api_check( cudaEventSynchronize(stop), "cudaEventSynchronize" );
    api_check( cudaEventElapsedTime(&elapsed, start, stop), "cudaEventElapsedTime" );
    if (!(elapsed > 0.0f))
        throw std::runtime_error("device timer returned no measurement for the Laplacian");
    return (double)elapsed * 1.0E-3;
}

template <typename T>
Correctness CUDAStencil<T>::check() const
{
    Correctness res{0.0, 0.0, 0.0, 0};

    const size_t snx = size_t(nx), sny = size_t(ny), snz = size_t(nz);
    const size_t slice = snx * sny;
    std::vector<T> h_f(slice);

    const double expected = 3.0;
    double sum_abs = 0.0;
    double sum_sq_err = 0.0;
    double sum_sq_exact = 0.0;
    double max_err = 0.0;
    size_t count = 0;

    for (size_t k = 1; k + 1 < snz; ++k) {
        api_check( cudaMemcpy(h_f.data(), d_f + slice * k, slice * sizeof(T), cudaMemcpyDeviceToHost),
                   "cudaMemcpy (f slice)" );
        for (size_t j = 1; j + 1 < sny; ++j) {
            for (size_t i = 1; i + 1 < snx; ++i) {
                size_t pos = i + snx * j;
                double val = Host<T>::to_double(h_f[pos]);
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

} // namespace

std::unique_ptr<Stencil> make_stencil(Precision p, const Dims& n, const Dims& block)
{
    switch (p)
    {
        case Precision::Double: return std::unique_ptr<Stencil>(new CUDAStencil<double>(n, block));
        case Precision::Float:  return std::unique_ptr<Stencil>(new CUDAStencil<float>(n, block));
        case Precision::Half:   return std::unique_ptr<Stencil>(new CUDAStencil<__half>(n, block));
    }
    throw std::runtime_error("unknown precision");
}

LaunchLimits launch_limits(int device)
{
    cudaDeviceProp props;
    api_check( cudaGetDeviceProperties(&props, device), "cudaGetDeviceProperties" );
    LaunchLimits l;
    l.max_block = {size_t(props.maxThreadsDim[0]), size_t(props.maxThreadsDim[1]), size_t(props.maxThreadsDim[2])};
    l.max_grid = {size_t(props.maxGridSize[0]), size_t(props.maxGridSize[1]), size_t(props.maxGridSize[2])};
    return l;
}
