// Implements the CUDA miniBUDE backend with precision-specific device math, parameter-level no-alias qualifiers, optional shared forcefield storage, and explicit allocation modes. Work is tiled by PPWI, with stragglers remapped to keep block synchronization valid and their redundant results suppressed. Reused CUDA events record kernel time while paired host timestamps capture launch and synchronization overhead; deck data is converted to the selected type before upload.

#pragma once

#include "bude.h"
#include <cmath>
#include <iostream>
#include <string>

#ifdef IMPL_CLS
  #error IMPL_CLS was already defined
#endif
#define IMPL_CLS CudaBude

#ifdef NO_RESTRICT
  #define BUDE_RESTRICT
#else
  #define BUDE_RESTRICT __restrict__
#endif

template <typename T> struct DeviceMath;

template <> struct DeviceMath<float> {
  __device__ static inline float sin(float x) { return sinf(x); }
  __device__ static inline float cos(float x) { return cosf(x); }
  __device__ static inline float sqrt(float x) { return sqrtf(x); }
  __device__ static inline float abs(float x) { return fabsf(x); }
  __device__ static inline float rcp(float x) { return 1.0f / x; }
};

template <> struct DeviceMath<double> {
  __device__ static inline double sin(double x) { return ::sin(x); }
  __device__ static inline double cos(double x) { return ::cos(x); }
  __device__ static inline double sqrt(double x) { return ::sqrt(x); }
  __device__ static inline double abs(double x) { return ::fabs(x); }
  __device__ static inline double rcp(double x) { return 1.0 / x; }
};

static __global__ void version(int *value) {
#ifdef __CUDA_ARCH__
  value[0] = __CUDA_ARCH__;
#else
  value[0] = -1;
#endif
}

template <typename T, size_t PPWI>
static __global__ void fasten_main(int natlig, int natpro, int ntypes,
                                   const AtomT<T> *BUDE_RESTRICT protein_molecule,
                                   const AtomT<T> *BUDE_RESTRICT ligand_molecule,
                                   const T *BUDE_RESTRICT transforms_0, const T *BUDE_RESTRICT transforms_1,
                                   const T *BUDE_RESTRICT transforms_2, const T *BUDE_RESTRICT transforms_3,
                                   const T *BUDE_RESTRICT transforms_4, const T *BUDE_RESTRICT transforms_5,
                                   T *BUDE_RESTRICT etotals,
                                   const FFParamsT<T> *BUDE_RESTRICT global_forcefield,
                                   int numTransforms) {
  using M = DeviceMath<T>;
  constexpr T ZERO = Consts<T>::ZERO;
  constexpr T QUARTER = Consts<T>::QUARTER;
  constexpr T HALF = Consts<T>::HALF;
  constexpr T ONE = Consts<T>::ONE;
  constexpr T TWO = Consts<T>::TWO;
  constexpr T FOUR = Consts<T>::FOUR;
  constexpr T CNSTNT = Consts<T>::CNSTNT;
  constexpr T HARDNESS = Consts<T>::HARDNESS;
  constexpr T NPNPDIST = Consts<T>::NPNPDIST;
  constexpr T NPPDIST = Consts<T>::NPPDIST;
  constexpr T MAXVAL = Consts<T>::MAXVAL;

  const int lsz = int(blockDim.x);

  const int base = int(blockIdx.x) * lsz * int(PPWI) + int(threadIdx.x);

  const int ix = base < numTransforms ? base : (numTransforms - int(PPWI) * lsz + int(threadIdx.x));

#ifdef USE_SHARED
  extern __shared__ char sharedRaw[];
  FFParamsT<T> *forcefield = reinterpret_cast<FFParamsT<T> *>(sharedRaw);
  for (int i = int(threadIdx.x); i < ntypes; i += lsz)
    forcefield[i] = global_forcefield[i];
  __syncthreads();
#else
  (void)ntypes;
  const FFParamsT<T> *BUDE_RESTRICT forcefield = global_forcefield;
#endif

  T etot[PPWI];
  Vec4<T> transform[PPWI][3];
  for (int i = 0; i < int(PPWI); i++) {
    const int index = ix + i * lsz;

    const T sx = M::sin(transforms_0[index]);
    const T cx = M::cos(transforms_0[index]);
    const T sy = M::sin(transforms_1[index]);
    const T cy = M::cos(transforms_1[index]);
    const T sz = M::sin(transforms_2[index]);
    const T cz = M::cos(transforms_2[index]);

    transform[i][0].x = cy * cz;
    transform[i][0].y = sx * sy * cz - cx * sz;
    transform[i][0].z = cx * sy * cz + sx * sz;
    transform[i][0].w = transforms_3[index];
    transform[i][1].x = cy * sz;
    transform[i][1].y = sx * sy * sz + cx * cz;
    transform[i][1].z = cx * sy * sz - sx * cz;
    transform[i][1].w = transforms_4[index];
    transform[i][2].x = -sy;
    transform[i][2].y = sx * cy;
    transform[i][2].z = cx * cy;
    transform[i][2].w = transforms_5[index];

    etot[i] = ZERO;
  }

  int il = 0;
  do {
    const AtomT<T> l_atom = ligand_molecule[il];

    const FFParamsT<T> l_params = forcefield[l_atom.type];
    const bool lhphb_ltz = l_params.hphb < ZERO;
    const bool lhphb_gtz = l_params.hphb > ZERO;

    Vec3<T> lpos[PPWI];
    const Vec4<T> linitpos{l_atom.x, l_atom.y, l_atom.z, ONE};
    for (int i = 0; i < int(PPWI); i++) {
      lpos[i].x = transform[i][0].w + linitpos.x * transform[i][0].x + linitpos.y * transform[i][0].y +
                  linitpos.z * transform[i][0].z;
      lpos[i].y = transform[i][1].w + linitpos.x * transform[i][1].x + linitpos.y * transform[i][1].y +
                  linitpos.z * transform[i][1].z;
      lpos[i].z = transform[i][2].w + linitpos.x * transform[i][2].x + linitpos.y * transform[i][2].y +
                  linitpos.z * transform[i][2].z;
    }

    int ip = 0;
    do {
      const AtomT<T> p_atom = protein_molecule[ip];

      const FFParamsT<T> p_params = forcefield[p_atom.type];

      const T radij = p_params.radius + l_params.radius;
      const T r_radij = M::rcp(radij);

      const T elcdst = (p_params.hbtype == HBTYPE_F && l_params.hbtype == HBTYPE_F) ? FOUR : TWO;
      const T elcdst1 = (p_params.hbtype == HBTYPE_F && l_params.hbtype == HBTYPE_F) ? QUARTER : HALF;
      const bool type_E = ((p_params.hbtype == HBTYPE_E || l_params.hbtype == HBTYPE_E));

      const bool phphb_ltz = p_params.hphb < ZERO;
      const bool phphb_gtz = p_params.hphb > ZERO;
      const bool phphb_nz = p_params.hphb != ZERO;
      const T p_hphb = p_params.hphb * (phphb_ltz && lhphb_gtz ? -ONE : ONE);
      const T l_hphb = l_params.hphb * (phphb_gtz && lhphb_ltz ? -ONE : ONE);
      const T distdslv = (phphb_ltz ? (lhphb_ltz ? NPNPDIST : NPPDIST) : (lhphb_ltz ? NPPDIST : -MAXVAL));
      const T r_distdslv = M::rcp(distdslv);

      const T chrg_init = l_params.elsc * p_params.elsc;
      const T dslv_init = p_hphb + l_hphb;

      for (int i = 0; i < int(PPWI); i++) {
        const T x = lpos[i].x - p_atom.x;
        const T y = lpos[i].y - p_atom.y;
        const T z = lpos[i].z - p_atom.z;
        const T distij = M::sqrt(x * x + y * y + z * z);

        const T distbb = distij - radij;
        const bool zone1 = (distbb < ZERO);

        etot[i] += (ONE - (distij * r_radij)) * (zone1 ? TWO * HARDNESS : ZERO);

        T chrg_e = chrg_init * ((zone1 ? ONE : (ONE - distbb * elcdst1)) * (distbb < elcdst ? ONE : ZERO));
        const T neg_chrg_e = -M::abs(chrg_e);
        chrg_e = type_E ? neg_chrg_e : chrg_e;
        etot[i] += chrg_e * CNSTNT;

        const T coeff = (ONE - (distbb * r_distdslv));
        T dslv_e = dslv_init * ((distbb < distdslv && phphb_nz) ? ONE : ZERO);
        dslv_e *= (zone1 ? ONE : coeff);
        etot[i] += dslv_e;
      }
    } while (++ip < natpro);
  } while (++il < natlig);

  if (base < numTransforms) {
    for (int i = 0; i < int(PPWI); i++) {
      etotals[base + i * lsz] = etot[i] * HALF;
    }
  }
}

template <typename T, size_t PPWI> class IMPL_CLS final : public Bude<T, PPWI> {

public:
  IMPL_CLS() = default;

  static inline void checkError(const cudaError_t err = cudaGetLastError()) {
    if (err != cudaSuccess) {
      throw std::runtime_error(std::string(cudaGetErrorName(err)) + ": " + std::string(cudaGetErrorString(err)));
    }
  }

  [[nodiscard]] std::string name() override { return "cuda"; };

  [[nodiscard]] std::vector<Device> enumerateDevices() override {
    int count = 0;
    checkError(cudaGetDeviceCount(&count));
    std::vector<Device> devices(count);
    for (int i = 0; i < count; ++i) {
      cudaDeviceProp props{};
      checkError(cudaGetDeviceProperties(&props, i));
      devices[i] = {i, std::string(props.name) + " (" +
                           std::to_string(props.totalGlobalMem / 1024 / 1024) + "MB;" +
                           "sm_" + std::to_string(props.major) + std::to_string(props.minor) +
                           ")"};
    }
    return devices;
  };

  template <typename U> [[nodiscard]] static U *allocate(size_t size) {
    U *data = nullptr;
#if defined(MANAGED)
    checkError(cudaMallocManaged(&data, sizeof(U) * size));
#elif defined(PAGEFAULT)
    data = (U *)std::malloc(sizeof(U) * size);
#else
    checkError(cudaMalloc(&data, sizeof(U) * size));
#endif
    return data;
  }

  template <typename U> [[nodiscard]] static U *allocate(const std::vector<U> &xs) {
    U *data = allocate<U>(xs.size());
    checkError(cudaMemcpy(data, xs.data(), xs.size() * sizeof(U), cudaMemcpyHostToDevice));
    return data;
  }

  template <typename U> static void deallocate(U *data) {
#if defined(PAGEFAULT)
    std::free(data);
#else
    checkError(cudaFree(data));
#endif
  }

  [[nodiscard]] bool compatible(const Params &p, size_t wgsize, size_t device) const override {
    checkError(cudaSetDevice(int(device)));
    auto data = allocate<int>(1);
    version<<<1, 1>>>(data);
    int rawKernelCC = -2;
    checkError(cudaMemcpy(&rawKernelCC, data, sizeof(rawKernelCC), cudaMemcpyDeviceToHost));
    checkError(cudaDeviceSynchronize());
    deallocate(data);
    cudaDeviceProp props{};
    checkError(cudaGetDeviceProperties(&props, int(device)));
    auto deviceCC = std::to_string(props.major * 10 + props.minor);
    std::string kernelCC;
    // clang-format off
    switch (rawKernelCC) {
    case 0: kernelCC = "(invalid, kernel returned bad result)"; break;
    case -1: kernelCC = "(unknown, device __CUDA_ARCH__ was not defined)"; break;
    case -2: kernelCC = "(unknown, kernel failed to execute)"; break;
    default: kernelCC = std::to_string(rawKernelCC / 10); break;
    }
    // clang-format on

    if (deviceCC != kernelCC) {
      std::cerr << "# [WARNING] Device cc (sm_" << deviceCC << ") != Kernel cc (sm" << kernelCC << ")\n"
                << "# Results may be incorrect, recompile with `--generate-code arch=compute_" << deviceCC
                << ",code=sm_" << deviceCC << "`" << std::endl;
      return false;
    } else {
      std::cout << "# Device and kernel cc: sm_" << deviceCC << std::endl;
      return true;
    }
  };

  [[nodiscard]] Sample fasten(const Params &p, size_t wgsize, size_t device) const override {

    checkError(cudaSetDevice(int(device)));

    Sample sample(PPWI, wgsize, PrecisionBits<T>::value, p.nposes());

    const auto hProtein = convertAtoms<T>(p.protein);
    const auto hLigand = convertAtoms<T>(p.ligand);
    const auto hForcefield = convertForcefield<T>(p.forcefield);

    auto contextStart = now();
    auto protein = allocate(hProtein);
    auto ligand = allocate(hLigand);
    auto transforms_0 = allocate(convertPoses<T>(p.poses[0]));
    auto transforms_1 = allocate(convertPoses<T>(p.poses[1]));
    auto transforms_2 = allocate(convertPoses<T>(p.poses[2]));
    auto transforms_3 = allocate(convertPoses<T>(p.poses[3]));
    auto transforms_4 = allocate(convertPoses<T>(p.poses[4]));
    auto transforms_5 = allocate(convertPoses<T>(p.poses[5]));
    auto forcefield = allocate(hForcefield);
    auto results = allocate<T>(sample.energies.size());
    checkError(cudaDeviceSynchronize());
    auto contextEnd = now();

    sample.contextTime = {contextStart, contextEnd};

    size_t global = size_t(std::ceil(double(p.nposes()) / double(PPWI)));
    global = size_t(std::ceil(double(global) / double(wgsize)));
    const size_t local = wgsize;

#ifdef USE_SHARED
    const size_t shared = p.ntypes() * sizeof(FFParamsT<T>);
#else
    const size_t shared = 0;
#endif

    cudaEvent_t startEvent{}, stopEvent{};
    checkError(cudaEventCreate(&startEvent));
    checkError(cudaEventCreate(&stopEvent));

    for (size_t i = 0; i < p.totalIterations(); ++i) {
      auto wallStart = now();
      checkError(cudaEventRecord(startEvent));
      fasten_main<T, PPWI><<<global, local, shared>>>(
          int(p.natlig()), int(p.natpro()), int(p.ntypes()), protein, ligand,
          transforms_0, transforms_1, transforms_2, transforms_3, transforms_4, transforms_5,
          results, forcefield, int(p.nposes()));
      checkError(cudaEventRecord(stopEvent));
      checkError(cudaEventSynchronize(stopEvent));
      auto wallEnd = now();

      float deviceMs = 0.0f;
      checkError(cudaEventElapsedTime(&deviceMs, startEvent, stopEvent));

      sample.kernelMillis.push_back(double(deviceMs));
      sample.wallMillis.push_back(elapsedMillis(wallStart, wallEnd));
    }

    checkError(cudaEventDestroy(startEvent));
    checkError(cudaEventDestroy(stopEvent));

    std::vector<T> raw(sample.energies.size());
    checkError(cudaMemcpy(raw.data(), results, raw.size() * sizeof(T), cudaMemcpyDeviceToHost));
    std::copy(raw.begin(), raw.end(), sample.energies.begin());

    deallocate(protein);
    deallocate(ligand);
    deallocate(transforms_0);
    deallocate(transforms_1);
    deallocate(transforms_2);
    deallocate(transforms_3);
    deallocate(transforms_4);
    deallocate(transforms_5);
    deallocate(forcefield);
    deallocate(results);

    checkError(cudaDeviceSynchronize());

    return sample;
  };
};
