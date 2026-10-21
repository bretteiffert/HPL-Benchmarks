// miniBUDE device code: precision-specific device math, the compile-check
// kernel and the fasten_main kernel. Everything below is the original
// fasten.hpp code, unchanged (lines 21-193 of the original file).
//
// Build-time options, as in the original (make EXTRA_FLAGS=...):
//   -DNO_RESTRICT   drop the __restrict__ qualifiers
//   -DUSE_SHARED    copy the forcefield into shared memory per block

#pragma once

#include <cstddef>
#include <cmath>

#include <cuda_runtime.h>

#include "bude.h"

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
