// Hartree-Fock device code: precision-specific math helpers and the Fock
// build kernel. The kernel body is the original hartree_fock kernel; the only
// changes are the ones run-time precision forces:
//   - the kernel is a template on real_t (float or double) and on the sqrt
//     mode (LegacySqrt), which used to be the PRECISION and LEGACY_SQRTF macros;
//   - dtol and dev_consts are per-type local constants with the original
//     literal values (the host global dtol and the __device__ dev_consts
//     struct cannot be typed per precision);
//   - r_sqrt_boys is called as r_sqrt_boys<LegacySqrt>.
// The math helpers are overloads that call exactly the functions the
// PRECISION == 32 / 64 branches used (sqrtf/expf/erff/powf/fabsf for float,
// sqrt/exp/erf/pow/fabs for double).

#pragma once

#include <cstdint>
#include <math.h>

#include <hip/hip_runtime.h>

__host__ __device__ inline float  r_sqrt(float x)            { return sqrtf(x); }
__host__ __device__ inline float  r_exp (float x)            { return expf(x);  }
__host__ __device__ inline float  r_erf (float x)            { return erff(x);  }
__host__ __device__ inline float  r_pow (float a, float b)   { return powf(a,b);}
__host__ __device__ inline float  r_abs (float x)            { return fabsf(x); }

__host__ __device__ inline double r_sqrt(double x)           { return sqrt(x);  }
__host__ __device__ inline double r_exp (double x)           { return exp(x);   }
__host__ __device__ inline double r_erf (double x)           { return erf(x);   }
__host__ __device__ inline double r_pow (double a, double b) { return pow(a,b); }
__host__ __device__ inline double r_abs (double x)           { return fabs(x);  }

template <class real_t>
__host__ __device__ inline real_t r_sq(real_t x) { return x * x; }

// LEGACY_SQRTF=1 behaviour (--legacy-sqrt): the Boys-function square roots go
// through sqrtf in float even in a float64 build. Native mode uses r_sqrt.
template <bool LegacySqrt, class real_t>
__host__ __device__ inline real_t r_sqrt_boys(real_t x)
{
	if (LegacySqrt) return (real_t)sqrtf((float)x);
	return r_sqrt(x);
}

template <class real_t>
struct DevConsts {
	real_t sqrtpi2;
	real_t dtol;
	real_t rcut;
};

template <class real_t, bool LegacySqrt>
__global__ void hartree_fock(uint64_t nnnn, uint32_t ngauss, uint32_t natoms,
		real_t *geom, real_t *xpnt, real_t *coef, real_t *dens,
		real_t *schwarz, real_t *fock)
{
	// Run-time precision: the screening constants are per-type locals with the
	// same literal values as the original globals (dtol) and dev_consts.
	const real_t dtol = (real_t)1.0e-12;
	const DevConsts<real_t> dev_consts = {(real_t)1.12837916709551255856, (real_t)1.0e-12, (real_t)1.0e-12};

	uint64_t ijkl = (uint64_t)blockIdx.x * blockDim.x + threadIdx.x;
	if (ijkl == 0 || ijkl > nnnn) return;

	uint64_t ij = (uint64_t)sqrtf(2*ijkl);
	uint64_t n = (ij*ij + ij) / 2;
	while (n < ijkl) {
		ij += 1;
		n = (ij*ij + ij) / 2;
	}
	uint64_t kl = ijkl - (ij * ij - ij) / 2;
	if (schwarz[ij] * schwarz[kl] > dtol) {
		uint64_t i = (uint64_t)sqrtf(2*ij)-1;
		n = (i * i + i) / 2;
		while (n < ij) {
			i += 1;
			n = (i * i + i) / 2;
		}
		uint64_t j = ij - (i * i - i) / 2;

		uint64_t k = (uint64_t)sqrtf(2*kl);
		n = (k * k + k) / 2;
		while (n < kl) {
			k += 1;
			n = (k * k + k ) / 2;
		}
		uint64_t l = kl - (k * k - k) / 2;
		i-= 1;
		j-= 1;
		k-= 1;
		l-= 1;

		real_t eri = 0.0;
		for (int ib = 0; ib < (int)ngauss; ib++) {
			for (int jb = 0; jb < (int)ngauss; jb++) {
				real_t aij = (real_t)1.0 / (xpnt[ib] + xpnt[jb]);
				real_t dij = coef[ib] * coef[jb] *
					r_exp(-xpnt[ib] * xpnt[jb] * aij *
							(r_sq(geom[i] - geom[j])
							 + r_sq(geom[1*natoms + i] - geom[1*natoms + j])
							 + r_sq(geom[2*natoms + i] - geom[2*natoms + j])))
					* r_pow(aij,(real_t)1.5);
				if (r_abs(dij) > dtol) {
					real_t xij = aij * (xpnt[ib] * geom[i] + xpnt[jb] * geom[j]);
					real_t yij = aij * (xpnt[ib] * geom[1*natoms + i] + xpnt[jb] * geom[1*natoms + j]);
					real_t zij = aij * (xpnt[ib] * geom[2*natoms + i] + xpnt[jb] * geom[2*natoms + j]);
					for (int kb = 0; kb < (int)ngauss; kb++) {
						for (int lb = 0; lb < (int)ngauss; lb++) {
						real_t akl = (real_t)1.0 / (xpnt[kb] + xpnt[lb]);
							real_t dkl = dij * coef[kb] * coef[lb] *
								r_exp(-xpnt[kb] * xpnt[lb] * akl *
										(r_sq(geom[k] - geom[l])
										 + r_sq(geom[natoms + k] - geom[natoms + l])
										 + r_sq(geom[2*natoms + k] - geom[2*natoms + l]))) * r_pow(akl,(real_t)1.5);
							if (r_abs(dkl) > dtol) {
								real_t aijkl = (xpnt[ib] + xpnt[jb])
									* (xpnt[kb] + xpnt[lb])
									/ (xpnt[ib] + xpnt[jb] + xpnt[kb] + xpnt[lb]);
								real_t tt = aijkl * (r_sq(xij - akl * (xpnt[kb] * geom[k] + xpnt[lb] * geom[l]))
										+ r_sq(yij - akl * (xpnt[kb] * geom[natoms + k] + xpnt[lb] * geom[natoms + l]))
										+ r_sq(zij - akl * (xpnt[kb] * geom[2*natoms + k] + xpnt[lb] * geom[2*natoms + l])));
								real_t f0t = dev_consts.sqrtpi2;
								if (tt > dev_consts.rcut) {
									f0t = (r_pow(tt, (real_t)-0.5) * r_erf(r_sqrt_boys<LegacySqrt>(tt)));
								}
								eri += dkl * f0t * r_sqrt_boys<LegacySqrt>(aijkl);
							}

						}

					}
				}

			}
		}
		if (i == j) {
			eri *= (real_t)0.5;
		}
		if (k == l) {
			eri *= (real_t)0.5;
		}
		if (i ==k && j == l) {
			eri *= (real_t)0.5;
		}
		atomicAdd(&fock[i*natoms + j], dens[k*natoms + l]*eri*(real_t)4.0);
		atomicAdd(&fock[k*natoms + l], dens[i*natoms + j]*eri*(real_t)4.0);
		atomicAdd(&fock[i*natoms + k], -dens[j*natoms + l]*eri);
		atomicAdd(&fock[i*natoms + l], -dens[j*natoms + k]*eri);
		atomicAdd(&fock[j*natoms + k], -dens[i*natoms + l]*eri);
		atomicAdd(&fock[j*natoms + l], -dens[i*natoms + k]*eri);
	}
}
