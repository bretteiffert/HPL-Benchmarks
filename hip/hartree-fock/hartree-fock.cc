// Implements the HIP Hartree-Fock benchmark with in-source fp32/fp64 selection, explicit device transfers, bounds-safe complete grid coverage, and precision-specific math. The optional legacy sqrt mode reproduces the former mixed-precision square roots, while the normal mode uses the selected arithmetic type. A host fp64 Fock build provides untimed diagnostics for supported input sizes; HIP-event-timed CSV iterations accumulate into the existing Fock buffer rather than re-zeroing it. Screening constants retain the same literal values in both precision modes, and the printed unsymmetrized energy remains a reproducible benchmark checksum rather than a physical two-electron energy.

#include "hip/hip_runtime.h"
#include <iostream>
#include <cstdio>
#include <cstdlib>
#include <iomanip>
#include <fstream>
#include <string>
#include <math.h>

#include <hip/hip_runtime.h>

#define PRECISION 64

#if PRECISION == 32
typedef float  real_t;
#define REAL_NAME "fp32"
#elif PRECISION == 64
typedef double real_t;
#define REAL_NAME "fp64"
#else
#error "PRECISION must be 32 or 64"
#endif

const real_t pi = (real_t)3.1415926535897931;
const real_t sqrtpi2 = (real_t)(pow(3.1415926535897931,-0.5) * 2.0);
const real_t dtol = (real_t)1.0e-12;
const real_t rcut = (real_t)1.0e-12;
const real_t tobohrs = (real_t)1.889725987722;

#if PRECISION == 32
__host__ __device__ inline real_t r_sqrt(real_t x)          { return sqrtf(x); }
__host__ __device__ inline real_t r_exp (real_t x)          { return expf(x);  }
__host__ __device__ inline real_t r_erf (real_t x)          { return erff(x);  }
__host__ __device__ inline real_t r_pow (real_t a, real_t b){ return powf(a,b);}
__host__ __device__ inline real_t r_abs (real_t x)          { return fabsf(x); }
#else
__host__ __device__ inline real_t r_sqrt(real_t x)          { return sqrt(x);  }
__host__ __device__ inline real_t r_exp (real_t x)          { return exp(x);   }
__host__ __device__ inline real_t r_erf (real_t x)          { return erf(x);   }
__host__ __device__ inline real_t r_pow (real_t a, real_t b){ return pow(a,b); }
__host__ __device__ inline real_t r_abs (real_t x)          { return fabs(x);  }
#endif

__host__ __device__ inline real_t r_sq(real_t x) { return x * x; }

#define LEGACY_SQRTF 0
#if LEGACY_SQRTF
__host__ __device__ inline real_t r_sqrt_boys(real_t x) { return (real_t)sqrtf((float)x); }
#define SQRT_MODE "legacy-sqrtf"
#else
__host__ __device__ inline real_t r_sqrt_boys(real_t x) { return r_sqrt(x); }
#define SQRT_MODE "native"
#endif

bool csv_output = false;
int num_iter = 10;

bool run_check = true;
bool force_check = false;
int  max_check_atoms = 128;

__device__ const struct {
	real_t sqrtpi2;
	real_t dtol;
	real_t rcut;
} dev_consts = {(real_t)1.12837916709551255856, (real_t)1.0e-12, (real_t)1.0e-12};

#define CUDA_CHECK(err)  __checkCudaErrors (err, __FILE__, __LINE__)

inline void __checkCudaErrors(hipError_t err, const char *file, const int line )
{
  if(hipSuccess != err){
	fprintf(stderr, "%s(%i) : CUDA Runtime API error %d: %s.\n",file, line, (int)err, hipGetErrorString( err ) );
	 exit(-1);
  }
}

real_t ssss(uint32_t i, uint32_t j, uint32_t k, uint32_t l,
		       uint32_t ngauss, real_t *xpnt, real_t *coef,
		       real_t *geom, uint32_t natoms) {
	real_t eri = 0.0;
	real_t aij, dij, akl, dkl, aijkl;
	real_t xij, yij, zij;
	real_t f0t, tt;

	for (int ib = 0; ib < (int)ngauss; ib++) {
	     for (int jb = 0; jb < (int)ngauss; jb++) {
		  aij = (real_t)1.0 / (xpnt[ib] + xpnt[jb]);
		  dij = coef[ib] * coef[jb] *
			r_exp(-xpnt[ib] * xpnt[jb] * aij *
			    ((r_sq(geom[i] - geom[j]))
			    + (r_sq(geom[natoms + i] - geom[natoms + j]))
			    + (r_sq(geom[2*natoms + i] - geom[2*natoms + j]))))
			* r_pow(aij,(real_t)1.5);
		  if (r_abs(dij) > dtol) {
		      xij = aij * (xpnt[ib] * geom[i] + xpnt[jb] * geom[j]);
		      yij = aij * (xpnt[ib] * geom[natoms + i] + xpnt[jb] * geom[natoms + j]);
		      zij = aij * (xpnt[ib] * geom[2*natoms + i] + xpnt[jb] * geom[2*natoms + j]);

		      for (int kb = 0; kb < (int)ngauss; kb++) {
		           for(int lb = 0; lb < (int)ngauss; lb++) {
				   akl = (real_t)1.0 / (xpnt[kb] + xpnt[lb]);
				   dkl = dij * coef[kb] * coef[lb] *
					 r_exp(-xpnt[kb] * xpnt[lb] * akl
				           * ((r_sq(geom[k] - geom[l]))
					       + (r_sq(geom[natoms + k] - geom[natoms + l]))
					       + (r_sq(geom[2*natoms + k] - geom[2*natoms + l]))))
					  * r_pow(akl,(real_t)1.5);

			           if (r_abs(dkl) > dtol) {
				       aijkl = (xpnt[ib] + xpnt[jb])
						* (xpnt[kb] + xpnt[lb])
						/ (xpnt[ib] + xpnt[jb]
					           + xpnt[kb] + xpnt[lb]);
				       tt = aijkl * (r_sq(xij - akl * (xpnt[kb] * geom[k] + xpnt[lb] * geom[l]))
				                     + r_sq(yij - akl * (xpnt[kb] * geom[natoms + k] + xpnt[lb] * geom[natoms + l]))
				                     + r_sq(zij - akl * (xpnt[kb] * geom[2*natoms + k] + xpnt[lb] * geom[2*natoms + l])));
				       f0t = sqrtpi2;
				       if (tt > rcut) {
				       	   f0t = r_pow(tt, (real_t)-0.5) * r_erf(r_sqrt_boys(tt));
				       }
				       eri += (dkl * f0t * r_sqrt_boys(aijkl));
				   }
			   }
		      }

		  }

	     }
	}
	return eri;

}

void read_file(const char *filename, uint32_t* ngauss, uint32_t* natoms,
	       real_t** xpnt, real_t** coef, real_t** geom,
	       double** xpnt_d, double** coef_d, double** geom_d) {

	std::ifstream inp(filename);

	if (!inp) {
            std::cout << "ERROR: Could not read file!" << std::endl;
	    exit(1);
	}

	if (!(inp >> *ngauss)) {
	    std::cout << "ERROR: Could not read # of gaussians" << std::endl;
	    exit(1);
	}
	if (!(inp >> *natoms)) {
	    std::cout << "ERROR: Could not read # of atoms" << std::endl;
	    exit(1);
	}

	double *x = new double[*ngauss];
	double *c = new double[*ngauss];
	double *g = new double[3 * *natoms];

	for (uint32_t i = 0; i < *ngauss; i++) {
	     inp >> x[i];
	     inp >> c[i];
	}

	for (uint32_t i = 0; i < *natoms; i++) {
	     for (int j = 0; j < 3; j++) {
	          inp >> g[j * *natoms + i];
	     }
	}

	*xpnt_d = new double[*ngauss];
	*coef_d = new double[*ngauss];
	*geom_d = new double[3 * *natoms];
	for (uint32_t i = 0; i < *ngauss; i++) { (*xpnt_d)[i] = x[i]; (*coef_d)[i] = c[i]; }
	for (uint32_t i = 0; i < 3 * *natoms; i++) (*geom_d)[i] = g[i];

	*xpnt = new real_t[*ngauss];
	*coef = new real_t[*ngauss];
	*geom = new real_t[3 * *natoms];
	for (uint32_t i = 0; i < *ngauss; i++)     { (*xpnt)[i] = (real_t)x[i]; (*coef)[i] = (real_t)c[i]; }
	for (uint32_t i = 0; i < 3 * *natoms; i++)   (*geom)[i] = (real_t)g[i];

	delete[] x;
	delete[] c;
	delete[] g;
}

__global__ void hartree_fock(uint64_t nnnn, uint32_t ngauss, uint32_t natoms,
		real_t *geom, real_t *xpnt, real_t *coef, real_t *dens,
		real_t *schwarz, real_t *fock)
{
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
									f0t = (r_pow(tt, (real_t)-0.5) * r_erf(r_sqrt_boys(tt)));
								}
								eri += dkl * f0t * r_sqrt_boys(aijkl);
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


static const double REF_SQRTPI2 = 1.12837916709551255856;
static const double REF_DTOL    = 1.0e-12;
static const double REF_RCUT    = 1.0e-12;

static inline double d_sq(double x) { return x * x; }

static double ssss_ref(uint32_t i, uint32_t j, uint32_t k, uint32_t l,
                       uint32_t ngauss, const double *xpnt, const double *coef,
                       const double *geom, uint32_t natoms)
{
	double eri = 0.0;
	for (uint32_t ib = 0; ib < ngauss; ib++) {
	for (uint32_t jb = 0; jb < ngauss; jb++) {
		double aij = 1.0 / (xpnt[ib] + xpnt[jb]);
		double dij = coef[ib] * coef[jb] *
			exp(-xpnt[ib] * xpnt[jb] * aij *
			    (d_sq(geom[i] - geom[j])
			   + d_sq(geom[natoms + i] - geom[natoms + j])
			   + d_sq(geom[2*natoms + i] - geom[2*natoms + j])))
			* pow(aij, 1.5);
		if (fabs(dij) > REF_DTOL) {
			double xij = aij * (xpnt[ib]*geom[i] + xpnt[jb]*geom[j]);
			double yij = aij * (xpnt[ib]*geom[natoms + i] + xpnt[jb]*geom[natoms + j]);
			double zij = aij * (xpnt[ib]*geom[2*natoms + i] + xpnt[jb]*geom[2*natoms + j]);
			for (uint32_t kb = 0; kb < ngauss; kb++) {
			for (uint32_t lb = 0; lb < ngauss; lb++) {
				double akl = 1.0 / (xpnt[kb] + xpnt[lb]);
				double dkl = dij * coef[kb] * coef[lb] *
					exp(-xpnt[kb] * xpnt[lb] * akl *
					    (d_sq(geom[k] - geom[l])
					   + d_sq(geom[natoms + k] - geom[natoms + l])
					   + d_sq(geom[2*natoms + k] - geom[2*natoms + l])))
					* pow(akl, 1.5);
				if (fabs(dkl) > REF_DTOL) {
					double aijkl = (xpnt[ib] + xpnt[jb]) * (xpnt[kb] + xpnt[lb])
						/ (xpnt[ib] + xpnt[jb] + xpnt[kb] + xpnt[lb]);
					double tt = aijkl * (
						  d_sq(xij - akl*(xpnt[kb]*geom[k] + xpnt[lb]*geom[l]))
						+ d_sq(yij - akl*(xpnt[kb]*geom[natoms + k] + xpnt[lb]*geom[natoms + l]))
						+ d_sq(zij - akl*(xpnt[kb]*geom[2*natoms + k] + xpnt[lb]*geom[2*natoms + l])));
					double f0t = REF_SQRTPI2;
					if (tt > REF_RCUT) {
						f0t = pow(tt, -0.5) * erf(sqrt(tt));
					}
					eri += dkl * f0t * sqrt(aijkl);
				}
			}}
		}
	}}
	return eri;
}

static void fock_build_cpu(uint32_t ngauss, uint32_t natoms,
                           const double *geom, const double *xpnt, const double *coef,
                           const double *dens, double *fock_out)
{
	for (uint64_t t = 0; t < (uint64_t)natoms * natoms; t++) fock_out[t] = 0.0;

	uint64_t nn_ref = (((uint64_t)natoms * natoms) + natoms) / 2;
	double *schwarz = new double[nn_ref + 1];
	for (uint64_t t = 0; t <= nn_ref; t++) schwarz[t] = 0.0;
	{
		uint64_t ijr = 0;
		for (uint32_t i = 0; i < natoms; i++)
			for (uint32_t j = 0; j <= i; j++) {
				ijr++;
				schwarz[ijr] = sqrt(fabs(ssss_ref(i,j,i,j,ngauss,xpnt,coef,geom,natoms)));
			}
	}

	for (uint32_t i = 0; i < natoms; i++) {
	for (uint32_t j = 0; j <= i; j++) {
		uint64_t ij = (uint64_t)i * (i + 1) / 2 + j + 1;
		for (uint32_t k = 0; k < natoms; k++) {
		for (uint32_t l = 0; l <= k; l++) {
			uint64_t kl = (uint64_t)k * (k + 1) / 2 + l + 1;
			if (kl > ij) continue;
			if (!(schwarz[ij] * schwarz[kl] > REF_DTOL)) continue;

			double eri = ssss_ref(i, j, k, l, ngauss, xpnt, coef, geom, natoms);
			if (i == j) eri *= 0.5;
			if (k == l) eri *= 0.5;
			if (i == k && j == l) eri *= 0.5;

			fock_out[i*natoms + j] += dens[k*natoms + l] * eri * 4.0;
			fock_out[k*natoms + l] += dens[i*natoms + j] * eri * 4.0;
			fock_out[i*natoms + k] -= dens[j*natoms + l] * eri;
			fock_out[i*natoms + l] -= dens[j*natoms + k] * eri;
			fock_out[j*natoms + k] -= dens[i*natoms + l] * eri;
			fock_out[j*natoms + l] -= dens[i*natoms + k] * eri;
		}}
	}}
	delete[] schwarz;
}

static double energy_from_fock(const double *fock, const double *dens, uint32_t natoms)
{
	double e = 0.0;
	for (uint32_t i = 0; i < natoms; i++)
		for (uint32_t j = 0; j < natoms; j++)
			e += fock[i*natoms + j] * dens[i*natoms + j];
	return e * 0.5;
}

static double energy_symmetrized(const double *fock, const double *dens, uint32_t natoms)
{
	double e = 0.0;
	for (uint32_t i = 0; i < natoms; i++)
		for (uint32_t j = 0; j < natoms; j++) {
			double f = (i == j) ? fock[i*natoms + j]
			                    : fock[i*natoms + j] + fock[j*natoms + i];
			e += f * dens[i*natoms + j];
		}
	return e * 0.5;
}

int main(int argc, char **argv){
	if (argc < 2) {
		std::cout << "ERROR: Please provide an input file" << std::endl;
		return 1;
	}

    for (int a = 2; a < argc; a++) {
        std::string s(argv[a]);
        if (s == "--csv")         csv_output = true;
        else if (s == "--check")  { run_check = true; force_check = true; }
        else if (s == "--no-check") run_check = false;
        else if (s.rfind("--max-check-atoms=", 0) == 0)
            max_check_atoms = atoi(s.c_str() + 18);
        else if (s.rfind("--iters=", 0) == 0)
            num_iter = atoi(s.c_str() + 8);
    }

	uint32_t natoms, ngauss;
	real_t *h_xpnt, *h_coef, *h_geom;
	double *xpnt_d, *coef_d, *geom_d;

	read_file(argv[1], &ngauss, &natoms, &h_xpnt, &h_coef, &h_geom,
	          &xpnt_d, &coef_d, &geom_d);

	uint64_t nn = ((((uint64_t)natoms * natoms) + natoms) / 2);
	size_t   nsq = (size_t)natoms * natoms;

	real_t *h_dens    = new real_t[nsq];
	real_t *h_fock    = new real_t[nsq];
	real_t *h_schwarz = new real_t[(size_t)(nn + 1)];
	for (uint64_t t = 0; t <= nn; t++) h_schwarz[t] = (real_t)0.0;

	double *dens_d = new double[nsq];
	for (uint32_t i = 0; i < natoms; i++){
		for (uint32_t j = 0; j < natoms; j++){
			h_dens[i*natoms + j] = (real_t)0.1;
			dens_d[i*natoms + j] = 0.1;
		}
		h_dens[i*natoms + i] = (real_t)1.0;
		dens_d[i*natoms + i] = 1.0;
	}

	for (uint32_t i = 0; i < ngauss; i++){
		h_coef[i] = h_coef[i] * r_pow(((real_t)2.0 * h_xpnt[i]), (real_t)0.75);
		coef_d[i] = coef_d[i] * pow((2.0 * xpnt_d[i]), 0.75);
	}

	for (uint32_t i = 0; i < natoms; i++){
		h_geom[0 * natoms + i] *= tobohrs;
		h_geom[1 * natoms + i] *= tobohrs;
		h_geom[2 * natoms + i] *= tobohrs;
		geom_d[0 * natoms + i] *= 1.889725987722;
		geom_d[1 * natoms + i] *= 1.889725987722;
		geom_d[2 * natoms + i] *= 1.889725987722;
	}

	for (uint32_t i = 0; i < natoms; i++){
		for (uint32_t j = 0; j < natoms; j++){
			h_fock[i*natoms + j] = (real_t)0.0;
		}
	}

	uint64_t ij = 0;
	real_t eri = 0.0;
	for (uint32_t i = 0; i < natoms; i++) {
		for (uint32_t j = 0; j <= i; j++) {
			ij = ij + 1;
			eri = ssss(i,j,i,j,ngauss,h_xpnt,h_coef,h_geom, natoms);
			h_schwarz[ij] = r_sqrt(r_abs(eri));
		}
	}

	uint64_t nnnn = ((nn * nn) + nn) / 2;
	uint32_t blk_size = 256;
	uint64_t n_blks64 = ((nnnn + 1) + blk_size - 1) / blk_size;
	if (n_blks64 > 2147483647ULL) {
		fprintf(stderr, "ERROR: grid size %llu exceeds gridDim.x limit\n",
		        (unsigned long long)n_blks64);
		return 1;
	}
	uint32_t n_blks = (uint32_t)n_blks64;

    int device;
    CUDA_CHECK(hipGetDevice(&device));
    hipDeviceProp_t props;
    CUDA_CHECK(hipGetDeviceProperties(&props, device));

	real_t *xpnt, *coef, *geom, *dens, *fock, *schwarz;
	CUDA_CHECK(hipMalloc(&xpnt,    sizeof(real_t) * ngauss));
	CUDA_CHECK(hipMalloc(&coef,    sizeof(real_t) * ngauss));
	CUDA_CHECK(hipMalloc(&geom,    sizeof(real_t) * (3 * (size_t)natoms)));
	CUDA_CHECK(hipMalloc(&dens,    sizeof(real_t) * nsq));
	CUDA_CHECK(hipMalloc(&fock,    sizeof(real_t) * nsq));
	CUDA_CHECK(hipMalloc(&schwarz, sizeof(real_t) * (size_t)(nn + 1)));

	CUDA_CHECK(hipMemcpy(xpnt,    h_xpnt,    sizeof(real_t) * ngauss, hipMemcpyHostToDevice));
	CUDA_CHECK(hipMemcpy(coef,    h_coef,    sizeof(real_t) * ngauss, hipMemcpyHostToDevice));
	CUDA_CHECK(hipMemcpy(geom,    h_geom,    sizeof(real_t) * (3 * (size_t)natoms), hipMemcpyHostToDevice));
	CUDA_CHECK(hipMemcpy(dens,    h_dens,    sizeof(real_t) * nsq, hipMemcpyHostToDevice));
	CUDA_CHECK(hipMemcpy(fock,    h_fock,    sizeof(real_t) * nsq, hipMemcpyHostToDevice));
	CUDA_CHECK(hipMemcpy(schwarz, h_schwarz, sizeof(real_t) * (size_t)(nn + 1), hipMemcpyHostToDevice));

	dim3 threads(blk_size, 1, 1);
	dim3 grid(n_blks, 1, 1);

	hartree_fock<<<grid, threads>>>(nnnn, ngauss, natoms,
			geom, xpnt, coef, dens,
			schwarz,fock);
	CUDA_CHECK(hipGetLastError());
	CUDA_CHECK(hipDeviceSynchronize());

	CUDA_CHECK(hipMemcpy(h_fock, fock, sizeof(real_t) * nsq, hipMemcpyDeviceToHost));

	real_t erep = 0.0;
	for (uint32_t i = 0; i < natoms; i++){
		for (uint32_t j = 0; j < natoms; j++){
			erep += h_fock[i*natoms +j] * h_dens[i*natoms + j];
		}
	}

    if (!csv_output) {
    	printf("2e- energy=%.16lf\n", (double)erep*0.5);

        if (run_check && (force_check || (int)natoms <= max_check_atoms)) {
            double *fock_gpu = new double[nsq];
            double *fock_ref = new double[nsq];
            for (size_t t = 0; t < nsq; t++) fock_gpu[t] = (double)h_fock[t];

            fock_build_cpu(ngauss, natoms, geom_d, xpnt_d, coef_d, dens_d, fock_ref);

            double e_gpu = energy_from_fock(fock_gpu, dens_d, natoms);
            double e_ref = energy_from_fock(fock_ref, dens_d, natoms);

            double denom = (fabs(e_ref) > 0.0) ? fabs(e_ref) : 1.0;
            double rel_loss = fabs(e_gpu - e_ref) / denom;

            double e_inpipe  = (double)erep * 0.5;
            double rel_inpipe = fabs(e_inpipe - e_ref) / denom;

            double maxd = 0.0, sumsq = 0.0, fscale = 0.0;
            for (size_t t = 0; t < nsq; t++) {
                double d = fabs(fock_gpu[t] - fock_ref[t]);
                if (d > maxd) maxd = d;
                sumsq += d * d;
                if (fabs(fock_ref[t]) > fscale) fscale = fabs(fock_ref[t]);
            }
            double rmsd = sqrt(sumsq / (double)nsq);
            if (fscale == 0.0) fscale = 1.0;
            double maxd_rel = maxd / fscale;

            double asym = 0.0;
            for (uint32_t i = 0; i < natoms; i++)
                for (uint32_t j = 0; j < natoms; j++) {
                    double d = fabs(fock_gpu[i*natoms + j] - fock_gpu[j*natoms + i]);
                    if (d > asym) asym = d;
                }

            printf("\n--- precision check (build: %s, sqrt: %s) ---\n", REAL_NAME, SQRT_MODE);
            printf("2e- energy (%s, %s reduction)      = %.16lf\n", REAL_NAME, REAL_NAME, e_inpipe);
            printf("2e- energy (%s Fock, fp64 reduction) = %.16lf\n", REAL_NAME, e_gpu);
            printf("2e- energy (fp64 CPU reference)       = %.16lf\n", e_ref);
            printf("precision loss, Fock build only       = %.6e\n", rel_loss);
            printf("precision loss, incl. %s reduction  = %.6e\n", REAL_NAME, rel_inpipe);
            printf("max |F - F_ref| (elementwise)         = %.6e  (rel %.6e)\n", maxd, maxd_rel);
            printf("rms |F - F_ref| (elementwise)         = %.6e\n", rmsd);
            printf("max |F(i,j) - F(j,i)|                 = %.6e\n", asym);
            printf("2e- energy from symmetrized F=A+A^T   = %.16lf   (diagnostic)\n",
                   energy_symmetrized(fock_gpu, dens_d, natoms));
            printf("---------------------------------------------------------------\n");

            delete[] fock_gpu; delete[] fock_ref;
        } else if (run_check) {
            printf("(correctness check skipped: natoms=%u > %d; use --check to force)\n",
                   natoms, max_check_atoms);
        }
    }
    else {
        printf("backend,GPU,precision,sqrt_mode,natoms,ngauss,exec_time_ms\n");

        float elapsed;
        hipEvent_t start, stop;
        CUDA_CHECK(hipEventCreate(&start));
        CUDA_CHECK(hipEventCreate(&stop));

        for (int i = 0; i < num_iter; ++i) {
            CUDA_CHECK(hipDeviceSynchronize());
            CUDA_CHECK(hipEventRecord(start));
            hartree_fock<<<grid, threads>>>(nnnn, ngauss, natoms,
                    geom, xpnt, coef, dens,
                    schwarz,fock);
            CUDA_CHECK(hipGetLastError());
            CUDA_CHECK(hipEventRecord(stop));
            CUDA_CHECK(hipEventSynchronize(stop));
            CUDA_CHECK(hipEventElapsedTime(&elapsed, start, stop));

            printf("CUDA,%s,%s,%s,%d,%d,%f\n", props.name, REAL_NAME,
            SQRT_MODE, natoms, ngauss, elapsed);
        }
    }

	CUDA_CHECK(hipFree(xpnt));
	CUDA_CHECK(hipFree(coef));
	CUDA_CHECK(hipFree(geom));
	CUDA_CHECK(hipFree(dens));
	CUDA_CHECK(hipFree(fock));
	CUDA_CHECK(hipFree(schwarz));
	delete[] h_xpnt; delete[] h_coef; delete[] h_geom;
	delete[] h_dens; delete[] h_fock; delete[] h_schwarz;
	delete[] xpnt_d; delete[] coef_d; delete[] geom_d; delete[] dens_d;
	return 0;
}
