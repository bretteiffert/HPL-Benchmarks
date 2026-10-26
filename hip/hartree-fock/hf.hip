// Hartree-Fock on HIP: host preparation in the run precision, device
// buffers, the untimed first Fock build, event-timed builds, and the fp64 CPU
// reference. ssss (Schwarz screening), ssss_ref, fock_build_cpu and the two
// energy functions are the original code; ssss is templated on precision and
// sqrt mode. Nothing here prints; failures throw.

#include "hf.h"
#include "kernels.hip.h"

#include <cmath>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

void api_check(hipError_t err, const char* what)
{
  if (err != hipSuccess)
    throw std::runtime_error(std::string("HIP ") + what + ": " + hipGetErrorString(err));
}

// The original host globals pi, sqrtpi2, dtol, rcut and tobohrs, per precision.
template <class real_t>
struct HostConsts
{
  static real_t sqrtpi2() { return (real_t)(pow(3.1415926535897931,-0.5) * 2.0); }
  static real_t dtol()    { return (real_t)1.0e-12; }
  static real_t rcut()    { return (real_t)1.0e-12; }
  static real_t tobohrs() { return (real_t)1.889725987722; }
};

template <class real_t, bool LegacySqrt>
real_t ssss(uint32_t i, uint32_t j, uint32_t k, uint32_t l,
		       uint32_t ngauss, real_t *xpnt, real_t *coef,
		       real_t *geom, uint32_t natoms) {
	// Host constants of the original, typed per precision.
	const real_t sqrtpi2 = HostConsts<real_t>::sqrtpi2();
	const real_t dtol = HostConsts<real_t>::dtol();
	const real_t rcut = HostConsts<real_t>::rcut();
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
				       	   f0t = r_pow(tt, (real_t)-0.5) * r_erf(r_sqrt_boys<LegacySqrt>(tt));
				       }
				       eri += (dkl * f0t * r_sqrt_boys<LegacySqrt>(aijkl));
				   }
			   }
		      }

		  }

	     }
	}
	return eri;

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

template <class real_t, bool LegacySqrt>
class HIPHartreeFock : public HartreeFock
{
  public:
    HIPHartreeFock(const Deck& deck, uint32_t block_size);
    ~HIPHartreeFock() override;

    void first_build() override;
    double checksum() const override { return checksum_; }
    double timed_build() override;
    FockCheck check() const override;

  private:
    uint32_t ngauss, natoms;
    uint64_t nn, nnnn;
    size_t nsq;
    uint32_t blk_size, n_blks;

    // Run-precision host data (as the original h_* arrays).
    std::vector<real_t> h_xpnt, h_coef, h_geom, h_dens, h_fock, h_schwarz;
    // fp64 copies for the CPU reference (as the original *_d arrays).
    std::vector<double> xpnt_d, coef_d, geom_d, dens_d;

    real_t *xpnt = nullptr, *coef = nullptr, *geom = nullptr;
    real_t *dens = nullptr, *fock = nullptr, *schwarz = nullptr;
    hipEvent_t start = nullptr, stop = nullptr;

    double checksum_ = 0.0;
    bool built_ = false;

    void launch();
    void release();
};

template <class real_t, bool LegacySqrt>
HIPHartreeFock<real_t, LegacySqrt>::HIPHartreeFock(const Deck& deck, uint32_t block_size)
  : ngauss(deck.ngauss), natoms(deck.natoms), blk_size(block_size)
{
	nn = pair_count(natoms);
	nsq = (size_t)natoms * natoms;
	nnnn = quartet_count(natoms);
	n_blks = (uint32_t)grid_blocks(natoms, blk_size);   // validated by main

	// Same preparation as the original main().
	h_xpnt.resize(ngauss); h_coef.resize(ngauss); h_geom.resize(3 * (size_t)natoms);
	xpnt_d = deck.xpnt; coef_d = deck.coef; geom_d = deck.geom;
	for (uint32_t i = 0; i < ngauss; i++)     { h_xpnt[i] = (real_t)deck.xpnt[i]; h_coef[i] = (real_t)deck.coef[i]; }
	for (uint32_t i = 0; i < 3 * natoms; i++)   h_geom[i] = (real_t)deck.geom[i];

	h_dens.resize(nsq); h_fock.resize(nsq); h_schwarz.assign((size_t)(nn + 1), (real_t)0.0);
	dens_d.resize(nsq);
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

	const real_t tobohrs = HostConsts<real_t>::tobohrs();
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
			eri = ssss<real_t, LegacySqrt>(i,j,i,j,ngauss,h_xpnt.data(),h_coef.data(),h_geom.data(), natoms);
			h_schwarz[ij] = r_sqrt(r_abs(eri));
		}
	}

	try
	{
		api_check(hipMalloc(&xpnt,    sizeof(real_t) * ngauss), "hipMalloc (xpnt)");
		api_check(hipMalloc(&coef,    sizeof(real_t) * ngauss), "hipMalloc (coef)");
		api_check(hipMalloc(&geom,    sizeof(real_t) * (3 * (size_t)natoms)), "hipMalloc (geom)");
		api_check(hipMalloc(&dens,    sizeof(real_t) * nsq), "hipMalloc (dens)");
		api_check(hipMalloc(&fock,    sizeof(real_t) * nsq), "hipMalloc (fock)");
		api_check(hipMalloc(&schwarz, sizeof(real_t) * (size_t)(nn + 1)), "hipMalloc (schwarz)");

		api_check(hipMemcpy(xpnt,    h_xpnt.data(),    sizeof(real_t) * ngauss, hipMemcpyHostToDevice), "hipMemcpy (xpnt)");
		api_check(hipMemcpy(coef,    h_coef.data(),    sizeof(real_t) * ngauss, hipMemcpyHostToDevice), "hipMemcpy (coef)");
		api_check(hipMemcpy(geom,    h_geom.data(),    sizeof(real_t) * (3 * (size_t)natoms), hipMemcpyHostToDevice), "hipMemcpy (geom)");
		api_check(hipMemcpy(dens,    h_dens.data(),    sizeof(real_t) * nsq, hipMemcpyHostToDevice), "hipMemcpy (dens)");
		api_check(hipMemcpy(fock,    h_fock.data(),    sizeof(real_t) * nsq, hipMemcpyHostToDevice), "hipMemcpy (fock)");
		api_check(hipMemcpy(schwarz, h_schwarz.data(), sizeof(real_t) * (size_t)(nn + 1), hipMemcpyHostToDevice), "hipMemcpy (schwarz)");

		api_check(hipEventCreate(&start), "hipEventCreate");
		api_check(hipEventCreate(&stop), "hipEventCreate");
	}
	catch (...)
	{
		release();
		throw;
	}
}

template <class real_t, bool LegacySqrt>
HIPHartreeFock<real_t, LegacySqrt>::~HIPHartreeFock()
{
	release();
}

template <class real_t, bool LegacySqrt>
void HIPHartreeFock<real_t, LegacySqrt>::release()
{
	// Best effort: never throw from cleanup.
	if (xpnt) (void)hipFree(xpnt);
	if (coef) (void)hipFree(coef);
	if (geom) (void)hipFree(geom);
	if (dens) (void)hipFree(dens);
	if (fock) (void)hipFree(fock);
	if (schwarz) (void)hipFree(schwarz);
	if (start) (void)hipEventDestroy(start);
	if (stop) (void)hipEventDestroy(stop);
	xpnt = coef = geom = dens = fock = schwarz = nullptr;
	start = stop = nullptr;
}

template <class real_t, bool LegacySqrt>
void HIPHartreeFock<real_t, LegacySqrt>::launch()
{
	dim3 threads(blk_size, 1, 1);
	dim3 grid(n_blks, 1, 1);

	hartree_fock<real_t, LegacySqrt><<<grid, threads>>>(nnnn, ngauss, natoms,
			geom, xpnt, coef, dens,
			schwarz,fock);
	api_check(hipGetLastError(), "hartree_fock launch");
}

template <class real_t, bool LegacySqrt>
void HIPHartreeFock<real_t, LegacySqrt>::first_build()
{
	launch();
	api_check(hipDeviceSynchronize(), "hipDeviceSynchronize (first build)");

	api_check(hipMemcpy(h_fock.data(), fock, sizeof(real_t) * nsq, hipMemcpyDeviceToHost), "hipMemcpy (fock)");

	real_t erep = 0.0;
	for (uint32_t i = 0; i < natoms; i++){
		for (uint32_t j = 0; j < natoms; j++){
			erep += h_fock[i*natoms +j] * h_dens[i*natoms + j];
		}
	}
	checksum_ = (double)erep*0.5;
	built_ = true;
}

template <class real_t, bool LegacySqrt>
double HIPHartreeFock<real_t, LegacySqrt>::timed_build()
{
	float elapsed = 0.0f;
	api_check(hipDeviceSynchronize(), "hipDeviceSynchronize");
	api_check(hipEventRecord(start), "hipEventRecord");
	launch();
	api_check(hipEventRecord(stop), "hipEventRecord");
	api_check(hipEventSynchronize(stop), "hipEventSynchronize");
	api_check(hipEventElapsedTime(&elapsed, start, stop), "hipEventElapsedTime");
	if (!(elapsed > 0.0f))
		throw std::runtime_error("device timer returned no measurement for the Fock build");
	return (double)elapsed * 1.0E-3;
}

template <class real_t, bool LegacySqrt>
FockCheck HIPHartreeFock<real_t, LegacySqrt>::check() const
{
	if (!built_) throw std::runtime_error("check() called before first_build()");

	// Same metrics as the original precision check.
	std::vector<double> fock_gpu(nsq), fock_ref(nsq);
	for (size_t t = 0; t < nsq; t++) fock_gpu[t] = (double)h_fock[t];

	fock_build_cpu(ngauss, natoms, geom_d.data(), xpnt_d.data(), coef_d.data(), dens_d.data(), fock_ref.data());

	FockCheck c;
	c.e_gpu = energy_from_fock(fock_gpu.data(), dens_d.data(), natoms);
	c.e_ref = energy_from_fock(fock_ref.data(), dens_d.data(), natoms);

	double denom = (fabs(c.e_ref) > 0.0) ? fabs(c.e_ref) : 1.0;
	c.rel_loss = fabs(c.e_gpu - c.e_ref) / denom;
	c.rel_inpipe = fabs(checksum_ - c.e_ref) / denom;

	double maxd = 0.0, sumsq = 0.0, fscale = 0.0;
	for (size_t t = 0; t < nsq; t++) {
		double d = fabs(fock_gpu[t] - fock_ref[t]);
		if (d > maxd) maxd = d;
		sumsq += d * d;
		if (fabs(fock_ref[t]) > fscale) fscale = fabs(fock_ref[t]);
	}
	c.rms_abs = sqrt(sumsq / (double)nsq);
	if (fscale == 0.0) fscale = 1.0;
	c.max_abs = maxd;
	c.max_rel = maxd / fscale;

	double asym = 0.0;
	for (uint32_t i = 0; i < natoms; i++)
		for (uint32_t j = 0; j < natoms; j++) {
			double d = fabs(fock_gpu[i*natoms + j] - fock_gpu[j*natoms + i]);
			if (d > asym) asym = d;
		}
	c.asym = asym;
	c.e_symmetrized = energy_symmetrized(fock_gpu.data(), dens_d.data(), natoms);
	return c;
}

} // namespace

std::unique_ptr<HartreeFock> make_hartree_fock(Precision p, SqrtMode m, const Deck& deck,
                                               uint32_t block_size)
{
	const bool legacy = (m == SqrtMode::Legacy);
	if (p == Precision::Double)
		return legacy ? std::unique_ptr<HartreeFock>(new HIPHartreeFock<double, true>(deck, block_size))
		              : std::unique_ptr<HartreeFock>(new HIPHartreeFock<double, false>(deck, block_size));
	return legacy ? std::unique_ptr<HartreeFock>(new HIPHartreeFock<float, true>(deck, block_size))
	              : std::unique_ptr<HartreeFock>(new HIPHartreeFock<float, false>(deck, block_size));
}

bool fp64_atomic_add_supported(int device)
{
	// atomicAdd on double is available on every ROCm-supported GPU.
	(void)device;
	return true;
}
