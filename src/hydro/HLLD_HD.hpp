#ifndef HLLD_HD_HPP_ // NOLINT
#define HLLD_HD_HPP_

#include "AMReX_Extension.H"
#include "AMReX_GpuQualifiers.H"
#include <AMReX.H>
#include <AMReX_REAL.H>
#include <algorithm>

#include "hydro/HydroState.hpp"
#include "util/ArrayView.hpp"
#include "util/valarray.hpp"

// HLLD solver based on Miyoshi & Kusano (2005), hereafter MK5.
// Corrections based on Minoshima & Miyoshi (2021), hereafter MM21.
namespace quokka::Riemann
{

AMREX_FORCE_INLINE AMREX_GPU_DEVICE auto get_u_E(const double gamma, const double vel_magn_sq, const double rho, const double P) -> double {
	// kinetic energy
	const double ke = 0.5 * rho * vel_magn_sq;
	return ke + P / (gamma - 1.0);
}

template <typename problem_t, int N_scalars, int N_mscalars, int fluxdim>
AMREX_FORCE_INLINE AMREX_GPU_DEVICE auto
getF(
	const double u_mx, const double u_E,
	const quokka::HydroState<N_scalars, N_mscalars>& s
) -> quokka::valarray<double, fluxdim> {
	
	// fluxes on the x side of the interface
	quokka::valarray<double, fluxdim> f = {
		u_mx,               // rho
		u_mx * s.u + s.P,   // mx
		s.rho * s.v * s.u,  // my
		s.rho * s.w * s.u,  // mz
		s.u * (u_E + s.P),  // E
		s.Eint * s.u        // Eint
	};
	// passive scalar fluxes
	for (int n = 0; n < N_scalars; ++n) {
		constexpr const int nHydroVars = fluxdim - N_scalars;
		f[nHydroVars + n] = s.scalar[n] * s.u;
	}
	return std::move(f);
}

template <typename problem_t, int N_scalars, int N_mscalars, int fluxdim>
AMREX_FORCE_INLINE AMREX_GPU_DEVICE auto
step_6(
	const double ptot_star, const double spd_C,
	const double spd_X, const double u_mx, const double u_E,
	const quokka::HydroState<N_scalars, N_mscalars>& s
) -> quokka::valarray<double, fluxdim> {
	const auto& f = getF<problem_t,N_scalars,N_mscalars,fluxdim>(u_mx, u_E, s);
	
	const double co_speed_ratio = (spd_C - s.u) / (spd_X - spd_C);
	const double rho = s.rho * co_speed_ratio;
	
	quokka::valarray<double, fluxdim> u_star_minus_u = {
		rho,         // rho
		rho * spd_X, // mx
		rho * s.v,   // my
		rho * s.w,   // mz
		co_speed_ratio * u_E + (ptot_star * spd_C - s.P * s.u) / (spd_X - spd_C), // E
		s.Eint * co_speed_ratio  // Eint
	};
	for (int n = 0; n < N_scalars; ++n) {
		constexpr const int nHydroVars = fluxdim - N_scalars;
		u_star_minus_u[nHydroVars + n] = s.scalar[n] * co_speed_ratio;
	}
	// return u_star
	return std::move(f + spd_X * u_star_minus_u);
}

template <typename problem_t, int N_scalars, int N_mscalars, int fluxdim>
AMREX_FORCE_INLINE AMREX_GPU_DEVICE auto HLLD_HD(
	quokka::HydroState<N_scalars, N_mscalars> const &sL,
	quokka::HydroState<N_scalars, N_mscalars> const &sR,
	const double gamma,
	const double perp_v_jump
) -> quokka::valarray<double, fluxdim> {
	//--- Step 1. Compute L/R states

	// frequently used term
	const double vel_magn_sq_L = SQUARE(sL.u) + (SQUARE(sL.v) + SQUARE(sL.w));
	const double vel_magn_sq_R = SQUARE(sR.u) + (SQUARE(sR.v) + SQUARE(sR.w));

	// set left and right conserved states
	const double u_L_mx = sL.u * sL.rho;
	const double u_R_mx = sR.u * sR.rho;

	//--- Step 2. Compute L & R wave speeds according to MK5, eqn. (67)

	const double spd_fms_L = std::sqrt(gamma * sL.P / sL.rho);
	const double spd_fms_R = std::sqrt(gamma * sR.P / sR.rho);
	const double spd_fms_max = std::max(spd_fms_L, spd_fms_R);
	const double u_min = std::min(sL.u, sR.u);
	const double u_max = std::max(sL.u, sR.u);
	// signal speeds (left to right), spd_C defined further down
	const double spd_L = std::min(0.0, u_min - spd_fms_max);
	const double spd_R = std::max(0.0, u_max + spd_fms_max);

	//--- Step 3. Compute L/R fluxes
	
	// u_L is supersonic
	if (spd_L >= 0.0)
		return getF<problem_t,N_scalars,N_mscalars,fluxdim>(u_L_mx, get_u_E(gamma, vel_magn_sq_L, sL.rho, sL.P), sL);
	
	// u_R is supersonic
	if (spd_R <= 0.0)
		return getF<problem_t,N_scalars,N_mscalars,fluxdim>(u_R_mx, get_u_E(gamma, vel_magn_sq_R, sR.rho, sR.P), sR);
	
	
	//--- Step 4. Compute middle and Alfven wave speeds

	// MK5: S_i - u_i (for i=L or R)
	const double siui_L = spd_L - sL.u;
	const double siui_R = spd_R - sR.u;
	// carbuncle detector
	const double para_v_jump = sR.u - sL.u; // negative -> compression
	// tp := shock anisotropy, clamped to [0, 1], with theta = tp^4
	const double denom_tp = std::max(1e-14, spd_fms_max - std::min(perp_v_jump, 0.0));
	const double tp = (spd_fms_max - std::min(para_v_jump, 0.0)) / denom_tp;
	const double theta = SQUARE(SQUARE(amrex::Clamp(tp, 0.0, 1.0)));
	// modified middle speed S_M from MK5 eqn 38 with theta from MM21 eqn 9
	const double sm_denom = siui_R * sR.rho - siui_L * sL.rho;
	const double spd_C = (siui_R * u_R_mx - siui_L * u_L_mx + theta * (sL.P - sR.P)) / sm_denom;

	//--- Step 5. Compute intermediate states

	// total pressure (w/ MM21 low-Mach correction)
	// cu := modified fast magnetosonic speed (MM21 eqn 14)
	const double spd_Cu_max = std::max(
		std::sqrt(vel_magn_sq_L),
		std::sqrt(vel_magn_sq_R)
	);
	// limiter (MM21 eqn 16)
	const double chi = std::min(1.0, spd_Cu_max / std::max(1e-14, spd_fms_max));
	const double phi = chi * (2.0 - chi);
	// corrected total star pressure (MM21 eqn 15)
	const double ptot_star_numer_term1 = siui_R * sR.rho * sL.P;
	const double ptot_star_numer_term2 = siui_L * sL.rho * sR.P;
	const double ptot_star_numer_term3 = phi * (sL.rho * sR.rho) * (siui_R * siui_L) * (sR.u - sL.u);
	const double ptot_star_numer = ptot_star_numer_term1 - ptot_star_numer_term2 + ptot_star_numer_term3;
	const double ptot_star = ptot_star_numer / sm_denom;
	
	
	//--- Step 6. Compute fluxes
	
	if (spd_C >= 0.0)
		return step_6<problem_t,N_scalars,N_mscalars,fluxdim>(ptot_star, spd_C, spd_L, u_L_mx, get_u_E(gamma, vel_magn_sq_L, sL.rho, sL.P), sL);
	 // else
		return step_6<problem_t,N_scalars,N_mscalars,fluxdim>(ptot_star, spd_C, spd_R, u_R_mx, get_u_E(gamma, vel_magn_sq_R, sR.rho, sR.P), sR);
}

} // namespace quokka::Riemann
#endif // HLLD_HD_HPP_