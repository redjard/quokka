//==============================================================================
// TwoMomentRad - a radiation transport library for patch-based AMR codes
// Copyright 2024 Benjamin Wibking.
// Released under the MIT license. See LICENSE file included in the GitHub repo.
//==============================================================================
/// \file testDiskGalaxy_no_mhd.cpp
/// \brief Defines a simulation using disk galaxy initial conditions.
///

#include <cmath>
#include <optional>

#include "AMReX_Array.H"
#include "AMReX_BLassert.H"
#include "AMReX_FabArrayBase.H"
#include "AMReX_GpuContainers.H"
#include "AMReX_GpuDevice.H"
#include "AMReX_MultiFab.H"
#include "AMReX_MultiFabUtil.H"
#include "AMReX_ParallelContext.H"
#include "AMReX_ParallelReduce.H"
#include "AMReX_Parser.H"
#include "AMReX_Print.H"
#include "AMReX_REAL.H"
#include "AMReX_Reduce.H"
#include "AMReX_iMultiFab.H"

#include "QuokkaSimulation.hpp"
#include "SimulationData.hpp"
#include "fundamental_constants.H"
#include "hydro/EOS.hpp"
#include "hydro/hydro_system.hpp"
#include "math/interpolate.hpp"
#include "math/quadrature.hpp"
#include "math/spherical_geometry.hpp"
// #include "particles/particle_types.hpp"
#include "physics_info.hpp"
// #include "util/BC.hpp"
#include "util/DataTable.hpp"

struct DiskGalaxy_no_mhd {
};

static_assert(AMREX_SPACEDIM == 3, "DiskGalaxy_no_mhd problem requires AMREX_SPACEDIM == 3.");

template <> struct quokka::EOS_Traits<DiskGalaxy_no_mhd> {
	static constexpr double gamma = 5. / 3.;
	static constexpr double mean_molecular_weight = 0.6 * C::m_u;
	// using EOSBackend = quokka::EOSTabulated<DiskGalaxy_no_mhd>;
};

template <> struct HydroSystem_Traits<DiskGalaxy_no_mhd> {
	static constexpr bool reconstruct_eint = true;
};

// struct DefaultPhysicsTraits {
// 	static constexpr bool is_hydro_enabled = false;
// 	static constexpr int numMassScalars = 0;
// 	// NOTE: numPassiveScalars is evaluated at the point of definition of DefaultPhysicsTraits, not
// 	// at the point of specialization. If you override numMassScalars, you MUST also explicitly
// 	// override numPassiveScalars, or it will silently inherit the pre-evaluated default of 0.
// 	static constexpr int numPassiveScalars = numMassScalars + 0;
// 	static constexpr bool is_radiation_enabled = false;
// 	static constexpr bool is_dust_enabled = false;
// 	static constexpr bool is_self_gravity_enabled = false;
// 	static constexpr bool is_mhd_enabled = false;
// 	static constexpr ResistivityModel resistivity_model = ResistivityModel::none;
// 	static constexpr int nGroups = 1;     // number of radiation groups
// 	static constexpr int nDustGroups = 1; // number of dust groups
// 	static constexpr UnitSystem unit_system = UnitSystem::CGS;
// 	static constexpr double boltzmann_constant = C::k_B;	    // Hydro, EOS
// 	static constexpr double gravitational_constant = C::Gconst; // gravity
// 	static constexpr double c_light = C::c_light;		    // radiation
// 	static constexpr double radiation_constant = C::a_rad;	    // radiation
// 	static constexpr double unit_length = 1.0;
// 	static constexpr double unit_mass = 1.0;
// 	static constexpr double unit_time = 1.0;
// 	static constexpr double unit_temperature = 1.0;
// };
template <> struct Physics_Traits<DiskGalaxy_no_mhd> : DefaultPhysicsTraits {
	static constexpr bool is_hydro_enabled = true;
	static constexpr bool is_self_gravity_enabled = true;
	// static constexpr bool is_mhd_enabled = true;
};

template <> struct Particle_Traits<DiskGalaxy_no_mhd> : DefaultParticleTraits {
	static constexpr ParticleSwitch particle_switch = ParticleSwitch::CIC | ParticleSwitch::StochasticStellarPop;
};

template <> struct SimulationData<DiskGalaxy_no_mhd> {
	amrex::Real r_inner{};
	amrex::Real r_outer{};
	amrex::Real vcirc_outer{};
	amrex::Real rho_outer{};
	amrex::Real velr_outer{};
	amrex::Real temp_outer{};

	amrex::Real vcirc_inner{};
	amrex::Real rho_inner{};
	amrex::Real velr_inner{};
	amrex::Real temp_inner{};

	amrex::Gpu::PinnedVector<amrex::Real> radius;
	amrex::Gpu::PinnedVector<amrex::Real> vcirc;
	amrex::Gpu::PinnedVector<amrex::Real> rho_halo;
	amrex::Gpu::PinnedVector<amrex::Real> velr_halo;
	amrex::Gpu::PinnedVector<amrex::Real> temp_halo;

	std::string haloVphiExpr;
	bool useHaloVphiParser = false;
	std::optional<amrex::Parser> haloVphiParser;
	std::optional<amrex::ParserExecutor<3>> haloVphiParserExe;
};


template <> void QuokkaSimulation<DiskGalaxy_no_mhd>::preCalculateInitialConditions()
{
	// amrex::Print() << "preCalculateInitialConditions\n";
	auto start = clock();
	// 1. read in circular velocity table "vcirc.dat"
	// get circular velocity profile filename from ParmParse
	amrex::ParmParse const pp("disk_galaxy");
	std::string filename;
	pp.query("vcirc_file", filename);
	double speed_factor = NAN;
	pp.query("speed_factor", speed_factor);
	AMREX_ALWAYS_ASSERT(!std::isnan(speed_factor));

	auto halo_table = quokka::DataTable<1, 4, quokka::OutOfBounds::clamp>::CSVReader(filename, quokka::TransformType::linear);
	auto const halo_table_const = halo_table.const_tables_host();
	AMREX_ALWAYS_ASSERT_WITH_MESSAGE(halo_table_const.sizes[0] > 0, "disk_galaxy.vcirc_file contained no numeric rows.");
	AMREX_ALWAYS_ASSERT_WITH_MESSAGE(halo_table_const.spacing_types[0] == quokka::TransformType::linear,
					 "disk_galaxy.vcirc_file must use linear spacing for the radius coordinate.");

	// 2. copy data to simData_.radius and simData_.vcirc
	const auto N = static_cast<size_t>(halo_table_const.sizes[0]);
	userData_.radius.resize(N);
	userData_.vcirc.resize(N);
	userData_.rho_halo.resize(N);
	userData_.velr_halo.resize(N);
	userData_.temp_halo.resize(N);

	const double length_unit = 1.0e3 * C::parsec; // kpc
	const double vel_unit = 1.0e5 * speed_factor; // km/s
	for (size_t i = 0; i < N; ++i) {
		amrex::Real const radius = halo_table_const.coord_min[0] + static_cast<amrex::Real>(i) * halo_table_const.dcoord[0];
		userData_.radius[i] = radius * length_unit;
		userData_.vcirc[i] = halo_table_const.dataViewArrays[0](static_cast<int>(i)) * vel_unit;
		userData_.rho_halo[i] = halo_table_const.dataViewArrays[1](static_cast<int>(i));
		userData_.velr_halo[i] = halo_table_const.dataViewArrays[2](static_cast<int>(i)) * speed_factor;
		userData_.temp_halo[i] = halo_table_const.dataViewArrays[3](static_cast<int>(i));
	}
	// amrex::Print() << "REDJARD: halo_table_const.dcoord[0] = " << halo_table_const.dcoord[0] << "\n";  // 0.02054794999
	// amrex::Print() << "REDJARD: halo_table_const.coord_min[0] = " << halo_table_const.coord_min[0] << "\n"; // 0.020548
	// amrex::Print() << "REDJARD: halo_table_const.coord_max[0] = " << halo_table_const.coord_max[0] << "\n"; // 205.4795

	// save min/max radii
	userData_.r_inner = halo_table_const.coord_min[0] * length_unit;
	userData_.vcirc_inner = halo_table_const.dataViewArrays[0](0) * vel_unit;
	userData_.rho_inner = halo_table_const.dataViewArrays[1](0);
	userData_.velr_inner = halo_table_const.dataViewArrays[2](0);
	userData_.temp_inner = halo_table_const.dataViewArrays[3](0);

	userData_.r_outer = halo_table_const.coord_max[0] * length_unit;
	userData_.vcirc_outer = halo_table_const.dataViewArrays[0](static_cast<int>(N - 1)) * vel_unit;
	userData_.rho_outer = halo_table_const.dataViewArrays[1](static_cast<int>(N - 1));
	userData_.velr_outer = halo_table_const.dataViewArrays[2](static_cast<int>(N - 1));
	userData_.temp_outer = halo_table_const.dataViewArrays[3](static_cast<int>(N - 1));

	// optional halo v_phi expression (variables: x, y, z)
	userData_.haloVphiExpr.clear();
	pp.query("halo_vphi_expr", userData_.haloVphiExpr);
	userData_.useHaloVphiParser = !userData_.haloVphiExpr.empty();
	if (userData_.useHaloVphiParser) {
		userData_.haloVphiParser.emplace(userData_.haloVphiExpr);
		userData_.haloVphiParser->registerVariables({"x", "y", "z"});
		userData_.haloVphiParserExe = userData_.haloVphiParser->compile<3>();
#ifdef AMREX_USE_GPU
		if (userData_.haloVphiParserExe->m_device_executor == nullptr) {
			amrex::Abort("disk_galaxy.halo_vphi_expr: device parser executor is null after compile<3>()");
		}
#endif
		userData_.haloVphiParser.reset();
	} else {
		userData_.haloVphiParser.reset();
		userData_.haloVphiParserExe.reset();
	}
	amrex::Print() << "REDJARD: ran preCalculateInitialConditions in " << float(clock() - start)/1e6 << " s\n";
}

template <> void QuokkaSimulation<DiskGalaxy_no_mhd>::setInitialConditionsOnGrid(quokka::grid const &grid_elem)
{
	auto start = clock();
	// amrex::Print() << "setInitialConditionsOnGrid\n";
	// read parameters
	//
	amrex::ParmParse const pp("disk_galaxy");

	// double magnetic_field_microgauss = 1.0; // default B-field strength
	// pp.query("magnetic_field_microgauss", magnetic_field_microgauss);
	// const double B_0 = magnetic_field_microgauss * 1.0e-6 / std::sqrt(4.0 * M_PI);

	// disc parameters
	double disk_gas_mass_Msun = NAN;     // disk mass
	double disk_Rscale_kpc = NAN;	     // disk scale length
	double disk_zscale_kpc = NAN;	     // disk scale height
	double T_disk = NAN;		     // K
	double disk_perturb_amplitude = NAN; // amplitude of harmonic mode perturbation
	double disk_perturb_Rmax_kpc = NAN;  // max radius (in kpc) for harmonic mode perturbations
	// double initial_scalar_density = NAN; // scalar density at cgs units (cm^-3)
	pp.query("disk_gas_mass_Msun", disk_gas_mass_Msun);
	pp.query("disk_Rscale_kpc", disk_Rscale_kpc);
	pp.query("disk_zscale_kpc", disk_zscale_kpc);
	pp.query("disk_temperature", T_disk);
	pp.query("disk_perturb_amplitude", disk_perturb_amplitude);
	pp.query("disk_perturb_Rmax_kpc", disk_perturb_Rmax_kpc);
	// pp.query("initial_scalar_density", initial_scalar_density);
	AMREX_ALWAYS_ASSERT(!std::isnan(disk_gas_mass_Msun));
	AMREX_ALWAYS_ASSERT(!std::isnan(disk_Rscale_kpc));
	AMREX_ALWAYS_ASSERT(!std::isnan(disk_zscale_kpc));
	AMREX_ALWAYS_ASSERT(!std::isnan(T_disk));
	AMREX_ALWAYS_ASSERT(!std::isnan(disk_perturb_amplitude));
	AMREX_ALWAYS_ASSERT(!std::isnan(disk_perturb_Rmax_kpc));
	// AMREX_ALWAYS_ASSERT(!std::isnan(initial_scalar_density));

	const double disk_gas_mass = disk_gas_mass_Msun * C::M_solar;
	const double R_d = disk_Rscale_kpc * (1.0e3 * C::parsec);
	const double z_d = disk_zscale_kpc * (1.0e3 * C::parsec);
	const double R_max_perturb = disk_perturb_Rmax_kpc * (1e3 * C::parsec);
	const double rho_0 = disk_gas_mass / 4. / M_PI / (R_d * R_d) / z_d; // normalization constant

	// read tables

	double const *R_table = userData_.radius.dataPtr();
	double const *vcirc_table = userData_.vcirc.dataPtr();
	double const *rhoH_table = userData_.rho_halo.dataPtr();
	double const *velr_table = userData_.velr_halo.dataPtr();
	double const *temp_table = userData_.temp_halo.dataPtr();

	auto const len_table = static_cast<int>(userData_.radius.size());
	const amrex::Real R_table_min = userData_.r_inner;
	const amrex::Real rho_inner = userData_.rho_inner;
	const amrex::Real vcirc_inner = userData_.vcirc_inner;
	const amrex::Real velr_inner = userData_.velr_inner;
	const amrex::Real temp_inner = userData_.temp_inner;

	const amrex::Real R_table_max = userData_.r_outer;
	const amrex::Real vcirc_outer = userData_.vcirc_outer;
	const amrex::Real rho_outer = userData_.rho_outer;
	const amrex::Real velr_outer = userData_.velr_outer;
	const amrex::Real temp_outer = userData_.temp_outer;
	const bool use_halo_vphi_parser = userData_.useHaloVphiParser;
	amrex::ParserExecutor<3> halo_vphi_parser{};
	if (use_halo_vphi_parser) {
		if (userData_.haloVphiParserExe.has_value()) {
			halo_vphi_parser = *userData_.haloVphiParserExe;
		} else {
			amrex::Abort("disk_galaxy.halo_vphi_expr: parser executor is missing after compile<3>()");
		}
	}

	const amrex::Box &indexRange = grid_elem.indexRange_;
	const amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> dx = grid_elem.dx_;
	const amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> prob_lo = grid_elem.prob_lo_;
	const amrex::Array4<double> &state_cc = grid_elem.array_;

	// particles.scalar_yield_per_SN must be set as well, and it should be greater than (initial_scalar_density * (128 pc)^3),
	// so that the SN ejected metal density in SN remnant is greater than the background density.
	// amrex::ParmParse const pp_particles("particles");
	// double scalar_yield_per_SN = NAN;
	// pp_particles.query("scalar_yield_per_SN", scalar_yield_per_SN);
	// AMREX_ALWAYS_ASSERT(!std::isnan(scalar_yield_per_SN));
	// const Real SNR_volume = std::pow(128.0 * C::parsec, 3);
	// AMREX_ALWAYS_ASSERT_WITH_MESSAGE(scalar_yield_per_SN > initial_scalar_density * SNR_volume,
	// 				 "particles.scalar_yield_per_SN must be greater than (initial_scalar_density * (128 pc)^3)");

	amrex::ParallelFor(indexRange, [=] AMREX_GPU_DEVICE(int i, int j, int k) {
		// Cartesian coordinates
		amrex::Real const x0 = prob_lo[0] + (i * dx[0]);
		amrex::Real const y0 = prob_lo[1] + (j * dx[1]);
		amrex::Real const z0 = prob_lo[2] + (k * dx[2]);

		amrex::Real const x1 = prob_lo[0] + ((i + 1) * dx[0]);
		amrex::Real const y1 = prob_lo[1] + ((j + 1) * dx[1]);
		amrex::Real const z1 = prob_lo[2] + ((k + 1) * dx[2]);

		// amrex::Real const x_mid = 0.5 * (x0 + x1);
		// amrex::Real const y_mid = 0.5 * (y0 + y1);
		// amrex::Real const z_mid = 0.5 * (z0 + z1);
		// amrex::Real const R_mid = std::sqrt((x_mid * x_mid) + (y_mid * y_mid));

		// amrex::Real const B_phi = B_0 * std::exp(-R_mid / R_d) * std::exp(-std::abs(z_mid) / z_d);
		// amrex::Real Bx = 0.0;
		// amrex::Real By = 0.0;
		// if (R_mid > 0.0) {
		// 	Bx = -B_phi * y_mid / R_mid;
		// 	By = B_phi * x_mid / R_mid;
		// }
		// amrex::Real const Emag = 0.5 * ((Bx * Bx) + (By * By));

		auto vcirc_exact = [R_table_min, R_table_max, R_table, vcirc_inner, vcirc_outer, vcirc_table, len_table](const amrex::Real R) {
			double vcirc = NAN;
			if (R > R_table_min && R < R_table_max) {
				vcirc = interpolate_value(R, R_table, vcirc_table, len_table);
			} else if (R >= R_table_max) {
				vcirc = vcirc_outer;
			} else if (R <= R_table_min) {
				vcirc = vcirc_inner;
			}
			return vcirc;
		};

		// compute velocity profiles
		auto vx_exact = [vcirc_exact](double x, double y, double _z ) {
			double const R = std::sqrt(std::pow(x, 2) + std::pow(y, 2));
			double const theta = std::atan2(x, y);
			return -vcirc_exact(R) * std::cos(theta); // vx
		};

		auto vy_exact = [vcirc_exact](double x, double y, double _z ) {
			double const R = std::sqrt(std::pow(x, 2) + std::pow(y, 2));
			double const theta = std::atan2(x, y);
			return vcirc_exact(R) * std::sin(theta); // vy
		};

		auto rhoHalo = [R_table_min, R_table, R_table_max, rho_inner, rho_outer, rhoH_table, len_table](const amrex::Real R) {
			double rho_H = NAN;
			if (R > R_table_min && R < R_table_max) {
				rho_H = interpolate_value(R, R_table, rhoH_table, len_table);
			} else if (R <= R_table_min) {
				rho_H = rho_inner;
			} else {
				rho_H = rho_outer;
			}

			return rho_H;
		};

		auto velHalo = [R_table_min, R_table, R_table_max, velr_inner, velr_outer, velr_table, len_table](const amrex::Real R) {
			double vel_H = NAN;
			if (R > R_table_min && R < R_table_max) {
				vel_H = interpolate_value(R, R_table, velr_table, len_table);
			} else if (R <= R_table_min) {
				vel_H = velr_inner;
			} else {
				vel_H = velr_outer;
			}
			return -vel_H;
		};

		auto tempHalo = [R_table_min, R_table, R_table_max, temp_inner, temp_outer, temp_table, len_table](const amrex::Real R) {
			double temp_H = NAN;
			if (R > R_table_min && R < R_table_max) {
				temp_H = interpolate_value(R, R_table, temp_table, len_table);
			} else if (R <= R_table_min) {
				temp_H = temp_inner;
			} else {
				temp_H = temp_outer;
			}
			return temp_H;
		};

		// compute density profiles
		auto rhoHalo_exact = [rhoHalo](double x, double y, double z) {
			double const r = std::sqrt(std::pow(x, 2) + std::pow(y, 2) + std::pow(z, 2));
			return rhoHalo(r);
		};

		auto tempHalo_exact = [tempHalo](double x, double y, double z) {
			double const r = std::sqrt(std::pow(x, 2) + std::pow(y, 2) + std::pow(z, 2));
			return tempHalo(r);
		};

		auto rhoDisk_exact = [rho_0, R_d, z_d, disk_perturb_amplitude, R_max_perturb](double x, double y, double z) {
			double const R = std::sqrt(std::pow(x, 2) + std::pow(y, 2));
			double const theta = std::atan2(x, y);
			double const drho_over_rho = disk_perturb_amplitude * jn(2, 5.1356 * R / R_max_perturb) * std::sin(2.0 * theta);
			return rho_0 * std::exp(-R / R_d) * std::exp(-std::abs(z) / z_d) * (1.0 + drho_over_rho);
		};

		// compute momenta profiles
		auto vphiHalo_exact = [=] AMREX_GPU_DEVICE(double x, double y, double z) {
			if (use_halo_vphi_parser) {
				return halo_vphi_parser(x, y, z);
			}
			return 0.0;
		};

		auto velx_exact = [velHalo, vphiHalo_exact](double x, double y, double z) {
			double const r = std::sqrt(std::pow(x, 2) + std::pow(y, 2) + std::pow(z, 2));
			double const R = std::sqrt(std::pow(x, 2) + std::pow(y, 2));
			double const vr_component = (r > 0.0) ? (velHalo(r) * x / r) : 0.0;
			double const vphi_component = (R > 0.0) ? (-vphiHalo_exact(x, y, z) * y / R) : 0.0;
			return vr_component + vphi_component; // vx
		};

		auto vely_exact = [velHalo, vphiHalo_exact](double x, double y, double z) {
			double const r = std::sqrt(std::pow(x, 2) + std::pow(y, 2) + std::pow(z, 2));
			double const R = std::sqrt(std::pow(x, 2) + std::pow(y, 2));
			double const vr_component = (r > 0.0) ? (velHalo(r) * y / r) : 0.0;
			double const vphi_component = (R > 0.0) ? (vphiHalo_exact(x, y, z) * x / R) : 0.0;
			return vr_component + vphi_component; // vy
		};

		auto velz_exact = [velHalo](double x, double y, double z) {
			double const r = std::sqrt(std::pow(x, 2) + std::pow(y, 2) + std::pow(z, 2));
			return (r > 0.0) ? (velHalo(r) * z / r) : 0.0; // vz
		};

		// integrate profiles over cell volume
		const double cell_vol = dx[0] * dx[1] * dx[2];
		constexpr double gamma_gas = quokka::EOS_Traits<DiskGalaxy_no_mhd>::gamma;
		// constexpr double gamma_gas = 5. / 3.;
		constexpr double mu = 0.61;

		auto rho_total_exact = [=] AMREX_GPU_DEVICE(double x, double y, double z) { return rhoDisk_exact(x, y, z) + rhoHalo_exact(x, y, z); };

		auto momx_total_exact = [=] AMREX_GPU_DEVICE(double x, double y, double z) {
			const double rho_disk_local = rhoDisk_exact(x, y, z);
			const double rho_halo_local = rhoHalo_exact(x, y, z);
			return rho_disk_local * vx_exact(x, y, z) + rho_halo_local * velx_exact(x, y, z);
		};

		auto momy_total_exact = [=] AMREX_GPU_DEVICE(double x, double y, double z) {
			const double rho_disk_local = rhoDisk_exact(x, y, z);
			const double rho_halo_local = rhoHalo_exact(x, y, z);
			return rho_disk_local * vy_exact(x, y, z) + rho_halo_local * vely_exact(x, y, z);
		};

		auto momz_total_exact = [=] AMREX_GPU_DEVICE(double x, double y, double z) {
			const double rho_halo_local = rhoHalo_exact(x, y, z);
			return rho_halo_local * velz_exact(x, y, z);
		};

		auto eint_total_exact = [=] AMREX_GPU_DEVICE(double x, double y, double z) {
			const double rho_disk_local = rhoDisk_exact(x, y, z);
			const double rho_halo_local = rhoHalo_exact(x, y, z);
			const double temp_halo_local = tempHalo_exact(x, y, z);
			const double eint_disk_local = (rho_disk_local > 0.0) ? (rho_disk_local * C::k_B * T_disk / (mu * C::m_p * (gamma_gas - 1.0))) : 0.0;
			const double eint_halo_local =
			    (rho_halo_local > 0.0) ? (rho_halo_local * C::k_B * temp_halo_local / (mu * C::m_p * (gamma_gas - 1.0))) : 0.0;
			return eint_disk_local + eint_halo_local;
		};

		const double rho = quad_3d(rho_total_exact, x0, x1, y0, y1, z0, z1) / cell_vol;
		AMREX_ALWAYS_ASSERT(rho > 0.0);
		const double momx = quad_3d(momx_total_exact, x0, x1, y0, y1, z0, z1) / cell_vol;
		const double momy = quad_3d(momy_total_exact, x0, x1, y0, y1, z0, z1) / cell_vol;
		const double momz = quad_3d(momz_total_exact, x0, x1, y0, y1, z0, z1) / cell_vol;
		const double Eint = quad_3d(eint_total_exact, x0, x1, y0, y1, z0, z1) / cell_vol;

		// Add up disk and halo contributions
		double const rho_disk_halo = rho;
		double const momx_disk_halo = momx;
		double const momy_disk_halo = momy;
		double const momz_disk_halo = momz;
		double const Ekin_disk_halo =
		    0.5 * (momx_disk_halo * momx_disk_halo + momy_disk_halo * momy_disk_halo + momz_disk_halo * momz_disk_halo) / rho_disk_halo;
		double const Eint_disk_halo = Eint;
		// double const Etot_disk_halo = Eint_disk_halo + Ekin_disk_halo + Emag;
		double const Etot_disk_halo = Eint_disk_halo + Ekin_disk_halo + 0;

		state_cc(i, j, k, HydroSystem<DiskGalaxy_no_mhd>::density_index) = rho_disk_halo;
		state_cc(i, j, k, HydroSystem<DiskGalaxy_no_mhd>::x1Momentum_index) = momx_disk_halo;
		state_cc(i, j, k, HydroSystem<DiskGalaxy_no_mhd>::x2Momentum_index) = momy_disk_halo;
		state_cc(i, j, k, HydroSystem<DiskGalaxy_no_mhd>::x3Momentum_index) = momz_disk_halo;
		state_cc(i, j, k, HydroSystem<DiskGalaxy_no_mhd>::energy_index) = Etot_disk_halo;
		state_cc(i, j, k, HydroSystem<DiskGalaxy_no_mhd>::internalEnergy_index) = Eint_disk_halo;

		// // first capture on device
		// const auto initial_scalar_density_d = initial_scalar_density;

		// // Initialize passive scalar field
		// if constexpr (Physics_Traits<DiskGalaxy_no_mhd>::numPassiveScalars > 0) {
		// 	state_cc(i, j, k, HydroSystem<DiskGalaxy_no_mhd>::scalar0_index) = initial_scalar_density_d;
		// }
	});
	amrex::Print() << "REDJARD: ran setInitialConditionsOnGrid in " << float(clock() - start)/1e6 << " s\n";
}

template <> void QuokkaSimulation<DiskGalaxy_no_mhd>::refineGrid(int lev, amrex::TagBoxArray &tags, amrex::Real _time, int _ngrow)
{
	// amrex::Print() << "refineGrid\n";
	auto start = clock();
	// geometrical refinement
	// tag cells within the cylinder defined by R < Rmax and abs(z) < zmax
	amrex::ParmParse const pp("disk_galaxy");
	amrex::Real refine_Rmax_kpc = NAN;
	amrex::Real refine_zmax_kpc = NAN;
	pp.query("refine_Rmax_kpc", refine_Rmax_kpc);
	pp.query("refine_zmax_kpc", refine_zmax_kpc);
	const amrex::Real refine_Rmax = refine_Rmax_kpc * (1.0e3 * C::parsec);
	const amrex::Real refine_zmax = refine_zmax_kpc * (1.0e3 * C::parsec);

	const auto prob_lo = geom[lev].ProbLoArray();
	const auto dx = geom[lev].CellSizeArray();
	const auto tag = tags.arrays();

	amrex::ParallelFor(tags, [=] AMREX_GPU_DEVICE(int bx, int i, int j, int k) noexcept {
		// NOTE: must check all nodes of the cell!
		// Otherwise, cells that are too big can completely prevent refinement.
		amrex::Real const x0 = prob_lo[0] + (i * dx[0]);
		amrex::Real const y0 = prob_lo[1] + (j * dx[1]);
		amrex::Real const z0 = prob_lo[2] + (k * dx[2]);

		amrex::Real const x1 = prob_lo[0] + ((i + 1) * dx[0]);
		amrex::Real const y1 = prob_lo[1] + ((j + 1) * dx[1]);
		amrex::Real const z1 = prob_lo[2] + ((k + 1) * dx[2]);

		auto tagIfPointInRegion = [=](amrex::Real x, amrex::Real y, amrex::Real z) {
			amrex::Real const R = std::sqrt(x * x + y * y);
			if ((R < refine_Rmax) && (std::abs(z) < refine_zmax)) {
				tag[bx](i, j, k) = amrex::TagBox::SET;
			}
		};

		for (auto const &x : {x0, x1}) {
			for (auto const &y : {y0, y1}) {
				for (auto const &z : {z0, z1}) {
					tagIfPointInRegion(x, y, z);
				}
			}
		}
	});
	amrex::Gpu::streamSynchronize();
	amrex::Print() << "REDJARD: ran refineGrid in " << float(clock() - start)/1e6 << " s\n";
}

auto problem_main() -> int
{
	auto start = clock();
	
	QuokkaSimulation<DiskGalaxy_no_mhd> sim;

	// initialize
	sim.setInitialConditions();
	
	amrex::Print() << "\nREDJARD: initialized in " << float(clock() - start)/1e6 << " s\n\n";

	// evolve
	sim.evolve();

	const int status = 0;
	return status;
}
