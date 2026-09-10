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
#include "AMReX_ParallelContext.H"
#include "AMReX_ParallelReduce.H"
#include "AMReX_Parser.H"
#include "AMReX_Print.H"
#include "AMReX_REAL.H"
#include "AMReX_Reduce.H"

#include "QuokkaSimulation.hpp"
#include "SimulationData.hpp"
#include "fundamental_constants.H"
#include "hydro/EOS.hpp"
#include "hydro/hydro_system.hpp"
#include "math/interpolate.hpp"
#include "math/quadrature.hpp"
#include "math/spherical_geometry.hpp"
#include "physics_info.hpp"
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
	static constexpr ParticleSwitch particle_switch = ParticleSwitch::CIC;
};

template <> struct SimulationData<DiskGalaxy_no_mhd> { // userData_
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
	std::optional<amrex::ParserExecutor<5>> haloVphiParserExe;
};


template <> void QuokkaSimulation<DiskGalaxy_no_mhd>::preCalculateInitialConditions()
{
	// 1. read in circular velocity table "vcirc.dat"
	// get circular velocity profile filename from ParmParse
	amrex::ParmParse const pp("disk_galaxy");
	std::string filename;
	pp.query("vcirc_file", filename);
	double length_factor = 1.0;
	pp.query("length_factor", length_factor);
	double speed_factor = 1.0;
	pp.query("speed_factor", speed_factor);
	double halo_density_factor = 1.0;
	pp.query("halo_density_factor", halo_density_factor);

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

	const double length_unit = 1.0e3 * C::parsec * length_factor; // kpc
	const double vel_unit = 1.0e5 * speed_factor; // km/s
	for (size_t i = 0; i < N; ++i) {
		amrex::Real const radius = halo_table_const.coord_min[0] + static_cast<amrex::Real>(i) * halo_table_const.dcoord[0];
		userData_.radius[i] = radius * length_unit;
		userData_.vcirc[i] = halo_table_const.dataViewArrays[0](static_cast<int>(i)) * vel_unit;
		userData_.rho_halo[i] = halo_table_const.dataViewArrays[1](static_cast<int>(i)) * halo_density_factor;
		userData_.velr_halo[i] = halo_table_const.dataViewArrays[2](static_cast<int>(i)) * speed_factor;
		userData_.temp_halo[i] = halo_table_const.dataViewArrays[3](static_cast<int>(i));
	}

	// save min/max radii
	userData_.r_inner = halo_table_const.coord_min[0] * length_unit;
	userData_.vcirc_inner = halo_table_const.dataViewArrays[0](0) * vel_unit;
	userData_.rho_inner = halo_table_const.dataViewArrays[1](0) * halo_density_factor;
	userData_.velr_inner = halo_table_const.dataViewArrays[2](0) * speed_factor;
	userData_.temp_inner = halo_table_const.dataViewArrays[3](0);

	userData_.r_outer = halo_table_const.coord_max[0] * length_unit;
	userData_.vcirc_outer = halo_table_const.dataViewArrays[0](static_cast<int>(N - 1)) * vel_unit;
	userData_.rho_outer = halo_table_const.dataViewArrays[1](static_cast<int>(N - 1)) * halo_density_factor;
	userData_.velr_outer = halo_table_const.dataViewArrays[2](static_cast<int>(N - 1)) * speed_factor;
	userData_.temp_outer = halo_table_const.dataViewArrays[3](static_cast<int>(N - 1));

	// optional halo v_phi expression (variables: x, y, z)
	userData_.haloVphiExpr.clear();
	pp.query("halo_vphi_expr", userData_.haloVphiExpr);
	userData_.useHaloVphiParser = !userData_.haloVphiExpr.empty();
	if (userData_.useHaloVphiParser) {
		userData_.haloVphiParser.emplace(userData_.haloVphiExpr);
		userData_.haloVphiParser->registerVariables({"x", "y", "z", "length_factor", "speed_factor"});
		userData_.haloVphiParserExe = userData_.haloVphiParser->compile<5>();
#ifdef AMREX_USE_GPU
		if (userData_.haloVphiParserExe->m_device_executor == nullptr) {
			amrex::Abort("disk_galaxy.halo_vphi_expr: device parser executor is null after compile<5>()");
		}
#endif
		userData_.haloVphiParser.reset();
	} else {
		userData_.haloVphiParser.reset();
		userData_.haloVphiParserExe.reset();
	}
}

template <> void QuokkaSimulation<DiskGalaxy_no_mhd>::setInitialConditionsOnGrid(quokka::grid const &grid_elem)
{
	amrex::ParmParse const pp("disk_galaxy");

	// disc parameters
	double disk_gas_mass_Msun = NAN;     // disk mass
	double disk_Rscale_kpc = NAN;	     // disk scale length
	double disk_zscale_kpc = NAN;	     // disk scale height
	double T_disk = NAN;		     // K
	double disk_perturb_amplitude = NAN; // amplitude of harmonic mode perturbation
	double disk_perturb_Rmax_kpc = NAN;  // max radius (in kpc) for harmonic mode perturbations
	pp.query("disk_gas_mass_Msun", disk_gas_mass_Msun);
	pp.query("disk_Rscale_kpc", disk_Rscale_kpc);
	pp.query("disk_zscale_kpc", disk_zscale_kpc);
	pp.query("disk_temperature", T_disk);
	pp.query("disk_perturb_amplitude", disk_perturb_amplitude);
	pp.query("disk_perturb_Rmax_kpc", disk_perturb_Rmax_kpc);
	AMREX_ALWAYS_ASSERT(!std::isnan(disk_gas_mass_Msun));
	AMREX_ALWAYS_ASSERT(!std::isnan(disk_Rscale_kpc));
	AMREX_ALWAYS_ASSERT(!std::isnan(disk_zscale_kpc));
	AMREX_ALWAYS_ASSERT(!std::isnan(T_disk));
	AMREX_ALWAYS_ASSERT(!std::isnan(disk_perturb_amplitude));
	AMREX_ALWAYS_ASSERT(!std::isnan(disk_perturb_Rmax_kpc));
	
	double length_factor = 1.0;
	pp.query("length_factor", length_factor);
	double speed_factor = 1.0;
	pp.query("speed_factor", speed_factor);
	// double halo_density_factor = 1.0;
	// pp.query("halo_density_factor", halo_density_factor);
	
	disk_Rscale_kpc *= length_factor;
	disk_zscale_kpc *= length_factor;
	disk_perturb_Rmax_kpc *= length_factor;
	disk_gas_mass_Msun *= length_factor * length_factor * length_factor;

	const double disk_gas_mass = disk_gas_mass_Msun * C::M_solar;
	const double R_d = disk_Rscale_kpc * (1.0e3 * C::parsec);
	const double z_d = disk_zscale_kpc * (1.0e3 * C::parsec);
	const double R_max_perturb = disk_perturb_Rmax_kpc * (1e3 * C::parsec);
	const double rho_0 = disk_gas_mass / 4. / M_PI / (R_d * R_d) / z_d; // normalization constant
	// we have a disk with exponential decay density of rate z_d in z, and R_d in radius

	// read tables

	double const *R_table = userData_.radius.dataPtr();
	double const *vcirc_table = userData_.vcirc.dataPtr();
	double const *rhoH_table = userData_.rho_halo.dataPtr();
	double const *velr_table = userData_.velr_halo.dataPtr();  // zeroed atm
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
	amrex::ParserExecutor<5> halo_vphi_parser{};
	if (use_halo_vphi_parser) {
		if (userData_.haloVphiParserExe.has_value()) {
			halo_vphi_parser = *userData_.haloVphiParserExe;
		} else {
			amrex::Abort("disk_galaxy.halo_vphi_expr: parser executor is missing after compile<5>()");
		}
	}
	
	// amrex::Print() << "REDJARD: R_table_min = " << R_table_min << "\n"; //
	// amrex::Print() << "REDJARD: R_table_max = " << R_table_max << "\n"; //
	// amrex::Print() << "REDJARD: rho_inner = "   << rho_inner   << "\n"; //
	// amrex::Print() << "REDJARD: rho_outer = "   << rho_outer   << "\n"; //

	const amrex::Box &indexRange = grid_elem.indexRange_;
	const amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> dx = grid_elem.dx_;
	const amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> prob_lo = grid_elem.prob_lo_;
	const amrex::Array4<double> &state_cc = grid_elem.array_;

	amrex::ParallelFor(indexRange, [=] AMREX_GPU_DEVICE(int i, int j, int k) {
		// Cartesian coordinates
		amrex::Real const x0 = prob_lo[0] + (i * dx[0]);
		amrex::Real const y0 = prob_lo[1] + (j * dx[1]);
		amrex::Real const z0 = prob_lo[2] + (k * dx[2]);

		amrex::Real const x1 = prob_lo[0] + ((i + 1) * dx[0]);
		amrex::Real const y1 = prob_lo[1] + ((j + 1) * dx[1]);
		amrex::Real const z1 = prob_lo[2] + ((k + 1) * dx[2]);

		auto vcirc_exact = [R_table_min, R_table_max, R_table, vcirc_inner, vcirc_outer, vcirc_table, len_table](const amrex::Real R) {
			double vcirc;
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
			double rho_H;
			if (R > R_table_min && R < R_table_max) {
				rho_H = interpolate_value(R, R_table, rhoH_table, len_table);
			} else if (R <= R_table_min) {
				rho_H = rho_inner;
			} else {
				rho_H = rho_outer;
			}

			return rho_H;
		};
		
		// zeroed atm
		auto velHalo = [R_table_min, R_table, R_table_max, velr_inner, velr_outer, velr_table, len_table](const amrex::Real R) {
			double vel_H;
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
			double temp_H;
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
				return halo_vphi_parser(x, y, z, length_factor, speed_factor);
			}
			return 0.0;
		};

		auto velx_exact = [velHalo, vphiHalo_exact](double x, double y, double z) {
			double const r = std::sqrt(std::pow(x, 2) + std::pow(y, 2) + std::pow(z, 2));
			double const R = std::sqrt(std::pow(x, 2) + std::pow(y, 2));
			double const vphi_component = (R > 0.0) ? (-vphiHalo_exact(x, y, z) * y / R) : 0.0;
			return velHalo(r) * x / r + vphi_component; // vx
		};

		auto vely_exact = [velHalo, vphiHalo_exact](double x, double y, double z) {
			double const r = std::sqrt(std::pow(x, 2) + std::pow(y, 2) + std::pow(z, 2));
			double const R = std::sqrt(std::pow(x, 2) + std::pow(y, 2));
			double const vphi_component = (R > 0.0) ? (vphiHalo_exact(x, y, z) * x / R) : 0.0;
			return velHalo(r) * y / r + vphi_component; // vy
		};

		auto velz_exact = [velHalo](double x, double y, double z) {
			double const r = std::sqrt(std::pow(x, 2) + std::pow(y, 2) + std::pow(z, 2));
			return (r > 0.0) ? (velHalo(r) * z / r) : 0.0; // vz
		};

		// integrate profiles over cell volume
		const double cell_vol = dx[0] * dx[1] * dx[2];
		constexpr double gamma_gas = quokka::EOS_Traits<DiskGalaxy_no_mhd>::gamma;
		constexpr double mu = 0.61;

		auto rho_total_exact = [=] AMREX_GPU_DEVICE(double x, double y, double z) {
			return rhoDisk_exact(x, y, z) + rhoHalo_exact(x, y, z);
		};

		auto momx_total_exact = [=] AMREX_GPU_DEVICE(double x, double y, double z) {
			return rhoDisk_exact(x, y, z) * vx_exact(x, y, z) + rhoHalo_exact(x, y, z) * velx_exact(x, y, z);
		};

		auto momy_total_exact = [=] AMREX_GPU_DEVICE(double x, double y, double z) {
			return rhoDisk_exact(x, y, z) * vy_exact(x, y, z) + rhoHalo_exact(x, y, z) * vely_exact(x, y, z);
		};

		auto momz_total_exact = [=] AMREX_GPU_DEVICE(double x, double y, double z) {
			return rhoHalo_exact(x, y, z) * velz_exact(x, y, z);
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
		double const Ekin_disk_halo = 0.5 * (momx_disk_halo * momx_disk_halo + momy_disk_halo * momy_disk_halo + momz_disk_halo * momz_disk_halo) / rho_disk_halo;
		double const Eint_disk_halo = Eint;
		// double const Etot_disk_halo = Eint_disk_halo + Ekin_disk_halo + Emag;
		double const Etot_disk_halo = Eint_disk_halo + Ekin_disk_halo + 0;

		state_cc(i, j, k, HydroSystem<DiskGalaxy_no_mhd>::density_index) = rho_disk_halo;
		state_cc(i, j, k, HydroSystem<DiskGalaxy_no_mhd>::x1Momentum_index) = momx_disk_halo;
		state_cc(i, j, k, HydroSystem<DiskGalaxy_no_mhd>::x2Momentum_index) = momy_disk_halo;
		state_cc(i, j, k, HydroSystem<DiskGalaxy_no_mhd>::x3Momentum_index) = momz_disk_halo;
		state_cc(i, j, k, HydroSystem<DiskGalaxy_no_mhd>::energy_index) = Etot_disk_halo;
		state_cc(i, j, k, HydroSystem<DiskGalaxy_no_mhd>::internalEnergy_index) = Eint_disk_halo;

	});
}

template <> void QuokkaSimulation<DiskGalaxy_no_mhd>::createInitialCICParticles()
{
	// read particles from ASCII file
	amrex::ParmParse const pp("disk_galaxy");
	std::string filename;
	pp.query("particle_file", filename);

	amrex::Print() << "\nReading particles from ASCII file " << filename << "...\n";
	CICParticles->SetVerbose(1);
	const int nreal_extra = 4; // mass vx vy vz
	CICParticles->InitFromAsciiFile(filename, nreal_extra, nullptr);
	amrex::Print() << "\n";
}

template <> void QuokkaSimulation<DiskGalaxy_no_mhd>::refineGrid(int lev, amrex::TagBoxArray &tags, amrex::Real _time, int _ngrow)
{
	// amrex::Print() << "refineGrid\n";
	// auto start = clock();
	// geometrical refinement
	// tag cells within the cylinder defined by R < Rmax and abs(z) < zmax
	amrex::ParmParse const pp("disk_galaxy");
	amrex::Real refine_Rmax_kpc = NAN;
	amrex::Real refine_zmax_kpc = NAN;
	pp.query("refine_Rmax_kpc", refine_Rmax_kpc);
	pp.query("refine_zmax_kpc", refine_zmax_kpc);
	
	double length_factor = 1.0;
	pp.query("length_factor", length_factor);
	// double speed_factor = 1.0;
	// pp.query("speed_factor", speed_factor);
	
	refine_Rmax_kpc *= length_factor;
	refine_zmax_kpc *= length_factor;
	
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
	// amrex::Print() << "REDJARD: ran refineGrid in " << float(clock() - start)/1e6 << " s\n";
}

void apply_dm_potential_on_grid( quokka::grid const &grid_elem, amrex::Real dt ) {
	// const amrex::Box &indexRange = grid_elem.indexRange_;
	const amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> dx = grid_elem.dx_;
	const amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> prob_lo = grid_elem.prob_lo_;
	// const amrex::Array4<double> &state = grid_elem.array_;
	
	// amrex::Print() << "REDJARD: level = " << level << "\n";
	// amrex::Print() << "REDJARD: iter.index() = " << iter.index() << "\n";
	amrex::Print() << "REDJARD: prob_lo = [" << prob_lo[0] << ", " << prob_lo[1] << ", " << prob_lo[2] << "]\n";
	amrex::Print() << "REDJARD: dx = [" << dx[0] << ", " << dx[1] << ", " << dx[2] << "]\n";
	
	/*
	// taken from lizmcole/MHDDisk
	amrex::ParallelFor(indexRange, [=] AMREX_GPU_DEVICE(int i, int j, int k) noexcept {
		const double x = prob_lo[0] + (i + 0.5) * dx[0];
		const double y = prob_lo[1] + (j + 0.5) * dx[1];
		const double z = prob_lo[2] + (k + 0.5) * dx[2];
		const double R2 = x * x + y * y;
		const double R = std::sqrt(R2);
		
		const double rho = state(i, j, k, HydroSystem<MHDGalaxy>::density_index);
		const double px = state(i, j, k, HydroSystem<MHDGalaxy>::x1Momentum_index);
		const double py = state(i, j, k, HydroSystem<MHDGalaxy>::x2Momentum_index);
		const double pz = state(i, j, k, HydroSystem<MHDGalaxy>::x3Momentum_index);
		const double Eint = state(i, j, k, HydroSystem<MHDGalaxy>::internalEnergy_index);
		const double Etot_old = state(i, j, k, HydroSystem<MHDGalaxy>::energy_index);
		const double Ekin_old = 0.5 * (px * px + py * py + pz * pz) / rho;
		const double Emag = Etot_old - Ekin_old - Eint;
		
		const double D = R2 + Rc * Rc + (z / q_flatten) * (z / q_flatten);
		const double g_R = (R > 0.0) ? -(vc * vc * R / D) : 0.0;
		const double g_z = -(vc * vc * z / (q_flatten * q_flatten * D));
		const double gx = (R > 0.0) ? g_R * x / R : 0.0;
		const double gy = (R > 0.0) ? g_R * y / R : 0.0;
		
		const double px_new = px + dt_lev * rho * gx;
		const double py_new = py + dt_lev * rho * gy;
		const double pz_new = pz + dt_lev * rho * g_z;
		const double Ekin_new = 0.5 * (px_new * px_new + py_new * py_new + pz_new * pz_new) / rho;
		
		state(i, j, k, HydroSystem<MHDGalaxy>::x1Momentum_index) = px_new;
		state(i, j, k, HydroSystem<MHDGalaxy>::x2Momentum_index) = py_new;
		state(i, j, k, HydroSystem<MHDGalaxy>::x3Momentum_index) = pz_new;
		state(i, j, k, HydroSystem<MHDGalaxy>::energy_index) = Ekin_new + Eint + Emag;
	});
	//*/
}

template <> void QuokkaSimulation<DiskGalaxy_no_mhd>::addStrangSplitSources(amrex::MultiFab &mf, int level, amrex::Real /*time*/, amrex::Real dt_lev) {
	// iterate over all grids (of this MultiFab)
	for (amrex::MFIter iter(mf); iter.isValid(); ++iter) {
		quokka::grid grid_elem(
			mf.array(iter),
			iter.validbox(),
			this->geom[level].CellSizeArray(),
			this->geom[level].ProbLoArray(),
			this->geom[level].ProbHiArray(),
			quokka::centering::cc,
			quokka::direction::na
		);
		
		// dark matter gravitational potential
		apply_dm_potential_on_grid(grid_elem,dt_lev);
		
	}
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
