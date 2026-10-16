"""
makeplots_fast.py — memory-efficient AMReX/Quokka plotfile diagnostics (CPU/MPI)
modified

1 command-line argument required:
    plotfile_path — which plotfile to load (required; passed straight to yt.load).
"""

import gc
import os
import re
import sys
import time
class bench:
    def __enter__(self):
        self.start = time.perf_counter()
    def __exit__(self, *_):
        print(f"{time.perf_counter()-self.start:.6f} s")
import yt
import unyt as u
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import SymLogNorm, LogNorm


scale = 1
# scale = 1/3

t_start = time.time()

yt.set_log_level("warning")

# ============================================================
# FIXED AXES — locked color-scale ranges for cross-run comparison
# ============================================================
# 12-panel B field: components are diverging (+-), |B| is sequential (0..max)
BFIELD_COMP_VMAX = None          # Bx, By, Bz shown on [-VMAX, +VMAX]
BFIELD_COMP_VMIN = None  # -BFIELD_COMP_VMAX
BFIELD_MAG_VMIN  = 0.0           # |B| shown on [VMIN, VMAX]
BFIELD_MAG_VMAX  = None

# Bphi 2-way comparison: XY (midplane) panels get a wider range than the
# XZ / YZ (edge-on) panels, since Bphi is concentrated near the midplane.
BPHI_XY_VMAX    = 1e-6
BPHI_EDGE_VMAX  = 1e-7
BPHI_LINTHRESH_FRACTION = 1e-2   # linthresh = VMAX * this fraction

# Density slices: floor is the simulation's density floor, ceiling is
# roughly the disk midplane peak.
DENS_LOG_VMIN = None            # log10(rho_floor [g/cm^3])
DENS_LOG_VMAX = None            # log10(rho_max   [g/cm^3])

# div B slices: symmetric linear range (NOT log)
DIVB_VMAX = None
DIVB_VMIN = None  # -DIVB_VMAX

# Velocity 12-panel + zoom: components use SymLogNorm about a linear core,
# magnitude uses LogNorm. VMAX_VEL/LINTHRESH set the FIXED full-domain
# scale; the zoomed SN panel below auto-scales instead (see that section).
# VMAX_VEL = 2e7 *scale   # example 200 km/s in cm/s
VMAX_VEL = 2000 * 1e5 *scale   # example 200 km/s in cm/s
LINTHRESH = 1e5

# ============================================================
# Load
# ============================================================
PLOTFILE = sys.argv[1]

ds = yt.load(PLOTFILE)

IS_MHD = ('boxlib', 'x-BField') in ds.field_list
# if not IS_MHD:
#     print("Detected 'hydro' in plotfile path/run tag -> skipping all magnetic-field "
#            "diagnostics (B-field 12-panel, plasma beta, div B, Bphi comparison).")


Msun   = u.Msun.to('g').v[()]
Gconst = 6.6743015e-8
m_u    = 1.6605390689252e-24  # g
k_B    = 1.380649e-16         # erg/K
pc     = 3085677587679310666.
kpc    = pc*1000
t_myr  = float(ds.current_time.v) / (3.15576e13)

width  = float(ds.domain_width[0].v)
extent_kpc = np.array([-1,1]*2) * width / kpc /2

print(f"Fields: {tuple(field if section == "boxlib" else f"{section}:{field}" for section, field in ds.field_list)}")
print(f"{ds.domain_dimensions} top level cells, refined {ds.index.max_level} levels")
print(f"Cell size: {ds.domain_width.v/ds.domain_dimensions/2**ds.index.max_level /pc} pc")
print(f"Time: {t_myr:.3f} Myr")

# ── Aphi table metadata ───────────────────────────────────────────────────────
# Override via $APHI_META_FILE if your run uses a different seed/tag than the hardcoded default.
if IS_MHD:
    aphi_meta_path = os.environ.get("APHI_META_FILE", "tests/input/Aphi_2d_meta_1.txt")
    meta = {}
    with open(aphi_meta_path) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            if "=" in line:
                key, val = line.split("=", 1)
                val_clean = val.split("#")[0].split("[")[0].strip()
                meta[key.strip()] = val_clean
    
    nR       = int(meta.get("nR",     meta.get("seed_nR")))
    nz       = int(meta.get("nz",     meta.get("seed_nz")))
    Rmax_cm  = float(meta.get("Rmax_cm", meta.get("seed_Rmax", meta.get("cyl_Rmax_cm"))))
    Lz_cm    = float(meta.get("Lz_cm",   meta.get("seed_Lz",   meta.get("cyl_Lz_cm"))))
    dR_seed  = float(meta.get("dR_fine_cm", Rmax_cm / nR))
    dz_seed  = float(meta.get("dz_fine_cm", Lz_cm  / nz))
    
    print(f"Table shape       : {nR} x {nz}")
    print(f"dR_table          = {dR_seed/3.086e21:.3f} kpc")
    print(f"dz_table          = {dz_seed/3.086e18:.1f} pc")
    print(f"Rmax              = {Rmax_cm/3.086e21:.1f} kpc")
    print(f"Lz                = {Lz_cm/3.086e18:.0f} pc")


RES = 800*4

# ============================================================
# Helpers
# ============================================================

xlabels = {"x": "y", "y": "x", "z": "x"}
ylabels = {"x": "z", "y": "z", "z": "y"}
titles = {"x": "YZ (x=0)", "y": "XZ (y=0)", "z": "XY (z=0)"}


def get_slice(normal, field, res=RES, type="SlicePlot", **kwargs):
    plane = yt.__getattribute__(type)(ds, normal, field, center=3*[kpc/1e12], **kwargs)
    plane.set_buff_size(res)
    plane.render()
    data = plane.frb[field].v
    if normal == "y":
        data = data.T
    return data

def get_proj(*args, **kwargs):
    return get_slice(*args, type="ProjectionPlot", **kwargs)


# ============================================================
# Derived field: Bphi reconstructed
# ============================================================
# Skipped entirely for hydro runs — x-BField/y-BField don't exist, and
# nothing downstream needs this field when IS_HYDRO is True.
if IS_MHD:
    def _Bphi_reconstructed(field, data):
        Bx = data[("boxlib", "x-BField")].v
        By = data[("boxlib", "y-BField")].v
        x  = data[("index",  "x")].v
        y  = data[("index",  "y")].v
        r2 = x**2 + y**2
        r  = np.where(r2 > 0, np.sqrt(r2), 1.0)
        return (By * x - Bx * y) / r * data.ds.quan(1.0, "G")
    
    if IS_MHD:
        ds.add_field(
            name=("boxlib", "Bphi_reconstructed"),
            function=_Bphi_reconstructed,
            sampling_type="cell",
            units="G",
        )

# ============================================================
# Derived fields: Velocity components and magnitude
# ============================================================
def _vel_comp(field, data, comp):
    # Retrieve as arrays
    p = data[("boxlib", f"{comp}-GasMomentum")]
    rho = data[("boxlib", "gasDensity")]

    # Perform division and explicitly multiply by the velocity unit
    # to ensure the result is a unyt_array with units 'cm/s'
    return (p / rho) * data.ds.quan(1.0, "cm/s")

for comp in ['x', 'y', 'z']:
    ds.add_field(
        name=("boxlib", f"{comp}-Velocity"),
        function=lambda field, data, c=comp: _vel_comp(field, data, c),
        sampling_type="cell",
        units="cm/s",
    )

def _vel_mag(field, data):
    # Retrieve the velocity component fields
    vx = data[("boxlib", "x-Velocity")]
    vy = data[("boxlib", "y-Velocity")]
    vz = data[("boxlib", "z-Velocity")]

    # Calculate magnitude: (v_x^2 + v_y^2 + v_z^2)^(0.5)
    # By extracting .v, we work with raw numpy arrays,
    # then attach the correct unit 'cm/s' at the end.
    mag_val = np.sqrt(vx.v**2 + vy.v**2 + vz.v**2)
    return mag_val * data.ds.quan(1.0, "cm/s")

ds.add_field(
    name=("boxlib", "velocity_mag"),
    function=_vel_mag,
    sampling_type="cell",
    units="cm/s",
)



X_XY = get_slice('z',("boxlib", "x"))
Y_XY = get_slice('z',("boxlib", "y"))
R_XY = np.sqrt(X_XY**2 + Y_XY**2 + 1e-200)

normals = ("z", "y", "x")

# Collect slices (Collective call)
ρ_slices = {}
for normal in normals:
    ρ_slices[normal] = get_slice(normal, ("boxlib", "gasDensity"))


# ============================================================
# B-field 12-panel
# ============================================================
if IS_MHD:
    print("\n--- B-field 12-panel ---")
    
    slices = {}
    plane_meta = {}
    for field in ["x-BField", "y-BField", "z-BField"]:
        for normal in normals:
            print(f"  Slicing {field} {normal}...")
            data, xlabel, ylabel, title = get_slice(normal, ("boxlib", field)), xlabels[normal], ylabels[normal], titles[normal]
            slices[(field, normal)] = data
            plane_meta[normal] = (xlabel, ylabel, title)

    for normal in normals:
        Bx = slices[("x-BField", normal)]
        By = slices[("y-BField", normal)]
        Bz = slices[("z-BField", normal)]
        slices[("Bmag", normal)] = np.sqrt(Bx**2 + By**2 + Bz**2)

    row_fields = ["x-BField", "y-BField", "z-BField", "Bmag"]
    row_labels  = [r"$B_x$", r"$B_y$", r"$B_z$", r"$|B|$"]

    fig, axes = plt.subplots(4, 3, figsize=(16, 18))
    for row, (field, rlabel) in enumerate(zip(row_fields, row_labels)):
        for col, normal in enumerate(normals):
            ax   = axes[row, col]
            data = slices[(field, normal)]
            xlabel, ylabel, plabel = plane_meta[normal]
            if row == 3:
                norm = plt.Normalize(vmin=BFIELD_MAG_VMIN, vmax=BFIELD_MAG_VMAX)
                cmap = "inferno"
            else:
                norm = plt.Normalize(vmin=BFIELD_COMP_VMIN, vmax=BFIELD_COMP_VMAX)
                cmap = "RdBu_r"
            im = ax.pcolormesh(
                np.linspace(extent_kpc[0], extent_kpc[1], data.shape[1]+1),
                np.linspace(extent_kpc[2], extent_kpc[3], data.shape[0]+1),
                data, norm=norm, cmap=cmap, shading="flat", rasterized=True,
            )
            plt.colorbar(im, ax=ax, label="G")
            ax.set_title(f"{rlabel} — {plabel}", fontsize=9)
            ax.set_xlabel(xlabel); ax.set_ylabel(ylabel)
            ax.set_aspect("equal")

    fig.suptitle(f"Magnetic field slices — t = {t_myr:.1f} Myr", fontsize=13)
    fig.tight_layout()
    fig.savefig("Bfield_12panel_ytb.png", dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: Bfield_12panel_ytb.png")

# ============================================================
# Density
# ============================================================

fig, axes = plt.subplots(1, len(normals), figsize=(len(normals)*6, 5))
for ax, normal in zip(axes, normals):
    log_data = np.log10(ρ_slices[normal])
    im = ax.imshow(log_data, origin="lower", extent=extent_kpc,
                   cmap="viridis", vmin=DENS_LOG_VMIN, vmax=DENS_LOG_VMAX,
                   aspect="equal")
    plt.colorbar(im, ax=ax,
                 label=r"$\log_{10}(\rho\ [\mathrm{g\ cm^{-3}}])$")
    ax.set_title(f"gasDensity — {titles[normal]}", fontsize=9)
    ax.set_xlabel(xlabels[normal]); ax.set_ylabel(ylabels[normal])

fig.suptitle(f"Density slices — t = {t_myr:.1f} Myr", fontsize=12)
fig.tight_layout()
fig.savefig("density_slices.png", dpi=150, bbox_inches="tight")
plt.close(fig)
print(f"Saved: density_slices.png")



fig, axes = plt.subplots(1, len(normals), figsize=(len(normals)*6, 5))
for ax, normal in zip(axes, normals):
    rho = ρ_slices[normal]
    Eint = get_slice(normal, ("boxlib", "gasInternalEnergy"))
    γ = 1.0001 # hardcoded
    µ = 0.6 * m_u
    p = Eint*(γ-1)
    T = (µ / k_B) * (p / rho)
    if normal == 'z':
        cs = (γ * p / rho)**.5
    im = ax.imshow(np.log10(T), origin="lower", extent=extent_kpc, cmap="viridis", aspect="equal")
    plt.colorbar(im, ax=ax, label=r"log$_{10}$ T [K]")
    ax.set_title(titles[normal], fontsize=9)
    ax.set_xlabel(xlabels[normal]); ax.set_ylabel(ylabels[normal])
fig.suptitle(f"Temperature slices — t = {t_myr:.1f} Myr", fontsize=12)
fig.tight_layout()
fig.savefig("T_slices.png", dpi=150, bbox_inches="tight")
plt.close(fig)
print(f"Saved: T_slices.png")



fig, axes = plt.subplots(1, 2, figsize=(2*7, 6))
for ax, normal in zip(axes, normals):
    im = ax.imshow(
        get_proj(normal, ("boxlib", "gasDensity")),
        origin="lower", extent=extent_kpc, cmap="viridis",
        norm=LogNorm(),
        aspect="equal",
    )
    plt.colorbar(im, ax=ax, label=r"$\Sigma$")
    ax.set_title(f"Column density, {normal}-projection — t = {t_myr:.1f} Myr")
    ax.set_xlabel(xlabels[normal]); ax.set_ylabel(ylabels[normal])
fig.tight_layout()
fig.savefig("density_projection.png", dpi=150, bbox_inches="tight")
plt.close(fig)
print(f"Saved: density_projection.png")



fig, axes = plt.subplots(1, 2, figsize=(2*7, 6))
for ax, normal in zip(axes, normals):
    im = ax.imshow(
        get_slice(normal,("index", "cell_volume")),
        origin="lower", extent=extent_kpc, cmap="gist_ncar",
        norm=LogNorm(),
        aspect="equal",
    )
    plt.colorbar(im, ax=ax, label=r"")
    ax.set_title(f"Grid cell volume — {titles[normal]}")
    ax.set_xlabel(xlabels[normal]); ax.set_ylabel(ylabels[normal])
fig.tight_layout()
fig.savefig("grid_layout.png", dpi=150, bbox_inches="tight")
plt.close(fig)
print(f"Saved: grid_layout.png")


# ============================================================
# Plasma beta slices  (locked — unchanged)
# ============================================================
if IS_MHD:
    print("\n--- Plasma beta (masked) ---")

    # Collective reads on all ranks first
    beta_slice_data = {}
    for normal in normals:
        beta_slice_data[normal] = get_slice(normal, ("boxlib", "plasma_beta")), xlabels[normal], ylabels[normal], titles[normal]
    
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    for ax, normal in zip(axes, normals):
        data, xlabel, ylabel, title = beta_slice_data[normal]
        log_data = np.log10(np.where(data > 0, data, 1e-300))
        
        mask_width = 8 * Rmax_cm / nR
        
        if normal == "z":
            mask = R_XY < mask_width
        else:
            mask = (np.abs(X_XY) < mask_width) | (np.abs(Y_XY) < mask_width)

        log_data_masked = np.where(mask, np.nan, log_data)
        im = ax.imshow(log_data_masked, origin="lower", extent=extent_kpc,
                       cmap="magma", vmin=-2, vmax=6, aspect="equal")
        plt.colorbar(im, ax=ax, label=r"$\log_{10}(\beta)$")
        ax.set_title(f"Plasma beta — {title}", fontsize=9)
        ax.set_xlabel(xlabel); ax.set_ylabel(ylabel)

        if normal == "z":
            ax.add_patch(plt.Circle((0, 0), mask_width/kpc, color="white", fill=False, lw=0.8, ls="--"))
        else:
            for val in [-mask_width, mask_width]:
                ax.axvline(val/kpc, color="white", lw=0.8, ls="--")
                ax.axhline(val/kpc, color="white", lw=0.8, ls="--")

    fig.suptitle(f"Plasma beta slices (axis-masked) — t = {t_myr:.1f} Myr", fontsize=12)
    fig.tight_layout()
    fig.savefig("plasma_beta_slices_masked.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: plasma_beta_slices_masked.png")

    # ============================================================
    # Volume-averaged plasma beta — CHUNKED Z-SLAB LOOP, MPI-DISTRIBUTED
    # ============================================================
    print("\n--- Volume averages (chunked z-slab, memory-efficient) ---")

    max_level = ds.index.max_level
    dims_full = ds.domain_dimensions * (2 ** max_level)
    n_slabs   = 128
    slab_nz   = max(1, -(-dims_full[2] // n_slabs))  # ceiling division so the last slab isn't dropped

    rho_transition = 1e-28

    rng            = np.random.default_rng(42)
    RESERVOIR_N    = 1_000_000 # Reduced slightly to ensure memory safety
    beta_reservoir = np.empty(RESERVOIR_N, dtype=np.float32)
    reservoir_fill = 0
    reservoir_full = False

    LE = ds.domain_left_edge.v
    dx = ds.domain_width.v / dims_full

    print(f"  Finest-level dims: {dims_full}")
    print(f"  n_slabs={n_slabs}, slab_nz={slab_nz} cells")

    my_slab_indices = list(range(0, n_slabs, 1))

    for slab_idx in my_slab_indices:
        z0_cell = slab_idx * slab_nz
        z1_cell = min(z0_cell + slab_nz, dims_full[2])
        if z0_cell >= dims_full[2]: continue

        slab_left  = [LE[0], LE[1], LE[2] + z0_cell * dx[2]]
        slab_right = [LE[0] + dims_full[0]*dx[0], LE[1] + dims_full[1]*dx[1], LE[2] + z1_cell * dx[2]]
        region = ds.box(slab_left, slab_right)

        for chunk in region.chunks([("boxlib", "plasma_beta"), ("boxlib", "gasDensity")], "io"):

            beta  = chunk[("boxlib", "plasma_beta")].v.ravel()
            rho   = chunk[("boxlib", "gasDensity")].v.ravel()

            mask_disk = rho > rho_transition

            # Reservoir sampling
            disk_beta = beta[mask_disk].astype(np.float32)
            n_new = len(disk_beta)
            if n_new > 0:
                if not reservoir_full:
                    space = RESERVOIR_N - reservoir_fill
                    take = min(n_new, space)
                    beta_reservoir[reservoir_fill:reservoir_fill + take] = disk_beta[:take]
                    reservoir_fill += take
                    if reservoir_fill == RESERVOIR_N: reservoir_full = True
                else:
                    idx = rng.integers(0, RESERVOIR_N, size=n_new)
                    beta_reservoir[idx] = disk_beta

            del beta, rho, mask_disk
        gc.collect()

    # comm.Barrier()
    # print("  All ranks finished. Reducing data...")

    # GATHER AND PERCENTILES
    # all_reservoirs = comm.gather(beta_reservoir[:reservoir_fill], root=0)
    all_reservoirs = beta_reservoir[:reservoir_fill]

    print("  Disk beta percentiles:")
    for p in [10, 25, 50, 75, 90]:
        print(f"    {p}th: {np.percentile(all_reservoirs, p):.3e}")

# ============================================================
# div B — per AMR level (free covering_grid immediately)
# ============================================================
if IS_MHD:
    print("\n--- div B ---")

    # covering_grid objects don't support .chunks() in this yt version
    # (YTDataSelectorNotImplemented). divB is already per-level, non-interpolated
    # data (differenced directly from the level's own face data), so iterate the
    # real AMR grid patches at that level instead -- also naturally chunked
    # (each patch is blocking-factor sized) and distributes cleanly across ranks.
    for lev in range(ds.index.max_level + 1):
        level_grids = ds.index.select_grids(lev)

        max_val = -np.inf
        sum_val = 0.0
        count = 0

        for g in level_grids:
            data = g[("boxlib", "divB")].v
            abs_data = np.abs(data)

            if abs_data.size > 0:
                max_val = max(max_val, np.max(abs_data))
            sum_val += np.sum(abs_data)
            count += data.size

            del data, abs_data

        # Gather results from all MPI ranks to rank 0
        global_max = max_val
        global_sum = sum_val
        global_count = count
        
        mean_val = global_sum / global_count if global_count > 0 else 0.0
        print(f"  Level {lev}:  max |divB| = {global_max:.3e}  mean |divB| = {mean_val:.3e}")

        gc.collect()

    print("  Plotting raw divB slices...")
    divB_panels = {}
    divB_meta = {}
    for normal in ["z", "y", "x"]:
        data, xlabel, ylabel, title = get_slice(normal, ("boxlib", "divB")), xlabels[normal], ylabels[normal], titles[normal]
        divB_panels[normal] = data
        divB_meta[normal] = (xlabel, ylabel, title)
    
    all_divB_max = max(np.max(np.abs(d)) for d in divB_panels.values())

    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    for ax, normal in zip(axes, normals):
        data = divB_panels[normal]
        xlabel, ylabel, title = divB_meta[normal]
        im = ax.imshow(data, origin="lower", extent=extent_kpc,
                       cmap="RdBu_r", vmin=DIVB_VMIN, vmax=DIVB_VMAX, aspect="equal")
        plt.colorbar(im, ax=ax, label=r"$\nabla\cdot B$")
        ax.set_title(f"div B — {title}", fontsize=9)
        ax.set_xlabel(xlabel); ax.set_ylabel(ylabel)

    fig.suptitle(
        # f"div B — t = {t_myr:.1f} Myr\nmax |divB| (this run) = {all_divB_max:.2e}  |  scale fixed to ±{DIVB_VMAX:.1e}",
        f"div B — t = {t_myr:.1f} Myr\nmax |divB| (this run) = {all_divB_max:.2e}",
        fontsize=12)
    fig.tight_layout()
    fig.savefig("divB_slices_raw.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: divB_slices_raw.png")

# ============================================================
# Bphi comparison (2-way)
# ============================================================
if IS_MHD:
    print("\n--- Bphi Comparison ---")

    bphi_fields  = [("boxlib", "Bphi"), ("boxlib", "Bphi_reconstructed")]
    bphi_labels  = ["Analytic Table Bphi", "Grid Reconstructed Bphi"]

    # Collective reads on all ranks first
    bphi_slice_data = {}
    for field in bphi_fields:
        for normal in normals:
            bphi_slice_data[(field, normal)] = get_slice(normal, ("boxlib", field)), xlabels[normal], ylabels[normal], titles[normal]
    
    fig, axes = plt.subplots(2, 3, figsize=(16, 9))
    for row, (field_tuple, rlabel) in enumerate(zip(bphi_fields, bphi_labels)):
        for col, normal in enumerate(normals):
            ax   = axes[row, col]
            data, xlabel, ylabel, title = bphi_slice_data[(field_tuple, normal)]

            vmax      = BPHI_XY_VMAX if normal == "z" else BPHI_EDGE_VMAX
            linthresh = vmax * BPHI_LINTHRESH_FRACTION

            im = ax.imshow(
                data, origin="lower", extent=extent_kpc, cmap="RdBu_r",
                norm=SymLogNorm(linthresh=linthresh, vmin=-vmax, vmax=vmax, base=10), aspect="equal",
            )
            plt.colorbar(im, ax=ax, label=r"$B_\phi$ [G]", format="%.1e")
            ax.set_title(f"{rlabel} — {title}", fontsize=9)
            ax.set_xlabel(xlabel); ax.set_ylabel(ylabel)

    fig.suptitle(f"$B_\\phi$ Initial Condition Comparison — t = {t_myr:.1f} Myr",
                 fontsize=12)
    fig.tight_layout()
    fig.savefig("Bphi_2way_comparison.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: Bphi_2way_comparison.png")

    bphi_xy  = get_slice("z", ("boxlib", "Bphi"))  # collective — all ranks call this
    bmag_xy  = slices[("Bmag", "z")]
    
    bphi_max = np.percentile(np.abs(bphi_xy[np.isfinite(bphi_xy)]), 99)
    bmag_rms = np.sqrt(np.mean(bmag_xy**2))
    bmag_max = np.percentile(bmag_xy[np.isfinite(bmag_xy)], 99)

    print("====================================================")
    print(f"Bphi 99th percentile (midplane)   : {bphi_max:.3e} G")
    print(f"|B|  rms             (midplane)   : {bmag_rms:.3e} G")
    print(f"|B|  99th percentile (midplane)   : {bmag_max:.3e} G")
    # print(f"Bphi/|B|_rms                      : {bphi_max/bmag_rms:.3e}  (target: < 1e-2)")
    # print(f"Bphi/|B|_max                      : {bphi_max/bmag_max:.3e}  (target: < 1e-2)")
    print("====================================================")

    n_cell    = ds.domain_dimensions[0]
    Lbox_cm   = float(ds.domain_width[0].v)
    dx_level0 = Lbox_cm / n_cell
    print(f"dR_table/dx_L0 = {dR_seed/dx_level0:.2f}  (want < 0.5)")
    print(f"dz_table/dx_L0 = {dz_seed/dx_level0:.2f}")

# ============================================================
# Velocity 12-panel
# ============================================================
# print("\n--- Velocity 12-panel ---")

vel_fields = ["x-Velocity", "y-Velocity", "z-Velocity", "velocity_mag"]
vel_labels = [r"$v_x$", r"$v_y$", r"$v_z$", r"$|v|$"]

# Collect slices (Collective call)
vel_slices = {}
for field in vel_fields:
    for normal in normals:
        vel_slices[(field, normal)] = get_slice(normal, ("boxlib", field))

fig, axes = plt.subplots(4, 3, figsize=(16, 18))

for row, (field, rlabel) in enumerate(zip(vel_fields, vel_labels)):
    for col, normal in enumerate(normals):
        ax = axes[row, col]
        data = vel_slices[(field, normal)]

        if field == "velocity_mag":
            # For magnitude, use LogNorm (exclude near-zero values to avoid artifacts)
            norm = LogNorm(vmin=1e5, vmax=VMAX_VEL)
            cmap = "magma"
        else:
            # For components (vx, vy, vz), use SymLogNorm for positive/negative range
            norm = SymLogNorm(linthresh=LINTHRESH, linscale=1.0,
                            vmin=-VMAX_VEL, vmax=VMAX_VEL, base=10)
            cmap = "RdBu_r"

        im = ax.pcolormesh(
            np.linspace(extent_kpc[0], extent_kpc[1], data.shape[1]+1),
            np.linspace(extent_kpc[2], extent_kpc[3], data.shape[0]+1),
            data, norm=norm, cmap=cmap, shading="flat", rasterized=True,
        )
        plt.colorbar(im, ax=ax, label="cm/s")
        ax.set_title(f"{rlabel} — {titles[normal]}", fontsize=9)
        ax.set_xlabel(xlabels[normal]); ax.set_ylabel(ylabels[normal])
        ax.set_aspect("equal")

fig.suptitle(f"Velocity slices — t = {t_myr:.1f} Myr", fontsize=13)
fig.tight_layout()
fig.savefig("velocity_12panel.png", dpi=300, bbox_inches="tight")
plt.close(fig)
print(f"Saved: velocity_12panel.png")


# ============================================================
# Velocity fluctuations (turbulent residual, midplane)
# ============================================================

Rc  = 0.5 * kpc *scale
vc  = float(ds.parameters.get("mhd_galaxy.vc_kms", 200 *scale)) * 1.0e5

vcirc_ideal_f = lambda R: vc * R / np.sqrt(R**2 + Rc**2)

vx_xy = vel_slices[("x-Velocity", "z")]
vy_xy = vel_slices[("y-Velocity", "z")]
vz_xy = vel_slices[("z-Velocity", "z")]

# Cylindrical decomposition
vR_xy = (vx_xy * X_XY + vy_xy * Y_XY) / R_XY
vφ_xy = (vy_xy * X_XY - vx_xy * Y_XY) / R_XY

dvφ_xy = vφ_xy - vcirc_ideal_f(R_XY) # azimuthal fluctuation (exact baseline)
dvR_xy = vR_xy                       # radial: mean ~0 by symmetry
dvz_xy = vz_xy                       # vertical: mean ~0 by symmetry

R_bins_f = np.linspace(0.0, width / 2, 60)
R_mid_f  = 0.5 * (R_bins_f[:-1] + R_bins_f[1:])
sigma_turb_1d = np.full(len(R_mid_f), np.nan)
for i, (rlo, rhi) in enumerate(zip(R_bins_f[:-1], R_bins_f[1:])):
    m = (R_XY >= rlo) & (R_XY < rhi)
    if m.sum() > 4:
        sigma_turb_1d[i] = np.sqrt(np.nanmean(dvφ_xy[m]**2 + dvR_xy[m]**2 + dvz_xy[m]**2))

fig, axes = plt.subplots(2, 2, figsize=(2*6, 2*5))
for ax, data, label in zip(axes.flat[:3],
                           [dvR_xy, dvφ_xy, dvz_xy],
                           [r"$\delta v_R$", r"$\delta v_\phi$", r"$\delta v_z$"]):
    fluct_vmax = np.nanpercentile(np.abs(data), 99)
    im = ax.imshow(
        data/1e5, origin="lower", extent=extent_kpc, cmap="RdBu_r",
        vmin=-fluct_vmax/1e5, vmax=fluct_vmax/1e5,
        aspect="equal",
    )
    plt.colorbar(im, label="km/s")
    ax.set_title(f"{label} (turbulent residual)", fontsize=10)
    ax.set_xlabel(xlabels["z"]); ax.set_ylabel(ylabels["z"])
ax4 = axes.flat[3]
ax4.plot(R_mid_f/kpc, sigma_turb_1d / 1e5, lw=2)
ax4.set_xlabel("R [kpc]")
ax4.set_ylabel(r"$\sigma_{turb}$ [km/s]")
ax4.set_title("Turbulent velocity dispersion vs R (midplane)")
fig.suptitle(f"Velocity fluctuations (midplane) — t = {t_myr:.1f} Myr", fontsize=13)
fig.tight_layout()
fig.savefig("velocity_fluctuations.png", dpi=150, bbox_inches="tight")
plt.close(fig)
print(f"Saved: velocity_fluctuations.png")


# ============================================================
# Rotation curve diagnostic
# ============================================================

# indefinite integral of √(r²-x²)
def DiskSegment(x,r):
    return 1/2*( x * np.sqrt(r**2-x**2) + r**2*np.arctan2( x, np.sqrt(r**2 - x**2)) )
# overlap area of 1×1 square with lower corner (x0,y0), and disk with radius r
def grid_disc_intersection(r, x0, y0):
    
    # mirror into +x+y quarter of space
    # if x0 < 0:
    #     x0 = -1-x0
    # if y0 < 0:
    #     y0 = -1-y0
    
    x1 = x0+1
    y1 = y0+1
    
    # # check if fully outside (lower corner outside)
    # if x0**2 + y0**2 >= r**2:
    #     return 0
    
    # # check if fully inside (upper corner inside)
    # if x1**2 + y1**2 <= r**2:
    #     return 1
    
    # Compute integration region which is not neccessarily [x0, x0 + 1]
    
    # intersection with upper cell bound
    xcross = np.sqrt(r**2 - np.min((y1,r),axis=0)**2)  # upper intersection x pos, or minimum 0
    x0t = np.max((x0,xcross),axis=0)
    
    xcross = np.sqrt(r**2 - y0**2)  # lower intersection x pos (has to exist, but may be right of cell)
    x1t = np.min((x1,xcross),axis=0)
    
    res = DiskSegment(x1t,r) - DiskSegment(x0t,r) - (x1t-x0t)*y0 + (x0t - x0)*(y1-y0)
    
    res[x0**2 + y0**2 >= r**2] = 0
    res[x1**2 + y1**2 <= r**2] = 1
    
    return res
# data must be an even-by-even square 2d grid of densities (g/cm²) of width width
def sample_plane_density_to_radii(data,res=RES,cutoff_radius=None,dosum=True):
    
    data_res = data.shape[0]
    
    if len(data.shape) != 2:
        raise Exception('data must be 2D')
    
    for dim in data.shape:
        if dim != data_res:
            raise Exception('data must be square')
    
    if data_res%2:
        raise Exception('data must have even width')
    
    if cutoff_radius is None:
        cutoff_radius = width/2
    
    bins_per_grid = res/cutoff_radius * (width/2)/(data_res//2)
    bins = np.linspace(0, cutoff_radius, res+1)
    midpoints = (bins[:-1] + bins[1:])/2
    # mass = np.zeros(res+1)
    # mass = np.zeros(4*res)  # TODO replace
    
    pos = np.arange(data_res)
    pos -= data_res//2
    pos = np.max([pos,-1-pos],axis=0)
    xs, ys = np.meshgrid(pos, pos)
    # xs, ys = xs.flatten(), ys.flatten()
    
    lowerR = (xs**2 + ys**2)**.5
    upperR = ((xs+1)**2 + (ys+1)**2)**.5
    
    lowerlbs = (lowerR * bins_per_grid).astype(int)
    upperlbs = (upperR * bins_per_grid).astype(int)
    
    # don't compute stuff that will be trimmed
    mask = lowerlbs<res
    xs = xs[mask]
    ys = ys[mask]
    lowerlbs = lowerlbs[mask]
    upperlbs = upperlbs[mask]
    data = data[mask]
    
    spans = upperlbs-lowerlbs+2
    Δbs = np.arange(spans.sum()) - np.repeat(np.cumsum(spans) - spans, spans)
    bs = Δbs + np.repeat(lowerlbs, spans)
    bs = np.repeat(lowerlbs, spans) + np.arange(spans.sum()) - np.repeat(np.cumsum(spans) - spans, spans)
    xs = np.repeat(xs, spans)
    ys = np.repeat(ys, spans)
    data = np.repeat(data, spans)
    with np.errstate(invalid='ignore'):
        areas = grid_disc_intersection(bs/bins_per_grid,xs,ys)
    ring_areas = ( areas[1:] - areas[:-1] * (Δbs[1:] > 0) )
    mass = ring_areas * data[1:]
    mass = np.bincount(bs[:-1], weights=mass, minlength=res)
    
    mass = mass[:res]
    
    if dosum:
        mass *= (width/data_res)**2  # multiply by old area per cell
        mass /= np.pi*( bins[1:]**2 - bins[:-1]**2 )  # divide by are of annuli to get back density
    else:
        mass /= np.bincount(bs[:-1], weights=ring_areas, minlength=res)[:res] # divide by number of cells to get average
    
    return midpoints, mass


bounds = (0,width/2)  # MUST start on 0

vol_dens_R, vol_dens = sample_plane_density_to_radii(ρ_slices["z"],1000,bounds[1])

toomre_res = 1000

rad_mass_data = get_proj("z",("boxlib", "gasDensity"))
rad_mass_R, rad_mass = sample_plane_density_to_radii(rad_mass_data,toomre_res,bounds[1])
Q_mass = rad_mass.copy()
rad_mass /= Msun/pc**2

Q_R, Q_cs = sample_plane_density_to_radii(cs,toomre_res,bounds[1],False)  # average cs on the ring

Q_Rv, Q_vcirc = sample_plane_density_to_radii(vφ_xy,2*toomre_res,bounds[1],False)  # average vcirc on the ring


# Q_Ω = lambda R: vcirc_ideal_f(R) / R
# ε = 0.2 * ( Q_R[1] - Q_R[0] )
# dΩdR = (Q_Ω(Q_R + ε) - Q_Ω(Q_R - ε)) / (2.0 * ε);  # numeric derivative of Ω
# inner_κ = 4*Q_Ω(Q_R)**2 + 2*Q_R*Q_Ω(Q_R)*dΩdR
# κ = np.sign(inner_κ) * np.sqrt(abs(inner_κ));
# ToomreQ = κ * Q_cs /( np.pi * 6.6743015e-8 * Q_mass )
Q_Ωr = Q_vcirc / Q_Rv
ε = Q_Rv[1] - Q_Rv[0]
dΩdR = (Q_Ωr[1::2] - Q_Ωr[0::2]) / ε;  # numeric derivative of Ω
Q_Ω  = (Q_Ωr[1::2] + Q_Ωr[0::2]) / 2
inner_κ = 4*Q_Ω**2 + 2*Q_R*Q_Ω*dΩdR
κ = np.sign(inner_κ) * np.sqrt(abs(inner_κ));
ToomreQ = κ * Q_cs /( np.pi * 6.6743015e-8 * Q_mass )


R_bins    = np.linspace(*bounds, 80)
R_mid     = 0.5 * (R_bins[:-1] + R_bins[1:])
vcirc_med = np.full(len(R_mid),np.nan)
vcirc_p16 = np.full(len(R_mid),np.nan)
vcirc_p84 = np.full(len(R_mid),np.nan)

for idx_r, (rlo, rhi) in enumerate(zip(R_bins[:-1], R_bins[1:])):
    mask = (R_XY >= rlo) & (R_XY < rhi)
    if mask.sum() > 4:
        vals             = vφ_xy[mask]
        vcirc_med[idx_r] = np.median(vals)
        vcirc_p16[idx_r] = np.percentile(vals, 16)
        vcirc_p84[idx_r] = np.percentile(vals, 84)


fig, axes = plt.subplots(1, 3, figsize=(3*6, 5))

ax = axes[0]
ax2 = ax.twinx()
ax2.axvline(Rc/kpc, color="gray", ls=":", lw=1, label=f"Rc = {Rc/kpc} kpc")
ax.plot(rad_mass_R/kpc, rad_mass, lw=2, label="surface density", color="C0")
# ax.plot(rad_mass_R/kpc, rad_mass[0] * rad_mass_f(rad_mass_R), lw=2, ls="--", label="MHDDisk surfaceDensityProfile", color="green")
ax2.plot(vol_dens_R/kpc, vol_dens/m_u, lw=2, label='volume density', color="C1")
ax.set_xlabel("R [kpc]"); ax.set_ylabel(r"ρ [M$_\odot$/pc$^2$]", color="C0"); ax2.set_ylabel(r"ρ [m$_\text{u}$/cm$^3$]", color="C1")
ax.tick_params(axis="y", colors="C0"); ax2.tick_params(axis="y", colors="C1")
ax.set_title("disk density")
ax.set_xlim(bounds[0]/kpc, bounds[1]/kpc); ax.set_ylim(bottom=0); ax2.set_ylim(bottom=0)
ax.legend(*zip(*( (line, line.get_label()) for line in ax.get_lines() + ax2.get_lines() )))

ax = axes[1]
ax.axhline(1, bounds[0]/kpc, bounds[1]/kpc, color="black", lw=1)
ax.plot(Q_R/kpc, ToomreQ, lw=2, label=f"Q")
ax.axvline(Rc/kpc, color="gray", ls=":", lw=1, label=f"Rc = {Rc/kpc} kpc")
ax.set_xlabel("R [kpc]"); # ax.set_ylabel(r"Q")
ax.set_title("Toomre stability")
ax.legend()
ax.set_xlim(bounds[0]/kpc, bounds[1]/kpc); ax.set_ylim(bottom=0, top=10)

ax = axes[2]
ax.fill_between(R_mid/kpc, vcirc_p16/1e5, vcirc_p84/1e5, alpha=0.3, label="16–84th pct")
ax.plot(R_mid/kpc, vcirc_med/1e5, lw=2, label="Median (sim)")
ax.plot(np.linspace(*bounds, 500)/kpc, vcirc_ideal_f(np.linspace(*bounds, 500))/1e5, lw=2, ls="--", label="Analytic", color="green")
ax.axvline(Rc/kpc,        color="gray", ls=":", lw=1, label=f"Rc = {Rc/kpc} kpc")
ax.set_xlabel("R [kpc]"); ax.set_ylabel("$v_{circ}$ [km/s]")
ax.set_title("Sim circular velocity vs analytic (midplane)")
ax.legend(fontsize=8); ax.set_xlim(bounds[0]/kpc, bounds[1]/kpc); ax.set_ylim(bottom=0)

fig.suptitle(f"Toomre stuff — t = {t_myr:.1f} Myr", fontsize=12)
fig.tight_layout()
fig.savefig("toomre.png", dpi=150, bbox_inches="tight")
plt.close(fig)
print(f"Saved: toomre.png")



print(f"All plots complete. Wall time: {time.time() - t_start:.1f} s")