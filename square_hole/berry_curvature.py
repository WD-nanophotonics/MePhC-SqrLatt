from pathlib import Path
import sys

case_root = Path(__file__).resolve().parent
project_root = case_root.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

import config
import importlib.util

_runner_path = project_root / "berry_curvature.py"
_runner_spec = importlib.util.spec_from_file_location("berry_curvature_runner", _runner_path)
_runner = importlib.util.module_from_spec(_runner_spec)
_runner_spec.loader.exec_module(_runner)
compute_berry_curvature = _runner.compute_berry_curvature
plot_berry_record = _runner.plot_berry_record


resolution = 64
num_bands = 3
# In c4q mode this is samples per axis in the first quadrant. Expansion yields
# a regular (2*grid_n - 1) by (2*grid_n - 1) full-square grid.
grid_n = 24
# The final plotted domain is [-grid_extent, grid_extent] on both k axes.
# Changing this changes simulation data and therefore the record identity.
grid_extent = 1.0
# Berry plaquette side length in Cartesian reciprocal coordinates.
step = 0.0005
# None computes and stores every band up to num_bands in one simulation record.
# Set a 0-based integer only when intentionally computing one band.
band_index = None
# Python 0-based band selected only for plotting; changing it does not recompute.
plot_band_index = 2
# "auto" verifies the complete canonical structure: identity resolves to c4q;
# a non-identity affine case resolves conservatively to raw_bz.
symmetry = "auto"
# True bypasses C4 reduction and independently computes the full square grid.
raw_full_grid = False

# "auto": reuse or compute; "compute": force overwrite; "plot_only": never run MPB.
run_mode = "auto"
# Add a timestamped archive while retaining the canonical working record.
archive_record = False
# Explicit existing .pkl to load, taking priority over automatic matching.
record_path = None
# True requires compute metadata such as resolution to match before reuse.
reuse_requires_compute_match = True

# Plot-only parameters below do not affect record identity or reuse.
plot_params = {
    "save": True,
    "show": False,
    "figsize": (5.2, 4.8),
    "dpi": 140,
    "title": None,
    "xlabel": "kx",
    "ylabel": "ky",
    "font_size": 11,
    "tick_size": 10,
    "grid": True,
    "grid_kwargs": {"linestyle": ":", "linewidth": 0.45, "alpha": 0.7},
    # mesh_size/interpolation are used only for scattered data. Complete C4
    # regular grids are reshaped directly with no Delaunay interpolation.
    "mesh_size": 80,
    "interpolation": "linear",
    "cmap": "RdBu_r",
    # vmin/vmax control normalization. Set both to None for automatic limits.
    "vmin": -1,
    "vmax": 1,
    "colorbar": True,
    "colorbar_label": "Berry curvature",
    # colorbar_kwargs controls appearance; for example add "ticks": [-0.5, 0, 0.5].
    "colorbar_kwargs": {"fraction": 0.046, "pad": 0.04},
}


def main():
    """Obtain a multi-band Berry record and plot one selected band."""
    record, output_record_path, tmp_path = compute_berry_curvature(
        config,
        resolution=resolution,
        num_bands=num_bands,
        grid_n=grid_n,
        grid_extent=grid_extent,
        step=step,
        band_index=band_index,
        symmetry=symmetry,
        raw_full_grid=raw_full_grid,
        run_mode=run_mode,
        archive=archive_record,
        reuse_requires_compute_match=reuse_requires_compute_match,
        record_path=record_path,
        source_case=str(case_root),
    )
    image_path = None
    if plot_params.get("save", True) or plot_params.get("show", False):
        _, ax, image_path = plot_berry_record(output_record_path, band_index=plot_band_index, plot_params=plot_params)
        print("plot title:", ax.get_title())
    print("record:", output_record_path)
    print("tmp record:", tmp_path)
    print("image:", image_path)
    print("berry curvature shape:", record["data"]["bcs"].shape)
    return record, output_record_path, image_path


if __name__ == "__main__":
    main()
