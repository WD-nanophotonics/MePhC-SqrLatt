from pathlib import Path
import sys

case_root = Path(__file__).resolve().parent
project_root = case_root.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

import config
import importlib.util

_runner_path = project_root / "band_structure.py"
_runner_spec = importlib.util.spec_from_file_location("band_structure_runner", _runner_path)
_runner = importlib.util.module_from_spec(_runner_spec)
_runner_spec.loader.exec_module(_runner)
compute_band_structure = _runner.compute_band_structure
plot_band_record = _runner.plot_band_record
preview_unit_cell = _runner.preview_unit_cell


resolution = 64
num_bands = 3
# Number of intervals on each Gamma-X, X-M, and M-Gamma segment.
# The complete path contains 3*n_per_segment + 1 sampled k-points.
n_per_segment = 10

# Preview switches do not affect records or the formal calculation.
preview_numpy = True
preview_mpb = False
# MPB resolution used only by the inexpensive dielectric preview.
preview_resolution = 32
# False means preview only; no record lookup, simulation, or plot is performed.
run_calculation = True

# True computes Berry curvature at every band-path point and uses it as color.
# This is substantially more expensive than an ordinary band calculation.
color_by_berry = True
# Side length of each Berry plaquette in Cartesian reciprocal coordinates.
berry_step = 0.0005
# Number of bands computed when color_by_berry is enabled.
berry_num_bands = num_bands

# "auto": reuse a matching record or compute it if missing.
# "compute": always recompute and overwrite the canonical record.
# "plot_only": require an existing matching record and never run MPB.
run_mode = "auto"
# Keep an additional timestamped record as well as the canonical working file.
archive_record = False
# Optional explicit .pkl path. When set, it is loaded before run_mode matching.
record_path = None
# True makes resolution and other compute metadata part of cache matching.
reuse_requires_compute_match = True

# Plot-only parameters below never participate in record matching.
plot_params = {
    # save writes a PNG derived from the record name; show opens a GUI window.
    "save": True,
    "show": False,
    "figsize": (6.0, 5.0),
    "dpi": 140,
    "title": "Square-hole band structure",
    # None keeps the high-symmetry tick labels without adding a separate x label.
    "xlabel": None,
    "ylabel": "Frequency (THz)",
    "font_size": 11,
    "tick_size": 10,
    "grid": True,
    # grid_kwargs is passed directly to Matplotlib Axes.grid.
    "grid_kwargs": {"axis": "y", "linestyle": ":", "linewidth": 0.5, "alpha": 0.75},
    "legend": True,
    "legend_kwargs": {"fontsize": 9, "loc": "best"},
    # line and scatter are independent; enabling both overlays markers on lines.
    "line": True,
    "scatter": True,
    # None uses Matplotlib's color cycle. A supplied list is cycled by band.
    "color_list": None,
    "linewidth": 1.5,
    # markersize follows scatter semantics: marker area in points squared.
    "markersize": 18,
    # Black marker outlines keep zero-BC (white) points visible.
    "scatter_edgecolor": "black",
    "scatter_linewidth": 0.5,
    # Scatter must stay above line so its outline is never cut by the line.
    "line_zorder": 2,
    "scatter_zorder": 3,
    "alpha": 1.0,
    "linestyle": "-",
    "marker": "o",
    "color_by_berry": color_by_berry,
    # Berry color limits can be fixed with bc_vmin/bc_vmax at this top level.
    "bc_cmap": "RdBu_r",
    "bc_vmin": None,
    "bc_vmax": None,
    "bc_label": "Berry curvature",
    "colorbar": color_by_berry,
    # colorbar_kwargs controls layout/ticks, not the colormap value limits.
    "colorbar_kwargs": {"fraction": 0.046, "pad": 0.04},
}


def main():
    """Preview, obtain a band record, then optionally render it."""
    if preview_numpy or preview_mpb:
        preview_unit_cell(
            config,
            resolution=preview_resolution,
            numpy_preview=preview_numpy,
            mpb_preview=preview_mpb,
            show=True,
        )

    if not run_calculation:
        print("preview complete; run_calculation is False")
        return None, None, None

    compute_num_bands = berry_num_bands if color_by_berry else num_bands
    record, output_record_path, tmp_path = compute_band_structure(
        config,
        resolution=resolution,
        num_bands=compute_num_bands,
        n_per_segment=n_per_segment,
        compute_bc=color_by_berry,
        berry_step=berry_step,
        run_mode=run_mode,
        archive=archive_record,
        reuse_requires_compute_match=reuse_requires_compute_match,
        record_path=record_path,
        source_case=str(case_root),
    )
    image_path = None
    if plot_params.get("save", True) or plot_params.get("show", False):
        _, ax, image_path = plot_band_record(output_record_path, plot_params=plot_params)
        print("plot title:", ax.get_title())
    print("record:", output_record_path)
    print("tmp record:", tmp_path)
    print("image:", image_path)
    print("freqs shape:", record["data"]["freqs"].shape)
    if record["data"].get("bcs") is not None:
        print("berry curvature shape:", record["data"]["bcs"].shape)
    return record, output_record_path, image_path


if __name__ == "__main__":
    main()
