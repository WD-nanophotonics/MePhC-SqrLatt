from pathlib import Path
import sys

case_root = Path(__file__).resolve().parent
project_root = case_root.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

import config
import importlib.util

_runner_path = project_root / "efs.py"
_runner_spec = importlib.util.spec_from_file_location("efs_runner", _runner_path)
_runner = importlib.util.module_from_spec(_runner_spec)
_runner_spec.loader.exec_module(_runner)
compute_efs = _runner.compute_efs
plot_efs_record = _runner.plot_efs_record


resolution = 6
num_bands = 3
# Number of k samples per axis in the square first Brillouin zone.
grid_n = 3
# Python 0-based band stored in the EFS task identity and plotted by default.
band_index = 0

# "auto": reuse or compute; "compute": force overwrite; "plot_only": never run MPB.
run_mode = "auto"
# Add a timestamped archive while retaining the canonical working record.
archive_record = False
# Explicit existing .pkl to load, taking priority over automatic matching.
record_path = None
# True requires compute metadata such as resolution to match before reuse.
reuse_requires_compute_match = True

# Plot-only parameters below do not participate in record matching.
plot_params = {
    "save": True,
    "show": False,
    # True plots converted THz values; False plots normalized MPB frequency.
    "use_actual": True,
    # Python 0-based band selected from the record for visualization.
    "band_index": band_index,
    "figsize": (5.4, 4.8),
    "dpi": 140,
    "title": "Square-hole EFS",
    "xlabel": "kx",
    "ylabel": "ky",
    "font_size": 11,
    "tick_size": 10,
    "grid": True,
    "grid_kwargs": {"linestyle": ":", "linewidth": 0.45, "alpha": 0.7},
    # Resolution and SciPy method of the interpolated contour mesh.
    "mesh_size": 80,
    "interpolation": "linear",
    # Integer contour count; a list/array would request exact frequency levels.
    "levels": 8,
    "cmap": "viridis",
    "linewidth": 1.2,
    "colorbar": True,
    # colorbar_kwargs controls appearance and ticks, not contour values.
    "colorbar_kwargs": {"fraction": 0.046, "pad": 0.04},
}


def main():
    """Obtain an EFS record and optionally render its selected band."""
    record, output_record_path, tmp_path = compute_efs(
        config,
        resolution=resolution,
        num_bands=num_bands,
        grid_n=grid_n,
        band_index=band_index,
        run_mode=run_mode,
        archive=archive_record,
        reuse_requires_compute_match=reuse_requires_compute_match,
        record_path=record_path,
        source_case=str(case_root),
    )
    image_path = None
    if plot_params.get("save", True) or plot_params.get("show", False):
        _, ax, image_path = plot_efs_record(output_record_path, plot_params=plot_params)
        print("plot title:", ax.get_title())
    print("record:", output_record_path)
    print("tmp record:", tmp_path)
    print("image:", image_path)
    print("freqs shape:", record["data"].freqs.shape)
    return record, output_record_path, image_path


if __name__ == "__main__":
    main()
