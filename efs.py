"""Generic SqrLatt EFS simulator and record visualizer."""

from __future__ import annotations

from pathlib import Path

from mephc.efs import plot_efs
from mephc.records import load_record, make_image_path, make_record
from mephc.workflows import resolve_record, save_record_outputs

project_root = Path(__file__).resolve().parent


def compute_efs(config, *, resolution: int, num_bands: int, grid_n: int, band_index: int, run_mode: str = "auto", archive: bool = False, reuse_requires_compute_match: bool = True, record_path=None, save: bool = True, save_tmp: bool = True, source_case=None):
    """Load or calculate EFS data on the identity/current-BZ domain."""
    if not isinstance(num_bands, int) or isinstance(num_bands, bool) or num_bands < 1:
        raise ValueError("num_bands must be an integer >= 1")
    if not isinstance(grid_n, int) or isinstance(grid_n, bool) or grid_n < 1:
        raise ValueError("grid_n must be an integer >= 1")
    if not isinstance(band_index, int) or isinstance(band_index, bool) or not 0 <= band_index < num_bands:
        raise ValueError(f"band_index must be between 0 and {num_bands - 1}")
    structure = config.canonical_structure()
    geometry_id = structure.geometry_id()
    points = config.square_grid(grid_n, extent=0.5)
    domain = "square_full_grid" if structure.identity else "current_bz"
    task_params = {"num_bands": int(num_bands), "grid_n": int(grid_n), "band_index": int(band_index)}
    if not structure.identity:
        task_params.update({"domain": domain, "canonical_structure": "sqrlatt.square_hole.canonical.v1", "stretch_factor": float(structure.stretch_factor), "stretch_angle_degrees": float(structure.stretch_angle_degrees)})
    compute_params = {"resolution": int(resolution), "lattice_type": config.lattice_type}
    if not structure.identity:
        compute_params.update({"direct_basis": structure.lattice.direct_basis.tolist(), "reciprocal_basis": structure.lattice.reciprocal_basis.tolist()})
    record, path = resolve_record(project_root, geometry_id, "efs", task_params=task_params, compute_params=compute_params, run_mode=run_mode, record_path=record_path, reuse_requires_compute_match=reuse_requires_compute_match)
    if record is not None:
        return record, path, None
    band = config.make_band(resolution=resolution)
    result = band.compute_efs(config.build_pattern(), points, num_bands=num_bands)
    result.metadata.update({"domain": domain, "canonical_structure": structure.metadata(), "domain_outline": structure.first_bz.vertices, "sampling_order": "kx_outer_ky_inner"})
    record = make_record("efs", geometry_id, task_params=task_params, compute_params=compute_params, data=result, source_case=source_case)
    canonical_path, tmp_path = save_record_outputs(project_root, geometry_id, "efs", task_params, record, archive=archive, archive_params={"band_index": band_index, "grid_n": grid_n}, save=save, save_tmp=save_tmp, tmp_name="efs_latest.pkl")
    return record, canonical_path, tmp_path


def plot_efs_record(record_or_path, *, show: bool = False, save: bool = True, use_actual: bool = True, band_index=None, mesh_size: int = 120, interpolation: str = "linear", levels=8, image_path=None, plot_params=None):
    """Render an EFS record without invoking MPB."""
    record_path = None
    if isinstance(record_or_path, (str, Path)):
        record_path = Path(record_or_path)
        record = load_record(record_path)
    else:
        record = record_or_path
    params = dict(plot_params or {})
    show = params.pop("show", show)
    save = params.pop("save", save)
    use_actual = params.pop("use_actual", use_actual)
    band_index = params.pop("band_index", band_index)
    mesh_size = params.pop("mesh_size", mesh_size)
    interpolation = params.pop("interpolation", interpolation)
    levels = params.pop("levels", levels)
    if band_index is None:
        band_index = int(record["task_params"].get("band_index", 0))
    if image_path is None and save:
        if record_path is None:
            raise ValueError("image_path is required for an in-memory record")
        image_path = make_image_path(project_root, record_path, record["geometry_id"])
    fig, ax = plot_efs(record["data"], band_index=band_index, use_actual=use_actual, mesh_size=mesh_size, interpolation=interpolation, levels=levels, save_path=image_path if save else None, show=show, **params)
    return fig, ax, image_path
