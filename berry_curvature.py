"""Generic SqrLatt Berry-curvature simulator and record visualizer."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from mephc.records import load_record, make_image_path, make_record
from mephc.workflows import resolve_record, save_record_outputs
from mephc.kspace import SquareKSpace

project_root = Path(__file__).resolve().parent


def _symmetry_token(symmetry, raw_full_grid, structure=None):
    """Resolve a compatibility token through canonical C4 verification."""
    if structure is None:
        if raw_full_grid or symmetry is None:
            return "raw"
        if symmetry in {"c4", "c4q"}:
            return "c4q"
        raise ValueError("symmetry must be None, 'c4', or 'c4q'.")
    return structure.resolve_symmetry(symmetry, raw_full_grid=raw_full_grid)


def _c4_expand(k_points, values, tolerance=1e-10):
    """Expand a verified first-quadrant result by proper C4 rotations."""
    points = np.asarray(k_points, dtype=float)
    if len(points) == 0:
        raise ValueError("cannot expand an empty C4 sample")
    N = max(1, len(np.unique(np.round(points[:, 0], 10))))
    return SquareKSpace(N).c4_expand(points, values, tolerance=tolerance)


def _compute_raw_berry_grid(band, pattern, k_points, *, step, num_bands, band_index):
    """Run the shared MePhC Berry calculator at explicit Cartesian k-points."""
    return band.compute_berry_grid(pattern, k_points, step=step, num_bands=num_bands, band_index=band_index, symmetry=None)


def compute_berry_curvature(config, *, resolution: int, num_bands: int, grid_n: int, step: float, grid_extent: float = 0.5, band_index=None, symmetry=None, raw_full_grid: bool = False, run_mode: str = "auto", archive: bool = False, reuse_requires_compute_match: bool = True, record_path=None, save: bool = True, save_tmp: bool = True, source_case=None):
    """Load or calculate one multi-band Berry record.

    Identity behavior retains first-quadrant C4q reduction and point order.
    Affine cases sample the canonical current BZ and reject C4 reduction.
    """
    if not isinstance(num_bands, int) or isinstance(num_bands, bool) or num_bands < 1:
        raise ValueError("num_bands must be an integer >= 1")
    if not isinstance(grid_n, int) or isinstance(grid_n, bool) or grid_n < 1:
        raise ValueError("grid_n must be an integer >= 1")
    if step <= 0:
        raise ValueError("step must be positive")
    if band_index is not None and (not isinstance(band_index, int) or band_index < 0 or band_index >= num_bands):
        raise ValueError(f"band_index must be None or between 0 and {num_bands - 1}")

    structure = config.canonical_structure()
    geometry_id = structure.geometry_id()
    symmetry_name = _symmetry_token(symmetry, raw_full_grid, structure)
    if symmetry_name == "c4q":
        raw_points = config.c4_quadrant(grid_n, extent=grid_extent)
        domain = "c4_quadrant"
    elif structure.identity:
        raw_points = config.square_grid(grid_n, extent=grid_extent)
        domain = "square_full_grid"
    else:
        raw_points = config.square_grid(grid_n, extent=grid_extent)
        domain = "current_bz"

    task_params = {"num_bands": int(num_bands), "grid_n": int(grid_n), "grid_extent": float(grid_extent), "symmetry": symmetry_name, "step": float(step), "band_index": band_index}
    if not structure.identity:
        task_params.update({"domain": domain, "canonical_structure": "sqrlatt.square_hole.canonical.v1", "stretch_factor": float(structure.stretch_factor), "stretch_angle_degrees": float(structure.stretch_angle_degrees)})
    compute_params = {"resolution": int(resolution), "lattice_type": config.lattice_type}
    if not structure.identity:
        compute_params.update({"direct_basis": structure.lattice.direct_basis.tolist(), "reciprocal_basis": structure.lattice.reciprocal_basis.tolist()})
    record, path = resolve_record(project_root, geometry_id, "bc", task_params=task_params, compute_params=compute_params, run_mode=run_mode, record_path=record_path, reuse_requires_compute_match=reuse_requires_compute_match)
    if record is not None:
        return record, path, None

    band = config.make_band(resolution=resolution)
    raw_result = _compute_raw_berry_grid(band, config.build_pattern(), raw_points, step=step, num_bands=num_bands, band_index=band_index)
    if symmetry_name == "c4q":
        expanded_points, expanded_values = _c4_expand(raw_result["k_points"], raw_result["bcs"])
        result = dict(raw_result)
        result.update({"raw_k_points": raw_result["k_points"], "raw_bcs": raw_result["bcs"], "k_points": expanded_points, "bcs": expanded_values})
    else:
        result = dict(raw_result)
        result.update({"raw_k_points": raw_result["k_points"], "raw_bcs": raw_result["bcs"], "k_points": raw_result["k_points"], "bcs": raw_result["bcs"]})
    result.update({"symmetry": symmetry_name, "domain": domain, "canonical_structure": structure.metadata(), "domain_outline": structure.first_bz.vertices, "grid_extent": float(grid_extent)})
    record = make_record("bc", geometry_id, task_params=task_params, compute_params=compute_params, data=result, source_case=source_case)
    canonical_path, tmp_path = save_record_outputs(project_root, geometry_id, "bc", task_params, record, archive=archive, archive_params={"num_bands": num_bands, "band_index": band_index, "grid_n": grid_n, "grid_extent": grid_extent, "symmetry": symmetry_name, "step": step}, save=save, save_tmp=save_tmp, tmp_name="bc_latest.pkl")
    return record, canonical_path, tmp_path


def _berry_image_path(record_path, geometry_id, band_index, multi_band):
    base = make_image_path(project_root, record_path, geometry_id)
    if multi_band:
        return base.with_name(f"{base.stem}_b{band_index + 1}{base.suffix}")
    return base


def plot_berry_record(record_or_path, *, band_index: int = 0, show: bool = False, save: bool = True, mesh_size: int = 120, interpolation: str = "linear", image_path=None, plot_params=None):
    """Plot one 0-based band from a Berry record without MPB."""
    record_path = None
    if isinstance(record_or_path, (str, Path)):
        record_path = Path(record_or_path)
        record = load_record(record_path)
    else:
        record = record_or_path
    params = dict(plot_params or {})
    show = params.pop("show", show)
    save = params.pop("save", save)
    mesh_size = params.pop("mesh_size", mesh_size)
    interpolation = params.pop("interpolation", interpolation)
    data = record["data"]
    values = np.asarray(data["bcs"], dtype=float)
    multi_band = values.ndim == 2
    if multi_band:
        if band_index < 0 or band_index >= values.shape[1]:
            raise ValueError(f"band_index must be between 0 and {values.shape[1] - 1}")
        values = values[:, band_index]
    elif band_index not in (0, None):
        raise ValueError("single-band Berry records only accept band_index=0")
    if image_path is None and save:
        if record_path is None:
            raise ValueError("image_path is required for an in-memory record")
        image_path = _berry_image_path(record_path, record["geometry_id"], int(band_index or 0), multi_band)
    params.setdefault("title", f"Berry curvature (Band {int(band_index or 0) + 1})")
    params.setdefault("colorbar_label", "Berry curvature")
    from mephc.plotting import plot_scalar_field
    fig, ax = plot_scalar_field(data["k_points"], values, mesh_size=mesh_size, interpolation=interpolation, save_path=image_path if save else None, show=show, **params)
    return fig, ax, image_path
