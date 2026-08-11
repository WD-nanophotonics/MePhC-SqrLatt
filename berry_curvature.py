from pathlib import Path
import sys

import numpy as np

project_root = Path(__file__).resolve().parent

from mephc.kspace import SquareKSpace
from mephc.records import (
    canonical_record_path,
    data_dir,
    find_matching_record,
    load_record,
    make_image_path,
    make_record,
    make_record_name,
    save_record,
    tmp_dir,
    update_archive_manifest,
)


def _resolve_existing_record(*, config, kind, task_params, compute_params, run_mode, record_path, reuse_requires_compute_match):
    if record_path is not None:
        path = Path(record_path)
        return load_record(path), path
    if run_mode not in {"auto", "compute", "plot_only"}:
        raise ValueError("run_mode must be 'auto', 'compute', or 'plot_only'.")
    if run_mode in {"auto", "plot_only"}:
        record, path = find_matching_record(
            project_root,
            config.geometry_id,
            kind,
            task_params=task_params,
            compute_params=compute_params,
            require_compute_match=reuse_requires_compute_match,
        )
        if record is not None:
            return record, path
        if run_mode == "plot_only":
            expected = canonical_record_path(project_root, config.geometry_id, kind, task_params)
            raise FileNotFoundError(f"No matching {kind!r} record found. Expected canonical path: {expected}")
    return None, None


def _square_c4_reduced_zone_points(N, extent):
    return np.asarray(SquareKSpace(N).c4_quadrant(extent=extent), dtype=float)


def _c4_expand(k_points, values, tolerance=1e-10):
    N = max(1, len(np.unique(np.round(np.asarray(k_points)[:, 0], 10))))
    return SquareKSpace(N).c4_expand(k_points, values, tolerance=tolerance)


def _symmetry_token(symmetry, raw_full_grid):
    if raw_full_grid or symmetry is None:
        return "raw"
    if symmetry in {"c4", "c4q"}:
        return "c4q"
    raise ValueError("symmetry must be None, 'c4', or 'c4q'.")


def _compute_raw_berry_grid(band, pattern, k_points, *, step, num_bands, band_index):
    return band.compute_berry_grid(pattern, k_points, step=step, num_bands=num_bands, band_index=band_index)


def compute_berry_curvature(
    config,
    *,
    resolution,
    num_bands,
    grid_n,
    step,
    grid_extent=0.5,
    band_index=None,
    symmetry=None,
    raw_full_grid=False,
    run_mode="auto",
    archive=False,
    reuse_requires_compute_match=True,
    record_path=None,
    save=True,
    save_tmp=True,
    source_case=None,
):
    """Load or compute a square-lattice Berry-curvature record.

    ``grid_n`` is samples per axis. With ``symmetry='c4q'`` it applies to the
    first quadrant and expands to ``(2*grid_n - 1)^2`` points. ``grid_extent``
    sets the final square bounds. ``step`` is the plaquette side in Cartesian
    reciprocal coordinates. ``band_index=None`` computes all ``num_bands``.

    ``raw_full_grid=True`` disables symmetry reduction. ``run_mode`` is
    ``auto``, ``compute``, or ``plot_only``.
    """
    symmetry_name = _symmetry_token(symmetry, raw_full_grid)
    task_params = {
        "num_bands": num_bands,
        "grid_n": grid_n,
        "grid_extent": grid_extent,
        "symmetry": symmetry_name,
        "step": step,
        "band_index": band_index,
    }
    compute_params = {"resolution": resolution, "lattice_type": config.lattice_type}
    record, path = _resolve_existing_record(
        config=config,
        kind="bc",
        task_params=task_params,
        compute_params=compute_params,
        run_mode=run_mode,
        record_path=record_path,
        reuse_requires_compute_match=reuse_requires_compute_match,
    )
    if record is not None:
        return record, path, None

    band = config.make_band(resolution=resolution)
    pattern = config.build_pattern()

    if symmetry_name == "c4q":
        raw_k_points = _square_c4_reduced_zone_points(grid_n, grid_extent)
        raw_result = _compute_raw_berry_grid(band, pattern, raw_k_points, step=step, num_bands=num_bands, band_index=band_index)
        expanded_k_points, expanded_bcs = _c4_expand(raw_result["k_points"], raw_result["bcs"])
        result = dict(raw_result)
        result.update(
            {
                "raw_k_points": raw_result["k_points"],
                "raw_bcs": raw_result["bcs"],
                "k_points": expanded_k_points,
                "bcs": expanded_bcs,
                "symmetry": "c4q",
            }
        )
    else:
        k_points = config.square_grid(grid_n, extent=grid_extent)
        result = _compute_raw_berry_grid(band, pattern, k_points, step=step, num_bands=num_bands, band_index=band_index)
        result["raw_k_points"] = result["k_points"]
        result["raw_bcs"] = result["bcs"]
        result["symmetry"] = "raw"

    record = make_record(
        "bc",
        config.geometry_id,
        task_params=task_params,
        compute_params=compute_params,
        data=result,
        source_case=source_case,
    )
    canonical_path = canonical_record_path(project_root, config.geometry_id, "bc", task_params)
    tmp_path = tmp_dir(project_root) / "bc_latest.pkl"
    if save:
        save_record(record, canonical_path)
        update_archive_manifest(project_root, canonical_path, record)
    if archive:
        record_name = make_record_name(
            "bc",
            num_bands=num_bands,
            band_index=band_index,
            grid_n=grid_n,
            grid_extent=grid_extent,
            symmetry=symmetry_name,
            step=step,
            created_at=record["created_at"],
        )
        archive_path = data_dir(project_root, config.geometry_id) / record_name
        save_record(record, archive_path)
        update_archive_manifest(project_root, archive_path, record)
    if save_tmp:
        save_record(record, tmp_path)
    return record, canonical_path, tmp_path


def _berry_image_path(record_path, geometry_id, band_index, multi_band):
    base = make_image_path(project_root, record_path, geometry_id)
    if multi_band:
        return base.with_name(f"{base.stem}_b{band_index + 1}{base.suffix}")
    return base


def plot_berry_record(record_or_path, *, band_index=0, show=False, save=True, mesh_size=120, interpolation="linear", image_path=None, plot_params=None):
    """Plot one Python 0-based band from an existing Berry record.

    Complete regular grids are reshaped directly. ``mesh_size`` and
    ``interpolation`` apply only to the scattered-data fallback.
    """
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

    from mephc.plotting import plot_scalar_field

    data = record["data"]
    values = data["bcs"]
    multi_band = getattr(values, "ndim", None) == 2
    if multi_band:
        if band_index < 0 or band_index >= values.shape[1]:
            raise ValueError(f"band_index must be between 0 and {values.shape[1] - 1}")
        values = values[:, band_index]
    elif band_index not in (0, None):
        raise ValueError("Single-band Berry record can only be plotted with band_index=0.")

    if image_path is None and save:
        if record_path is None:
            raise ValueError("image_path is required when plotting an in-memory record with save=True.")
        image_path = _berry_image_path(record_path, record["geometry_id"], int(band_index or 0), multi_band)
    if params.get("title") is None:
        params["title"] = f"Berry curvature (Band {int(band_index or 0) + 1})"
    if params.get("colorbar_label") is None:
        params["colorbar_label"] = "Berry curvature"
    fig, ax = plot_scalar_field(
        data["k_points"],
        values,
        mesh_size=mesh_size,
        interpolation=interpolation,
        save_path=image_path if save else None,
        show=show,
        **params,
    )
    return fig, ax, image_path
