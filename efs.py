from pathlib import Path
import sys

project_root = Path(__file__).resolve().parent

from mephc.efs import plot_efs
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


def compute_efs(
    config,
    *,
    resolution,
    num_bands,
    grid_n,
    band_index,
    run_mode="auto",
    archive=False,
    reuse_requires_compute_match=True,
    record_path=None,
    save=True,
    save_tmp=True,
    source_case=None,
):
    """Load or compute EFS frequencies on an ``grid_n x grid_n`` square grid.

    ``band_index`` is 0-based and identifies the intended contour task.
    ``run_mode`` is ``auto``, ``compute``, or ``plot_only``. Plot styling is
    not part of this function or the record matching key.
    """
    task_params = {"num_bands": num_bands, "grid_n": grid_n, "band_index": band_index}
    compute_params = {"resolution": resolution, "lattice_type": config.lattice_type}
    record, path = _resolve_existing_record(
        config=config,
        kind="efs",
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
    result = band.compute_square_efs(pattern, N=grid_n, num_bands=num_bands)
    record = make_record(
        "efs",
        config.geometry_id,
        task_params=task_params,
        compute_params=compute_params,
        data=result,
        source_case=source_case,
    )
    canonical_path = canonical_record_path(project_root, config.geometry_id, "efs", task_params)
    tmp_path = tmp_dir(project_root) / "efs_latest.pkl"
    if save:
        save_record(record, canonical_path)
        update_archive_manifest(project_root, canonical_path, record)
    if archive:
        record_name = make_record_name("efs", band_index=band_index, grid_n=grid_n, created_at=record["created_at"])
        archive_path = data_dir(project_root, config.geometry_id) / record_name
        save_record(record, archive_path)
        update_archive_manifest(project_root, archive_path, record)
    if save_tmp:
        save_record(record, tmp_path)
    return record, canonical_path, tmp_path


def plot_efs_record(record_or_path, *, show=False, save=True, use_actual=True, band_index=None, mesh_size=120, interpolation="linear", levels=8, image_path=None, plot_params=None):
    """Render EFS contours from an existing record without running MPB.

    ``use_actual`` chooses THz versus normalized frequency. ``levels`` is a
    contour count or explicit values; ``plot_params`` overrides matching named
    arguments and is forwarded to :func:`mephc.efs.plot_efs`.
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
    use_actual = params.pop("use_actual", use_actual)
    band_index = params.pop("band_index", band_index)
    mesh_size = params.pop("mesh_size", mesh_size)
    interpolation = params.pop("interpolation", interpolation)
    levels = params.pop("levels", levels)
    if band_index is None:
        band_index = record["task_params"].get("band_index", 0)
    if image_path is None and save:
        if record_path is None:
            raise ValueError("image_path is required when plotting an in-memory record with save=True.")
        image_path = make_image_path(project_root, record_path, record["geometry_id"])
    fig, ax = plot_efs(
        record["data"],
        band_index=band_index,
        use_actual=use_actual,
        mesh_size=mesh_size,
        interpolation=interpolation,
        levels=levels,
        save_path=image_path if save else None,
        show=show,
        **params,
    )
    return fig, ax, image_path
