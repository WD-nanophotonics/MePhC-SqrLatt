from pathlib import Path
import sys

project_root = Path(__file__).resolve().parent

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
from mephc.plotting import plot_band_path
from mephc.workflows import save_record_outputs
from mephc.preview import preview_mpb_dielectric, preview_pattern


def preview_unit_cell(config, *, resolution, mpb_preview=True, numpy_preview=True, show=True, preview_num_bands=1):
    """Preview case geometry without reading or writing simulation records.

    ``numpy_preview`` draws normalized pattern data. ``mpb_preview`` runs one
    lightweight MPB solve and shows raw plus rectified dielectric arrays.
    """
    band = config.make_band(resolution=resolution)
    pattern = config.build_pattern()
    figures = {}
    if numpy_preview:
        outline = config.unit_cell_outline() if hasattr(config, "unit_cell_outline") else None
        figures["numpy"] = preview_pattern(
            config.preview_pattern_data() if hasattr(config, "preview_pattern_data") else pattern,
            outline=outline,
            show=show,
        )
    if mpb_preview:
        figures["mpb"] = preview_mpb_dielectric(
            band,
            pattern,
            num_bands=preview_num_bands,
            k_point=(0.0, 0.0),
            show=show,
        )
    return figures


def _resolve_existing_record(
    *,
    config,
    kind,
    task_params,
    compute_params,
    run_mode,
    record_path,
    reuse_requires_compute_match,
):
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


def compute_band_structure(
    config,
    *,
    resolution,
    num_bands,
    n_per_segment,
    compute_bc=False,
    berry_step=0.0005,
    run_mode="auto",
    archive=False,
    reuse_requires_compute_match=True,
    record_path=None,
    save=True,
    save_tmp=True,
    source_case=None,
):
    """Load or compute one band-path record.

    ``n_per_segment`` is intervals per high-symmetry segment. ``compute_bc``
    adds Berry values at every path point using plaquette side ``berry_step``.
    ``run_mode`` is ``auto``, ``compute``, or ``plot_only``. Plot parameters are
    intentionally absent because they never define simulation data.

    Returns ``(record, canonical_or_loaded_path, tmp_path)``. ``tmp_path`` is
    ``None`` when an existing record was reused.
    """
    task_params = {
        "num_bands": num_bands,
        "path": "gxm",
        "n_per_segment": n_per_segment,
        "compute_bc": bool(compute_bc),
        "berry_step": berry_step if compute_bc else None,
    }
    compute_params = {"resolution": resolution, "lattice_type": config.lattice_type}
    record, path = _resolve_existing_record(
        config=config,
        kind="band",
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
    result = band.compute_band_path_with_berry(
        pattern,
        path=config.band_path(),
        n_per_segment=n_per_segment,
        step=berry_step,
        num_bands=num_bands,
        compute_bc=compute_bc,
    )
    record = make_record(
        "band",
        config.geometry_id,
        task_params=task_params,
        compute_params=compute_params,
        data=result,
        source_case=source_case,
    )
    canonical_path, tmp_path = save_record_outputs(
        project_root,
        config.geometry_id,
        "band",
        task_params,
        record,
        archive=archive,
        archive_params={"num_bands": num_bands, "path": "gxm"},
        save=save,
        save_tmp=save_tmp,
        tmp_name="band_latest.pkl",
    )
    return record, canonical_path, tmp_path


def plot_band_record(record_or_path, *, show=False, save=True, use_actual=True, image_path=None, plot_params=None):
    """Render an existing band record without running MPB.

    ``plot_params`` is forwarded to :func:`mephc.plotting.plot_band_path`.
    When ``color_by_berry`` is true, the record must already contain ``bcs``.
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
    color_by_berry = params.pop("color_by_berry", False)
    if color_by_berry:
        bcs = record["data"].get("bcs")
        if bcs is None:
            raise ValueError("This band record does not contain Berry curvature data. Recompute with compute_bc=True.")
        params.setdefault("bc_values", bcs)
    if image_path is None and save:
        if record_path is None:
            raise ValueError("image_path is required when plotting an in-memory record with save=True.")
        image_path = make_image_path(project_root, record_path, record["geometry_id"])
    fig, ax = plot_band_path(record["data"], use_actual=use_actual, save_path=image_path if save else None, show=show, **params)
    return fig, ax, image_path
