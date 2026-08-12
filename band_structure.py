"""Generic SqrLatt band simulator and record visualizer."""

from __future__ import annotations

from pathlib import Path
import importlib.util

from mephc.plotting import plot_band_path
from mephc.preview import preview_mpb_dielectric, preview_pattern
from mephc.records import load_record, make_image_path, make_record
from mephc.workflows import resolve_record, save_record_outputs

project_root = Path(__file__).resolve().parent


def preview_unit_cell(config, *, resolution: int, mpb_preview: bool = True, numpy_preview: bool = True, show: bool = True, preview_num_bands: int = 1):
    """Preview the canonical NumPy motif and optional MPB dielectric.

    Preview is side-effect free: it does not read or write records or images.
    Both previews consume the same pattern and current-cell outline as MPB.
    """
    band = config.make_band(resolution=resolution)
    pattern = config.build_pattern()
    figures = {}
    if numpy_preview:
        figures["numpy"] = preview_pattern(config.preview_pattern_data(), outline=config.unit_cell_outline(), show=show)
    if mpb_preview:
        figures["mpb"] = preview_mpb_dielectric(band, pattern, num_bands=preview_num_bands, k_point=(0.0, 0.0), show=show)
    return figures


def _case_structure(config):
    """Return the case's single canonical geometry authority."""
    return config.canonical_structure()


def _compute_params(config, resolution: int) -> dict:
    """Return cache metadata without changing identity task filenames."""
    structure = _case_structure(config)
    params = {"resolution": int(resolution), "lattice_type": config.lattice_type}
    if not structure.identity:
        params.update({
            "canonical_structure": "sqrlatt.square_hole.canonical.v1",
            "direct_basis": structure.lattice.direct_basis.tolist(),
            "reciprocal_basis": structure.lattice.reciprocal_basis.tolist(),
        })
    return params


def compute_band_structure(config, *, resolution: int, num_bands: int, n_per_segment: int, compute_bc: bool = False, berry_step: float = 0.0005, run_mode: str = "auto", archive: bool = False, reuse_requires_compute_match: bool = True, record_path=None, save: bool = True, save_tmp: bool = True, source_case=None):
    """Load or calculate one canonical band-path record.

    Identity cases retain the historical task key band_nb_gxm. Affine cases
    use a generic current-BZ path and never claim unverified X or M labels.
    """
    structure = _case_structure(config)
    geometry_id = structure.geometry_id()
    task_params = {
        "num_bands": int(num_bands),
        "path": structure.band_path_policy(),
        "n_per_segment": int(n_per_segment),
        "compute_bc": bool(compute_bc),
        "berry_step": float(berry_step) if compute_bc else None,
    }
    if not structure.identity:
        task_params.update({
            "domain": "current_bz",
            "symmetry": "none",
            "stretch_factor": float(structure.stretch_factor),
            "stretch_angle_degrees": float(structure.stretch_angle_degrees),
        })
    compute_params = _compute_params(config, resolution)
    record, path = resolve_record(project_root, geometry_id, "band", task_params=task_params, compute_params=compute_params, run_mode=run_mode, record_path=record_path, reuse_requires_compute_match=reuse_requires_compute_match)
    if record is not None:
        return record, path, None

    band = config.make_band(resolution=resolution)
    result = band.compute_band_path_with_berry(config.build_pattern(), path=config.band_path(), n_per_segment=n_per_segment, step=berry_step, num_bands=num_bands, compute_bc=compute_bc)
    result["canonical_structure"] = structure.metadata()
    result["path_policy"] = structure.band_path_policy()
    record = make_record("band", geometry_id, task_params=task_params, compute_params=compute_params, data=result, source_case=source_case)
    canonical_path, tmp_path = save_record_outputs(project_root, geometry_id, "band", task_params, record, archive=archive, archive_params={"num_bands": num_bands, "path": task_params["path"]}, save=save, save_tmp=save_tmp, tmp_name="band_latest.pkl")
    return record, canonical_path, tmp_path


def plot_band_record(record_or_path, *, show: bool = False, save: bool = True, use_actual: bool = True, image_path=None, plot_params=None):
    """Render a band record without invoking MPB."""
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
            raise ValueError("band record has no Berry data; recompute with compute_bc=True")
        params.setdefault("bc_values", bcs)
    if image_path is None and save:
        if record_path is None:
            raise ValueError("image_path is required for an in-memory record")
        image_path = make_image_path(project_root, record_path, record["geometry_id"])
    fig, ax = plot_band_path(record["data"], use_actual=use_actual, save_path=image_path if save else None, show=show, **params)
    return fig, ax, image_path


def main():
    """Execute the case-level band script when called at project root."""
    case_path = project_root / "square_hole" / "band_structure.py"
    spec = importlib.util.spec_from_file_location("sqrlatt_case_band_structure", case_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load case script: {case_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.main()


if __name__ == "__main__":
    main()
