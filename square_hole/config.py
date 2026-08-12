"""Geometry-only configuration for the SqrLatt square-hole case."""

from __future__ import annotations

from pathlib import Path
import sys

case_root = Path(__file__).resolve().parent
if str(case_root) not in sys.path:
    sys.path.insert(0, str(case_root))
try:
    from .canonical import SquareHoleStructure
except ImportError:
    from canonical import SquareHoleStructure
project_root = case_root.parent

# Physical geometry/material parameters. Task scripts own resolution, bands,
# k-grid, Berry step, workflow mode, and plotting settings.
lattice_type = "square"
hole_shape = "square_hole"
a = 400
d = 200
n_eff = 2.7
height = 1
polygon_sides = 4
polygon_rotation_degrees = 45.0
stretch_factor = 1.0
stretch_angle_degrees = 0.0

# Legacy identity namespace is kept byte-identical. Runtime callers use
# canonical_structure() so edits to the geometry block are reflected.
geometry_id = "SQR_LATT_SQR_HOLE_A400_D200_NEFF2p7"


def canonical_structure() -> SquareHoleStructure:
    """Build the single canonical source consumed by all workflows."""

    return SquareHoleStructure(
        a=a,
        d=d,
        n_eff=n_eff,
        height=height,
        stretch_factor=stretch_factor,
        stretch_angle_degrees=stretch_angle_degrees,
        polygon_sides=polygon_sides,
        polygon_rotation_degrees=polygon_rotation_degrees,
    )


def get_geometry_id() -> str:
    """Return the active canonical geometry ID, preserving the legacy variable."""

    return canonical_structure().geometry_id()


def geometry_parameters() -> dict[str, object]:
    """Return geometry-only parameters and canonical basis/BZ metadata."""

    return canonical_structure().geometry_parameters()


def validate_geometry() -> None:
    """Validate the active geometry-only configuration."""

    canonical_structure()


def make_band(*, resolution):
    """Build Band from the canonical current direct basis."""

    return canonical_structure().make_band(resolution=resolution)


def build_pattern():
    """Return the rigid motif consumed by preview and solver conversion."""

    return canonical_structure().build_pattern()


def band_path():
    """Return legacy G-X-M-G or an honestly named current-BZ path."""

    return canonical_structure().band_path()


def band_path_policy() -> str:
    """Return the task identity for the selected path policy."""

    return canonical_structure().band_path_policy()


def square_grid(n, extent=0.5):
    """Return identity square samples or current-BZ samples after deformation."""

    return canonical_structure().sample_grid(n, extent=extent)


def c4_quadrant(n, extent=1.0):
    """Return the identity first-quadrant grid used for verified C4 reduction."""

    return canonical_structure().c4_quadrant(n, extent=extent)


def resolve_symmetry(symmetry, raw_full_grid=False):
    """Resolve a Berry mode through the complete-structure C4 verifier."""

    return canonical_structure().resolve_symmetry(symmetry, raw_full_grid=raw_full_grid)


def verify_c4():
    """Return detailed C4 verification evidence for this complete structure."""

    return canonical_structure().verify_c4()


def unit_cell_outline():
    """Return the current-cell outline from the canonical direct basis."""

    return canonical_structure().unit_cell_outline()


def preview_pattern_data():
    """Return the exact normalized motif used by MPB geometry conversion."""

    return canonical_structure().build_pattern()
