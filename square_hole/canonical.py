"""Canonical square-hole lattice/structure authority for SqrLatt.

The supported physical model is a globally periodic square Bravais lattice
under an invertible uniaxial affine map. The square hole remains rigid in
laboratory Cartesian coordinates: its side length, orientation, center, and
material do not deform.
"""

from __future__ import annotations

from dataclasses import dataclass
from numbers import Real, Integral
import math

import numpy as np

from mephc.affine import AffineTransform2D
from mephc.bravais import BravaisLattice2D
from mephc.bz import first_brillouin_zone
from mephc.geometry import regular_polygon_vertices
from mephc.kspace import SquareKSpace, generic_bz_path, square_gxm_path
from mephc.records import make_geometry_id


_ROT90 = np.array([[0.0, -1.0], [1.0, 0.0]])


def _compact_number(value) -> str:
    if isinstance(value, Integral) and not isinstance(value, bool):
        text = str(int(value))
    else:
        text = f"{float(value):g}"
    return text.replace("-", "m").replace(".", "p")


def _validate_positive(name: str, value) -> None:
    if isinstance(value, bool) or not isinstance(value, Real) or not np.isfinite(float(value)) or float(value) <= 0:
        raise ValueError(f"{name} must be a positive finite number.")


def _periodic_point_match(point: np.ndarray, candidates: np.ndarray, direct_basis: np.ndarray, tolerance: float) -> bool:
    fractional = (candidates - point) @ np.linalg.inv(direct_basis).T
    return bool(np.any(np.linalg.norm(fractional - np.rint(fractional), axis=1) <= tolerance))


def verify_c4_structure(
    lattice: BravaisLattice2D,
    motif_vertices,
    *,
    material: str = "air",
    tolerance: float = 1e-9,
) -> dict[str, object]:
    """Verify a requested proper C4 operation on a complete normalized structure.

    The check tests the current direct Bravais basis and every motif vertex
    with periodic equivalence. It is an asserted candidate verifier, not a
    point-group search. reference_family, polygon side count, and a
    user-supplied token are never consulted as authority.
    """

    if not isinstance(lattice, BravaisLattice2D):
        raise TypeError("lattice must be a BravaisLattice2D")
    if not np.isfinite(tolerance) or tolerance <= 0:
        raise ValueError("tolerance must be positive and finite")
    vertices = np.asarray(motif_vertices, dtype=float)
    if vertices.ndim != 2 or vertices.shape[1] != 2 or len(vertices) < 1 or not np.all(np.isfinite(vertices)):
        raise ValueError("motif_vertices must have shape (N, 2) and finite values")

    direct = np.asarray(lattice.direct_basis, dtype=float)
    rotated_basis = _ROT90 @ direct
    integer_action = np.linalg.solve(direct, rotated_basis)
    basis_ok = bool(np.allclose(integer_action, np.rint(integer_action), atol=tolerance, rtol=0.0))

    rotated_vertices = vertices @ _ROT90.T
    motif_ok = all(_periodic_point_match(point, vertices, direct, tolerance) for point in rotated_vertices)
    area_ok = bool(np.isclose(abs(np.linalg.det(direct)), lattice.cell_area, atol=tolerance, rtol=0.0))
    material_ok = isinstance(material, str) and bool(material)

    reasons = []
    if not basis_ok:
        reasons.append("current direct basis is not invariant under the requested C4 rotation")
    if not motif_ok:
        reasons.append("motif vertices are not periodic-equivalent after C4 rotation")
    if not area_ok:
        reasons.append("direct-basis area invariant failed")
    if not material_ok:
        reasons.append("motif material identity is invalid")

    return {
        "verified": not reasons,
        "rotation_order": 4,
        "rotation_matrix": _ROT90.tolist(),
        "tolerance": float(tolerance),
        "basis_integer_action": integer_action.tolist(),
        "basis_ok": basis_ok,
        "motif_ok": motif_ok,
        "area_ok": area_ok,
        "material_ok": material_ok,
        "material": material,
        "reasons": reasons,
    }


@dataclass(frozen=True)
class SquareHoleStructure:
    """Canonical square-hole geometry and affine lattice for one case.

    Parameters are geometry/material inputs only. Solver resolution, band
    counts, k-grid density, Berry step, and plotting controls live in the
    individual task scripts.
    """

    a: float = 400.0
    d: float = 200.0
    n_eff: float = 2.7
    height: float = 1.0
    stretch_factor: float = 1.0
    stretch_angle_degrees: float = 0.0
    polygon_sides: int = 4
    polygon_rotation_degrees: float = 45.0
    tolerance: float = 1e-9

    def __post_init__(self):
        _validate_positive("a", self.a)
        _validate_positive("d", self.d)
        _validate_positive("n_eff", self.n_eff)
        _validate_positive("height", self.height)
        if isinstance(self.polygon_sides, bool) or not isinstance(self.polygon_sides, Integral) or self.polygon_sides < 3:
            raise ValueError("polygon_sides must be an integer >= 3")
        if not np.isfinite(float(self.polygon_rotation_degrees)):
            raise ValueError("polygon_rotation_degrees must be finite")
        if not np.isfinite(float(self.tolerance)) or self.tolerance <= 0:
            raise ValueError("tolerance must be positive and finite")
        AffineTransform2D.uniaxial(float(self.stretch_factor), float(self.stretch_angle_degrees))

    @property
    def transform(self) -> AffineTransform2D:
        """Return the canonical direct-space uniaxial transform."""

        return AffineTransform2D.uniaxial(self.stretch_factor, self.stretch_angle_degrees)

    @property
    def lattice(self) -> BravaisLattice2D:
        """Return the current square Bravais lattice with transformed basis."""

        return BravaisLattice2D.square().transformed(self.transform)

    @property
    def motif_vertices(self) -> np.ndarray:
        """Return the rigid, centered square-hole vertices in normalized Cartesian coordinates."""

        radius = float(self.d) / (math.sqrt(2.0) * float(self.a))
        return regular_polygon_vertices(
            (0.0, 0.0),
            radius,
            int(self.polygon_sides),
            math.radians(float(self.polygon_rotation_degrees)),
        )

    @property
    def first_bz(self):
        """Return the validated current reciprocal Wigner-Seitz cell."""

        return first_brillouin_zone(self.lattice)

    @property
    def identity(self) -> bool:
        """Whether this case is the legacy undeformed square identity."""

        return self.transform.is_identity

    def geometry_id(self) -> str:
        """Return the stable geometry directory identifier.

        Identity inputs intentionally use the exact legacy namespace. A
        non-identity transform adds only physical deformation parameters.
        """

        base = make_geometry_id("square", "square_hole", a=self.a, d=self.d, n_eff=self.n_eff)
        if self.identity:
            return base
        return f"{base}_S{_compact_number(self.stretch_factor)}_ANG{_compact_number(self.stretch_angle_degrees)}"

    def geometry_parameters(self) -> dict[str, object]:
        """Return geometry/material and canonical lattice metadata for records."""

        return {
            "a": float(self.a),
            "d": float(self.d),
            "n_eff": float(self.n_eff),
            "height": float(self.height),
            "stretch_factor": float(self.stretch_factor),
            "stretch_angle_degrees": 0.0 if self.identity else float(self.stretch_angle_degrees),
            "polygon_sides": int(self.polygon_sides),
            "polygon_rotation_degrees": float(self.polygon_rotation_degrees),
            "lattice": self.lattice.metadata(),
            "bz": self.first_bz.metadata(),
        }

    def build_pattern(self) -> np.ndarray:
        """Return the exact motif consumed by both preview and Meep conversion."""

        return np.asarray(self.motif_vertices, dtype=float)

    def unit_cell_outline(self) -> np.ndarray:
        """Return the transformed primitive-cell outline from the current basis."""

        a1, a2 = self.lattice.direct_basis.T
        return np.asarray(
            [
                -0.5 * a1 - 0.5 * a2,
                0.5 * a1 - 0.5 * a2,
                0.5 * a1 + 0.5 * a2,
                -0.5 * a1 + 0.5 * a2,
            ],
            dtype=float,
        )

    def verify_c4(self) -> dict[str, object]:
        """Verify C4 using the complete current lattice plus rigid motif."""

        return verify_c4_structure(
            self.lattice,
            self.motif_vertices,
            material="air",
            tolerance=self.tolerance,
        )

    def resolve_symmetry(self, requested: str | None, *, raw_full_grid: bool = False) -> str:
        """Resolve or reject Berry symmetry modes against the verified structure.

        Identity auto resolves to legacy-compatible c4q. Non-identity auto
        resolves to raw_bz. Explicit C4 modes are rejected unless the
        complete verifier passes.
        """

        if raw_full_grid:
            return "raw" if self.identity else "raw_bz"
        mode = "raw" if requested is None else str(requested).lower()
        if mode not in {"auto", "raw", "raw_bz", "c4", "c4q"}:
            raise ValueError("symmetry must be one of None, 'auto', 'raw', 'c4', or 'c4q'")
        verification = self.verify_c4()
        if mode == "auto":
            return "c4q" if verification["verified"] else "raw_bz"
        if mode in {"c4", "c4q"}:
            if not verification["verified"]:
                detail = "; ".join(verification["reasons"])
                raise ValueError(f"explicit C4 reduction is invalid for this structure: {detail}")
            return "c4q"
        return "raw" if self.identity else "raw_bz"

    def sample_grid(self, grid_n: int, *, extent: float = 0.5) -> np.ndarray:
        """Return identity full-square or non-identity current-BZ samples."""

        if not isinstance(grid_n, Integral) or isinstance(grid_n, bool) or grid_n < 1:
            raise ValueError("grid_n must be an integer >= 1")
        if self.identity:
            return np.asarray(SquareKSpace(int(grid_n)).full_grid(extent=extent), dtype=float)
        return np.asarray(
            SquareKSpace(int(grid_n), lattice_model=self.lattice).current_bz(),
            dtype=float,
        )

    def c4_quadrant(self, grid_n: int, *, extent: float = 1.0) -> np.ndarray:
        """Return the legacy-compatible first-quadrant C4 reduced grid."""

        if not self.identity:
            raise ValueError("C4 quadrant sampling is only valid for the identity structure")
        return np.asarray(SquareKSpace(int(grid_n)).c4_quadrant(extent=extent), dtype=float)

    def band_path(self):
        """Return G-X-M-G for identity, otherwise a generic current-BZ path."""

        return square_gxm_path() if self.identity else generic_bz_path(self.lattice)

    def band_path_policy(self) -> str:
        """Return the honest path identity stored in task metadata."""

        return "gxm" if self.identity else "generic_current_bz_vertices"

    def symmetry_capabilities(self) -> dict[str, object]:
        """Return verified, not inferred, symmetry capabilities."""

        verification = self.verify_c4()
        return {
            "requested_operation": "C4 proper rotation",
            "c4_verified": bool(verification["verified"]),
            "c4q_eligible": bool(verification["verified"]),
            "verification": verification,
            "identity_legacy": self.identity,
            "reference_family": "square",
            "authority": "complete_canonical_structure_verifier",
        }

    def metadata(self) -> dict[str, object]:
        """Return the complete canonical structure payload."""

        return {
            "schema": "sqrlatt.square_hole.canonical.v1",
            "geometry_id": self.geometry_id(),
            "geometry_parameters": self.geometry_parameters(),
            "symmetry_capabilities": self.symmetry_capabilities(),
            "band_path_policy": self.band_path_policy(),
            "motif_policy": "rigid_local_square_cartesian",
        }

    def make_band(self, *, resolution: int):
        """Construct MePhC Band from this structure's current lattice authority."""

        from mephc.band import Band

        return Band(
            a=self.a,
            r1=self.d / 2.0,
            r2=None,
            n_eff=self.n_eff,
            h=self.height,
            resolution=resolution,
            lattice_model=self.lattice,
        )
