"""SqrLatt integration through the shared MePhC R5 field authority."""

from __future__ import annotations

import numpy as np

from mephc.affine import AffineTransform2D
from mephc.deformation import PeriodicSupercellField, canonicalize_field, periodic_supercell_field
from mephc.deformation_geometry import replicated_rigid_pattern
from mephc.r5 import primitive_guard, record_identity, supercell_metadata


def global_field(structure):
    return canonicalize_field(structure.transform)


def finite_patch_preview(structure, field, replication=(3, 3), pattern=None):
    return replicated_rigid_pattern(
        structure.motif_vertices if pattern is None else pattern,
        structure.lattice,
        replication=replication,
        field=field,
    )


def periodic_supercell_preview(structure, field, replication=(2, 2)):
    field = periodic_supercell_field(field, structure.lattice, replication)
    return {
        "pattern": finite_patch_preview(structure, field, replication=replication),
        "supercell": supercell_metadata(field),
        "record_identity": record_identity(field, reference_lattice=structure.lattice, replication=replication),
    }

def build_supercell_solver(structure, field, *, q_points, resolution, num_bands=6):
    """Run the production periodic-supercell adapter for canonical SqrLatt."""
    if not isinstance(field, PeriodicSupercellField):
        raise TypeError("R6 adapter requires a declared PeriodicSupercellField")
    field.require_verified()
    matrix = field.supercell.matrix
    if not np.array_equal(matrix, np.diag(np.diag(matrix))):
        raise ValueError("R6 adapter currently requires diagonal replication")
    replication = tuple(int(value) for value in np.diag(matrix))
    pattern = finite_patch_preview(structure, field, replication=replication)
    band = structure.make_band(resolution=int(resolution))
    solver = band.run_supercell(
        pattern,
        field,
        q_points=q_points,
        num_bands=int(num_bands),
        resolution=int(resolution),
        polarization="TE",
    )
    return solver, {"band": band, "field": field, "pattern": pattern, "replication": list(replication)}


__all__ = ["build_supercell_solver", "finite_patch_preview", "global_field", "periodic_supercell_preview", "primitive_guard"]
