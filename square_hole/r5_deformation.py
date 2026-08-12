"""SqrLatt integration through the shared MePhC R5 field authority."""

from __future__ import annotations

from mephc.affine import AffineTransform2D
from mephc.deformation import canonicalize_field, periodic_supercell_field
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


__all__ = ["finite_patch_preview", "global_field", "periodic_supercell_preview", "primitive_guard"]
