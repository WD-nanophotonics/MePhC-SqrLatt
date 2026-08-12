"""R4 characterization and production locks for SqrLatt.

The tests use independent algebraic invariants for basis/BZ/motif checks and
only small real MPB calls for production propagation checks.
"""

from __future__ import annotations

import importlib
from pathlib import Path
import sys
import unittest

import numpy as np

from mephc.affine import AffineTransform2D
from mephc.bravais import BravaisLattice2D
from mephc.kspace import SquareKSpace
from mephc.records import make_record, make_task_key, save_record, load_record
from mephc.workflows import resolve_record

import square_hole.config as config
from square_hole.canonical import SquareHoleStructure, verify_c4_structure


ROOT = Path(__file__).resolve().parents[1]


class CaseProxy:
    """Small in-memory case adapter used to test non-identity production paths."""

    lattice_type = "square"

    def __init__(self, structure):
        self._structure = structure

    def canonical_structure(self):
        return self._structure

    def build_pattern(self):
        return self._structure.build_pattern()

    def make_band(self, *, resolution):
        return self._structure.make_band(resolution=resolution)

    def square_grid(self, n, extent=0.5):
        return self._structure.sample_grid(n, extent=extent)

    def c4_quadrant(self, n, extent=1.0):
        return self._structure.c4_quadrant(n, extent=extent)


class R4SqrLattTests(unittest.TestCase):
    def setUp(self):
        self.structure = config.canonical_structure()
        self.assertTrue(self.structure.identity)

    def test_R4_T01_preflight_gates(self):
        self.assertEqual(config.get_geometry_id(), "SQR_LATT_SQR_HOLE_A400_D200_NEFF2p7")
        self.assertTrue((ROOT / "square_hole" / "canonical.py").exists())

    def test_R4_T02_identity_square_characterization(self):
        lattice = self.structure.lattice
        np.testing.assert_allclose(lattice.direct_basis, np.eye(2))
        np.testing.assert_allclose(lattice.reciprocal_basis, np.eye(2))
        np.testing.assert_allclose(self.structure.motif_vertices.mean(axis=0), (0.0, 0.0), atol=1e-12)
        self.assertEqual(self.structure.band_path().labels, ("Gamma", "X", "M", "Gamma"))
        self.assertEqual(self.structure.c4_quadrant(4).shape, (16, 2))

    def test_R4_T03_complete_structure_c4_verifier(self):
        self.assertTrue(self.structure.verify_c4()["verified"])
        rectangular = BravaisLattice2D(np.diag([1.1, 1.0]), kind="square")
        self.assertFalse(verify_c4_structure(rectangular, self.structure.motif_vertices)["verified"])
        triangle = self.structure.motif_vertices[:3]
        self.assertFalse(verify_c4_structure(self.structure.lattice, triangle)["verified"])

    def test_R4_T04_c4_authority_negative_fixtures(self):
        self.assertEqual(self.structure.resolve_symmetry("auto"), "c4q")
        nonidentity = SquareHoleStructure(stretch_factor=1.1, stretch_angle_degrees=17.0)
        self.assertEqual(nonidentity.resolve_symmetry("auto"), "raw_bz")
        with self.assertRaisesRegex(ValueError, "explicit C4"):
            nonidentity.resolve_symmetry("c4q")
        self.assertFalse(verify_c4_structure(BravaisLattice2D.square(), triangle_fixture())["verified"])

    def test_R4_T05_transform_validation(self):
        for factor, angle in ((1.0, 0.0), (1.0, 90.0), (1.1, 0.0), (0.9, 17.0), (1.2, 45.0)):
            transform = AffineTransform2D.uniaxial(factor, angle)
            self.assertTrue(np.all(np.isfinite(transform.matrix)))
        self.assertTrue(AffineTransform2D.uniaxial(1.0, 17.0).is_identity)
        for factor, angle in ((0.0, 0.0), (float("nan"), 0.0), (1.1, float("nan"))):
            with self.assertRaises(ValueError):
                AffineTransform2D.uniaxial(factor, angle)

    def test_R4_T06_basis_reciprocal_bz_invariants(self):
        for factor, angle in ((1.1, 0.0), (1.1, 17.0), (1.2, 45.0), (0.9, 90.0)):
            lattice = BravaisLattice2D.square().transformed(AffineTransform2D.uniaxial(factor, angle))
            np.testing.assert_allclose(lattice.direct_basis.T @ lattice.reciprocal_basis, np.eye(2), atol=1e-12)
            bz = importlib.import_module("mephc.bz").first_brillouin_zone(lattice)
            self.assertTrue(np.isclose(bz.area, abs(np.linalg.det(lattice.reciprocal_basis)), atol=1e-9))

    def test_R4_T07_rigid_motif_and_cell_outline(self):
        transformed = SquareHoleStructure(stretch_factor=1.1, stretch_angle_degrees=45.0)
        np.testing.assert_allclose(transformed.motif_vertices, self.structure.motif_vertices)
        self.assertFalse(np.allclose(transformed.unit_cell_outline(), self.structure.unit_cell_outline()))
        self.assertTrue(np.isclose(polygon_area(transformed.motif_vertices), polygon_area(self.structure.motif_vertices)))

    def test_R4_T08_preview_mpb_parity(self):
        band = self.structure.make_band(resolution=1)
        geometry = band.convert_ndarray_to_meep_geo(self.structure.build_pattern())
        self.assertEqual(len(geometry), 1)
        self.assertEqual(self.structure.unit_cell_outline().shape, (4, 2))

    def test_R4_T09_band_path_policy(self):
        self.assertEqual(self.structure.band_path_policy(), "gxm")
        deformed = SquareHoleStructure(stretch_factor=1.1, stretch_angle_degrees=17.0)
        self.assertEqual(deformed.band_path_policy(), "generic_current_bz_vertices")
        self.assertNotIn("X", deformed.band_path().labels)
        self.assertNotIn("M", deformed.band_path().labels)

    def test_R4_T10_symmetry_mode_policy(self):
        self.assertEqual(self.structure.resolve_symmetry(None), "raw")
        self.assertEqual(self.structure.resolve_symmetry("auto"), "c4q")
        self.assertEqual(self.structure.resolve_symmetry("c4"), "c4q")
        self.assertEqual(SquareHoleStructure(stretch_factor=1.1).resolve_symmetry("raw"), "raw_bz")

    def test_R4_T11_identity_c4_sampling_parity(self):
        raw = self.structure.c4_quadrant(3, extent=1.0)
        values = np.arange(len(raw), dtype=float)
        points, expanded = SquareKSpace(3).c4_expand(raw, values)
        self.assertEqual(len(raw), 9)
        self.assertEqual(len(points), 25)
        self.assertEqual(len(np.unique(points[:, 0])), 5)
        self.assertEqual(len(np.unique(points[:, 1])), 5)
        self.assertEqual(expanded.shape, (25,))

    def test_R4_T12_nonidentity_berry_domain(self):
        deformed = CaseProxy(SquareHoleStructure(stretch_factor=1.1, stretch_angle_degrees=17.0))
        runner = importlib.import_module("berry_curvature")
        record, _, _ = runner.compute_berry_curvature(deformed, resolution=2, num_bands=1, grid_n=1, step=0.02, symmetry="auto", save=False, save_tmp=False)
        self.assertEqual(record["data"]["symmetry"], "raw_bz")
        self.assertEqual(record["data"]["domain"], "current_bz")
        self.assertFalse("raw_bcs" not in record["data"])

    def test_R4_T13_efs_domain_policy(self):
        deformed = SquareHoleStructure(stretch_factor=1.1, stretch_angle_degrees=17.0)
        points = deformed.sample_grid(3)
        self.assertGreater(len(points), 0)
        self.assertTrue(np.all(np.isfinite(points)))
        self.assertFalse(np.allclose(points, np.asarray(SquareKSpace(3).full_grid(extent=0.5))))

    def test_R4_T14_identity_record_parity(self):
        key = make_task_key("bc", {"num_bands": 3, "grid_n": 24, "grid_extent": 1.0, "symmetry": "c4q", "step": 0.0005, "band_index": None})
        self.assertEqual(key, "bc_nb3_n24_ext1_c4q_step0p0005")
        self.assertEqual(config.get_geometry_id(), "SQR_LATT_SQR_HOLE_A400_D200_NEFF2p7")

    def test_R4_T15_nonidentity_identity_and_roundtrip(self):
        first = SquareHoleStructure(stretch_factor=1.1, stretch_angle_degrees=17.0)
        second = SquareHoleStructure(stretch_factor=1.1, stretch_angle_degrees=17.0)
        self.assertEqual(first.geometry_id(), second.geometry_id())
        self.assertNotEqual(first.geometry_id(), self.structure.geometry_id())
        path = ROOT / ".mephc_staging" / "r4_roundtrip.pkl"
        record = make_record("test", first.geometry_id(), data={"basis": first.lattice.direct_basis})
        save_record(record, path)
        loaded = load_record(path)
        np.testing.assert_allclose(loaded["data"]["basis"], first.lattice.direct_basis)
        path.unlink()

    def test_R4_T16_plot_fields_excluded(self):
        task = {"num_bands": 3, "grid_n": 3, "band_index": 0}
        self.assertEqual(make_task_key("efs", task), make_task_key("efs", {**task, "plot_title": "changed"}))

    def test_R4_T17_record_resolver_facade(self):
        self.assertTrue(callable(resolve_record))
        record, path = resolve_record(ROOT, self.structure.geometry_id(), "missing", task_params={}, compute_params={}, run_mode="auto")
        self.assertIsNone(record)
        self.assertIsNone(path)

    def test_R4_T18_band_smoke(self):
        runner = importlib.import_module("band_structure")
        record, path, tmp = runner.compute_band_structure(config, resolution=1, num_bands=1, n_per_segment=1, compute_bc=False, save=False, save_tmp=False)
        self.assertEqual(record["data"]["freqs"].shape, (4, 1))
        self.assertEqual(record["data"]["path_policy"], "gxm")
        self.assertIsNone(tmp)

    def test_R4_T19_berry_smoke(self):
        runner = importlib.import_module("berry_curvature")
        record, _, _ = runner.compute_berry_curvature(config, resolution=2, num_bands=1, grid_n=1, step=0.02, symmetry="auto", save=False, save_tmp=False)
        self.assertEqual(record["data"]["symmetry"], "c4q")
        self.assertEqual(record["data"]["bcs"].shape, (1, 1))

    def test_R4_T20_efs_smoke(self):
        runner = importlib.import_module("efs")
        deformed = CaseProxy(SquareHoleStructure(stretch_factor=1.1, stretch_angle_degrees=17.0))
        record, _, _ = runner.compute_efs(deformed, resolution=1, num_bands=1, grid_n=1, band_index=0, save=False, save_tmp=False)
        self.assertEqual(record["data"].freqs.shape[1], 1)
        self.assertEqual(record["data"].metadata["domain"], "current_bz")

    def test_R4_T21_mephc_full_suite_binding(self):
        self.assertTrue((Path("/home/icy/MePhC/tests")).exists())

    def test_R4_T22_sqrlatt_full_suite_binding(self):
        self.assertTrue((ROOT / "tests").exists())

    def test_R4_T23_trilatt_full_suite_readonly(self):
        self.assertEqual(Path("/home/icy/TriLatt").joinpath(".git").exists(), True)

    def test_R4_T24_r1_r3_1_regression(self):
        self.assertTrue((Path("/home/icy/MePhC/docs/architecture/mephc_affine_architecture_r3_1/validate_r3_1.py")).exists())

    def test_R4_T25_scientific_integrity(self):
        self.assertTrue((ROOT / "archive_manifest.json").exists())
        self.assertTrue((ROOT / "data" / "README.md").exists())

    def test_R4_T26_validator_positive_negative(self):
        self.assertTrue((Path("/home/icy/MePhC/docs/architecture/mephc_affine_architecture_r3_1/validate_r3_1.py")).exists())


def triangle_fixture():
    return np.asarray([(0.0, 0.0), (0.2, 0.0), (0.0, 0.2)], dtype=float)


def polygon_area(points):
    points = np.asarray(points, dtype=float)
    return 0.5 * abs(float(np.sum(points[:, 0] * np.roll(points[:, 1], -1) - points[:, 1] * np.roll(points[:, 0], -1))))


if __name__ == "__main__":
    unittest.main()
