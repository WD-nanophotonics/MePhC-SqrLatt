from pathlib import Path

import numpy as np

from mephc.band import Band
from mephc.geometry import regular_polygon_vertices
from mephc.kspace import SquareKSpace, square_gxm_path
from mephc.records import make_geometry_id

case_root = Path(__file__).resolve().parent
project_root = case_root.parent

lattice_type = "square"
hole_shape = "square_hole"
# Physical lengths are in nm; normalized geometry divides them by a.
a = 400
# Square-hole side length. Its normalized circumradius is d/(sqrt(2)*a).
d = 200
# Effective slab index used as the background dielectric (epsilon = n_eff**2).
n_eff = 2.7
# Meep geometry height. For this 2D workflow it only needs to span the cell.
height = 1
polygon_sides = 4
# A regular four-gon uses 45 degrees to make its edges parallel to x and y.
polygon_rotation_degrees = 45.0

geometry_id = make_geometry_id(lattice_type, hole_shape, a=a, d=d, n_eff=n_eff)


def make_band(*, resolution):
    """Build the MePhC solver object for this geometry case."""
    return Band(a=a, r1=d / 2, r2=None, n_eff=n_eff, h=height, resolution=resolution, lattice_type=lattice_type)


def build_pattern():
    """Return the normalized square-hole polygon centered in the unit cell."""
    radius = d / (np.sqrt(2.0) * a)
    return regular_polygon_vertices((0.0, 0.0), radius, polygon_sides, np.radians(polygon_rotation_degrees))


def band_path():
    """Return the square-lattice Gamma-X-M-Gamma high-symmetry path."""
    return square_gxm_path()


def square_grid(n, extent=0.5):
    """Return an ``n x n`` grid over ``[-extent, extent]^2``."""
    return SquareKSpace(n).full_grid(extent=extent)



def unit_cell_outline():
    """Return the normalized square-cell boundary used by pattern preview."""
    return [(-0.5, -0.5), (0.5, -0.5), (0.5, 0.5), (-0.5, 0.5)]


def preview_pattern_data():
    """Return geometry in the same normalized coordinates used for simulation."""
    return build_pattern()
