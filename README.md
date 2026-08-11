# SqrLatt

Case-based square-lattice MPB workflows built on the public
`WD-nanophotonics/MePhC` package.

Install the pinned public MePhC release in the `mp` environment:

```bash
/home/icy/miniconda3/envs/mp/bin/python -m pip install \\
  "mephc @ git+https://github.com/WD-nanophotonics/MePhC.git@v0.1.1"
```

Root files are reusable runners:

- `band_structure.py`
- `berry_curvature.py`
- `efs.py`

Specific geometries live in case folders. The first case is `square_hole/`.
Its `config.py` contains only geometry/material parameters. Each calculation script owns its own resolution, band count, grid, step, and plot settings.

Run from the case folder or by VS Code "Run Python File":

```bash
cd /home/icy/SqrLatt/square_hole
/home/icy/miniconda3/envs/mp/bin/python band_structure.py
/home/icy/miniconda3/envs/mp/bin/python berry_curvature.py
/home/icy/miniconda3/envs/mp/bin/python efs.py
```

Calculation records are saved as pickle files under `data/<geometry_id>/` and overwritten temporary records under `data/_tmp/`. Images are derived from pickle records and saved under `image/<geometry_id>/`.

Pickle records, generated images, and temporary outputs are local artifacts and
are ignored by Git. The tracked `archive_manifest.json` records the relative
record path, geometry/task/compute metadata, creation time, and SHA-256 hash so
the calculation archive remains auditable without publishing binary data.
