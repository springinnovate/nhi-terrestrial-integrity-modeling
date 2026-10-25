"""Build and install the callable model without checkout-only dependencies."""

from __future__ import annotations

from email.parser import Parser
from pathlib import Path
import re
import shutil
import subprocess
import sys
import tarfile
import tempfile
import unittest
from zipfile import ZipFile


class PackagingTests(unittest.TestCase):
    """Check source archives, wheel contents and a real installed calculation."""

    def test_installed_source_archive_runs_without_the_checkout(self) -> None:
        """Install a wheel rebuilt from the source archive and calculate a map."""
        repository = Path(__file__).resolve().parents[1]
        package = "nhi_terrestrial_integrity_modeling"
        modules = {
            "habitat_condition", "habitat_training", "prepared_rasters",
            "prepared_runner", "prepared_sampling", "spatial_sampling",
        }

        def run_python(arguments: list[str], directory: Path) -> None:
            """Run a build or smoke command and show its output on failure.

            Args:
                arguments: Python arguments, following isolated-mode selection.
                directory: Working directory outside the original checkout.
            """
            result = subprocess.run(
                [sys.executable, "-I", *arguments], cwd=directory,
                capture_output=True, text=True,
            )
            self.assertEqual(result.returncode, 0, result.stdout + result.stderr)

        with tempfile.TemporaryDirectory() as temporary:
            workspace = Path(temporary)
            source = workspace / "source"
            (source / "scripts").mkdir(parents=True)
            for name in ("pyproject.toml", "README.md", "LICENSE", "MANIFEST.in"):
                shutil.copyfile(repository / name, source / name)
            for path in (repository / "scripts").glob("*.py"):
                shutil.copyfile(path, source / "scripts" / path.name)
            (source / "scripts" / "private_raster.tif").write_bytes(b"not package data")
            (source / "data").mkdir()
            (source / "data" / "private.sqlite").write_bytes(b"not package data")
            (source / "tests").mkdir()
            (source / "tests" / "test_private.py").write_text('"Not package code."\n')
            run_python(["-c", (
                "from setuptools.build_meta import build_sdist, build_wheel; "
                "build_wheel('dist'); build_sdist('dist')"
            )], source)
            archive, = (source / "dist").glob("*.tar.gz")
            with tarfile.open(archive) as built:
                names = {name.partition("/")[2] for name in built.getnames()}
                self.assertEqual(
                    {name for name in names if name.endswith(".py")},
                    {f"scripts/{name}.py" for name in modules | {"__init__"}},
                )
                self.assertFalse(any(name.endswith((".tif", ".sqlite")) for name in names))
            rebuilt = workspace / "rebuilt"
            run_python([
                "-m", "pip", "wheel", "--no-cache-dir", "--no-index",
                "--disable-pip-version-check", "--no-build-isolation", "--no-deps",
                "--wheel-dir", str(rebuilt), str(archive),
            ], workspace)
            wheel, = rebuilt.glob("*.whl")
            original_wheel, = (source / "dist").glob("*.whl")
            for artifact in (original_wheel, wheel):
                with ZipFile(artifact) as built:
                    names = built.namelist()
                    self.assertEqual(
                        {name for name in names if name.endswith(".py")},
                        {f"{package}/{name}.py" for name in modules | {"__init__"}},
                    )
                    self.assertFalse(any(name.endswith((".tif", ".sqlite")) for name in names))
                    metadata_path, = [name for name in names if name.endswith("/METADATA")]
                    metadata = Parser().parsestr(built.read(metadata_path).decode())
                    self.assertEqual(
                        {re.split(r"[<>=!~; ]", requirement)[0]
                         for requirement in metadata.get_all("Requires-Dist", [])},
                        {"numpy", "pandas", "pyproj", "rasterio", "scikit-learn", "shapely"},
                    )
            target = workspace / "installed"
            run_python([
                "-m", "pip", "install", "--no-cache-dir", "--no-index",
                "--disable-pip-version-check", "--no-deps", "--no-compile",
                "--target", str(target), str(wheel),
            ], workspace)
            smoke = workspace / "smoke.py"
            smoke.write_text('''from pathlib import Path
import sys
target = Path(sys.argv[1]).resolve()
sys.path.insert(0, str(target))
import numpy as np
import rasterio
from rasterio.transform import from_origin
from shapely.geometry import box
from nhi_terrestrial_integrity_modeling.habitat_training import IntegrityConfiguration
from nhi_terrestrial_integrity_modeling.prepared_rasters import RasterBand
from nhi_terrestrial_integrity_modeling.prepared_runner import run_habitat_condition
rows, columns = np.indices((16, 24))
x, category, reference = rows * .3 + columns * .07, (rows + columns) % 3, rows < 6
response = 1 + .3*x + .1*category + .04*np.sin(rows + columns) + .2*(columns % 3)*~reference
path = Path('inputs.tif')
with rasterio.open(path, 'w', driver='GTiff', count=4, width=24, height=16,
                   dtype='float64', crs='EPSG:6933', transform=from_origin(0, 16000, 1000, 1000)) as source:
    source.write(np.stack((x, category, response, reference)))
configuration = IntegrityConfiguration(
    fold_count=3, sampling_block_size_meters=2000, validation_block_size_meters=4000,
    minimum_predictor_coverage=.8, maximum_row_missing_fraction=.5, spline_knot_count=3,
    minimum_response_coverage=.5, ridge_alpha=1.)
result = run_habitat_condition(
    [RasterBand('x', path), RasterBand('category', path, 2)], [RasterBand('condition', path, 3)],
    RasterBand('reference', path, 4), box(100, 100, 23900, 15900), Path('output'), configuration,
    categorical_predictor_name='category', samples_per_class_per_block=100, random_seed=17,
    tile_size_pixels=8, window_size_pixels=5, covariance_shrinkage=.1)
for field in ('reference_similarity', 'expected_reference', 'observed_minus_expected',
              'standardized_deviation', 'inference_status'):
    assert getattr(result, field).is_file(), field
with rasterio.open(result.reference_similarity) as source:
    values = source.read(1, masked=True)
    assert source.shape == (16, 24) and source.count == 1 and source.nodata == -9999
    assert values.count() > 0 and values.min() >= 0 and values.max() <= 1
    assert values.mask[reference].all()
for name, module in tuple(sys.modules.items()):
    if name == 'nhi_terrestrial_integrity_modeling' or name.startswith('nhi_terrestrial_integrity_modeling.'):
        assert Path(module.__file__).resolve().is_relative_to(target), module.__file__
assert 'scripts' not in sys.modules
''', encoding="utf-8")
            run_python([str(smoke), str(target)], workspace)


if __name__ == "__main__":
    unittest.main()
