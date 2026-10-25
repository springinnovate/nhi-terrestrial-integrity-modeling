"""Verify spatial sample selection and weights without raster-cache dependencies."""

from __future__ import annotations

import subprocess
import sys
import unittest
from pathlib import Path

import numpy as np

from scripts import spatial_sampling as sampling


def _sample_two_tiles() -> tuple[
    dict[tuple[int, int, int], sampling.SamplingStratumState],
    list[tuple[int, int]],
]:
    """Sample two neighboring tiles whose pixels share one spatial block.

    Returns:
        Reference/background strata and retained-count increases after each
        tile. The arrays match the existing sampler's two-tile fixture.
    """

    strata: dict[tuple[int, int, int], sampling.SamplingStratumState] = {}
    increases: list[tuple[int, int]] = []
    local_rows, local_columns = np.indices((4, 4))
    for tile_sequence in range(2):
        global_columns = tile_sequence * 4 + local_columns
        reference_classes = ((global_columns + local_rows) % 3 == 0).astype(
            np.uint8
        )
        increases.append(
            sampling.retain_tile_candidates(
                strata,
                tile_id=f"x{tile_sequence:+07d}_y+000000",
                tile_sequence=tile_sequence,
                tile_pixel_count=16,
                eligible_pixel_indices=np.arange(16, dtype=np.int64),
                sampling_block_columns=np.zeros(16, dtype=np.int64),
                sampling_block_rows=np.zeros(16, dtype=np.int64),
                reference_site_classes=reference_classes.ravel(),
                random_seed=17,
                samples_per_class_per_block=3,
            )
        )
    return strata, increases


class SpatialSamplingTest(unittest.TestCase):
    """Preserve global sample caps, deterministic addresses and area weights."""

    def test_merges_tile_candidates_and_weights_the_complete_strata(self) -> None:
        """Keep one cap per block/class and preserve the caller's row order."""

        strata, increases = _sample_two_tiles()
        background = strata[(0, 0, 0)]
        reference = strata[(0, 0, 1)]
        self.assertEqual([(3, 3), (0, 0)], increases)
        self.assertEqual(21, background.available_pixel_count)
        self.assertEqual(11, reference.available_pixel_count)
        # Captured from the original sampler at a88198a.
        self.assertEqual(
            [(1, 4), (1, 10), (0, 8)],
            [(item.tile_sequence, item.local_pixel_index)
             for item in background.candidates],
        )
        self.assertEqual(
            [(1, 14), (0, 15), (0, 9)],
            [(item.tile_sequence, item.local_pixel_index)
             for item in reference.candidates],
        )
        candidates = [
            reference.candidates[0], background.candidates[2],
            reference.candidates[2], background.candidates[0],
            reference.candidates[1], background.candidates[1],
        ]
        columns = sampling.build_sample_weight_columns(
            candidates, strata, pixel_area_m2=1_000_000.0
        )

        np.testing.assert_array_equal(columns["reference_site"], [1, 0] * 3)
        np.testing.assert_array_equal(columns["sampling_block_id"], [1] * 6)
        np.testing.assert_array_equal(
            columns["available_pixels_in_block_class"], [11, 21] * 3
        )
        np.testing.assert_array_equal(
            columns["sampled_pixels_in_block_class"], [3] * 6
        )
        np.testing.assert_allclose(columns["sampling_weight"], [11 / 3, 7] * 3)
        np.testing.assert_allclose(
            columns["sampling_probability"], [3 / 11, 1 / 7] * 3
        )
        np.testing.assert_allclose(
            columns["area_weight_m2"], [11_000_000 / 3, 7_000_000] * 3
        )
        self.assertAlmostEqual(32_000_000.0, columns["area_weight_m2"].sum())

    def test_sparse_eligibility_keeps_original_full_tile_priorities(self) -> None:
        """Keep full-tile priorities when only a few source pixels are eligible."""

        strata: dict[tuple[int, int, int], sampling.SamplingStratumState] = {}
        increase = sampling.retain_tile_candidates(
            strata,
            tile_id="x+000000_y+000000",
            tile_sequence=0,
            tile_pixel_count=16,
            eligible_pixel_indices=np.array([1, 6, 9, 14], dtype=np.int64),
            sampling_block_columns=np.zeros(4, dtype=np.int64),
            sampling_block_rows=np.zeros(4, dtype=np.int64),
            reference_site_classes=np.zeros(4, dtype=np.uint8),
            random_seed=17,
            samples_per_class_per_block=2,
        )

        self.assertEqual((2, 0), increase)
        selected = strata[(0, 0, 0)]
        self.assertEqual(4, selected.available_pixel_count)
        self.assertEqual(
            [9, 6], [item.local_pixel_index for item in selected.candidates]
        )
        # Captured by running a88198a with only these four pixels eligible.
        np.testing.assert_array_equal(
            [item.random_priority for item in selected.candidates],
            [0.018284974543618326, 0.2090094834971844],
        )

    def test_calculates_without_cache_raster_or_plotting_imports(self) -> None:
        """Select and weight pixels without importing raster or plotting libraries."""

        script = f'''
import runpy
import sys
namespace = runpy.run_path({str(Path(__file__).resolve())!r})
strata, _ = namespace["_sample_two_tiles"]()
candidates = [item for state in strata.values() for item in state.candidates]
namespace["sampling"].build_sample_weight_columns(candidates, strata, 1_000_000.0)
for name in sys.modules:
    assert not (
        name.startswith("scripts.") and name != "scripts.spatial_sampling"
    ), name
    assert name.split(".")[0] not in {{
        "matplotlib", "rasterio", "cartopy", "ee", "tomllib", "pyproj"
    }}, name
'''
        result = subprocess.run(
            [sys.executable, "-B", "-c", script],
            cwd=Path(__file__).resolve().parents[1],
            capture_output=True,
            text=True,
            check=False,
        )
        self.assertEqual(0, result.returncode, result.stdout + result.stderr)


if __name__ == "__main__":
    unittest.main()
