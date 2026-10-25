"""Select spatially balanced pixels and calculate their represented-area weights."""

from __future__ import annotations

import hashlib
from collections.abc import Sequence
from dataclasses import dataclass, field

import numpy as np


@dataclass(frozen=True)
class SamplingCandidate:
    """Identify one retained source pixel before its band values are reread.

    Attributes:
        tile_sequence: Position of the source tile in deterministic AOI order.
        local_pixel_index: Row-major pixel position inside the source tile.
        sampling_block_column: Global equal-area sampling-block column.
        sampling_block_row: Global equal-area sampling-block row.
        reference_site_class: Zero for background or one for reference.
        random_priority: Seeded priority used for sampling without replacement.
    """

    tile_sequence: int
    local_pixel_index: int
    sampling_block_column: int
    sampling_block_row: int
    reference_site_class: int
    random_priority: float


@dataclass
class SamplingStratumState:
    """Accumulate one global sampling-block and reference-class stratum.

    A stratum is the complete group of eligible pixels sharing one global
    sampling block and one binary reference class. Its sampling cap and weight
    are calculated independently from every other such group.

    Attributes:
        available_pixel_count: Eligible source pixels encountered in the
            stratum.
        candidates: Lowest-priority pixels retained up to the configured cap.
    """

    available_pixel_count: int = 0
    candidates: list[SamplingCandidate] = field(default_factory=list)


def retain_tile_candidates(
    strata: dict[tuple[int, int, int], SamplingStratumState],
    *,
    tile_id: str,
    tile_sequence: int,
    tile_pixel_count: int,
    eligible_pixel_indices: np.ndarray,
    sampling_block_columns: np.ndarray,
    sampling_block_rows: np.ndarray,
    reference_site_classes: np.ndarray,
    random_seed: int,
    samples_per_class_per_block: int,
) -> tuple[int, int]:
    """Merge one tile's lowest-priority pixels into the global block/class sample.

    A block crossing tile boundaries receives one sampling cap. Random priorities
    are generated for the complete tile before selecting eligible positions.
    A position's priority depends on the tile identity, dimensions and seed,
    independently of which other positions are eligible.

    Args:
        strata: Population counts and retained candidates from previous tiles;
            updated in place, including new block/class entries.
        tile_id: Stable spatial tile address used with the random seed.
        tile_sequence: Tile's position in the caller's deterministic traversal.
        tile_pixel_count: Number of cells in the complete row-major tile.
        eligible_pixel_indices: Nonempty array of eligible flat tile positions,
            after the source reader applies AOI and data-availability rules.
        sampling_block_columns: Global sampling-block column for each eligible
            position, in the same order.
        sampling_block_rows: Corresponding global sampling-block rows.
        reference_site_classes: Corresponding binary classes, 0 for background
            and 1 for reference sites.
        random_seed: Preset seed for deterministic sampling.
        samples_per_class_per_block: Positive scientific sample cap per global
            block/class combination.

    Returns:
        Increases in retained background and reference counts, respectively.
        Replacing a retained pixel without growing its stratum returns no increase.
    """

    retained_background_candidate_count = 0
    retained_reference_candidate_count = 0
    # Python's built-in hash is randomly salted for each process, so it
    # would select different pixels on another run. SHA-256 combines the
    # configured seed and tile address into a repeatable per-tile NumPy
    # seed across processes and machines.
    tile_seed_material = f"{random_seed}:{tile_id}".encode("utf-8")
    tile_seed = int.from_bytes(
        hashlib.sha256(tile_seed_material).digest()[:8],
        byteorder="little",
        signed=False,
    )
    tile_random_generator = np.random.default_rng(tile_seed)
    all_tile_priorities = tile_random_generator.random(tile_pixel_count)
    eligible_priorities = all_tile_priorities[eligible_pixel_indices]

    stratum_records = np.empty(
        len(eligible_pixel_indices),
        dtype=[
            ("block_column", np.int64),
            ("block_row", np.int64),
            ("reference_class", np.uint8),
        ],
    )
    stratum_records["block_column"] = sampling_block_columns
    stratum_records["block_row"] = sampling_block_rows
    stratum_records["reference_class"] = reference_site_classes
    unique_strata, stratum_offsets = np.unique(
        stratum_records,
        return_inverse=True,
    )
    positions_sorted_by_stratum = np.argsort(
        stratum_offsets,
        kind="stable",
    )
    pixels_per_tile_stratum = np.bincount(
        stratum_offsets,
        minlength=len(unique_strata),
    )
    stratum_start_offset = 0
    for stratum_offset, unique_stratum in enumerate(unique_strata):
        available_in_tile = int(
            pixels_per_tile_stratum[stratum_offset]
        )
        positions_in_stratum = positions_sorted_by_stratum[
            stratum_start_offset : (
                stratum_start_offset + available_in_tile
            )
        ]
        stratum_start_offset += available_in_tile
        stratum_key = (
            int(unique_stratum["block_column"]),
            int(unique_stratum["block_row"]),
            int(unique_stratum["reference_class"]),
        )
        stratum_state = strata.setdefault(stratum_key, SamplingStratumState())
        stratum_state.available_pixel_count += available_in_tile
        retained_from_tile = min(
            samples_per_class_per_block,
            available_in_tile,
        )
        if retained_from_tile < available_in_tile:
            local_priority_offsets = np.argpartition(
                eligible_priorities[positions_in_stratum],
                retained_from_tile - 1,
            )[:retained_from_tile]
            retained_positions = positions_in_stratum[
                local_priority_offsets
            ]
        else:
            retained_positions = positions_in_stratum
        new_candidates = [
            SamplingCandidate(
                tile_sequence=tile_sequence,
                local_pixel_index=int(eligible_pixel_indices[position]),
                sampling_block_column=stratum_key[0],
                sampling_block_row=stratum_key[1],
                reference_site_class=stratum_key[2],
                random_priority=float(eligible_priorities[position]),
            )
            for position in retained_positions
        ]
        previous_candidate_count = len(stratum_state.candidates)
        # Break retained-priority ties by tile order and pixel position.
        stratum_state.candidates = sorted(
            [*stratum_state.candidates, *new_candidates],
            key=lambda candidate: (
                candidate.random_priority,
                candidate.tile_sequence,
                candidate.local_pixel_index,
            ),
        )[: samples_per_class_per_block]
        retained_candidate_delta = (
            len(stratum_state.candidates) - previous_candidate_count
        )
        if stratum_key[2] == 1:
            retained_reference_candidate_count += retained_candidate_delta
        else:
            retained_background_candidate_count += retained_candidate_delta

    return retained_background_candidate_count, retained_reference_candidate_count


def build_sample_weight_columns(
    candidates: Sequence[SamplingCandidate],
    strata: dict[tuple[int, int, int], SamplingStratumState],
    pixel_area_m2: float,
) -> dict[str, np.ndarray]:
    """Build block identifiers, reference labels and expansion weights per sample.

    Each selected pixel represents its stratum's population divided by the number
    selected from that stratum. Multiplying by the pixel area gives its
    represented area. Block IDs follow sorted block-coordinate pairs, starting at 1.

    Args:
        candidates: Retained pixels in the caller's desired output row order.
        strata: Completed population counts and selections from
            ``retain_tile_candidates``.
        pixel_area_m2: Constant pixel area for the admitted equal-area grid.

    Returns:
        Arrays in candidate order: block ID/column/row, reference class, pixel
        area, available/selected stratum counts, selection probability, population
        expansion weight and represented-area weight. Names and dtypes match the
        existing spatial sample table.
    """

    block_keys = sorted({key[:2] for key in strata})
    block_id_by_key = {
        block_key: block_offset + 1
        for block_offset, block_key in enumerate(block_keys)
    }
    sampled_row_count = len(candidates)
    table_columns: dict[str, np.ndarray] = {
        "sampling_block_id": np.empty(sampled_row_count, dtype=np.int64),
        "sampling_block_column": np.empty(sampled_row_count, dtype=np.int64),
        "sampling_block_row": np.empty(sampled_row_count, dtype=np.int64),
        "reference_site": np.empty(sampled_row_count, dtype=np.uint8),
        "pixel_area_m2": np.empty(sampled_row_count, dtype=np.float64),
        "available_pixels_in_block_class": np.empty(
            sampled_row_count,
            dtype=np.int64,
        ),
        "sampled_pixels_in_block_class": np.empty(
            sampled_row_count,
            dtype=np.int32,
        ),
        "sampling_probability": np.empty(
            sampled_row_count,
            dtype=np.float64,
        ),
        "sampling_weight": np.empty(sampled_row_count, dtype=np.float64),
        "area_weight_m2": np.empty(sampled_row_count, dtype=np.float64),
    }
    for destination_row, candidate in enumerate(candidates):
        stratum_key = (
            candidate.sampling_block_column,
            candidate.sampling_block_row,
            candidate.reference_site_class,
        )
        stratum_state = strata[stratum_key]
        sampled_in_stratum = len(stratum_state.candidates)
        sampling_weight = (
            stratum_state.available_pixel_count / sampled_in_stratum
        )
        table_columns["sampling_block_id"][destination_row] = (
            block_id_by_key[stratum_key[:2]]
        )
        table_columns["sampling_block_column"][destination_row] = (
            candidate.sampling_block_column
        )
        table_columns["sampling_block_row"][destination_row] = (
            candidate.sampling_block_row
        )
        table_columns["reference_site"][destination_row] = (
            candidate.reference_site_class
        )
        table_columns["pixel_area_m2"][destination_row] = pixel_area_m2
        table_columns[
            "available_pixels_in_block_class"
        ][destination_row] = stratum_state.available_pixel_count
        table_columns[
            "sampled_pixels_in_block_class"
        ][destination_row] = sampled_in_stratum
        table_columns["sampling_probability"][destination_row] = (
            sampled_in_stratum
            / stratum_state.available_pixel_count
        )
        table_columns["sampling_weight"][destination_row] = sampling_weight
        table_columns["area_weight_m2"][destination_row] = (
            pixel_area_m2 * sampling_weight
        )

    return table_columns
