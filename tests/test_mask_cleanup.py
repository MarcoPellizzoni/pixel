"""Tests for the refinements applied to the cut-out mask."""

import numpy as np

from pixel.steps.mask_cleanup import feather_mask_edges, keep_largest_region


class TestKeepLargestRegion:
    """Only the largest connected region must survive."""

    def test_the_smaller_isolated_blob_is_removed(self) -> None:
        mask = np.zeros((40, 40), dtype=np.uint8)
        mask[5:25, 5:25] = 255  # subject: 400 pixels
        mask[32:36, 32:36] = 255  # isolated leftover: 16 pixels

        result = keep_largest_region(mask, alpha_threshold=8)

        assert np.all(result[5:25, 5:25] == 255)
        assert np.all(result[32:36, 32:36] == 0)

    def test_a_thin_connected_appendage_survives(self) -> None:
        # This reproduces the whiskers case: a thin stroke, but attached to the body.
        mask = np.zeros((40, 40), dtype=np.uint8)
        mask[10:30, 10:30] = 255
        mask[19, 30:38] = 255

        result = keep_largest_region(mask, alpha_threshold=8)

        assert np.all(result[19, 30:38] == 255)

    def test_a_semi_transparent_connected_appendage_survives(self) -> None:
        # Whiskers are only faintly opaque: the low threshold must still count
        # them as part of the body, instead of detaching and deleting them.
        mask = np.zeros((40, 40), dtype=np.uint8)
        mask[10:30, 10:30] = 255
        mask[19, 30:38] = 20

        result = keep_largest_region(mask, alpha_threshold=8)

        assert np.all(result[19, 30:38] == 20)

    def test_too_high_a_threshold_detaches_the_faint_parts(self) -> None:
        # Documents why the threshold has to be kept low.
        mask = np.zeros((40, 40), dtype=np.uint8)
        mask[10:30, 10:30] = 255
        mask[19, 30:38] = 20

        result = keep_largest_region(mask, alpha_threshold=100)

        assert np.all(result[19, 30:38] == 0)

    def test_the_soft_values_of_surviving_pixels_do_not_change(self) -> None:
        mask = np.zeros((20, 20), dtype=np.uint8)
        mask[5:15, 5:15] = 200
        mask[5, 5] = 37

        result = keep_largest_region(mask, alpha_threshold=8)

        assert result[5, 5] == 37
        assert result[10, 10] == 200

    def test_an_empty_mask_stays_empty(self) -> None:
        mask = np.zeros((10, 10), dtype=np.uint8)

        result = keep_largest_region(mask, alpha_threshold=8)

        assert np.all(result == 0)

    def test_a_full_mask_stays_full(self) -> None:
        mask = np.full((10, 10), 255, dtype=np.uint8)

        result = keep_largest_region(mask, alpha_threshold=8)

        assert np.all(result == 255)

    def test_diagonally_connected_regions_count_as_one(self) -> None:
        mask = np.zeros((20, 20), dtype=np.uint8)
        mask[2:10, 2:10] = 255
        # Touches the previous one only at a corner.
        mask[10:18, 10:18] = 255

        result = keep_largest_region(mask, alpha_threshold=8)

        assert np.all(result[2:10, 2:10] == 255)
        assert np.all(result[10:18, 10:18] == 255)


class TestFeatherMaskEdges:
    """Feathering must soften the edge without moving it."""

    def test_a_zero_radius_leaves_the_mask_unchanged(self) -> None:
        mask = np.zeros((20, 20), dtype=np.uint8)
        mask[5:15, 5:15] = 255

        result = feather_mask_edges(mask, radius=0)

        assert np.array_equal(result, mask)

    def test_feathering_creates_intermediate_values_at_the_edge(self) -> None:
        mask = np.zeros((20, 20), dtype=np.uint8)
        mask[5:15, 5:15] = 255

        result = feather_mask_edges(mask, radius=2)

        intermediate = (result > 0) & (result < 255)
        assert intermediate.any()

    def test_the_centre_and_the_outside_stay_saturated(self) -> None:
        mask = np.zeros((30, 30), dtype=np.uint8)
        mask[10:20, 10:20] = 255

        result = feather_mask_edges(mask, radius=1)

        assert result[15, 15] == 255
        assert result[0, 0] == 0

    def test_the_dimensions_are_unchanged(self) -> None:
        mask = np.zeros((13, 17), dtype=np.uint8)

        result = feather_mask_edges(mask, radius=3)

        assert result.shape == (13, 17)
