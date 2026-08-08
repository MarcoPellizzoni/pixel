"""Tests for turning an image into something displayable."""

from io import BytesIO

import numpy as np
from conftest import solid_image
from PIL import Image

from pixel.domain import RGBAImage
from pixel.ui.preview import (
    CHECKER_DARK,
    CHECKER_LIGHT,
    CHECKER_SQUARE_SIZE,
    to_png_bytes,
)


def decode(png_bytes: bytes) -> np.ndarray:
    """Decode PNG bytes back into an RGB array, so the result can be inspected."""
    return np.array(Image.open(BytesIO(png_bytes)).convert("RGB"))


class TestEncoding:
    """The result must be a PNG the interface can display."""

    def test_it_produces_a_readable_png(self) -> None:
        result = to_png_bytes(solid_image((10, 20, 30)))

        assert Image.open(BytesIO(result)).format == "PNG"

    def test_an_opaque_image_keeps_its_colours(self) -> None:
        result = to_png_bytes(solid_image((10, 20, 30)))

        assert list(decode(result)[0, 0]) == [10, 20, 30]

    def test_the_result_has_no_alpha_channel(self) -> None:
        # The chequerboard has already been flattened in, so there is nothing
        # left for the display to be transparent against.
        result = to_png_bytes(solid_image((10, 20, 30), alpha=0))

        assert decode(result).shape[2] == 3


class TestScaling:
    """Oversized images must be scaled down, small ones left alone."""

    def test_a_large_image_is_shrunk(self) -> None:
        image = solid_image((10, 20, 30), width=3000, height=1500)

        result = decode(to_png_bytes(image, max_side=600))

        assert max(result.shape[0], result.shape[1]) == 600

    def test_the_aspect_ratio_is_kept(self) -> None:
        image = solid_image((10, 20, 30), width=2000, height=1000)

        result = decode(to_png_bytes(image, max_side=500))

        assert result.shape[1] / result.shape[0] == 2.0

    def test_a_small_image_is_not_enlarged(self) -> None:
        image = solid_image((10, 20, 30), width=40, height=20)

        result = decode(to_png_bytes(image, max_side=1000))

        assert (result.shape[1], result.shape[0]) == (40, 20)


class TestCheckerboard:
    """Transparency must show up as the usual grey chequerboard."""

    def test_a_fully_transparent_image_shows_the_board(self) -> None:
        # Wide enough to span several squares in both directions.
        image = solid_image((255, 0, 0), width=64, height=64, alpha=0)

        result = decode(to_png_bytes(image))

        assert set(np.unique(result)) == {CHECKER_DARK, CHECKER_LIGHT}

    def test_the_squares_alternate(self) -> None:
        image = solid_image((255, 0, 0), width=64, height=64, alpha=0)

        result = decode(to_png_bytes(image))

        # The pixel one square to the right belongs to the other colour.
        first_square = result[0, 0, 0]
        next_square = result[0, CHECKER_SQUARE_SIZE, 0]
        assert first_square != next_square

    def test_an_opaque_area_hides_the_board(self) -> None:
        # Left half opaque red, right half fully transparent.
        data = np.zeros((32, 64, 4), dtype=np.uint8)
        data[:, :32, :3] = (255, 0, 0)
        data[:, :32, 3] = 255

        result = decode(to_png_bytes(RGBAImage(data)))

        assert list(result[0, 0]) == [255, 0, 0]
        assert result[0, 60, 0] in (CHECKER_DARK, CHECKER_LIGHT)

    def test_a_half_transparent_pixel_blends_with_the_board(self) -> None:
        data = np.zeros((32, 32, 4), dtype=np.uint8)
        data[:, :, :3] = 255
        data[:, :, 3] = 128

        result = decode(to_png_bytes(RGBAImage(data)))

        # Halfway between white and the board's grey: neither pure white nor the
        # board showing through untouched.
        assert CHECKER_LIGHT < int(result[0, 0, 0]) < 255
