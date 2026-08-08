"""Tests for the `RGBAImage` domain model."""

import numpy as np
import pytest

from pixel.domain import RGBAImage, to_uint8


def make_rgba(height: int = 4, width: int = 6, fill: int = 128) -> np.ndarray:
    """Create a valid RGBA array, a useful starting point for the tests."""
    return np.full((height, width, 4), fill, dtype=np.uint8)


class TestToUint8:
    """Converting from floating point to 8 bits must round, not truncate."""

    def test_rounds_to_the_nearest_value(self) -> None:
        result = to_uint8(np.array([9.6, 9.4], dtype=np.float32))

        assert list(result) == [10, 9]

    def test_a_value_just_below_an_integer_is_not_truncated(self) -> None:
        # This is the real case: floating-point imprecision produces 189.9999
        # where the exact arithmetic would give 190. Truncating, every neutral
        # step would darken the image by one level, and the error would compound.
        result = to_uint8(np.array([189.99999], dtype=np.float32))

        assert result[0] == 190

    def test_out_of_range_values_are_clipped(self) -> None:
        result = to_uint8(np.array([-40.0, 300.0], dtype=np.float32))

        assert list(result) == [0, 255]

    def test_the_returned_dtype_is_uint8(self) -> None:
        assert to_uint8(np.array([1.0], dtype=np.float32)).dtype == np.uint8

    def test_the_shape_is_unchanged(self) -> None:
        result = to_uint8(np.zeros((3, 4, 3), dtype=np.float32))

        assert result.shape == (3, 4, 3)


class TestValidation:
    """Construction must reject malformed arrays immediately."""

    def test_accepts_a_valid_rgba_array(self) -> None:
        image = RGBAImage(make_rgba())
        assert image.height == 4
        assert image.width == 6

    def test_rejects_a_two_dimensional_array(self) -> None:
        with pytest.raises(ValueError, match="3 dimensions"):
            RGBAImage(np.zeros((4, 6), dtype=np.uint8))

    def test_rejects_a_wrong_channel_count(self) -> None:
        with pytest.raises(ValueError, match="channels"):
            RGBAImage(np.zeros((4, 6, 3), dtype=np.uint8))

    def test_rejects_a_dtype_other_than_uint8(self) -> None:
        with pytest.raises(ValueError, match="uint8"):
            RGBAImage(np.zeros((4, 6, 4), dtype=np.float32))


class TestFromRgb:
    """An image without transparency must become fully opaque."""

    def test_adds_an_opaque_alpha_channel(self) -> None:
        rgb = np.full((3, 3, 3), 200, dtype=np.uint8)

        image = RGBAImage.from_rgb(rgb)

        assert np.all(image.alpha == 255)
        assert np.array_equal(image.rgb, rgb)

    def test_rejects_a_non_rgb_array(self) -> None:
        with pytest.raises(ValueError, match="RGB"):
            RGBAImage.from_rgb(np.zeros((3, 3, 4), dtype=np.uint8))


class TestImmutability:
    """Modifying what you get out of an image must not alter it."""

    def test_modifying_the_rgb_copy_does_not_touch_the_original(self) -> None:
        image = RGBAImage(make_rgba(fill=100))

        extracted = image.rgb
        extracted[:] = 0

        assert np.all(image.rgb == 100)

    def test_modifying_the_alpha_copy_does_not_touch_the_original(self) -> None:
        image = RGBAImage(make_rgba(fill=100))

        extracted = image.alpha
        extracted[:] = 0

        assert np.all(image.alpha == 100)


class TestTransformations:
    """`with_rgb` and `with_alpha` change one part and leave the other."""

    def test_with_rgb_preserves_the_alpha_channel(self) -> None:
        image = RGBAImage(make_rgba(fill=50))
        new_rgb = np.full((4, 6, 3), 210, dtype=np.uint8)

        result = image.with_rgb(new_rgb)

        assert np.array_equal(result.rgb, new_rgb)
        assert np.all(result.alpha == 50)

    def test_with_alpha_preserves_the_colours(self) -> None:
        image = RGBAImage(make_rgba(fill=50))
        new_alpha = np.full((4, 6), 255, dtype=np.uint8)

        result = image.with_alpha(new_alpha)

        assert np.all(result.rgb == 50)
        assert np.all(result.alpha == 255)

    def test_with_rgb_rejects_different_dimensions(self) -> None:
        image = RGBAImage(make_rgba(height=4, width=6))

        with pytest.raises(ValueError, match="dimensions"):
            image.with_rgb(np.zeros((9, 9, 3), dtype=np.uint8))

    def test_with_alpha_rejects_different_dimensions(self) -> None:
        image = RGBAImage(make_rgba(height=4, width=6))

        with pytest.raises(ValueError, match="dimensions"):
            image.with_alpha(np.zeros((9, 9), dtype=np.uint8))


class TestCompositeOver:
    """Blending onto an opaque background must follow the alpha blending formula."""

    def test_an_opaque_pixel_keeps_its_own_colour(self) -> None:
        data = np.zeros((1, 1, 4), dtype=np.uint8)
        data[0, 0] = [10, 20, 30, 255]

        result = RGBAImage(data).composite_over((255, 255, 255))

        assert list(result[0, 0]) == [10, 20, 30]

    def test_a_transparent_pixel_takes_the_background_colour(self) -> None:
        data = np.zeros((1, 1, 4), dtype=np.uint8)
        data[0, 0] = [10, 20, 30, 0]

        result = RGBAImage(data).composite_over((255, 255, 255))

        assert list(result[0, 0]) == [255, 255, 255]

    def test_a_half_transparent_pixel_mixes_the_two_colours(self) -> None:
        data = np.zeros((1, 1, 4), dtype=np.uint8)
        # Alpha 128/255 ~= 0.502: the result must sit halfway between 0 and 255.
        data[0, 0] = [0, 0, 0, 128]

        result = RGBAImage(data).composite_over((255, 255, 255))

        assert result[0, 0, 0] == pytest.approx(127, abs=1)

    def test_the_result_no_longer_has_an_alpha_channel(self) -> None:
        result = RGBAImage(make_rgba()).composite_over((0, 0, 0))

        assert result.shape == (4, 6, 3)
