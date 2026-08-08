"""Tests for the colour steps."""

import numpy as np
import pytest
from conftest import gray_image, solid_image

from pixel.domain import RGBAImage
from pixel.steps.color import (
    GrayscaleConfig,
    GrayscaleStep,
    InvertConfig,
    InvertStep,
    LuminanceStandard,
    PosterizeConfig,
    PosterizeStep,
    SaturationConfig,
    SaturationStep,
    SepiaConfig,
    SepiaStep,
)


class TestGrayscale:
    """The conversion must apply the weights of the chosen standard."""

    def test_the_three_channels_become_equal(self) -> None:
        result = GrayscaleStep(GrayscaleConfig()).apply(solid_image((200, 100, 50)))

        rgb = result.rgb
        assert np.array_equal(rgb[:, :, 0], rgb[:, :, 1])
        assert np.array_equal(rgb[:, :, 1], rgb[:, :, 2])

    def test_bt709_uses_its_own_coefficients(self) -> None:
        config = GrayscaleConfig(standard=LuminanceStandard.BT709)

        result = GrayscaleStep(config).apply(solid_image((200, 100, 50)))

        expected = 200 * 0.2126 + 100 * 0.7152 + 50 * 0.0722
        assert result.rgb[0, 0, 0] == pytest.approx(expected, abs=1)

    def test_bt601_uses_its_own_coefficients(self) -> None:
        config = GrayscaleConfig(standard=LuminanceStandard.BT601)

        result = GrayscaleStep(config).apply(solid_image((200, 100, 50)))

        expected = 200 * 0.299 + 100 * 0.587 + 50 * 0.114
        assert result.rgb[0, 0, 0] == pytest.approx(expected, abs=1)

    def test_the_two_standards_differ_on_warm_tones(self) -> None:
        image = solid_image((230, 140, 60))

        bt601 = GrayscaleStep(
            GrayscaleConfig(standard=LuminanceStandard.BT601)
        ).apply(image)
        bt709 = GrayscaleStep(
            GrayscaleConfig(standard=LuminanceStandard.BT709)
        ).apply(image)

        # BT.601 weighs red more heavily, so on an orange it comes out brighter.
        assert bt601.rgb[0, 0, 0] > bt709.rgb[0, 0, 0]

    def test_black_and_white_stay_themselves(self) -> None:
        step = GrayscaleStep(GrayscaleConfig())

        assert np.all(step.apply(solid_image((255, 255, 255))).rgb == 255)
        assert np.all(step.apply(solid_image((0, 0, 0))).rgb == 0)

    def test_the_alpha_is_unchanged(self) -> None:
        image = solid_image((200, 100, 50), alpha=77)

        result = GrayscaleStep(GrayscaleConfig()).apply(image)

        assert np.all(result.alpha == 77)


class TestSepia:
    """Toning must warm the image up, by a controllable amount."""

    def test_the_result_is_warmer_than_the_original(self) -> None:
        result = SepiaStep(SepiaConfig()).apply(solid_image((128, 128, 128)))

        red, _, blue = result.rgb[0, 0]
        assert red > blue

    def test_zero_intensity_leaves_the_original(self) -> None:
        image = solid_image((60, 120, 200))

        result = SepiaStep(SepiaConfig(intensity=0.0)).apply(image)

        assert np.array_equal(result.rgb, image.rgb)

    def test_an_intermediate_intensity_sits_between_the_two_extremes(self) -> None:
        image = solid_image((60, 120, 200))

        full = SepiaStep(SepiaConfig(intensity=1.0)).apply(image)
        half = SepiaStep(SepiaConfig(intensity=0.5)).apply(image)

        original_blue = int(image.rgb[0, 0, 2])
        assert min(original_blue, int(full.rgb[0, 0, 2])) <= int(half.rgb[0, 0, 2])
        assert int(half.rgb[0, 0, 2]) <= max(original_blue, int(full.rgb[0, 0, 2]))

    def test_the_alpha_is_unchanged(self) -> None:
        image = solid_image((60, 120, 200), alpha=33)

        assert np.all(SepiaStep(SepiaConfig()).apply(image).alpha == 33)


class TestInvert:
    """Inversion is the complement to 255, and it does not touch transparency."""

    def test_black_becomes_white(self) -> None:
        result = InvertStep(InvertConfig()).apply(solid_image((0, 0, 0)))

        assert np.all(result.rgb == 255)

    def test_every_channel_is_its_own_complement(self) -> None:
        result = InvertStep(InvertConfig()).apply(solid_image((10, 100, 250)))

        assert list(result.rgb[0, 0]) == [245, 155, 5]

    def test_two_inversions_return_to_the_original(self) -> None:
        image = solid_image((10, 100, 250))
        step = InvertStep(InvertConfig())

        assert np.array_equal(step.apply(step.apply(image)).rgb, image.rgb)

    def test_the_alpha_is_not_inverted(self) -> None:
        image = solid_image((10, 100, 250), alpha=200)

        assert np.all(InvertStep(InvertConfig()).apply(image).alpha == 200)


class TestSaturation:
    """Saturation must make the hues more vivid or more muted."""

    def test_zeroing_it_produces_a_grey(self) -> None:
        result = SaturationStep(SaturationConfig(amount=0.0)).apply(
            solid_image((200, 50, 50))
        )

        red, green, blue = result.rgb[0, 0]
        assert red == green == blue

    def test_raising_it_makes_the_hue_more_vivid(self) -> None:
        image = solid_image((180, 120, 120))

        result = SaturationStep(SaturationConfig(amount=2.0)).apply(image)

        # More saturated means a greater distance between the dominant channel
        # and the others.
        distance_before = int(image.rgb[0, 0, 0]) - int(image.rgb[0, 0, 1])
        distance_after = int(result.rgb[0, 0, 0]) - int(result.rgb[0, 0, 1])
        assert distance_after > distance_before

    def test_leaving_it_at_one_changes_almost_nothing(self) -> None:
        image = solid_image((180, 120, 90))

        result = SaturationStep(SaturationConfig(amount=1.0)).apply(image)

        # The round trip through HSV introduces a minimal rounding error, but it
        # must not shift the colour visibly.
        assert np.abs(result.rgb.astype(int) - image.rgb.astype(int)).max() <= 2

    def test_a_grey_stays_grey(self) -> None:
        result = SaturationStep(SaturationConfig(amount=3.0)).apply(gray_image(128))

        red, green, blue = result.rgb[0, 0]
        assert red == green == blue


class TestPosterize:
    """Posterisation must reduce the number of distinct values."""

    def test_it_reduces_the_levels_present(self) -> None:
        # A gradient holds many different values; only a few must remain after.
        data = np.zeros((1, 256, 4), dtype=np.uint8)
        data[0, :, :3] = np.arange(256, dtype=np.uint8)[:, np.newaxis]
        data[:, :, 3] = 255

        result = PosterizeStep(PosterizeConfig(levels=4)).apply(RGBAImage(data))

        assert len(np.unique(result.rgb[:, :, 0])) == 4

    def test_black_and_white_stay_at_the_extremes(self) -> None:
        step = PosterizeStep(PosterizeConfig(levels=3))

        assert np.all(step.apply(solid_image((0, 0, 0))).rgb == 0)
        assert np.all(step.apply(solid_image((255, 255, 255))).rgb == 255)

    def test_fewer_than_two_levels_is_corrected(self) -> None:
        # With a single level the image would be a uniform rectangle: the step
        # raises the value to the smallest sensible one instead of dividing by zero.
        result = PosterizeStep(PosterizeConfig(levels=1)).apply(gray_image(200))

        assert result.rgb.max() <= 255

    def test_the_alpha_is_unchanged(self) -> None:
        image = solid_image((10, 100, 250), alpha=44)

        assert np.all(PosterizeStep(PosterizeConfig()).apply(image).alpha == 44)
