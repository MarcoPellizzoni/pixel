"""Tests for the tonal steps."""

import numpy as np
from conftest import gradient_image, gray_image, solid_image, split_image

from pixel.domain import RGBAImage
from pixel.steps.tone import (
    AutoContrastConfig,
    AutoContrastStep,
    BrightnessContrastConfig,
    BrightnessContrastStep,
    GammaConfig,
    GammaStep,
    ThresholdConfig,
    ThresholdMethod,
    ThresholdStep,
)


class TestBrightnessContrast:
    """Brightness and contrast must act independently of each other."""

    def test_raising_the_brightness_lightens(self) -> None:
        result = BrightnessContrastStep(
            BrightnessContrastConfig(brightness=0.3)
        ).apply(gray_image(100))

        assert result.rgb[0, 0, 0] > 100

    def test_lowering_the_brightness_darkens(self) -> None:
        result = BrightnessContrastStep(
            BrightnessContrastConfig(brightness=-0.3)
        ).apply(gray_image(100))

        assert result.rgb[0, 0, 0] < 100

    def test_the_default_values_change_nothing(self) -> None:
        image = gradient_image()

        result = BrightnessContrastStep(BrightnessContrastConfig()).apply(image)

        assert np.abs(result.rgb.astype(int) - image.rgb.astype(int)).max() <= 1

    def test_contrast_pushes_tones_away_from_mid_grey(self) -> None:
        light = gray_image(200)
        dark = gray_image(50)
        step = BrightnessContrastStep(BrightnessContrastConfig(contrast=1.5))

        assert step.apply(light).rgb[0, 0, 0] > 200
        assert step.apply(dark).rgb[0, 0, 0] < 50

    def test_contrast_pivots_about_mid_grey(self) -> None:
        # Mid grey is the pivot: it must not move.
        result = BrightnessContrastStep(BrightnessContrastConfig(contrast=2.0)).apply(
            gray_image(128)
        )

        assert abs(int(result.rgb[0, 0, 0]) - 128) <= 1

    def test_the_values_do_not_leave_the_scale(self) -> None:
        result = BrightnessContrastStep(
            BrightnessContrastConfig(brightness=5.0, contrast=10.0)
        ).apply(gradient_image())

        assert result.rgb.min() >= 0
        assert result.rgb.max() <= 255


class TestGamma:
    """Gamma must act on the midtones and leave the extremes alone."""

    def test_below_one_it_lifts_the_midtones(self) -> None:
        result = GammaStep(GammaConfig(gamma=0.5)).apply(gray_image(64))

        assert result.rgb[0, 0, 0] > 64

    def test_above_one_it_deepens_the_midtones(self) -> None:
        result = GammaStep(GammaConfig(gamma=2.0)).apply(gray_image(180))

        assert result.rgb[0, 0, 0] < 180

    def test_gamma_one_changes_nothing(self) -> None:
        image = gradient_image()

        result = GammaStep(GammaConfig(gamma=1.0)).apply(image)

        assert np.abs(result.rgb.astype(int) - image.rgb.astype(int)).max() <= 1

    def test_black_and_white_do_not_move(self) -> None:
        step = GammaStep(GammaConfig(gamma=2.5))

        assert step.apply(solid_image((0, 0, 0))).rgb.max() == 0
        assert step.apply(solid_image((255, 255, 255))).rgb.min() == 255

    def test_an_invalid_gamma_is_treated_as_neutral(self) -> None:
        image = gray_image(120)

        result = GammaStep(GammaConfig(gamma=0.0)).apply(image)

        assert abs(int(result.rgb[0, 0, 0]) - 120) <= 1


class TestAutoContrast:
    """Equalisation must widen the tonal range without shifting the colours."""

    def test_it_widens_the_range_of_a_flat_image(self) -> None:
        # A gradient squeezed into a few levels, like a photo shot in low light.
        data = np.zeros((64, 64, 4), dtype=np.uint8)
        data[:, :, :3] = np.linspace(100, 130, 64, dtype=np.uint8)[None, :, None]
        data[:, :, 3] = 255

        image = RGBAImage(data)
        result = AutoContrastStep(AutoContrastConfig(clip_limit=4.0)).apply(image)

        range_before = int(image.rgb.max()) - int(image.rgb.min())
        range_after = int(result.rgb.max()) - int(result.rgb.min())
        assert range_after > range_before

    def test_a_zero_limit_disables_the_equalisation(self) -> None:
        image = gradient_image()

        result = AutoContrastStep(AutoContrastConfig(clip_limit=0.0)).apply(image)

        assert np.array_equal(result.data, image.data)

    def test_a_grey_stays_grey(self) -> None:
        # Proof that the equalisation happens on brightness alone: acting on the
        # three RGB channels separately would shift a grey off-colour.
        result = AutoContrastStep(AutoContrastConfig()).apply(gradient_image())

        rgb = result.rgb.astype(int)
        assert np.abs(rgb[:, :, 0] - rgb[:, :, 1]).max() <= 2
        assert np.abs(rgb[:, :, 1] - rgb[:, :, 2]).max() <= 2

    def test_the_alpha_is_unchanged(self) -> None:
        image = solid_image((100, 120, 140), alpha=88)

        assert np.all(AutoContrastStep(AutoContrastConfig()).apply(image).alpha == 88)


class TestThreshold:
    """Binarisation must produce black and white only."""

    def test_only_two_values_remain(self) -> None:
        result = ThresholdStep(ThresholdConfig()).apply(gradient_image())

        assert set(np.unique(result.rgb)).issubset({0, 255})

    def test_the_fixed_threshold_splits_at_the_given_level(self) -> None:
        config = ThresholdConfig(method=ThresholdMethod.FIXED, level=128)
        step = ThresholdStep(config)

        assert step.apply(gray_image(200)).rgb[0, 0, 0] == 255
        assert step.apply(gray_image(50)).rgb[0, 0, 0] == 0

    def test_otsu_separates_two_crisp_populations(self) -> None:
        image = split_image((30, 30, 30), (220, 220, 220))

        result = ThresholdStep(ThresholdConfig(method=ThresholdMethod.OTSU)).apply(image)

        assert result.rgb[0, 0, 0] == 0
        assert result.rgb[0, -1, 0] == 255

    def test_inversion_swaps_black_and_white(self) -> None:
        image = gray_image(200)
        config = ThresholdConfig(method=ThresholdMethod.FIXED, level=128)

        straight = ThresholdStep(config).apply(image)
        inverted = ThresholdStep(
            ThresholdConfig(method=ThresholdMethod.FIXED, level=128, invert=True)
        ).apply(image)

        assert straight.rgb[0, 0, 0] != inverted.rgb[0, 0, 0]

    def test_the_adaptive_threshold_copes_with_an_even_window(self) -> None:
        # OpenCV demands an odd window: the step must correct it instead of
        # letting the error propagate.
        config = ThresholdConfig(method=ThresholdMethod.ADAPTIVE, block_size=20)

        result = ThresholdStep(config).apply(gradient_image())

        assert set(np.unique(result.rgb)).issubset({0, 255})

    def test_the_adaptive_threshold_copes_with_a_tiny_window(self) -> None:
        config = ThresholdConfig(method=ThresholdMethod.ADAPTIVE, block_size=1)

        result = ThresholdStep(config).apply(gradient_image())

        assert set(np.unique(result.rgb)).issubset({0, 255})

    def test_the_result_is_monochrome(self) -> None:
        result = ThresholdStep(ThresholdConfig()).apply(solid_image((200, 100, 50)))

        rgb = result.rgb
        assert np.array_equal(rgb[:, :, 0], rgb[:, :, 2])
