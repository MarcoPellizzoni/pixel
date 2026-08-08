"""Tests for the spatial filtering steps."""

import numpy as np
from conftest import gray_image, noisy_image, solid_image, split_image

from pixel.steps.filters import (
    BlurConfig,
    BlurStep,
    DenoiseConfig,
    DenoiseMethod,
    DenoiseStep,
    EdgesConfig,
    EdgesStep,
    SharpenConfig,
    SharpenStep,
)


class TestBlur:
    """Blurring must soften the crisp transitions."""

    def test_it_softens_an_edge(self) -> None:
        image = split_image((0, 0, 0), (255, 255, 255))

        result = BlurStep(BlurConfig(radius=4.0)).apply(image)

        # After the blur, intermediate values appear around the edge.
        edge_column = result.rgb[:, 32, 0]
        assert 0 < int(edge_column.mean()) < 255

    def test_a_zero_radius_changes_nothing(self) -> None:
        image = noisy_image()

        result = BlurStep(BlurConfig(radius=0.0)).apply(image)

        assert np.array_equal(result.data, image.data)

    def test_it_reduces_the_spread_of_a_noisy_image(self) -> None:
        image = noisy_image()

        result = BlurStep(BlurConfig(radius=3.0)).apply(image)

        assert result.rgb.std() < image.rgb.std()

    def test_a_solid_colour_stays_the_same(self) -> None:
        result = BlurStep(BlurConfig(radius=5.0)).apply(solid_image((120, 130, 140)))

        assert list(result.rgb[8, 8]) == [120, 130, 140]

    def test_the_alpha_is_unchanged(self) -> None:
        image = solid_image((120, 130, 140), alpha=66)

        assert np.all(BlurStep(BlurConfig()).apply(image).alpha == 66)


class TestSharpen:
    """Sharpening must increase the contrast on the edges."""

    def test_it_increases_the_contrast_on_an_edge(self) -> None:
        # An already blurred edge: sharpening must make it crisper.
        image = BlurStep(BlurConfig(radius=3.0)).apply(
            split_image((60, 60, 60), (190, 190, 190))
        )

        result = SharpenStep(SharpenConfig(amount=1.5)).apply(image)

        assert result.rgb.std() > image.rgb.std()

    def test_a_zero_amount_changes_nothing(self) -> None:
        image = noisy_image()

        result = SharpenStep(SharpenConfig(amount=0.0)).apply(image)

        assert np.array_equal(result.data, image.data)

    def test_a_solid_colour_stays_the_same(self) -> None:
        # With no detail to enhance, the unsharp mask is zero.
        result = SharpenStep(SharpenConfig(amount=2.0)).apply(
            solid_image((100, 110, 120))
        )

        assert list(result.rgb[8, 8]) == [100, 110, 120]

    def test_the_values_do_not_leave_the_scale(self) -> None:
        result = SharpenStep(SharpenConfig(amount=10.0)).apply(noisy_image())

        assert result.rgb.min() >= 0
        assert result.rgb.max() <= 255


class TestDenoise:
    """Attenuation must clean up without blunting the edges."""

    def test_the_bilateral_filter_reduces_the_noise(self) -> None:
        image = noisy_image()

        result = DenoiseStep(
            DenoiseConfig(method=DenoiseMethod.BILATERAL, strength=60.0)
        ).apply(image)

        assert result.rgb.std() < image.rgb.std()

    def test_non_local_means_reduces_the_noise(self) -> None:
        image = noisy_image(width=32, height=32)

        result = DenoiseStep(
            DenoiseConfig(method=DenoiseMethod.NON_LOCAL_MEANS, strength=20.0)
        ).apply(image)

        assert result.rgb.std() < image.rgb.std()

    def test_the_edge_stays_crisp(self) -> None:
        # This is the bilateral filter's advantage over a blur: the jump between
        # the two halves must stay nearly intact.
        image = split_image((20, 20, 20), (230, 230, 230))

        result = DenoiseStep(DenoiseConfig(strength=60.0)).apply(image)

        jump = int(result.rgb[16, 40, 0]) - int(result.rgb[16, 20, 0])
        assert jump > 180

    def test_the_alpha_is_unchanged(self) -> None:
        image = solid_image((100, 110, 120), alpha=55)

        assert np.all(DenoiseStep(DenoiseConfig()).apply(image).alpha == 55)


class TestEdges:
    """Detection must find the edges and ignore the uniform areas."""

    def test_a_solid_colour_has_no_contours(self) -> None:
        result = EdgesStep(EdgesConfig()).apply(gray_image(120))

        # On a white background, no contours means an all-white sheet.
        assert np.all(result.rgb == 255)

    def test_a_crisp_edge_is_found(self) -> None:
        image = split_image((0, 0, 0), (255, 255, 255))

        result = EdgesStep(EdgesConfig()).apply(image)

        assert result.rgb.min() == 0

    def test_the_contour_lands_where_the_edge_is(self) -> None:
        image = split_image((0, 0, 0), (255, 255, 255), width=64)

        result = EdgesStep(EdgesConfig()).apply(image)

        grey = result.rgb[:, :, 0]
        # The black line falls around column 32, not at the margins.
        assert grey[:, 28:36].min() == 0
        assert grey[:, :10].min() == 255

    def test_disabling_on_white_inverts_the_tones(self) -> None:
        image = split_image((0, 0, 0), (255, 255, 255))

        on_black = EdgesStep(EdgesConfig(on_white=False)).apply(image)

        # White lines on a black background: most of the image is black.
        assert on_black.rgb.mean() < 128

    def test_the_result_is_monochrome(self) -> None:
        result = EdgesStep(EdgesConfig()).apply(
            split_image((200, 30, 30), (30, 30, 200))
        )

        rgb = result.rgb
        assert np.array_equal(rgb[:, :, 0], rgb[:, :, 2])
