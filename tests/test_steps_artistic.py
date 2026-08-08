"""Tests for the artistic steps (the pen effect has a file of its own)."""

import numpy as np
from conftest import gray_image, noisy_image, solid_image, split_image

from pixel.steps.artistic import (
    CartoonConfig,
    CartoonStep,
    PencilSketchConfig,
    PencilSketchStep,
    VignetteConfig,
    VignetteStep,
)


class TestPencilSketch:
    """The pencil sketch must leave white paper where there is nothing."""

    def test_a_solid_colour_becomes_white_paper(self) -> None:
        result = PencilSketchStep(PencilSketchConfig()).apply(gray_image(140))

        assert result.rgb.min() > 250

    def test_an_edge_leaves_a_mark(self) -> None:
        image = split_image((30, 30, 30), (225, 225, 225))

        result = PencilSketchStep(PencilSketchConfig()).apply(image)

        assert result.rgb.min() < 200

    def test_raising_the_strength_darkens_the_stroke(self) -> None:
        image = split_image((30, 30, 30), (225, 225, 225))

        light = PencilSketchStep(PencilSketchConfig(strength=0.5)).apply(image)
        heavy = PencilSketchStep(PencilSketchConfig(strength=2.0)).apply(image)

        assert heavy.rgb.mean() < light.rgb.mean()

    def test_the_result_is_monochrome(self) -> None:
        result = PencilSketchStep(PencilSketchConfig()).apply(
            split_image((200, 30, 30), (30, 30, 200))
        )

        rgb = result.rgb
        assert np.array_equal(rgb[:, :, 0], rgb[:, :, 1])
        assert np.array_equal(rgb[:, :, 1], rgb[:, :, 2])

    def test_the_alpha_is_unchanged(self) -> None:
        image = solid_image((100, 110, 120), alpha=91)

        assert np.all(PencilSketchStep(PencilSketchConfig()).apply(image).alpha == 91)


class TestCartoon:
    """The cartoon effect must flatten the colours and mark the contours."""

    def test_it_reduces_the_number_of_colours(self) -> None:
        image = noisy_image()

        result = CartoonStep(CartoonConfig(color_levels=6)).apply(image)

        colors_before = len(np.unique(image.rgb.reshape(-1, 3), axis=0))
        colors_after = len(np.unique(result.rgb.reshape(-1, 3), axis=0))
        assert colors_after < colors_before

    def test_it_traces_black_contours_on_an_edge(self) -> None:
        image = split_image((40, 40, 200), (220, 220, 60))

        result = CartoonStep(CartoonConfig()).apply(image)

        assert result.rgb.min() < 40

    def test_a_solid_colour_gets_no_contours(self) -> None:
        result = CartoonStep(CartoonConfig()).apply(solid_image((150, 90, 60)))

        # With no edges to ink, the colour remains (up to the quantisation).
        assert result.rgb.min() > 20

    def test_the_dimensions_are_unchanged(self) -> None:
        image = noisy_image(width=40, height=24)

        result = CartoonStep(CartoonConfig()).apply(image)

        assert (result.width, result.height) == (40, 24)

    def test_the_alpha_is_unchanged(self) -> None:
        image = solid_image((150, 90, 60), alpha=17)

        assert np.all(CartoonStep(CartoonConfig()).apply(image).alpha == 17)


class TestVignette:
    """The vignette must darken the edges and leave the centre bright."""

    def test_the_corners_are_darker_than_the_centre(self) -> None:
        result = VignetteStep(VignetteConfig()).apply(gray_image(200, 64, 64))

        centre = int(result.rgb[32, 32, 0])
        corner = int(result.rgb[0, 0, 0])
        assert corner < centre

    def test_the_centre_stays_almost_intact(self) -> None:
        result = VignetteStep(VignetteConfig()).apply(gray_image(200, 64, 64))

        assert result.rgb[32, 32, 0] >= 195

    def test_a_zero_strength_changes_nothing(self) -> None:
        image = gray_image(200, 32, 32)

        result = VignetteStep(VignetteConfig(strength=0.0)).apply(image)

        assert np.array_equal(result.rgb, image.rgb)

    def test_raising_the_strength_darkens_more(self) -> None:
        image = gray_image(200, 64, 64)

        light = VignetteStep(VignetteConfig(strength=0.3)).apply(image)
        heavy = VignetteStep(VignetteConfig(strength=0.9)).apply(image)

        assert heavy.rgb[0, 0, 0] < light.rgb[0, 0, 0]

    def test_the_effect_is_symmetric(self) -> None:
        result = VignetteStep(VignetteConfig()).apply(gray_image(200, 64, 64))

        rgb = result.rgb[:, :, 0]
        assert rgb[0, 0] == rgb[0, -1] == rgb[-1, 0] == rgb[-1, -1]

    def test_the_alpha_is_unchanged(self) -> None:
        image = solid_image((200, 200, 200), alpha=123)

        assert np.all(VignetteStep(VignetteConfig()).apply(image).alpha == 123)
