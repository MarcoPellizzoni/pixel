"""Tests for the pen rendering step."""

import numpy as np

from pixel.domain import RGBAImage
from pixel.steps.pen_sketch import PenSketchConfig, PenSketchStep

# Hatching must be turned off when checking the behaviour of the stroke alone,
# otherwise it would add lines XDoG did not produce.
WITHOUT_HATCHING = PenSketchConfig(hatching=False)


def grey_square(level: int, side: int = 64) -> RGBAImage:
    """Create an opaque square image of a single grey level."""
    data = np.zeros((side, side, 4), dtype=np.uint8)
    data[:, :, :3] = level
    data[:, :, 3] = 255
    return RGBAImage(data)


def image_with_an_edge(side: int = 64) -> RGBAImage:
    """Create an image split in half: light on the left, dark on the right."""
    data = np.zeros((side, side, 4), dtype=np.uint8)
    data[:, : side // 2, :3] = 230
    data[:, side // 2 :, :3] = 25
    data[:, :, 3] = 255
    return RGBAImage(data)


class TestUniformSurfaces:
    """Where there is nothing to draw, the sheet must stay white."""

    def test_a_solid_colour_produces_no_ink(self) -> None:
        result = PenSketchStep(WITHOUT_HATCHING).apply(grey_square(140))

        assert np.all(result.rgb == 255)

    def test_even_a_dark_solid_colour_stays_white(self) -> None:
        # Checks that the base tone is not mistaken for a contour: that is the
        # difference between a line drawing and a blackened silhouette.
        result = PenSketchStep(WITHOUT_HATCHING).apply(grey_square(20))

        assert np.all(result.rgb == 255)


class TestContours:
    """A crisp edge must become a pen stroke."""

    def test_an_edge_produces_ink(self) -> None:
        result = PenSketchStep(WITHOUT_HATCHING).apply(image_with_an_edge())

        assert result.rgb.min() < 128

    def test_the_ink_gathers_on_the_edge(self) -> None:
        side = 64
        result = PenSketchStep(WITHOUT_HATCHING).apply(image_with_an_edge(side))

        grey = result.rgb[:, :, 0]
        # A narrow band around the dividing line.
        on_the_edge = grey[:, side // 2 - 3 : side // 2 + 3]
        # An area far from the edge, where the colour is uniform.
        far_away = grey[:, : side // 4]

        assert on_the_edge.min() < far_away.min()

    def test_a_higher_threshold_leaves_more_white(self) -> None:
        image = image_with_an_edge()

        sparse = PenSketchStep(
            PenSketchConfig(ink_threshold=0.85, hatching=False)
        ).apply(image)
        dense = PenSketchStep(
            PenSketchConfig(ink_threshold=0.25, hatching=False)
        ).apply(image)

        assert sparse.rgb.mean() > dense.rgb.mean()


class TestHatching:
    """Hatching must darken the shadows, and only those."""

    def test_hatching_darkens_a_shadowed_area(self) -> None:
        # A uniform dark grey: without hatching it would stay a white sheet.
        image = grey_square(20)

        without = PenSketchStep(WITHOUT_HATCHING).apply(image)
        with_hatching = PenSketchStep(PenSketchConfig(hatching=True)).apply(image)

        assert with_hatching.rgb.mean() < without.rgb.mean()

    def test_hatching_does_not_touch_the_light_areas(self) -> None:
        # Well above the default shadow threshold (95).
        image = grey_square(240)

        result = PenSketchStep(PenSketchConfig(hatching=True)).apply(image)

        assert np.all(result.rgb == 255)

    def test_hatching_leaves_white_gaps_between_the_lines(self) -> None:
        # Hatching is made of separate lines: if it blackened everything it would
        # be a solid fill, not a pen drawing.
        result = PenSketchStep(PenSketchConfig(hatching=True)).apply(grey_square(10))

        grey = result.rgb[:, :, 0]
        assert grey.max() == 255
        assert grey.min() < 255


class TestInvariants:
    """Properties the step must honour in every case."""

    def test_the_alpha_is_unchanged(self) -> None:
        data = np.zeros((32, 32, 4), dtype=np.uint8)
        data[:, :, :3] = 120
        data[:, :, 3] = 42
        image = RGBAImage(data)

        result = PenSketchStep(PenSketchConfig()).apply(image)

        assert np.all(result.alpha == 42)

    def test_the_result_is_monochrome(self) -> None:
        result = PenSketchStep(PenSketchConfig()).apply(image_with_an_edge())

        rgb = result.rgb
        assert np.array_equal(rgb[:, :, 0], rgb[:, :, 1])
        assert np.array_equal(rgb[:, :, 1], rgb[:, :, 2])

    def test_the_dimensions_are_unchanged(self) -> None:
        image = image_with_an_edge(side=48)

        result = PenSketchStep(PenSketchConfig()).apply(image)

        assert (result.height, result.width) == (48, 48)
