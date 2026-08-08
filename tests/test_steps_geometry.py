"""Tests for the geometric steps."""

import numpy as np
from conftest import solid_image

from pixel.domain import RGBAImage
from pixel.steps.geometry import (
    CropConfig,
    CropStep,
    FitMode,
    FlipConfig,
    FlipStep,
    ResizeConfig,
    ResizeStep,
    RotateConfig,
    RotateStep,
)


class TestResize:
    """Resizing must honour the request and the aspect ratio."""

    def test_the_scale_factor_multiplies_both_sides(self) -> None:
        image = solid_image((10, 20, 30), width=100, height=50)

        result = ResizeStep(ResizeConfig(scale=0.5)).apply(image)

        assert (result.width, result.height) == (50, 25)

    def test_width_alone_preserves_the_aspect_ratio(self) -> None:
        image = solid_image((10, 20, 30), width=100, height=50)

        result = ResizeStep(ResizeConfig(width=60)).apply(image)

        assert (result.width, result.height) == (60, 30)

    def test_height_alone_preserves_the_aspect_ratio(self) -> None:
        image = solid_image((10, 20, 30), width=100, height=50)

        result = ResizeStep(ResizeConfig(height=100)).apply(image)

        assert (result.width, result.height) == (200, 100)

    def test_contain_fits_the_image_inside_the_box(self) -> None:
        image = solid_image((10, 20, 30), width=100, height=50)

        result = ResizeStep(
            ResizeConfig(width=40, height=40, fit=FitMode.CONTAIN)
        ).apply(image)

        assert result.width <= 40
        assert result.height <= 40

    def test_cover_fills_the_whole_box(self) -> None:
        image = solid_image((10, 20, 30), width=100, height=50)

        result = ResizeStep(ResizeConfig(width=40, height=40, fit=FitMode.COVER)).apply(
            image
        )

        assert result.width >= 40
        assert result.height >= 40

    def test_stretch_forces_the_exact_dimensions(self) -> None:
        image = solid_image((10, 20, 30), width=100, height=50)

        result = ResizeStep(
            ResizeConfig(width=33, height=77, fit=FitMode.STRETCH)
        ).apply(image)

        assert (result.width, result.height) == (33, 77)

    def test_with_no_request_the_image_does_not_change(self) -> None:
        image = solid_image((10, 20, 30), width=20, height=20)

        result = ResizeStep(ResizeConfig()).apply(image)

        assert np.array_equal(result.data, image.data)

    def test_it_never_goes_below_one_pixel(self) -> None:
        image = solid_image((10, 20, 30), width=10, height=10)

        result = ResizeStep(ResizeConfig(scale=0.001)).apply(image)

        assert result.width >= 1
        assert result.height >= 1

    def test_the_alpha_is_resized_along_with_the_colour(self) -> None:
        image = solid_image((10, 20, 30), width=40, height=40, alpha=128)

        result = ResizeStep(ResizeConfig(scale=0.5)).apply(image)

        assert result.alpha.shape == (20, 20)
        assert abs(int(result.alpha.mean()) - 128) <= 1


class TestCrop:
    """Cropping must keep exactly the requested area."""

    def test_crops_the_given_rectangle(self) -> None:
        image = solid_image((10, 20, 30), width=50, height=40)

        result = CropStep(CropConfig(x=5, y=10, width=20, height=15)).apply(image)

        assert (result.width, result.height) == (20, 15)

    def test_zero_dimensions_reach_the_edge(self) -> None:
        image = solid_image((10, 20, 30), width=50, height=40)

        result = CropStep(CropConfig(x=10, y=10)).apply(image)

        assert (result.width, result.height) == (40, 30)

    def test_an_oversized_rectangle_is_clamped(self) -> None:
        image = solid_image((10, 20, 30), width=50, height=40)

        result = CropStep(CropConfig(x=0, y=0, width=999, height=999)).apply(image)

        assert (result.width, result.height) == (50, 40)

    def test_it_keeps_the_right_pixels(self) -> None:
        data = np.zeros((10, 10, 4), dtype=np.uint8)
        data[:, :, 3] = 255
        data[3, 4, :3] = (111, 112, 113)
        image = RGBAImage(data)

        result = CropStep(CropConfig(x=4, y=3, width=1, height=1)).apply(image)

        assert list(result.rgb[0, 0]) == [111, 112, 113]

    def test_the_crop_is_independent_of_the_original(self) -> None:
        image = solid_image((10, 20, 30), width=20, height=20)

        result = CropStep(CropConfig(x=0, y=0, width=5, height=5)).apply(image)
        result.data[:] = 0

        assert image.rgb[0, 0, 0] == 10


class TestRotate:
    """Rotation must widen the canvas and not invent opaque pixels."""

    def test_a_right_angle_swaps_the_sides(self) -> None:
        image = solid_image((10, 20, 30), width=40, height=20)

        result = RotateStep(RotateConfig(degrees=90.0)).apply(image)

        assert (result.width, result.height) == (20, 40)

    def test_without_expansion_the_dimensions_stay(self) -> None:
        image = solid_image((10, 20, 30), width=40, height=20)

        result = RotateStep(RotateConfig(degrees=90.0, expand=False)).apply(image)

        assert (result.width, result.height) == (40, 20)

    def test_a_slanted_rotation_widens_the_canvas(self) -> None:
        image = solid_image((10, 20, 30), width=40, height=40)

        result = RotateStep(RotateConfig(degrees=45.0)).apply(image)

        assert result.width > 40

    def test_the_empty_corners_stay_transparent(self) -> None:
        image = solid_image((200, 200, 200), width=40, height=40)

        result = RotateStep(RotateConfig(degrees=45.0)).apply(image)

        # The top-left corner of the widened canvas is not covered by any
        # original pixel.
        assert result.alpha[0, 0] == 0

    def test_four_right_angles_come_back_to_the_start(self) -> None:
        image = solid_image((10, 20, 30), width=30, height=20)
        step = RotateStep(RotateConfig(degrees=90.0))

        result = image
        for _ in range(4):
            result = step.apply(result)

        assert (result.width, result.height) == (30, 20)


class TestFlip:
    """Flipping must invert the requested axis and only that one."""

    def _image_with_a_mark(self) -> RGBAImage:
        """A black image with one white pixel in the top-left corner."""
        data = np.zeros((10, 10, 4), dtype=np.uint8)
        data[:, :, 3] = 255
        data[0, 0, :3] = 255
        return RGBAImage(data)

    def test_horizontal_moves_the_mark_to_the_right(self) -> None:
        result = FlipStep(FlipConfig(horizontal=True)).apply(self._image_with_a_mark())

        assert result.rgb[0, 9, 0] == 255
        assert result.rgb[0, 0, 0] == 0

    def test_vertical_moves_the_mark_to_the_bottom(self) -> None:
        result = FlipStep(FlipConfig(horizontal=False, vertical=True)).apply(
            self._image_with_a_mark()
        )

        assert result.rgb[9, 0, 0] == 255

    def test_both_axes_move_the_mark_to_the_opposite_corner(self) -> None:
        result = FlipStep(FlipConfig(horizontal=True, vertical=True)).apply(
            self._image_with_a_mark()
        )

        assert result.rgb[9, 9, 0] == 255

    def test_no_axis_leaves_the_image_unchanged(self) -> None:
        image = self._image_with_a_mark()

        result = FlipStep(FlipConfig(horizontal=False, vertical=False)).apply(image)

        assert np.array_equal(result.data, image.data)

    def test_two_identical_flips_cancel_out(self) -> None:
        image = self._image_with_a_mark()
        step = FlipStep(FlipConfig(horizontal=True))

        assert np.array_equal(step.apply(step.apply(image)).data, image.data)
