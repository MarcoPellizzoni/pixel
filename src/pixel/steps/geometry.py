"""Geometric steps: they change the image's shape and size.

Single responsibility of this module: the transformations that move pixels
around without interpreting their colour.

These are the only steps that alter the image's dimensions, and for that reason
they work on all four channels at once: colour and opacity have to move
together, otherwise the cut-out silhouette would come unstuck from the subject.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
from typing import Annotated

import cv2
import numpy as np

from pixel.domain import RGBAImage


class FitMode(StrEnum):
    """How to interpret the requested width/height pair."""

    # Shrink until the whole image fits inside the box: the aspect ratio is
    # preserved and margins may be left over.
    CONTAIN = "contain"

    # Enlarge until the image covers the whole box: the aspect ratio stays
    # correct but some of the image falls outside the edges.
    COVER = "cover"

    # Force exactly the requested dimensions, distorting the image.
    STRETCH = "stretch"


@dataclass(frozen=True)
class ResizeConfig:
    """Resizing parameters."""

    width: Annotated[
        int,
        "Requested width in pixels. 0 means 'not specified': give width alone "
        "and the height follows, preserving the aspect ratio.",
    ] = 0
    height: Annotated[
        int,
        "Requested height in pixels. 0 means 'not specified': give height "
        "alone and the width follows, preserving the aspect ratio.",
    ] = 0

    scale: Annotated[
        float,
        "Alternative scale factor: 0.5 halves, 2.0 doubles. If greater than "
        "zero it takes precedence over width and height.",
    ] = 0.0

    fit: Annotated[
        FitMode,
        "How to fit the image when both dimensions are requested.",
    ] = FitMode.CONTAIN


class ResizeStep:
    """Resizes the image."""

    def __init__(self, config: ResizeConfig) -> None:
        """Prepare the step."""
        self._config = config

    @property
    def name(self) -> str:
        """Step name."""
        return "resize"

    def apply(self, image: RGBAImage) -> RGBAImage:
        """Resize the image according to the configuration."""
        target_width, target_height = self._compute_target_size(
            image.width, image.height
        )

        if (target_width, target_height) == (image.width, image.height):
            # No change requested: skip a pointless interpolation, which would
            # slightly degrade the image anyway.
            return image

        # The right interpolation depends on the direction: when shrinking,
        # INTER_AREA averages the discarded pixels and avoids aliasing; when
        # enlarging, INTER_CUBIC gives smoother transitions than plain linear
        # interpolation.
        is_shrinking = target_width * target_height < image.width * image.height
        interpolation = cv2.INTER_AREA if is_shrinking else cv2.INTER_CUBIC

        resized = cv2.resize(
            image.data,
            (target_width, target_height),
            interpolation=interpolation,
        )

        return RGBAImage(resized)

    def _compute_target_size(self, width: int, height: int) -> tuple[int, int]:
        """Work out the final dimensions from the configuration.

        Args:
            width: current width in pixels.
            height: current height in pixels.

        Returns:
            The requested (width, height) pair, never below 1 pixel.
        """
        config = self._config

        if config.scale > 0:
            # The scale factor is the most explicit request: it wins over the rest.
            return max(1, round(width * config.scale)), max(
                1, round(height * config.scale)
            )

        has_width = config.width > 0
        has_height = config.height > 0

        if not has_width and not has_height:
            # Nothing requested: the image stays as it is.
            return width, height

        if has_width and not has_height:
            # Only one dimension: the other follows the original aspect ratio.
            return config.width, max(1, round(height * config.width / width))

        if has_height and not has_width:
            return max(1, round(width * config.height / height)), config.height

        if config.fit is FitMode.STRETCH:
            # Distortion allowed: take the dimensions exactly as given.
            return config.width, config.height

        # CONTAIN picks the smaller factor (the whole image fits in the box),
        # COVER the larger one (the box gets covered).
        width_ratio = config.width / width
        height_ratio = config.height / height
        ratio = (
            min(width_ratio, height_ratio)
            if config.fit is FitMode.CONTAIN
            else max(width_ratio, height_ratio)
        )

        return max(1, round(width * ratio)), max(1, round(height * ratio))


@dataclass(frozen=True)
class CropConfig:
    """Rectangular crop parameters."""

    x: Annotated[
        int,
        "Left edge of the rectangle to keep, in pixels from the left border.",
    ] = 0
    y: Annotated[
        int,
        "Top edge of the rectangle to keep, in pixels from the top border.",
    ] = 0

    width: Annotated[
        int,
        "Width of the rectangle to keep. 0 means 'as far as the right border'.",
    ] = 0
    height: Annotated[
        int,
        "Height of the rectangle to keep. 0 means 'as far as the bottom border'.",
    ] = 0


class CropStep:
    """Crops a rectangular region of the image."""

    def __init__(self, config: CropConfig) -> None:
        """Prepare the step."""
        self._config = config

    @property
    def name(self) -> str:
        """Step name."""
        return "crop"

    def apply(self, image: RGBAImage) -> RGBAImage:
        """Keep only the given rectangle."""
        config = self._config

        # The rectangle is clamped to the image bounds rather than raising an
        # error: asking for a crop larger than the original is a common case and
        # the intent ("keep that area") remains clear.
        left = min(max(config.x, 0), image.width - 1)
        top = min(max(config.y, 0), image.height - 1)

        right = (
            image.width if config.width <= 0 else min(left + config.width, image.width)
        )
        bottom = (
            image.height
            if config.height <= 0
            else min(top + config.height, image.height)
        )

        # `copy()` detaches the crop from the original array: without it the
        # result would stay a view onto it, and the source image would no longer
        # be independent.
        return RGBAImage(image.data[top:bottom, left:right].copy())


@dataclass(frozen=True)
class RotateConfig:
    """Rotation parameters."""

    degrees: Annotated[
        float,
        "Angle in degrees, counter-clockwise.",
    ] = 90.0

    expand: Annotated[
        bool,
        "When enabled the canvas is widened so the rotated image fits "
        "entirely; otherwise the corners that fall outside the frame are cut "
        "off.",
    ] = True


class RotateStep:
    """Rotates the image by an arbitrary angle."""

    def __init__(self, config: RotateConfig) -> None:
        """Prepare the step."""
        self._config = config

    @property
    def name(self) -> str:
        """Step name."""
        return "rotate"

    def apply(self, image: RGBAImage) -> RGBAImage:
        """Rotate the image about its own centre."""
        center = (image.width / 2.0, image.height / 2.0)

        # 2x3 rotation matrix: OpenCV measures the angle counter-clockwise.
        matrix = cv2.getRotationMatrix2D(center, self._config.degrees, scale=1.0)

        output_width, output_height = image.width, image.height

        if self._config.expand:
            output_width, output_height = self._compute_expanded_size(matrix, image)

            # Widening the canvas moves the centre: the matrix's last column
            # holds the translation, which has to be corrected accordingly.
            matrix[0, 2] += (output_width - image.width) / 2.0
            matrix[1, 2] += (output_height - image.height) / 2.0

        # The areas of canvas no original pixel reaches stay fully transparent,
        # not black.
        rotated = cv2.warpAffine(
            image.data,
            matrix,
            (output_width, output_height),
            flags=cv2.INTER_CUBIC,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=(0, 0, 0, 0),
        )

        return RGBAImage(rotated)

    @staticmethod
    def _compute_expanded_size(
        matrix: np.ndarray, image: RGBAImage
    ) -> tuple[int, int]:
        """Work out the canvas needed to hold the rotated image.

        Args:
            matrix: the 2x3 rotation matrix.
            image: the image to rotate.

        Returns:
            The (width, height) pair of the widened canvas.
        """
        # The rotated bounding box is as large as the projection of the original
        # sides onto the two axes: the absolute sine and cosine are enough.
        absolute_cosine = abs(matrix[0, 0])
        absolute_sine = abs(matrix[0, 1])

        width = int(image.height * absolute_sine + image.width * absolute_cosine)
        height = int(image.height * absolute_cosine + image.width * absolute_sine)

        return width, height


@dataclass(frozen=True)
class FlipConfig:
    """Flip parameters."""

    horizontal: Annotated[
        bool,
        "Flip left to right (mirror effect).",
    ] = True

    vertical: Annotated[
        bool,
        "Flip top to bottom.",
    ] = False


class FlipStep:
    """Flips the image as in a mirror."""

    def __init__(self, config: FlipConfig) -> None:
        """Prepare the step."""
        self._config = config

    @property
    def name(self) -> str:
        """Step name."""
        return "flip"

    def apply(self, image: RGBAImage) -> RGBAImage:
        """Flip the image about the requested axes."""
        config = self._config

        if not config.horizontal and not config.vertical:
            # No axis requested: nothing to do.
            return image

        # `cv2.flip` codes: 1 = horizontal, 0 = vertical, -1 = both.
        if config.horizontal and config.vertical:
            flip_code = -1
        elif config.horizontal:
            flip_code = 1
        else:
            flip_code = 0

        return RGBAImage(cv2.flip(image.data, flip_code))
