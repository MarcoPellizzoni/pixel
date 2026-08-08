"""Colour steps: they reinterpret colour while leaving pixels where they are.

Single responsibility of this module: the transformations acting on the three
colour channels. All of them leave the alpha channel, that is the cut-out
silhouette, untouched.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
from typing import Annotated

import cv2
import numpy as np

from pixel.domain import RGBAImage, to_uint8

# Maximum value of an 8-bit channel.
MAX_CHANNEL_VALUE: int = 255


class LuminanceStandard(StrEnum):
    """Standard formulas for the perceived brightness of a colour.

    The human eye is not equally sensitive to the three primary colours: green
    weighs far more than blue. An arithmetic mean (R+G+B)/3 would give an
    unnatural grey, so a weighted mean defined by a standard is used instead.
    """

    # ITU-R BT.601: the analogue television standard, still used by OpenCV. It
    # renders warm tones brighter.
    BT601 = "bt601"

    # ITU-R BT.709: the HD/sRGB standard, the most accurate one for digital photos.
    BT709 = "bt709"


# Coefficients (red, green, blue) for each luminance standard.
# Each triple sums to 1.0, so the resulting grey stays within 0-255.
LUMINANCE_WEIGHTS: dict[LuminanceStandard, tuple[float, float, float]] = {
    LuminanceStandard.BT601: (0.299, 0.587, 0.114),
    LuminanceStandard.BT709: (0.2126, 0.7152, 0.0722),
}


def compute_luminance(rgb: np.ndarray, standard: LuminanceStandard) -> np.ndarray:
    """Compute perceived brightness as a weighted mean of R, G and B.

    A shared function: the greyscale step and the pen step both need the same
    notion of "how bright is this pixel", and they must use a single one to stay
    consistent with each other.

    Args:
        rgb: array of shape (height, width, 3) and dtype uint8.
        standard: the luminance formula to apply.

    Returns:
        A (height, width) uint8 array of grey levels.
    """
    red_weight, green_weight, blue_weight = LUMINANCE_WEIGHTS[standard]

    # Work in floating point so no precision is lost in the weighted sum; only
    # the final result goes back to 8 bits.
    rgb_float = rgb.astype(np.float32)

    luminance = (
        rgb_float[:, :, 0] * red_weight
        + rgb_float[:, :, 1] * green_weight
        + rgb_float[:, :, 2] * blue_weight
    )

    return to_uint8(luminance)


@dataclass(frozen=True)
class GrayscaleConfig:
    """Greyscale conversion parameters."""

    standard: Annotated[
        LuminanceStandard,
        "The luminance formula to apply.",
    ] = LuminanceStandard.BT709


class GrayscaleStep:
    """Converts the image to greyscale."""

    def __init__(self, config: GrayscaleConfig) -> None:
        """Prepare the step."""
        self._config = config

    @property
    def name(self) -> str:
        """Step name."""
        return "grayscale"

    def apply(self, image: RGBAImage) -> RGBAImage:
        """Replace the colours with their greyscale equivalent."""
        luminance = compute_luminance(image.rgb, self._config.standard)

        # A greyscale image is still technically RGB: replicate the same value
        # across the three channels. This way the data type does not change and
        # later steps do not have to handle two different formats.
        return image.with_rgb(np.dstack([luminance] * 3))


@dataclass(frozen=True)
class SepiaConfig:
    """Sepia toning parameters."""

    intensity: Annotated[
        float,
        "How strongly to apply the effect: 0.0 leaves the original, 1.0 tones "
        "fully.",
    ] = 1.0


# Sepia toning matrix, in the classic form used by photo editors. Each row says
# how to compose one output channel from the three input ones: the result is a
# monochrome image shifted towards the warm brown of old prints.
SEPIA_MATRIX: np.ndarray = np.array(
    [
        [0.393, 0.769, 0.189],
        [0.349, 0.686, 0.168],
        [0.272, 0.534, 0.131],
    ],
    dtype=np.float32,
)


class SepiaStep:
    """Tones the image towards the brown of vintage prints."""

    def __init__(self, config: SepiaConfig) -> None:
        """Prepare the step."""
        self._config = config

    @property
    def name(self) -> str:
        """Step name."""
        return "sepia"

    def apply(self, image: RGBAImage) -> RGBAImage:
        """Apply sepia toning."""
        original = image.rgb.astype(np.float32)

        # Row-wise product between each pixel and the transposed matrix: the
        # vectorised way of applying the same combination to every pixel.
        toned = original @ SEPIA_MATRIX.T

        # Intensity linearly blends the original and the toned version.
        intensity = float(np.clip(self._config.intensity, 0.0, 1.0))
        blended = original * (1.0 - intensity) + toned * intensity

        return image.with_rgb(to_uint8(blended))


@dataclass(frozen=True)
class InvertConfig:
    """Inversion has no parameters: it is the same operation for every image."""


class InvertStep:
    """Inverts the colours, like a film negative."""

    def __init__(self, config: InvertConfig) -> None:
        """Prepare the step."""
        self._config = config

    @property
    def name(self) -> str:
        """Step name."""
        return "invert"

    def apply(self, image: RGBAImage) -> RGBAImage:
        """Replace every value with its complement to 255."""
        # Colour channels only: inverting the alpha as well would swap the
        # subject with the background.
        return image.with_rgb(MAX_CHANNEL_VALUE - image.rgb)


@dataclass(frozen=True)
class SaturationConfig:
    """Saturation adjustment parameters."""

    amount: Annotated[
        float,
        "Multiplicative factor: 0.0 removes all colour, 1.0 leaves it "
        "unchanged, values above 1.0 make the hues more vivid.",
    ] = 1.5


class SaturationStep:
    """Makes the colours more vivid or more muted."""

    def __init__(self, config: SaturationConfig) -> None:
        """Prepare the step."""
        self._config = config

    @property
    def name(self) -> str:
        """Step name."""
        return "saturation"

    def apply(self, image: RGBAImage) -> RGBAImage:
        """Multiply every pixel's saturation."""
        # We move to HSV because there saturation is a channel of its own: in RGB
        # it would take a formula touching all three channels together.
        hsv = cv2.cvtColor(image.rgb, cv2.COLOR_RGB2HSV).astype(np.float32)

        hsv[:, :, 1] = np.clip(
            hsv[:, :, 1] * self._config.amount, 0, MAX_CHANNEL_VALUE
        )

        adjusted = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2RGB)

        return image.with_rgb(adjusted)


@dataclass(frozen=True)
class PosterizeConfig:
    """Posterisation parameters."""

    levels: Annotated[
        int,
        "How many colour levels to keep per channel. The fewer the levels, "
        "the more the image looks like a flat-colour screen print.",
    ] = 4


class PosterizeStep:
    """Reduces the image to a few flat colours, like a screen print."""

    def __init__(self, config: PosterizeConfig) -> None:
        """Prepare the step."""
        self._config = config

    @property
    def name(self) -> str:
        """Step name."""
        return "posterize"

    def apply(self, image: RGBAImage) -> RGBAImage:
        """Quantise each channel to the requested number of levels."""
        # With fewer than two levels the image would become a uniform rectangle.
        levels = max(2, self._config.levels)

        # Width of each step on the 0-255 scale.
        step_size = MAX_CHANNEL_VALUE / (levels - 1)

        original = image.rgb.astype(np.float32)

        # Round to the nearest step and return to the original scale, so black
        # stays black and white stays white.
        quantized = np.round(original / step_size) * step_size

        return image.with_rgb(to_uint8(quantized))
