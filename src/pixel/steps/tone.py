"""Tonal steps: they redistribute brightness without changing the hues.

Single responsibility of this module: the corrections acting on the tone curve
(how bright a pixel is), not on the colour itself.

Where brightness and colour need to be kept apart, the code moves to the Lab
colour space, whose L channel holds perceived brightness alone: correcting it
there leaves the hues exactly where they were, whereas acting on the three RGB
channels would shift them.
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

# Every possible value of an 8-bit channel: the domain over which the lookup
# tables (LUTs) are built.
CHANNEL_DOMAIN: np.ndarray = np.arange(256, dtype=np.float32)


@dataclass(frozen=True)
class BrightnessContrastConfig:
    """Brightness and contrast adjustment parameters."""

    brightness: Annotated[
        float,
        "Brightness shift, from -1.0 (full black) to +1.0 (full white).",
    ] = 0.0

    contrast: Annotated[
        float,
        "Contrast factor about mid grey: 1.0 leaves it unchanged, above 1.0 "
        "widens the distance between lights and darks, below 1.0 narrows it.",
    ] = 1.0


class BrightnessContrastStep:
    """Adjusts brightness and contrast."""

    def __init__(self, config: BrightnessContrastConfig) -> None:
        """Prepare the step."""
        self._config = config

    @property
    def name(self) -> str:
        """Step name."""
        return "brightness-contrast"

    def apply(self, image: RGBAImage) -> RGBAImage:
        """Apply the correction to every colour channel."""
        # Contrast pivots the values about mid grey, so a well-exposed image
        # stays centred instead of getting brighter or darker as a whole.
        middle_gray = MAX_CHANNEL_VALUE / 2.0
        brightness_offset = self._config.brightness * MAX_CHANNEL_VALUE

        # Precompute the conversion for all 256 possible values and then apply it
        # with a single table lookup: much faster than repeating the arithmetic
        # on every pixel.
        lookup_table = (
            (CHANNEL_DOMAIN - middle_gray) * self._config.contrast
            + middle_gray
            + brightness_offset
        )

        return image.with_rgb(cv2.LUT(image.rgb, to_uint8(lookup_table)))


@dataclass(frozen=True)
class GammaConfig:
    """Gamma correction parameters."""

    gamma: Annotated[
        float,
        "Exponent of the curve: below 1.0 it lifts the shadows, above 1.0 it "
        "deepens them. Unlike brightness, it leaves black and white almost "
        "untouched.",
    ] = 1.0


class GammaStep:
    """Lifts or deepens the midtones with a gamma curve."""

    def __init__(self, config: GammaConfig) -> None:
        """Prepare the step."""
        self._config = config

    @property
    def name(self) -> str:
        """Step name."""
        return "gamma"

    def apply(self, image: RGBAImage) -> RGBAImage:
        """Apply the gamma curve to the colour channels."""
        # A zero or negative gamma makes no mathematical sense: fall back to the
        # neutral value instead of producing infinities.
        gamma = self._config.gamma if self._config.gamma > 0 else 1.0

        # The curve is applied to values normalised to 0-1 and then scaled back
        # to 0-255: that is the very definition of gamma correction.
        normalized = CHANNEL_DOMAIN / MAX_CHANNEL_VALUE
        lookup_table = np.power(normalized, gamma) * MAX_CHANNEL_VALUE

        return image.with_rgb(cv2.LUT(image.rgb, to_uint8(lookup_table)))


@dataclass(frozen=True)
class AutoContrastConfig:
    """Parameters of the local contrast equalisation (CLAHE)."""

    clip_limit: Annotated[
        float,
        "Histogram clip limit: the higher it is, the stronger the effect. It "
        "exists to stop uniform areas having nothing but their noise "
        "amplified.",
    ] = 2.0

    tile_size: Annotated[
        int,
        "Side in pixels of the tiles over which the local histogram is "
        "computed.",
    ] = 8


class AutoContrastStep:
    """Brings out detail hidden in the shadows and highlights.

    A global equalisation would brighten the whole image at once. CLAHE
    (Contrast Limited Adaptive Histogram Equalization) instead splits the image
    into tiles and redistributes the tones within each: exactly what is needed to
    rescue a photo shot in low light without blowing out the already-lit areas.
    """

    def __init__(self, config: AutoContrastConfig) -> None:
        """Prepare the step."""
        self._config = config

    @property
    def name(self) -> str:
        """Step name."""
        return "auto-contrast"

    def apply(self, image: RGBAImage) -> RGBAImage:
        """Equalise the local contrast of the brightness alone."""
        if self._config.clip_limit <= 0:
            # Equalisation disabled.
            return image

        clahe = cv2.createCLAHE(
            clipLimit=self._config.clip_limit,
            tileGridSize=(self._config.tile_size, self._config.tile_size),
        )

        # Move to Lab to equalise brightness alone: applying CLAHE to the three
        # RGB channels separately would alter their ratios, and the colours would
        # shift.
        lab = cv2.cvtColor(image.rgb, cv2.COLOR_RGB2LAB)
        lightness, green_red, blue_yellow = cv2.split(lab)

        equalized_lab = cv2.merge([clahe.apply(lightness), green_red, blue_yellow])

        return image.with_rgb(cv2.cvtColor(equalized_lab, cv2.COLOR_LAB2RGB))


class ThresholdMethod(StrEnum):
    """How to pick the threshold separating white from black."""

    # A single threshold chosen by hand, the same across the whole image.
    FIXED = "fixed"

    # A single threshold computed by Otsu's method: it picks the value that best
    # separates the two populations in the histogram. Ideal with even lighting.
    OTSU = "otsu"

    # A threshold recomputed in every small window: the right choice when the
    # lighting varies from one area to another, as in a photo of a document.
    ADAPTIVE = "adaptive"


@dataclass(frozen=True)
class ThresholdConfig:
    """Binarisation parameters."""

    method: Annotated[
        ThresholdMethod,
        "How to determine the threshold.",
    ] = ThresholdMethod.OTSU

    level: Annotated[
        int,
        "Threshold to use with the `fixed` method (0-255).",
    ] = 128

    block_size: Annotated[
        int,
        "Side of the window used by the `adaptive` method; it must be odd.",
    ] = 31

    offset: Annotated[
        float,
        "Correction subtracted from the local mean in the `adaptive` method: "
        "raising it yields more white.",
    ] = 10.0

    invert: Annotated[
        bool,
        "Swap white and black in the result.",
    ] = False


class ThresholdStep:
    """Reduces the image to just two values: black and white."""

    def __init__(self, config: ThresholdConfig) -> None:
        """Prepare the step."""
        self._config = config

    @property
    def name(self) -> str:
        """Step name."""
        return "threshold"

    def apply(self, image: RGBAImage) -> RGBAImage:
        """Binarise the image using the configured method."""
        # Binarisation only makes sense on a single channel: if the image is in
        # colour we reason about its brightness.
        luminance = cv2.cvtColor(image.rgb, cv2.COLOR_RGB2GRAY)

        # The threshold type decides which side the above-threshold pixels end up on.
        threshold_type = (
            cv2.THRESH_BINARY_INV if self._config.invert else cv2.THRESH_BINARY
        )

        if self._config.method is ThresholdMethod.ADAPTIVE:
            binary = self._apply_adaptive(luminance, threshold_type)
        elif self._config.method is ThresholdMethod.OTSU:
            # Passing 0 as the threshold and adding the flag makes OpenCV compute
            # it by itself.
            _, binary = cv2.threshold(
                luminance, 0, MAX_CHANNEL_VALUE, threshold_type | cv2.THRESH_OTSU
            )
        else:
            _, binary = cv2.threshold(
                luminance, self._config.level, MAX_CHANNEL_VALUE, threshold_type
            )

        return image.with_rgb(np.dstack([binary] * 3))

    def _apply_adaptive(self, luminance: np.ndarray, threshold_type: int) -> np.ndarray:
        """Apply the adaptive threshold, with an always-valid window."""
        # OpenCV demands an odd window of at least 3 pixels.
        block_size = max(3, self._config.block_size)
        if block_size % 2 == 0:
            block_size += 1

        return cv2.adaptiveThreshold(
            luminance,
            MAX_CHANNEL_VALUE,
            # A Gaussian-weighted mean gives more regular edges than a plain
            # mean, which tends to produce jagged contours.
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            threshold_type,
            block_size,
            self._config.offset,
        )
