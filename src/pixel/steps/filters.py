"""Spatial filtering steps: every pixel is recomputed from its neighbours.

Single responsibility of this module: the convolutions and filters that blur,
sharpen, clean up or highlight the image's local variations.
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


@dataclass(frozen=True)
class BlurConfig:
    """Blur parameters."""

    radius: Annotated[
        float,
        "Radius of the Gaussian blur in pixels: the larger it is, the more "
        "detail the image loses.",
    ] = 3.0


class BlurStep:
    """Blurs the image with a Gaussian filter."""

    def __init__(self, config: BlurConfig) -> None:
        """Prepare the step."""
        self._config = config

    @property
    def name(self) -> str:
        """Step name."""
        return "blur"

    def apply(self, image: RGBAImage) -> RGBAImage:
        """Blur the colour channels."""
        if self._config.radius <= 0:
            # Zero radius: no blur.
            return image

        # `ksize=(0, 0)` lets OpenCV derive the kernel size from the standard
        # deviation, which is the parameter with a visual meaning.
        blurred = cv2.GaussianBlur(
            image.rgb, ksize=(0, 0), sigmaX=self._config.radius
        )

        return image.with_rgb(blurred)


@dataclass(frozen=True)
class SharpenConfig:
    """Detail enhancement parameters."""

    amount: Annotated[
        float,
        "How much to sharpen: 0.0 changes nothing, 1.0 is a decisive "
        "sharpening.",
    ] = 1.0

    radius: Annotated[
        float,
        "Radius of the comparison blur: small radii sharpen fine detail, "
        "large radii the contrast between broad areas.",
    ] = 2.0


class SharpenStep:
    """Enhances detail with an unsharp mask.

    The name comes from the darkroom: the image is compared with a blurred copy
    of itself, and the difference is amplified. Where the two frames agree
    (uniform areas) nothing changes; where they differ (the edges) the contrast
    increases and the detail looks sharper.
    """

    def __init__(self, config: SharpenConfig) -> None:
        """Prepare the step."""
        self._config = config

    @property
    def name(self) -> str:
        """Step name."""
        return "sharpen"

    def apply(self, image: RGBAImage) -> RGBAImage:
        """Sharpen the image's edges."""
        if self._config.amount <= 0 or self._config.radius <= 0:
            # Sharpening disabled.
            return image

        original = image.rgb.astype(np.float32)
        blurred = cv2.GaussianBlur(
            original, ksize=(0, 0), sigmaX=self._config.radius
        )

        # original + amount * (original - blurred): the difference holds only the
        # detail the blur destroyed, and it is put back amplified.
        sharpened = original + self._config.amount * (original - blurred)

        return image.with_rgb(to_uint8(sharpened))


class DenoiseMethod(StrEnum):
    """How to attenuate noise."""

    # Bilateral filter: averages only neighbours of similar colour, so it smooths
    # surfaces while keeping edges crisp. Fast, ideal before looking for contours.
    BILATERAL = "bilateral"

    # Non-Local Means: compares whole neighbourhoods scattered across the image
    # and averages the ones that resemble each other. Far slower, but it
    # preserves fine texture better.
    NON_LOCAL_MEANS = "nlmeans"


@dataclass(frozen=True)
class DenoiseConfig:
    """Noise attenuation parameters."""

    method: Annotated[
        DenoiseMethod,
        "Which algorithm to use.",
    ] = DenoiseMethod.BILATERAL

    strength: Annotated[
        float,
        "Strength of the attenuation. In the bilateral filter it is how far "
        "two colours may differ and still be averaged; in Non-Local Means it "
        "is the filter's overall intensity.",
    ] = 25.0

    radius: Annotated[
        int,
        "Reach of the bilateral filter, in pixels.",
    ] = 7


class DenoiseStep:
    """Attenuates noise while keeping contours crisp."""

    def __init__(self, config: DenoiseConfig) -> None:
        """Prepare the step."""
        self._config = config

    @property
    def name(self) -> str:
        """Step name."""
        return "denoise"

    def apply(self, image: RGBAImage) -> RGBAImage:
        """Clean the noise out of the colour channels."""
        if self._config.method is DenoiseMethod.NON_LOCAL_MEANS:
            denoised = cv2.fastNlMeansDenoisingColored(
                image.rgb,
                None,
                h=self._config.strength,
                hColor=self._config.strength,
                templateWindowSize=7,
                searchWindowSize=21,
            )
        else:
            denoised = cv2.bilateralFilter(
                image.rgb,
                d=max(1, self._config.radius),
                sigmaColor=self._config.strength,
                # The spatial radius is tied to the strength: this keeps a single
                # knob to turn, and the two effects stay in proportion.
                sigmaSpace=self._config.strength,
            )

        return image.with_rgb(denoised)


@dataclass(frozen=True)
class EdgesConfig:
    """Edge detection parameters."""

    low_threshold: Annotated[
        int,
        "Canny's low threshold: below this gradient a pixel is never an edge.",
    ] = 80

    high_threshold: Annotated[
        int,
        "High threshold: above this gradient a pixel is certainly an edge. "
        "Between the two, only what touches a certain edge survives, and that "
        "is where Canny's line continuity comes from.",
    ] = 180

    on_white: Annotated[
        bool,
        "When enabled, draw black lines on white rather than white on black.",
    ] = True


class EdgesStep:
    """Reduces the image to its contours alone, using Canny's algorithm."""

    def __init__(self, config: EdgesConfig) -> None:
        """Prepare the step."""
        self._config = config

    @property
    def name(self) -> str:
        """Step name."""
        return "edges"

    def apply(self, image: RGBAImage) -> RGBAImage:
        """Replace the image with a map of its contours."""
        luminance = cv2.cvtColor(image.rgb, cv2.COLOR_RGB2GRAY)

        # Canny returns white contours (255) on a black background (0).
        edges: np.ndarray = cv2.Canny(
            luminance,
            threshold1=self._config.low_threshold,
            threshold2=self._config.high_threshold,
        )

        if self._config.on_white:
            # Inverted is far more readable: dark lines on light paper.
            edges = MAX_CHANNEL_VALUE - edges

        return image.with_rgb(np.dstack([edges] * 3))
