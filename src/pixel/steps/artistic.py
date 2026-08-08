"""Artistic steps: stylisations that imitate a manual technique.

Single responsibility of this module: the effects that do not correct the image
but reinterpret it, the way someone redrawing it would.

The pen effect, far more elaborate than the others, has a module of its own
(`pen_sketch`).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Annotated

import cv2
import numpy as np

from pixel.domain import RGBAImage, to_uint8

# Maximum value of an 8-bit channel.
MAX_CHANNEL_VALUE: int = 255


@dataclass(frozen=True)
class PencilSketchConfig:
    """Pencil effect parameters."""

    radius: Annotated[
        float,
        "Blur radius: it governs how far the shading spreads around the "
        "edges, that is how soft the stroke is.",
    ] = 12.0

    strength: Annotated[
        float,
        "How pronounced the stroke is: above 1.0 the greys darken, below 1.0 "
        "the drawing stays lighter.",
    ] = 1.0


class PencilSketchStep:
    """Turns the image into a pencil sketch.

    It uses the colour-dodge technique borrowed from the darkroom: the greyscale
    image is divided by the blurred negative of itself. Where the two values
    resemble each other the result saturates to white (the paper), and grey
    remains only where there was a change: the edges and the shadows.
    """

    def __init__(self, config: PencilSketchConfig) -> None:
        """Prepare the step."""
        self._config = config

    @property
    def name(self) -> str:
        """Step name."""
        return "pencil-sketch"

    def apply(self, image: RGBAImage) -> RGBAImage:
        """Render the image as a pencil sketch."""
        luminance = cv2.cvtColor(image.rgb, cv2.COLOR_RGB2GRAY).astype(np.float32)

        # Blurred negative: the "second frame" of the comparison.
        inverted = MAX_CHANNEL_VALUE - luminance
        blurred_inverted = cv2.GaussianBlur(
            inverted, ksize=(0, 0), sigmaX=max(self._config.radius, 0.1)
        )

        # The dodge division. The small term added to the denominator avoids a
        # division by zero where the blurred negative is exactly 255.
        denominator = MAX_CHANNEL_VALUE - blurred_inverted + 1e-3
        dodged = luminance * MAX_CHANNEL_VALUE / denominator

        # Strength acts on the distance from white: raising it darkens the
        # stroke without moving the white of the paper.
        strengthened = MAX_CHANNEL_VALUE - (
            MAX_CHANNEL_VALUE - np.clip(dodged, 0, MAX_CHANNEL_VALUE)
        ) * max(self._config.strength, 0.0)

        sketch = to_uint8(strengthened)

        return image.with_rgb(np.dstack([sketch] * 3))


@dataclass(frozen=True)
class CartoonConfig:
    """Cartoon effect parameters."""

    smoothing_passes: Annotated[
        int,
        "How many times to apply the bilateral filter. Repeating it with a "
        "small radius flattens surfaces far more than one wide pass, and it "
        "is faster too.",
    ] = 4

    smoothing_strength: Annotated[
        float,
        "How much to flatten the colour gradations on each pass.",
    ] = 45.0

    edge_block_size: Annotated[
        int,
        "Side of the window used to look for the contours to ink.",
    ] = 9

    edge_offset: Annotated[
        float,
        "Correction to the contour threshold: raising it leaves only the "
        "strongest lines, and the drawing gets simpler.",
    ] = 6.0

    color_levels: Annotated[
        int,
        "Colour levels per channel: flat colours are what distinguishes a "
        "cartoon from a merely smoothed photograph.",
    ] = 8


class CartoonStep:
    """Turns the image into a comic-book drawing.

    It combines the two hallmarks of animated drawing: flat colour fills
    (obtained by smoothing and quantising) and heavy black contours (obtained
    with an adaptive threshold).
    """

    def __init__(self, config: CartoonConfig) -> None:
        """Prepare the step."""
        self._config = config

    @property
    def name(self) -> str:
        """Step name."""
        return "cartoon"

    def apply(self, image: RGBAImage) -> RGBAImage:
        """Render the image as a frame of a cartoon."""
        flat_colors = self._flatten_colors(image.rgb)
        edge_mask = self._detect_edges(image.rgb)

        # The contour mask is 0 on the lines and 255 elsewhere: using it as a
        # multiplier blackens the lines and leaves everything else untouched.
        edge_ratio = (edge_mask.astype(np.float32) / MAX_CHANNEL_VALUE)[
            :, :, np.newaxis
        ]
        inked = flat_colors.astype(np.float32) * edge_ratio

        return image.with_rgb(to_uint8(inked))

    def _flatten_colors(self, rgb: np.ndarray) -> np.ndarray:
        """Smooth the surfaces and reduce the colours to a few flat tones."""
        smoothed = rgb

        # Several light passes instead of one heavy pass: the result looks more
        # "drawn" and the computational cost stays low.
        for _ in range(max(self._config.smoothing_passes, 1)):
            smoothed = cv2.bilateralFilter(
                smoothed,
                d=9,
                sigmaColor=self._config.smoothing_strength,
                sigmaSpace=self._config.smoothing_strength,
            )

        levels = max(2, self._config.color_levels)
        step_size = MAX_CHANNEL_VALUE / (levels - 1)
        quantized = np.round(smoothed.astype(np.float32) / step_size) * step_size

        return to_uint8(quantized)

    def _detect_edges(self, rgb: np.ndarray) -> np.ndarray:
        """Find the contours to ink.

        Returns:
            A (height, width) uint8 mask: 0 on the lines, 255 elsewhere.
        """
        luminance = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)

        # The median filter removes salt-and-pepper noise before the threshold:
        # without it, every speck of noise would turn into a black dot.
        cleaned = cv2.medianBlur(luminance, 7)

        block_size = max(3, self._config.edge_block_size)
        if block_size % 2 == 0:
            block_size += 1

        # The adaptive threshold follows the changes in lighting, so it finds the
        # contours in shadowed areas too, where a fixed threshold would fail.
        return cv2.adaptiveThreshold(
            cleaned,
            MAX_CHANNEL_VALUE,
            cv2.ADAPTIVE_THRESH_MEAN_C,
            cv2.THRESH_BINARY,
            block_size,
            self._config.edge_offset,
        )


@dataclass(frozen=True)
class VignetteConfig:
    """Vignetting parameters."""

    strength: Annotated[
        float,
        "How much to darken the corners: 0.0 does nothing, 1.0 takes them to "
        "black.",
    ] = 0.6

    radius: Annotated[
        float,
        "Radius of the area that stays bright, as a fraction of the half- "
        "diagonal. Small values tighten the halo around the centre.",
    ] = 0.8

    softness: Annotated[
        float,
        "Softness of the transition between centre and edges: high values "
        "make it gradual, low values create an almost sharp boundary.",
    ] = 2.0


class VignetteStep:
    """Progressively darkens the edges, like an old lens."""

    def __init__(self, config: VignetteConfig) -> None:
        """Prepare the step."""
        self._config = config

    @property
    def name(self) -> str:
        """Step name."""
        return "vignette"

    def apply(self, image: RGBAImage) -> RGBAImage:
        """Apply the vignette to the colour channels."""
        brightness_field = self._build_brightness_field(image.height, image.width)

        # The brightness field has a single colour axis, so NumPy broadcasts it
        # automatically across the three channels.
        darkened = image.rgb.astype(np.float32) * brightness_field[:, :, np.newaxis]

        return image.with_rgb(to_uint8(darkened))

    def _build_brightness_field(self, height: int, width: int) -> np.ndarray:
        """Build the map of how bright each pixel remains.

        Args:
            height: image height in pixels.
            width: image width in pixels.

        Returns:
            A float32 (height, width) array of factors between 0.0 and 1.0.
        """
        # Normalised coordinates: 0 at the centre, 1 at the short edges.
        rows = np.linspace(-1.0, 1.0, height, dtype=np.float32)[:, np.newaxis]
        columns = np.linspace(-1.0, 1.0, width, dtype=np.float32)[np.newaxis, :]

        # Distance from the centre, relative to the requested radius: beyond 1.0
        # we are outside the area meant to stay bright.
        radius = max(self._config.radius, 1e-3)
        distance = np.sqrt(rows**2 + columns**2) / radius

        # The exponent governs how abruptly we go from the centre to the edges.
        falloff = np.power(
            np.clip(distance, 0.0, None), max(self._config.softness, 0.1)
        )

        strength = float(np.clip(self._config.strength, 0.0, 1.0))
        brightness = 1.0 - strength * np.clip(falloff, 0.0, 1.0)

        return np.clip(brightness, 0.0, 1.0).astype(np.float32)
