"""Pen drawing effect: ink contours and hatched shadows.

Single responsibility: turn an image into its "drawn in pen" equivalent, that is
strokes of black ink on white paper.

The underlying algorithm is XDoG (eXtended Difference of Gaussians,
Winnemoeller 2012), the reference technique for ink stylisation. The idea:

1. blur the image twice, with different radii;
2. the difference between the two blurs is nearly zero in uniform areas and
   large wherever there is an edge, because a wider blur smears the contrast
   more than a narrower one;
3. that difference, amplified and passed through a soft threshold, becomes the
   pen stroke.

On top of XDoG comes hatching of the shadows: a pen cannot produce greys, it
fills dark areas with more or less densely packed lines. In the deepest shadows
the lines cross (cross-hatching), exactly as in an ink drawing.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Annotated

import cv2
import numpy as np

from pixel.domain import RGBAImage, to_uint8
from pixel.steps.color import LuminanceStandard, compute_luminance

# Conversion factor between the 8-bit domain (0-255) and the normalised one
# (0.0-1.0) the formulas work in.
BYTE_RANGE: float = 255.0

# Percentile used as "the strongest edge in the image", to make the thresholds
# independent of the shot's contrast. The absolute maximum is not used because a
# single noisy pixel would then set the scale.
STRENGTH_PERCENTILE: float = 99.0

# Width of the blur used to judge where the shadows are. Shadows are a matter of
# overall shape, not of fine detail.
SHADOW_FIELD_SIGMA: float = 4.0


@dataclass(frozen=True)
class PenSketchConfig:
    """Pen drawing effect parameters."""

    # --- Noise attenuation, before looking for the contours ---

    bilateral_diameter: Annotated[
        int,
        "Diameter of the bilateral filter, which removes JPEG noise while "
        "keeping edges crisp. Without it, the pen effect would draw the "
        "compression artefacts as well.",
    ] = 9

    bilateral_sigma: Annotated[
        float,
        "How far two colours may differ and still be averaged together: high "
        "values flatten gradations into uniform fills.",
    ] = 60.0

    # --- XDoG core ---

    dog_sigma: Annotated[
        float,
        "Standard deviation of the 'narrow' Gaussian blur: it governs the "
        "stroke width. Small values give thin, nervous lines.",
    ] = 1.3

    dog_sigma_ratio: Annotated[
        float,
        "Ratio between the 'wide' blur and the 'narrow' one. 1.6 is the value "
        "that best approximates the Laplacian of Gaussian, standard in the "
        "literature.",
    ] = 1.6

    sharpness: Annotated[
        float,
        "Contour sharpening. It acts on edge strength already scaled against "
        "the image's strongest edge: raising it lets even weak edges (a fine "
        "texture) reach the strength of crisp ones, and the drawing gets "
        "denser. Useful values lie between 1 and 4; beyond that, dirt appears "
        "in the flat areas.",
    ] = 2.0

    ink_threshold: Annotated[
        float,
        "The threshold separating white paper from ink, on the 0.0-1.0 scale "
        "of edge strength. Raising it leaves more white, lowering it darkens "
        "the drawing.",
    ] = 0.55

    ink_softness: Annotated[
        float,
        "Steepness of the white/black transition around the threshold: high "
        "values give a decisive, almost binary stroke, like a ballpoint pen; "
        "low values give soft gradations, like a pencil.",
    ] = 10.0

    # --- Stroke clean-up ---

    despeckle_size: Annotated[
        int,
        "Side of the median filter window that removes isolated specks (it "
        "must be odd). 0 or 1 disables the clean-up.",
    ] = 3

    # --- Shadow hatching ---

    hatching: Annotated[
        bool,
        "When enabled, fills the shadowed areas with diagonal hatching, as a "
        "pen would, being unable to produce greys but only denser or sparser "
        "lines.",
    ] = True

    hatching_spacing: Annotated[
        int,
        "Distance in pixels between two hatching lines.",
    ] = 6

    hatching_line_width: Annotated[
        int,
        "Thickness in pixels of each line. It must stay smaller than the "
        "spacing, otherwise the lines touch and the shadow becomes a solid "
        "fill.",
    ] = 1

    hatching_angle: Annotated[
        float,
        "Slant of the lines, in degrees.",
    ] = 45.0

    shadow_threshold: Annotated[
        int,
        "Grey level (0-255) below which an area counts as shadow and is "
        "therefore hatched.",
    ] = 95

    crosshatch_ratio: Annotated[
        float,
        "Shadow depth (0.0-1.0) beyond which a second, perpendicular hatching "
        "is laid over the first: this is the cross-hatching used to render "
        "deep blacks in an ink drawing.",
    ] = 0.55

    hatching_strength: Annotated[
        float,
        "Opacity of the hatching, from 0.0 (invisible) to 1.0 (solid black).",
    ] = 0.55


class PenSketchStep:
    """Renders an image as though drawn in pen on white paper."""

    def __init__(self, config: PenSketchConfig) -> None:
        """Prepare the step.

        Args:
            config: stroke and hatching parameters.
        """
        self._config = config

    @property
    def name(self) -> str:
        """Step name."""
        return "pen-sketch"

    def apply(self, image: RGBAImage) -> RGBAImage:
        """Turn the image into a pen drawing.

        Args:
            image: the input image, in colour or already greyscale.

        Returns:
            The drawing, with the alpha channel unchanged.
        """
        # We work on a single channel: if the image is already grey the three
        # channels are identical and the conversion changes nothing.
        luminance = compute_luminance(image.rgb, LuminanceStandard.BT709)

        # 1. Attenuate the noise while preserving the edges.
        smoothed = self._reduce_noise(luminance)

        # 2. Extract the stroke proper.
        ink = self._extract_ink_strokes(smoothed)

        # 3. Remove the isolated specks left by the previous stage.
        cleaned_ink = self._remove_speckles(ink)

        # 4. Hatch the shadowed areas.
        drawing = self._add_shadow_hatching(cleaned_ink, smoothed)

        # The drawing is monochrome: replicate it across the three RGB channels.
        return image.with_rgb(np.dstack([drawing] * 3))

    # ------------------------------------------------------------------
    # 1. Noise attenuation
    # ------------------------------------------------------------------

    def _reduce_noise(self, luminance: np.ndarray) -> np.ndarray:
        """Smooth uniform surfaces without blunting the edges.

        The bilateral filter averages a pixel only with the neighbours that also
        resemble it in colour. The result: JPEG compression noise and fine
        texture disappear, while the outlines stay crisp. This stage is
        essential: without it, XDoG would draw genuine contours and image defects
        with equal emphasis.

        Args:
            luminance: greyscale image.

        Returns:
            The smoothed image.
        """
        return cv2.bilateralFilter(
            luminance,
            d=self._config.bilateral_diameter,
            sigmaColor=self._config.bilateral_sigma,
            sigmaSpace=self._config.bilateral_sigma,
        )

    # ------------------------------------------------------------------
    # 2. Stroke extraction (XDoG)
    # ------------------------------------------------------------------

    def _extract_ink_strokes(self, luminance: np.ndarray) -> np.ndarray:
        """Compute the pen strokes with the XDoG algorithm.

        Args:
            luminance: greyscale image, already smoothed.

        Returns:
            A (height, width) uint8 image: 255 = paper, 0 = ink.
        """
        # We work in normalised floating point: the formulas below add and
        # subtract values that fall outside the 0-255 range.
        normalized = luminance.astype(np.float32) / BYTE_RANGE

        narrow_blur = cv2.GaussianBlur(
            normalized, ksize=(0, 0), sigmaX=self._config.dog_sigma
        )
        wide_blur = cv2.GaussianBlur(
            normalized,
            ksize=(0, 0),
            sigmaX=self._config.dog_sigma * self._config.dog_sigma_ratio,
        )

        # Difference of Gaussians: about zero in uniform areas (where the two
        # blurs coincide) and away from zero around the edges.
        difference_of_gaussians = narrow_blur - wide_blur

        # Every edge produces a negative difference on its dark side and a
        # positive one on its light side. We keep only the dark side: that is
        # where a pen would lay its stroke, and taking both sides would double
        # every line. Subtracting the base tone (which XDoG would use for solid
        # fills) is what separates a line drawing from a blotchy silhouette: the
        # shadows will be rendered by the hatching.
        stroke_response = np.maximum(-difference_of_gaussians, 0.0)

        # Edge strength depends on the photo's contrast. We scale it against the
        # strongest edges present in the image, so that the thresholds below mean
        # the same thing on any shot. A high percentile is used instead of the
        # absolute maximum because a single noisy pixel must not dictate the
        # scale of the whole image.
        reference_strength = float(np.percentile(stroke_response, STRENGTH_PERCENTILE))
        if reference_strength < 1e-6:
            # An image with no edges at all (a solid colour): the sheet stays white.
            return np.full(luminance.shape, 255, dtype=np.uint8)

        relative_strength = np.clip(stroke_response / reference_strength, 0.0, 1.0)

        # Sharpening compresses the scale upwards: the higher it is, the more
        # weak edges reach the strength of crisp ones, and the denser and more
        # pronounced the drawing becomes.
        boosted_strength = np.tanh(self._config.sharpness * relative_strength)

        # Soft threshold with a hyperbolic tangent: below the threshold white
        # paper remains, above it we move to ink the more abruptly the higher the
        # stroke's hardness.
        ink_amount = 0.5 * (
            1.0
            + np.tanh(
                self._config.ink_softness
                * (boosted_strength - self._config.ink_threshold)
            )
        )

        # `ink_amount` is 1 on solid ink: the paper is its negative.
        paper = 1.0 - ink_amount

        return to_uint8(np.clip(paper, 0.0, 1.0) * BYTE_RANGE)

    # ------------------------------------------------------------------
    # 3. Clean-up
    # ------------------------------------------------------------------

    def _remove_speckles(self, ink: np.ndarray) -> np.ndarray:
        """Remove isolated black specks.

        The median filter replaces each pixel with the median of its neighbours:
        a black dot surrounded by white disappears, whereas a continuous line
        survives because its neighbours are black too. It is the right choice
        here, where a blur would soften the strokes as well.

        Args:
            ink: the raw drawing.

        Returns:
            The cleaned-up drawing.
        """
        size = self._config.despeckle_size
        if size <= 1:
            # Clean-up disabled.
            return ink

        # OpenCV requires an odd window side.
        odd_size = size if size % 2 == 1 else size + 1

        return cv2.medianBlur(ink, odd_size)

    # ------------------------------------------------------------------
    # 4. Shadow hatching
    # ------------------------------------------------------------------

    def _add_shadow_hatching(
        self, ink: np.ndarray, luminance: np.ndarray
    ) -> np.ndarray:
        """Fill the dark areas with pen hatching.

        Args:
            ink: the contour drawing produced by XDoG.
            luminance: the greyscale image, used to know where the shadows are.

        Returns:
            The drawing with the hatching laid over it.
        """
        if not self._config.hatching:
            return ink

        # How deep in shadow each pixel is, from 0.0 (full light) to 1.0 (black).
        shadow_amount = self._compute_shadow_amount(luminance)

        # Diagonal lines across the whole image, in the main direction.
        primary_lines = self._build_line_pattern(
            shape=ink.shape, angle_degrees=self._config.hatching_angle
        )

        # A second hatching, perpendicular to the first: crossing each other they
        # create the classic cross-hatching that renders deeper shadows.
        crossing_lines = self._build_line_pattern(
            shape=ink.shape, angle_degrees=self._config.hatching_angle + 90.0
        )

        # The plain hatching covers every shadow; the crossing one only comes
        # into play in the densest shadows, where `shadow_amount` is high.
        crosshatch_zone = shadow_amount >= self._config.crosshatch_ratio
        hatch_pattern = primary_lines | (crossing_lines & crosshatch_zone)

        # Per-pixel hatching opacity: proportional to the depth of the shadow, so
        # barely dark areas get a hint of lines and very dark ones a solid stroke.
        hatch_opacity = np.where(
            hatch_pattern, shadow_amount * self._config.hatching_strength, 0.0
        )

        # Laying the (black) hatching over the existing drawing:
        #   result = drawing * (1 - opacity)
        # The hatching can only darken, never lighten: this way the contours
        # already traced by XDoG stay intact.
        hatched = ink.astype(np.float32) * (1.0 - hatch_opacity)

        return to_uint8(hatched)

    def _compute_shadow_amount(self, luminance: np.ndarray) -> np.ndarray:
        """Measure how deep in shadow each pixel is, on a 0.0-1.0 scale.

        Args:
            luminance: greyscale image.

        Returns:
            A float32 (height, width) array: 0.0 where there is light, 1.0 at black.
        """
        threshold = float(self._config.shadow_threshold)
        if threshold <= 0:
            # No area counts as shadow.
            return np.zeros(luminance.shape, dtype=np.float32)

        # Shadows are judged on the overall shape, not on fine detail: a wide
        # blur stops the hatching from switching on and off pixel by pixel and
        # producing a dirty effect.
        shadow_field = cv2.GaussianBlur(
            luminance.astype(np.float32), ksize=(0, 0), sigmaX=SHADOW_FIELD_SIGMA
        )

        # Linear mapping: 0.0 at the threshold, 1.0 at absolute black.
        amount = (threshold - shadow_field) / threshold

        return np.clip(amount, 0.0, 1.0).astype(np.float32)

    def _build_line_pattern(
        self, shape: tuple[int, ...], angle_degrees: float
    ) -> np.ndarray:
        """Generate a grid of slanted parallel lines.

        Every pixel is projected onto the direction perpendicular to the lines;
        the remainder of dividing that projection by the spacing tells us where
        in the "line / empty gap" cycle we are.

        Args:
            shape: dimensions (height, width) of the grid to generate.
            angle_degrees: slant of the lines, in degrees.

        Returns:
            A boolean (height, width) mask: True where a line passes.
        """
        height, width = shape[0], shape[1]

        # Coordinates of every pixel. `indexing="ij"` keeps the (row, column)
        # order NumPy images use.
        rows, columns = np.meshgrid(
            np.arange(height, dtype=np.float32),
            np.arange(width, dtype=np.float32),
            indexing="ij",
        )

        angle_radians = np.deg2rad(angle_degrees)

        # Projection onto the normal to the lines: pixels with the same
        # projection lie on the same line.
        projection = rows * np.cos(angle_radians) + columns * np.sin(angle_radians)

        spacing = float(max(self._config.hatching_spacing, 1))
        position_in_cycle = np.mod(projection, spacing)

        return position_in_cycle < float(self._config.hatching_line_width)
