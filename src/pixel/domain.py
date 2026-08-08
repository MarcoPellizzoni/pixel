"""Domain model: the data type that flows through the whole pipeline.

This module has a single responsibility: to define *what an image is* for this
program, and to guarantee it is always in a valid, predictable shape (RGBA,
8 bits per channel).

It knows nothing about files on disk (see `image_io`) or about processing
algorithms (see the `steps` package).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

# Number of channels in an RGBA image: red, green, blue, alpha (opacity).
RGBA_CHANNELS: int = 4

# Number of channels in an RGB image (no transparency).
RGB_CHANNELS: int = 3

# Maximum value of an 8-bit channel: alpha 255 means a fully opaque pixel.
MAX_CHANNEL_VALUE: int = 255


def to_uint8(values: np.ndarray) -> np.ndarray:
    """Bring a floating-point array back to the 8-bit 0-255 scale.

    Nearly every algorithm computes in floating point and then has to return to
    8 bits. This is the only conversion allowed, and everything goes through it
    for two reasons:

    - clipping at the extremes prevents an out-of-range value from wrapping
      around, turning a blown-out white into black;
    - rounding is essential: NumPy's cast truncates, so a 189.9999 produced by
      floating-point imprecision would become 189. On a step that should be
      neutral that shows up as a one-level darkening, and it compounds at every
      stage of the pipeline.

    Args:
        values: an array of any shape, in floating point.

    Returns:
        The same array as uint8.
    """
    return np.clip(np.rint(values), 0, MAX_CHANNEL_VALUE).astype(np.uint8)


@dataclass(frozen=True)
class RGBAImage:
    """An immutable RGBA image.

    Every pipeline step receives an `RGBAImage` and returns a new one:
    immutability (`frozen=True`) prevents a step from accidentally modifying
    another step's input, and makes keeping the intermediate results trivial.

    Attributes:
        data: NumPy array of shape (height, width, 4) and dtype `uint8`. The
            first three channels are the colour, the fourth is the opacity.
    """

    data: np.ndarray

    # ------------------------------------------------------------------
    # Validation
    # ------------------------------------------------------------------

    def __post_init__(self) -> None:
        """Check the type's invariants right after construction.

        Failing here, as early as possible, is far clearer than seeing an
        inscrutable NumPy error three functions later.
        """
        if self.data.ndim != 3:
            raise ValueError(
                f"Expected 3 dimensions (height, width, channels), "
                f"got {self.data.ndim}."
            )
        if self.data.shape[2] != RGBA_CHANNELS:
            raise ValueError(
                f"Expected {RGBA_CHANNELS} channels (RGBA), got {self.data.shape[2]}."
            )
        if self.data.dtype != np.uint8:
            raise ValueError(f"Expected dtype uint8 (0-255), got {self.data.dtype}.")

    # ------------------------------------------------------------------
    # Alternative constructors
    # ------------------------------------------------------------------

    @classmethod
    def from_rgb(cls, rgb: np.ndarray) -> RGBAImage:
        """Build an RGBA image from an RGB array, making it opaque.

        Used when loading a JPEG, which by definition has no transparency.

        Args:
            rgb: array of shape (height, width, 3) and dtype uint8.

        Returns:
            The same image with an alpha channel added, all set to 255.
        """
        if rgb.ndim != 3 or rgb.shape[2] != RGB_CHANNELS:
            raise ValueError(
                f"Expected an RGB array (height, width, 3), got shape {rgb.shape}."
            )

        height, width = rgb.shape[:2]
        # Fully opaque alpha channel: no pixel is transparent.
        opaque_alpha = np.full((height, width), MAX_CHANNEL_VALUE, dtype=np.uint8)

        # `dstack` stacks channels along the last axis: (H, W, 3) + (H, W) -> (H, W, 4).
        return cls(np.dstack([rgb.astype(np.uint8), opaque_alpha]))

    # ------------------------------------------------------------------
    # Read-only access to the components
    # ------------------------------------------------------------------

    @property
    def rgb(self) -> np.ndarray:
        """The colour channels alone, as a (height, width, 3) uint8 copy."""
        # The copy protects immutability: whoever receives it can modify it
        # freely without corrupting this instance.
        return self.data[:, :, :RGB_CHANNELS].copy()

    @property
    def alpha(self) -> np.ndarray:
        """The opacity channel alone, as a (height, width) uint8 copy."""
        return self.data[:, :, 3].copy()

    @property
    def height(self) -> int:
        """Image height in pixels."""
        return int(self.data.shape[0])

    @property
    def width(self) -> int:
        """Image width in pixels."""
        return int(self.data.shape[1])

    # ------------------------------------------------------------------
    # Transformations (always return a new instance)
    # ------------------------------------------------------------------

    def with_rgb(self, rgb: np.ndarray) -> RGBAImage:
        """Return a copy with new colour channels and the same alpha.

        This is the typical operation for steps that change the appearance
        (greyscale, pen effect) without touching the cut-out silhouette.
        """
        if rgb.shape[:2] != (self.height, self.width):
            raise ValueError(
                f"New RGB dimensions {rgb.shape[:2]} do not match the image's "
                f"({self.height}, {self.width})."
            )
        return RGBAImage(np.dstack([rgb.astype(np.uint8), self.alpha]))

    def with_alpha(self, alpha: np.ndarray) -> RGBAImage:
        """Return a copy with a new alpha channel and the same colour.

        This is the typical operation for background removal, which decides
        which pixels belong to the subject and which do not.
        """
        if alpha.shape != (self.height, self.width):
            raise ValueError(
                f"New alpha dimensions {alpha.shape} do not match the image's "
                f"({self.height}, {self.width})."
            )
        return RGBAImage(np.dstack([self.rgb, alpha.astype(np.uint8)]))

    def composite_over(self, background: tuple[int, int, int]) -> np.ndarray:
        """Blend the image onto a solid background, removing transparency.

        Needed by algorithms that reason about opaque images: if we handed them
        the transparent pixels as they are, the colour "underneath" the
        transparency (often black) would create false edges and contours.

        The formula is standard alpha compositing over an opaque background:
            result = foreground * alpha + background * (1 - alpha)

        Args:
            background: background colour as an (R, G, B) triple in 0-255.

        Returns:
            An RGB (height, width, 3) uint8 array, without transparency.
        """
        # Bring alpha into [0.0, 1.0] and give it a third axis, so NumPy
        # broadcasts it automatically across the three colour channels.
        alpha_ratio = (self.alpha.astype(np.float32) / MAX_CHANNEL_VALUE)[
            :, :, np.newaxis
        ]

        foreground = self.rgb.astype(np.float32)
        background_plane = np.array(background, dtype=np.float32)

        blended = foreground * alpha_ratio + background_plane * (1.0 - alpha_ratio)

        return to_uint8(blended)
