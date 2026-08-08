"""Turning an image into something the screen can show.

Single responsibility: encode an `RGBAImage` into the PNG bytes the Flet `Image`
control accepts, and make it suitable for display along the way.

Two things happen here that must never happen to the image being edited:

- transparency is drawn over a chequerboard, the usual editor convention, so the
  user can actually see where the background was removed;
- oversized photos are scaled down, because encoding and shipping a 24-megapixel
  PNG on every single edit would make the interface crawl.

Both are presentation only. The session keeps the picture at full resolution and
full transparency, and that is what gets saved.
"""

from __future__ import annotations

from io import BytesIO

import numpy as np
from PIL import Image

from pixel.domain import MAX_CHANNEL_VALUE, RGBAImage
from pixel.steps.geometry import ResizeConfig, ResizeStep

# Longest side, in pixels, of the image handed to the interface. Large enough to
# stay sharp on a high-resolution display, small enough to encode in a few
# milliseconds so that editing still feels immediate.
PREVIEW_MAX_SIDE: int = 1400

# Side of the chequerboard squares, in pixels.
CHECKER_SQUARE_SIZE: int = 12

# The two greys of the chequerboard. Deliberately close together: the pattern
# has to read as "nothing here" without competing with the picture on top.
CHECKER_LIGHT: int = 0x50
CHECKER_DARK: int = 0x3A


def to_png_bytes(image: RGBAImage, max_side: int = PREVIEW_MAX_SIDE) -> bytes:
    """Render an image as PNG bytes ready for a Flet `Image` control.

    Args:
        image: the image to show, transparency and all.
        max_side: longest side allowed in the result; larger images are scaled
            down proportionally. Smaller ones are left alone, never enlarged.

    Returns:
        The encoded PNG, as bytes.
    """
    scaled = _scale_to_fit(image, max_side)
    flattened = _composite_over_checkerboard(scaled)

    buffer = BytesIO()
    Image.fromarray(flattened, mode="RGB").save(buffer, format="PNG")

    return buffer.getvalue()


def _scale_to_fit(image: RGBAImage, max_side: int) -> RGBAImage:
    """Shrink an image so its longest side fits within `max_side`.

    Args:
        image: the image to scale.
        max_side: the limit, in pixels.

    Returns:
        The scaled image, or the original when it already fits.
    """
    longest_side = max(image.width, image.height)
    if longest_side <= max_side or max_side <= 0:
        # Already small enough. Resizing anyway would only cost time and soften
        # the picture for nothing.
        return image

    # `ResizeStep` is reused rather than calling OpenCV again here: it already
    # knows to pick the interpolation that avoids aliasing when shrinking.
    scale = max_side / longest_side
    return ResizeStep(ResizeConfig(scale=scale)).apply(image)


def _composite_over_checkerboard(image: RGBAImage) -> np.ndarray:
    """Draw the image over a grey chequerboard, removing transparency.

    Args:
        image: the image to flatten.

    Returns:
        An RGB (height, width, 3) uint8 array, with no alpha channel left.
    """
    board = _build_checkerboard(image.height, image.width)

    # Standard alpha compositing, the same formula as `RGBAImage.composite_over`,
    # except the background varies from pixel to pixel instead of being one
    # colour, so it cannot reuse that method.
    alpha_ratio = (image.alpha.astype(np.float32) / MAX_CHANNEL_VALUE)[:, :, np.newaxis]
    blended = image.rgb.astype(np.float32) * alpha_ratio + board * (1.0 - alpha_ratio)

    return np.clip(np.rint(blended), 0, MAX_CHANNEL_VALUE).astype(np.uint8)


def _build_checkerboard(height: int, width: int) -> np.ndarray:
    """Build the grey chequerboard shown through transparent pixels.

    Args:
        height: image height in pixels.
        width: image width in pixels.

    Returns:
        A float32 (height, width, 3) array of grey levels.
    """
    # Integer-dividing the coordinates by the square size numbers the squares;
    # the parity of the sum of those numbers alternates exactly like a chessboard.
    rows = (np.arange(height) // CHECKER_SQUARE_SIZE)[:, np.newaxis]
    columns = (np.arange(width) // CHECKER_SQUARE_SIZE)[np.newaxis, :]
    is_light_square = (rows + columns) % 2 == 0

    board = np.where(is_light_square, CHECKER_LIGHT, CHECKER_DARK)

    # The same grey on all three channels, with the extra axis the compositing
    # formula needs.
    return np.repeat(board[:, :, np.newaxis], 3, axis=2).astype(np.float32)
