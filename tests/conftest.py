"""Image factories shared by every test.

Single responsibility: build test images with known content, so each test can
state precisely what it expects to come out.
"""

from __future__ import annotations

import numpy as np
import pytest

from pixel.domain import RGBAImage


def solid_image(
    color: tuple[int, int, int],
    width: int = 16,
    height: int = 16,
    alpha: int = 255,
) -> RGBAImage:
    """Create a single-colour image, with the given opacity."""
    data = np.zeros((height, width, 4), dtype=np.uint8)
    data[:, :, :3] = color
    data[:, :, 3] = alpha
    return RGBAImage(data)


def gray_image(level: int, width: int = 16, height: int = 16) -> RGBAImage:
    """Create an opaque image of a single grey level."""
    return solid_image((level, level, level), width=width, height=height)


def split_image(
    left_color: tuple[int, int, int],
    right_color: tuple[int, int, int],
    width: int = 64,
    height: int = 64,
) -> RGBAImage:
    """Create an image split into two vertical halves of different colour.

    This is the simplest shape containing a crisp edge: every test that checks
    edge detection needs it.
    """
    data = np.zeros((height, width, 4), dtype=np.uint8)
    data[:, : width // 2, :3] = left_color
    data[:, width // 2 :, :3] = right_color
    data[:, :, 3] = 255
    return RGBAImage(data)


def gradient_image(width: int = 64, height: int = 64) -> RGBAImage:
    """Create a horizontal gradient from black to white."""
    ramp = np.linspace(0, 255, width, dtype=np.uint8)
    data = np.zeros((height, width, 4), dtype=np.uint8)
    data[:, :, :3] = ramp[np.newaxis, :, np.newaxis]
    data[:, :, 3] = 255
    return RGBAImage(data)


def noisy_image(width: int = 64, height: int = 64, seed: int = 0) -> RGBAImage:
    """Create a random image, reproducible thanks to the fixed seed."""
    generator = np.random.default_rng(seed=seed)
    data = np.zeros((height, width, 4), dtype=np.uint8)
    data[:, :, :3] = generator.integers(0, 256, size=(height, width, 3), dtype=np.uint8)
    data[:, :, 3] = 255
    return RGBAImage(data)


@pytest.fixture
def photo() -> RGBAImage:
    """A test image with colours, edges and gradations.

    It stands for the ordinary case: steps that only have to "not break
    anything" can use it without building an image of their own.
    """
    data = np.zeros((32, 48, 4), dtype=np.uint8)
    # Red band on the left, green in the middle, blue on the right: three crisp edges.
    data[:, :16, :3] = (200, 60, 40)
    data[:, 16:32, :3] = (40, 180, 70)
    data[:, 32:, :3] = (50, 70, 210)
    data[:, :, 3] = 255
    return RGBAImage(data)
