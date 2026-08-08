"""Finding the outlines in an image.

Single responsibility: work out which pixels are inside the shape, then walk the
border between inside and outside to produce closed outlines.

This is the search the whole feature rests on. The algorithm is border
following, in the form described by Suzuki and Abe: starting from a pixel whose
neighbour is background, it steps around the region always keeping the inside on
the same hand, until it arrives back where it began. Following the border also
reveals which outlines sit inside which, so holes can be told apart from outer
edges — the inside of a letter O is found as a border in its own right, and
marked as a hole.

The implementation used is OpenCV's, which is that same algorithm. Where the
region comes from is decided here too, and it matters: after `remove-background`
the alpha channel already holds exactly what the user selected, which is the
editor's equivalent of Photoshop turning a selection into a work path.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum

import cv2
import numpy as np

from pixel.domain import MAX_CHANNEL_VALUE, RGBAImage
from pixel.paths.geometry import Outline, Point
from pixel.steps.color import LuminanceStandard, compute_luminance

# Index into OpenCV's hierarchy rows holding the parent contour, or -1 when the
# contour has none.
PARENT_INDEX: int = 3


class TraceSource(StrEnum):
    """Which part of the image decides what counts as inside the shape."""

    # The alpha channel: whatever a cut-out left behind. This is the one to
    # reach for after `remove-background`, and the closest thing the editor has
    # to Photoshop's "make work path from selection".
    ALPHA = "alpha"

    # Brightness: everything darker than the threshold is inside. Suits line
    # art, scanned drawings and the output of `threshold` or `edges`.
    DARK = "dark"

    # Brightness the other way up: everything lighter than the threshold is
    # inside.
    LIGHT = "light"


@dataclass(frozen=True)
class TraceConfig:
    """How to decide what is inside the shape, and what to keep.

    Attributes:
        source: which part of the image marks the shape.
        threshold: the level, 0-255, separating inside from outside.
        minimum_area: outlines enclosing fewer pixels than this are dropped.
            Without it, every speck of noise becomes a path of its own.
        include_holes: whether to keep the borders found inside other regions.
    """

    source: TraceSource = TraceSource.ALPHA
    threshold: int = 128
    minimum_area: float = 24.0
    include_holes: bool = True


def find_outlines(image: RGBAImage, config: TraceConfig) -> tuple[Outline, ...]:
    """Trace the borders of the shape in an image.

    Args:
        image: the image to trace.
        config: what counts as inside, and what is worth keeping.

    Returns:
        The outlines found, outer ones and holes together, largest first. Sorting
        by size means that when only some can be shown or used, the ones that
        carry the shape come first.
    """
    mask = build_mask(image, config)

    # `RETR_CCOMP` returns borders in two levels — outer edges and the holes
    # directly inside them — which is exactly the distinction a path needs, and
    # cheaper than the full nesting tree that `RETR_TREE` builds.
    # `CHAIN_APPROX_NONE` keeps every border pixel: thinning them out is the job
    # of the simplifying stage, which does it in a way the user can control.
    contours, hierarchy = cv2.findContours(
        mask, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_NONE
    )

    if not contours:
        # Nothing was inside the shape at all. Checking the contours rather than
        # the hierarchy is deliberate: OpenCV returns no hierarchy in this case,
        # and returning here means nothing ever has to reason about that.
        return ()

    found: list[Outline] = []

    # `hierarchy` comes back wrapped in an extra axis; its rows line up with the
    # contours one for one.
    rows = hierarchy[0]

    for contour, row in zip(contours, rows, strict=True):
        # A border found inside another one bounds a hole.
        is_hole = bool(row[PARENT_INDEX] >= 0)

        if is_hole and not config.include_holes:
            continue

        # `contourArea` is the enclosed area, so a long thin scratch is dropped
        # while a small solid blob is kept — which is the distinction that
        # matters when deciding what is noise.
        if cv2.contourArea(contour) < config.minimum_area:
            continue

        points = _to_points(contour)
        if len(points) < 3:
            # Too few corners to enclose anything, whatever the area said.
            continue

        found.append(Outline(points=points, is_hole=is_hole))

    return tuple(sorted(found, key=_enclosed_area, reverse=True))


def build_mask(image: RGBAImage, config: TraceConfig) -> np.ndarray:
    """Decide, for every pixel, whether it is inside the shape.

    Args:
        image: the image to examine.
        config: which part of it to look at, and at what level to cut.

    Returns:
        A (height, width) uint8 array: 255 inside the shape, 0 outside.
    """
    if config.source is TraceSource.ALPHA:
        # What a cut-out left behind. Semi-transparent edges count as inside
        # above the threshold, so a feathered mask still traces where the eye
        # sees the edge.
        channel = image.alpha
        inside = channel >= config.threshold
    else:
        brightness = compute_luminance(image.rgb, LuminanceStandard.BT709)
        inside = (
            brightness < config.threshold
            if config.source is TraceSource.DARK
            else brightness >= config.threshold
        )

    return np.where(inside, MAX_CHANNEL_VALUE, 0).astype(np.uint8)


def _to_points(contour: np.ndarray) -> tuple[Point, ...]:
    """Turn one OpenCV contour into plain points.

    Args:
        contour: the contour as OpenCV returns it, of shape (n, 1, 2).

    Returns:
        Its points, in order.
    """
    # The middle axis is OpenCV's own; squeezing it leaves (n, 2).
    flattened = contour.reshape(-1, 2)

    return tuple((float(x), float(y)) for x, y in flattened)


def _enclosed_area(outline: Outline) -> float:
    """Measure the area an outline encloses, for ranking by size.

    Uses the shoelace formula, whose sign depends on the winding direction, so
    the absolute value is taken.

    Args:
        outline: the outline to measure.

    Returns:
        The area in square pixels.
    """
    points = outline.points
    total = 0.0

    for index, (x, y) in enumerate(points):
        next_x, next_y = points[(index + 1) % len(points)]
        total += x * next_y - next_x * y

    return abs(total) / 2.0
