"""Thinning an outline down to the points that matter.

Single responsibility: drop the points of an outline that say nothing about its
shape, keeping the ones that do.

A border traced pixel by pixel has a point on every step, most of them sitting in
a straight line with their neighbours. Douglas-Peucker removes them by asking one
question over and over: of all the points between these two ends, which strays
furthest from the straight line joining them? If even that one is closer than the
tolerance, the whole stretch is a straight line and everything between the ends
goes; otherwise the stretch is split at that point and each half is asked the
same question. That recursive narrowing is what makes it a search rather than a
filter.

The tolerance is the same control Photoshop offers when converting a selection to
a work path, and behaves the same way: small values follow every wobble, large
ones give a few clean corners.
"""

from __future__ import annotations

import cv2
import numpy as np

from pixel.paths.geometry import Outline, Point

# Below this the tolerance is treated as zero and every point is kept: OpenCV
# refuses a negative one, and a vanishingly small one only wastes time.
MINIMUM_TOLERANCE: float = 0.01


def simplify(outline: Outline, tolerance: float) -> Outline:
    """Remove the points of an outline that do not change its shape.

    Args:
        outline: the outline to thin out.
        tolerance: how far, in pixels, the simplified outline may stray from the
            original. 0 keeps every point.

    Returns:
        The simplified outline, with its hole marking carried over. If
        simplifying would leave too few points to enclose anything, the original
        is returned instead: a triangle is the least an outline can be.
    """
    if tolerance < MINIMUM_TOLERANCE:
        return outline

    # OpenCV wants the shape it hands out itself: (n, 1, 2), floating point.
    points = np.array(outline.points, dtype=np.float32).reshape(-1, 1, 2)

    # `closed=True` tells it the last point joins the first, so the closing edge
    # is considered too and the join does not end up with a spurious corner.
    simplified = cv2.approxPolyDP(points, tolerance, True)

    kept = tuple(
        (float(x), float(y)) for x, y in simplified.reshape(-1, 2)
    )

    if len(kept) < 3:
        # The tolerance was large enough to collapse the shape entirely. Keeping
        # the original is more useful than returning something that cannot be
        # drawn.
        return outline

    return Outline(points=kept, is_hole=outline.is_hole)


def total_points(outlines: tuple[Outline, ...]) -> int:
    """Count the points across several outlines.

    Useful for telling the user how much a tolerance actually bought them.

    Args:
        outlines: the outlines to count.

    Returns:
        The total number of points.
    """
    return sum(len(outline.points) for outline in outlines)


def perimeter(points: tuple[Point, ...]) -> float:
    """Measure the length all the way round a closed outline.

    Args:
        points: the outline's points, in order.

    Returns:
        The perimeter in pixels.
    """
    total = 0.0

    for index, (x, y) in enumerate(points):
        next_x, next_y = points[(index + 1) % len(points)]
        total += float(np.hypot(next_x - x, next_y - y))

    return total
