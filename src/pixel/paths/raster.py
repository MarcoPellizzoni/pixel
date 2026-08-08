"""Drawing traced paths back onto an image.

Single responsibility: turn Bézier paths into pixels, so they can be seen on the
canvas rather than only written to a file.

A curve has to be broken into short straight pieces before it can be drawn. How
many pieces is the one judgement here: too few and a long sweeping curve shows
its corners, too many and the drawing slows down for a difference no one can see.
Rather than fixing a count, each segment gets a number of pieces in proportion to
how far it travels, so a curve spanning half the picture is smooth and one
spanning four pixels is not paid for.
"""

from __future__ import annotations

import cv2
import numpy as np

from pixel.domain import RGBAImage, to_uint8
from pixel.paths.geometry import BezierPath, Point, TracedPaths

# How many straight pieces to use per pixel of a curve's rough length. A third
# means a piece every three pixels, which is finer than a screen shows.
PIECES_PER_PIXEL: float = 1.0 / 3.0

# Fewest and most pieces one curve may be broken into, whatever its length.
MINIMUM_PIECES: int = 2
MAXIMUM_PIECES: int = 48


def draw(
    image: RGBAImage,
    traced: TracedPaths,
    color: tuple[int, int, int],
    width: int,
    on_blank: bool,
) -> RGBAImage:
    """Draw traced paths onto an image.

    Args:
        image: the image to draw on.
        traced: the paths to draw.
        color: the line colour, as (R, G, B).
        width: the line thickness in pixels.
        on_blank: True to draw on a plain white sheet instead of over the
            picture, which is how a path is usually inspected on its own.

    Returns:
        The image with the paths drawn on it.
    """
    canvas = (
        np.full((image.height, image.width, 3), 255, dtype=np.uint8)
        if on_blank
        else image.rgb
    )

    # OpenCV draws into whatever array it is given, so it must be one this
    # function owns rather than a view onto the image being edited.
    surface = canvas.copy()

    for path in traced.paths:
        points = flatten(path)
        if len(points) < 2:
            continue

        cv2.polylines(
            surface,
            [np.array(points, dtype=np.int32)],
            isClosed=True,
            color=color,
            thickness=max(1, width),
            # Anti-aliased, or a diagonal path would come out as a staircase and
            # look nothing like the curve it is.
            lineType=cv2.LINE_AA,
        )

    return image.with_rgb(to_uint8(surface.astype(np.float32)))


def flatten(path: BezierPath) -> list[Point]:
    """Break a path's curves into the short straight pieces a drawer needs.

    Args:
        path: the path to break up.

    Returns:
        The points along it, first to last. The closing point is not repeated:
        the drawing routine is told the shape is closed instead.
    """
    points: list[Point] = [path.start]
    current = path.start

    for segment in path.segments:
        pieces = _pieces_for(current, segment.end)

        # `t` runs from just past 0 to 1: the piece at 0 is the point the
        # previous segment already finished on.
        for step in range(1, pieces + 1):
            points.append(
                _at(
                    current,
                    segment.control_start,
                    segment.control_end,
                    segment.end,
                    step / pieces,
                )
            )

        current = segment.end

    return points


def _pieces_for(start: Point, end: Point) -> int:
    """Decide how many straight pieces one curve deserves.

    The straight-line distance between the ends stands in for the curve's true
    length, which would cost more to work out than it would save.

    Args:
        start: where the curve begins.
        end: where it finishes.

    Returns:
        The number of pieces, within the fixed bounds.
    """
    span = float(np.hypot(end[0] - start[0], end[1] - start[1]))

    return int(
        min(MAXIMUM_PIECES, max(MINIMUM_PIECES, round(span * PIECES_PER_PIXEL)))
    )


def _at(
    start: Point, control_start: Point, control_end: Point, end: Point, t: float
) -> Point:
    """Find the point a given fraction of the way along a cubic Bézier curve.

    Args:
        start: the curve's first anchor.
        control_start: the handle leaving it.
        control_end: the handle arriving at the far anchor.
        end: the far anchor.
        t: how far along, from 0.0 at the start to 1.0 at the end.

    Returns:
        The point on the curve.
    """
    # The Bernstein form: each of the four points pulls on the result, most
    # strongly near its own end of the curve.
    remaining = 1.0 - t
    a = remaining**3
    b = 3.0 * remaining**2 * t
    c = 3.0 * remaining * t**2
    d = t**3

    return (
        a * start[0] + b * control_start[0] + c * control_end[0] + d * end[0],
        a * start[1] + b * control_start[1] + c * control_end[1] + d * end[1],
    )
