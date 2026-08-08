"""Turning a polygon into the smooth curves an editor draws.

Single responsibility: give every corner of an outline a pair of Bézier handles,
so the straight-edged polygon becomes the flowing path Photoshop shows.

The handles come from the Catmull-Rom construction: each anchor's tangent points
along the line joining the anchor before it to the anchor after it. That is the
direction the shape is heading at that point, so the curve leaves and arrives
without a kink, and it needs nothing more than the neighbouring points to work
out — no fitting, no iteration, and the same answer every time.

How far the handles reach is the one control worth having. At zero they collapse
onto the anchors and the path is exactly the polygon it came from, corners and
all; the further out they go, the rounder the corners become. A third of the way
to the neighbour is the usual choice and is what keeps a circle looking like a
circle rather than a bulging one.
"""

from __future__ import annotations

from pixel.paths.geometry import BezierPath, CubicSegment, Outline, Point

# How far the handles reach, as a fraction of the distance to the neighbouring
# anchor. One third reproduces a circle almost exactly, which is the case the eye
# is quickest to judge.
DEFAULT_SMOOTHNESS: float = 1.0 / 3.0


def to_bezier(outline: Outline, smoothness: float = DEFAULT_SMOOTHNESS) -> BezierPath:
    """Turn a polygon outline into a closed path of cubic Bézier curves.

    Args:
        outline: the polygon to convert.
        smoothness: how far the handles reach, as a fraction of the distance to
            the neighbouring anchor. 0 keeps the corners sharp, giving back
            exactly the polygon; around a third gives the rounded look of a path
            drawn by hand.

    Returns:
        The path, with one curve per edge of the polygon.
    """
    points = outline.points
    count = len(points)

    # The tangent at each anchor, pointing the way the outline is heading there.
    tangents = tuple(
        _tangent(points[(index - 1) % count], points[(index + 1) % count], smoothness)
        for index in range(count)
    )

    segments: list[CubicSegment] = []

    for index in range(count):
        start = points[index]
        # The last edge closes the shape by running back to the first point.
        end_index = (index + 1) % count
        end = points[end_index]

        segments.append(
            CubicSegment(
                # Leaving `start`, the handle follows its tangent forwards.
                control_start=_offset(start, tangents[index], 1.0),
                # Arriving at `end`, it comes in against that anchor's tangent.
                control_end=_offset(end, tangents[end_index], -1.0),
                end=end,
            )
        )

    return BezierPath(
        start=points[0], segments=tuple(segments), is_hole=outline.is_hole
    )


def _tangent(before: Point, after: Point, smoothness: float) -> Point:
    """Work out an anchor's tangent from the anchors on either side.

    Args:
        before: the anchor preceding this one.
        after: the anchor following it.
        smoothness: how far the handles should reach.

    Returns:
        The offset to add to an anchor to reach its forward handle.
    """
    # The chord from the previous anchor to the next one is the direction the
    # outline is travelling through the anchor between them.
    return (
        (after[0] - before[0]) * smoothness * 0.5,
        (after[1] - before[1]) * smoothness * 0.5,
    )


def _offset(anchor: Point, tangent: Point, direction: float) -> Point:
    """Place a handle beside an anchor, along its tangent.

    Args:
        anchor: the anchor point.
        tangent: the anchor's tangent.
        direction: 1.0 for the handle leaving the anchor, -1.0 for the one
            arriving at it.

    Returns:
        The handle's position.
    """
    return (
        anchor[0] + tangent[0] * direction,
        anchor[1] + tangent[1] * direction,
    )
