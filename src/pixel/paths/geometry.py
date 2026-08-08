"""The shapes a traced path is made of.

Single responsibility: define what a path *is* — points, outlines and Bézier
curves — and nothing about how one is found or drawn.

The vocabulary follows the one image editors use. An **outline** is a closed
polygon of points, straight from the pixels. A **path** is that outline turned
into cubic Bézier curves, which is what Photoshop shows when it converts a
selection to a work path, and what an SVG or a printer wants.

Holes matter and are kept: the inside of a letter O is an outline in its own
right, marked as a hole, so that filling the path leaves it empty rather than
solid.
"""

from __future__ import annotations

from dataclasses import dataclass

# A point on the image, in pixels from the top-left corner. Kept as floats
# because simplifying and smoothing move points off the pixel grid, and rounding
# at every stage would slowly eat the shape.
Point = tuple[float, float]


@dataclass(frozen=True)
class Outline:
    """A closed polygon following the edge of one region.

    Attributes:
        points: the corners, in order. The closing edge back to the first point
            is implied rather than repeated.
        is_hole: whether this outline bounds a hole inside another region, such
            as the inside of a letter O.
    """

    points: tuple[Point, ...]
    is_hole: bool = False

    def __post_init__(self) -> None:
        """Reject an outline too small to bound anything."""
        if len(self.points) < 3:
            raise ValueError(
                f"An outline needs at least 3 points, got {len(self.points)}."
            )


@dataclass(frozen=True)
class CubicSegment:
    """One cubic Bézier curve, running from the previous anchor to `end`.

    Attributes:
        control_start: the handle leaving the previous anchor point.
        control_end: the handle arriving at `end`.
        end: the anchor point this segment finishes on.
    """

    control_start: Point
    control_end: Point
    end: Point


@dataclass(frozen=True)
class BezierPath:
    """A closed path of cubic Bézier curves, as an editor would draw it.

    Attributes:
        start: the first anchor point.
        segments: the curves, each carrying on from where the last one ended.
        is_hole: whether the path bounds a hole inside another one.
    """

    start: Point
    segments: tuple[CubicSegment, ...]
    is_hole: bool = False

    @property
    def anchors(self) -> tuple[Point, ...]:
        """Every anchor point, the starting one first."""
        return (self.start, *(segment.end for segment in self.segments))


@dataclass(frozen=True)
class TracedPaths:
    """Everything traced out of one image.

    Attributes:
        paths: the paths found, outer ones and holes together.
        width: width of the image they were traced from, in pixels.
        height: its height, so the paths can be placed on a canvas of the right
            size without the image itself having to be carried along.
    """

    paths: tuple[BezierPath, ...]
    width: int
    height: int

    def __len__(self) -> int:
        """How many paths were traced."""
        return len(self.paths)

    @property
    def is_empty(self) -> bool:
        """Whether nothing at all was traced."""
        return not self.paths
