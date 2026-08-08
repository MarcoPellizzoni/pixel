"""Tracing an image into paths, from start to finish.

Single responsibility: run the three stages in order — find the borders, thin
them out, curve them — and hand back the paths.

Each stage lives in its own module and knows nothing of the others; this is the
one place that knows they go together, and in which order. Splitting them that
way is what lets the tolerance be tried at several values without tracing the
borders again, which is the expensive part.

The whole thing mirrors what Photoshop does when a selection becomes a work path,
and takes the same two controls: how far the path may stray from the pixels, and
how rounded its corners are.
"""

from __future__ import annotations

from dataclasses import dataclass

from pixel.domain import RGBAImage
from pixel.paths.curves import DEFAULT_SMOOTHNESS, to_bezier
from pixel.paths.finder import TraceConfig, find_outlines
from pixel.paths.geometry import Outline, TracedPaths
from pixel.paths.simplify import simplify


@dataclass(frozen=True)
class PathStyle:
    """How closely the path should follow the pixels, and how smoothly.

    Attributes:
        tolerance: how far, in pixels, the path may stray from the traced border.
            The same control Photoshop offers: small values follow every wobble,
            large ones give a few clean corners.
        smoothness: how far the Bézier handles reach, as a fraction of the
            distance between anchors. 0 keeps the corners sharp.
    """

    tolerance: float = 2.0
    smoothness: float = DEFAULT_SMOOTHNESS


@dataclass(frozen=True)
class TraceReport:
    """What the tracing found, alongside the paths themselves.

    Attributes:
        paths: the traced paths.
        outlines_found: how many borders the search turned up.
        points_before: how many points those borders had, pixel by pixel.
        points_after: how many were left once the tolerance had been applied.
    """

    paths: TracedPaths
    outlines_found: int = 0
    points_before: int = 0
    points_after: int = 0

    @property
    def reduction(self) -> float:
        """The share of points the tolerance removed, from 0.0 to 1.0."""
        if not self.points_before:
            return 0.0

        return 1.0 - (self.points_after / self.points_before)


def trace(
    image: RGBAImage,
    config: TraceConfig | None = None,
    style: PathStyle | None = None,
) -> TraceReport:
    """Trace an image into Bézier paths.

    Args:
        image: the image to trace.
        config: what counts as inside the shape, and what is worth keeping.
        style: how closely and how smoothly the paths should follow it.

    Returns:
        The paths, with a note of how much work the tolerance saved.
    """
    settings = config or TraceConfig()
    shape = style or PathStyle()

    outlines = find_outlines(image, settings)

    if not outlines:
        return TraceReport(paths=TracedPaths((), image.width, image.height))

    simplified = tuple(simplify(outline, shape.tolerance) for outline in outlines)

    paths = tuple(to_bezier(outline, shape.smoothness) for outline in simplified)

    return TraceReport(
        paths=TracedPaths(paths, image.width, image.height),
        outlines_found=len(outlines),
        points_before=_count(outlines),
        points_after=_count(simplified),
    )


def _count(outlines: tuple[Outline, ...]) -> int:
    """Count the points across a group of outlines."""
    return sum(len(outline.points) for outline in outlines)
