"""Turning an image into vector paths, the way an editor's Path tool does.

Single responsibility of this package: find the outline of a shape in an image
and give it back as Bézier curves that can be drawn, edited or exported.

    geometry.py   what a path is made of: points, outlines, curves
    finder.py     the search: border following, and what counts as inside
    simplify.py   Douglas-Peucker, which is the tolerance control
    curves.py     polygons into Bézier curves
    svg.py        writing paths out as SVG
    tracer.py     the three stages in order

Nothing here knows about the pipeline, the command line or the window. The step
that draws paths onto an image lives in `pixel.steps.trace`, and depends on this;
nothing here depends on it.
"""

from pixel.paths.curves import DEFAULT_SMOOTHNESS, to_bezier
from pixel.paths.finder import TraceConfig, TraceSource, build_mask, find_outlines
from pixel.paths.geometry import BezierPath, CubicSegment, Outline, Point, TracedPaths
from pixel.paths.simplify import simplify
from pixel.paths.svg import to_path_data, to_svg
from pixel.paths.tracer import PathStyle, TraceReport, trace

__all__ = [
    "Point",
    "Outline",
    "CubicSegment",
    "BezierPath",
    "TracedPaths",
    "TraceSource",
    "TraceConfig",
    "PathStyle",
    "TraceReport",
    "find_outlines",
    "build_mask",
    "simplify",
    "to_bezier",
    "DEFAULT_SMOOTHNESS",
    "trace",
    "to_svg",
    "to_path_data",
]
