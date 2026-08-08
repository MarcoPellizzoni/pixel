"""Path tracing: the outline of the shape, drawn as a curve.

Single responsibility: find the paths in an image and draw them, so that what an
editor's Path tool would show can be seen on the canvas and placed anywhere in a
pipeline.

The work of finding the paths belongs to `pixel.paths`; this step only settles
what to trace and how to show it. The two are kept apart because a path is
useful beyond being looked at — the same paths are what gets written to an SVG —
and tying the finding to the drawing would make that awkward.

Used straight after `remove-background`, this traces the cut-out subject, which
is the editor's version of turning a selection into a work path.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Annotated

from pixel.domain import RGBAImage
from pixel.paths.finder import TraceConfig, TraceSource
from pixel.paths.raster import draw
from pixel.paths.tracer import PathStyle, TraceReport, trace


@dataclass(frozen=True)
class TraceStepConfig:
    """Path tracing parameters."""

    source: Annotated[
        TraceSource,
        "Which part of the image marks the shape to trace. 'alpha' follows what "
        "a cut-out left behind, and is the one to use after remove-background. "
        "'dark' treats everything darker than the threshold as the shape, which "
        "suits line art and scans; 'light' does the opposite.",
    ] = TraceSource.ALPHA

    threshold: Annotated[
        int,
        "The level, 0-255, separating inside the shape from outside. With "
        "'alpha' it is how opaque a pixel must be to count as part of the "
        "subject; with 'dark' or 'light', how bright.",
    ] = 128

    tolerance: Annotated[
        float,
        "How far, in pixels, the path may stray from the traced border. Small "
        "values follow every wobble of the edge and produce hundreds of points; "
        "large ones give a few clean corners. This is the same control an image "
        "editor offers when converting a selection to a path.",
    ] = 2.0

    smoothness: Annotated[
        float,
        "How rounded the corners are, as a fraction of the distance between "
        "points. 0 leaves the path a sharp-cornered polygon; around 0.33 gives "
        "the flowing look of a curve drawn by hand. Above 0.5 it starts to "
        "bulge past its own corners.",
    ] = 1.0 / 3.0

    minimum_area: Annotated[
        float,
        "Shapes enclosing fewer pixels than this are ignored. Raise it to drop "
        "specks of noise that would otherwise each become a path of their own.",
    ] = 24.0

    include_holes: Annotated[
        bool,
        "Whether to trace the borders found inside a shape, such as the inside "
        "of a letter O. Turn it off to keep only the outer silhouette.",
    ] = True

    line_color: Annotated[
        tuple[int, int, int],
        "Colour the path is drawn in.",
    ] = (0, 0, 0)

    line_width: Annotated[
        int,
        "Thickness of the drawn path, in pixels.",
    ] = 2

    on_blank: Annotated[
        bool,
        "Draw the path on a plain white sheet instead of over the picture. On "
        "is the way to judge the path itself; off shows how well it follows "
        "what is underneath.",
    ] = True


class TraceStep:
    """Draws the outline of the image's shape as a path."""

    def __init__(self, config: TraceStepConfig) -> None:
        """Prepare the step.

        Args:
            config: what to trace, and how to draw it.
        """
        self._config = config

    @property
    def name(self) -> str:
        """Step name."""
        return "trace"

    def apply(self, image: RGBAImage) -> RGBAImage:
        """Trace the image and draw the paths that were found.

        Args:
            image: the image to trace.

        Returns:
            The image with its paths drawn on, leaving the alpha channel alone so
            a cut-out keeps its shape.
        """
        report = self.trace(image)

        return draw(
            image,
            report.paths,
            color=self._config.line_color,
            width=self._config.line_width,
            on_blank=self._config.on_blank,
        )

    def trace(self, image: RGBAImage) -> TraceReport:
        """Find the paths without drawing them.

        Offered separately because the paths themselves are worth having: this is
        what the SVG export calls, so that exporting and drawing cannot disagree
        about what was traced.

        Args:
            image: the image to trace.

        Returns:
            The tracing report, paths included.
        """
        return trace(
            image,
            config=TraceConfig(
                source=self._config.source,
                threshold=self._config.threshold,
                minimum_area=self._config.minimum_area,
                include_holes=self._config.include_holes,
            ),
            style=PathStyle(
                tolerance=self._config.tolerance,
                smoothness=self._config.smoothness,
            ),
        )
