"""Tests for tracing an image into vector paths.

The shapes used are ones whose answer is known in advance — a square has four
corners, a ring has an outside and a hole — so that what the border search found
can be judged rather than merely observed.
"""

from __future__ import annotations

import numpy as np
import pytest

from pixel.domain import RGBAImage
from pixel.paths import (
    Outline,
    PathStyle,
    TraceConfig,
    TraceSource,
    build_mask,
    find_outlines,
    simplify,
    to_bezier,
    to_svg,
    trace,
)
from pixel.paths.raster import flatten


def square(side: int = 40, size: int = 100, opaque: bool = True) -> RGBAImage:
    """An image holding one filled square, centred.

    Args:
        side: the square's side, in pixels.
        size: the image's width and height.
        opaque: whether the square is marked in the alpha channel (True) or drawn
            dark on a light background (False).

    Returns:
        The image.
    """
    data = np.zeros((size, size, 4), dtype=np.uint8)
    start = (size - side) // 2
    end = start + side

    if opaque:
        data[:, :, :3] = 200
        data[start:end, start:end, 3] = 255
    else:
        data[:, :, :3] = 255
        data[:, :, 3] = 255
        data[start:end, start:end, :3] = 0

    return RGBAImage(data)


def ring(size: int = 120) -> RGBAImage:
    """An image holding a ring: a disc with a hole through the middle."""
    data = np.zeros((size, size, 4), dtype=np.uint8)
    data[:, :, :3] = 200

    rows, columns = np.mgrid[0:size, 0:size]
    radius = np.hypot(rows - size / 2, columns - size / 2)
    data[:, :, 3] = np.where((radius < size * 0.42) & (radius > size * 0.18), 255, 0)

    return RGBAImage(data)


class TestFindingTheOutline:
    """The border search must find the shape that is there."""

    def test_a_square_gives_one_outline(self) -> None:
        outlines = find_outlines(square(), TraceConfig())

        assert len(outlines) == 1
        assert not outlines[0].is_hole

    def test_the_outline_follows_the_square(self) -> None:
        # A 40-pixel square centred in 100 sits between 30 and 69 inclusive.
        outlines = find_outlines(square(side=40, size=100), TraceConfig())

        xs = [x for x, _ in outlines[0].points]
        ys = [y for _, y in outlines[0].points]
        assert (min(xs), max(xs)) == (30.0, 69.0)
        assert (min(ys), max(ys)) == (30.0, 69.0)

    def test_a_ring_gives_an_outside_and_a_hole(self) -> None:
        outlines = find_outlines(ring(), TraceConfig())

        assert len(outlines) == 2
        assert [outline.is_hole for outline in outlines] == [False, True]

    def test_holes_can_be_left_out(self) -> None:
        outlines = find_outlines(ring(), TraceConfig(include_holes=False))

        assert len(outlines) == 1
        assert not outlines[0].is_hole

    def test_an_empty_image_gives_nothing(self) -> None:
        blank = RGBAImage(np.zeros((40, 40, 4), dtype=np.uint8))

        assert find_outlines(blank, TraceConfig()) == ()

    def test_specks_smaller_than_the_minimum_are_dropped(self) -> None:
        # A big square and a two-pixel dot; only the square should survive.
        image = square(side=40, size=100)
        image.data[5:7, 5:7, 3] = 255

        outlines = find_outlines(image, TraceConfig(minimum_area=24.0))

        assert len(outlines) == 1

    def test_the_largest_outline_comes_first(self) -> None:
        image = square(side=40, size=100)
        image.data[5:20, 5:20, 3] = 255

        outlines = find_outlines(image, TraceConfig(minimum_area=1.0))

        assert len(outlines[0].points) >= len(outlines[1].points)


class TestWhatCountsAsInside:
    """Each source must pick out the region it claims to."""

    def test_alpha_follows_the_cut_out(self) -> None:
        mask = build_mask(square(opaque=True), TraceConfig(source=TraceSource.ALPHA))

        assert mask[50, 50] == 255
        assert mask[2, 2] == 0

    def test_dark_finds_ink_on_paper(self) -> None:
        mask = build_mask(
            square(opaque=False), TraceConfig(source=TraceSource.DARK, threshold=128)
        )

        assert mask[50, 50] == 255
        assert mask[2, 2] == 0

    def test_light_is_dark_the_other_way_up(self) -> None:
        dark = build_mask(square(opaque=False), TraceConfig(source=TraceSource.DARK))
        light = build_mask(square(opaque=False), TraceConfig(source=TraceSource.LIGHT))

        assert np.array_equal(light, 255 - dark)

    def test_the_threshold_decides_where_the_edge_falls(self) -> None:
        # A soft edge: at a low threshold more of it counts as inside.
        data = np.zeros((1, 256, 4), dtype=np.uint8)
        data[0, :, 3] = np.arange(256, dtype=np.uint8)
        image = RGBAImage(data)

        low = build_mask(image, TraceConfig(threshold=50)).sum()
        high = build_mask(image, TraceConfig(threshold=200)).sum()

        assert low > high


class TestSimplifying:
    """The tolerance must be the control it claims to be."""

    def test_a_square_traced_pixel_by_pixel_reduces_to_four_corners(self) -> None:
        outline = find_outlines(square(), TraceConfig())[0]

        simplified = simplify(outline, tolerance=2.0)

        assert len(simplified.points) == 4

    def test_a_larger_tolerance_keeps_fewer_points(self) -> None:
        outline = find_outlines(ring(), TraceConfig())[0]

        loose = simplify(outline, tolerance=6.0)
        tight = simplify(outline, tolerance=0.5)

        assert len(loose.points) < len(tight.points)

    def test_zero_tolerance_keeps_every_point(self) -> None:
        outline = find_outlines(square(), TraceConfig())[0]

        assert simplify(outline, tolerance=0.0).points == outline.points

    def test_a_hole_stays_a_hole(self) -> None:
        hole = find_outlines(ring(), TraceConfig())[1]

        assert simplify(hole, tolerance=3.0).is_hole

    def test_a_tolerance_that_would_collapse_the_shape_is_refused(self) -> None:
        # Better to hand back the original than something that cannot be drawn.
        outline = find_outlines(square(), TraceConfig())[0]

        simplified = simplify(outline, tolerance=10_000.0)

        assert len(simplified.points) >= 3


class TestCurves:
    """Turning corners into curves must not move the shape."""

    def test_every_edge_becomes_a_curve(self) -> None:
        outline = Outline(points=((0.0, 0.0), (10.0, 0.0), (10.0, 10.0)))

        path = to_bezier(outline)

        # One curve per edge, the closing edge included.
        assert len(path.segments) == 3

    def test_the_anchors_are_the_original_points(self) -> None:
        points = ((0.0, 0.0), (10.0, 0.0), (10.0, 10.0), (0.0, 10.0))

        path = to_bezier(Outline(points=points))

        assert path.anchors[: len(points)] == points

    def test_zero_smoothness_leaves_the_handles_on_the_anchors(self) -> None:
        # With no smoothing the path is exactly the polygon it came from.
        outline = Outline(points=((0.0, 0.0), (10.0, 0.0), (10.0, 10.0)))

        path = to_bezier(outline, smoothness=0.0)

        assert path.segments[0].control_start == path.start
        assert path.segments[0].control_end == path.segments[0].end

    def test_smoothing_moves_the_handles_off_the_anchors(self) -> None:
        outline = Outline(points=((0.0, 0.0), (10.0, 0.0), (10.0, 10.0)))

        path = to_bezier(outline, smoothness=0.5)

        assert path.segments[0].control_start != path.start

    def test_a_hole_stays_a_hole(self) -> None:
        outline = Outline(points=((0.0, 0.0), (5.0, 0.0), (5.0, 5.0)), is_hole=True)

        assert to_bezier(outline).is_hole

    def test_the_curve_passes_through_its_anchors(self) -> None:
        # Whatever the handles do, the drawn curve must start and finish on the
        # points that were traced.
        outline = Outline(points=((0.0, 0.0), (20.0, 0.0), (20.0, 20.0)))
        path = to_bezier(outline)

        drawn = flatten(path)

        assert drawn[0] == pytest.approx(path.start)
        assert drawn[-1] == pytest.approx(path.start, abs=1e-6)


class TestTracingEndToEnd:
    """The three stages together must produce a usable path."""

    def test_a_square_traces_to_a_four_cornered_path(self) -> None:
        report = trace(square(), style=PathStyle(tolerance=2.0))

        assert len(report.paths) == 1
        assert len(report.paths.paths[0].segments) == 4

    def test_the_report_says_how_much_was_saved(self) -> None:
        report = trace(square(), style=PathStyle(tolerance=2.0))

        assert report.points_before > report.points_after
        assert 0.0 < report.reduction < 1.0

    def test_the_canvas_size_is_carried_along(self) -> None:
        report = trace(square(size=100))

        assert (report.paths.width, report.paths.height) == (100, 100)

    def test_an_empty_image_traces_to_nothing(self) -> None:
        blank = RGBAImage(np.zeros((40, 40, 4), dtype=np.uint8))

        report = trace(blank)

        assert report.paths.is_empty
        assert len(report.paths) == 0


class TestSvg:
    """The SVG written must be one another program would accept."""

    def test_it_is_a_complete_document(self) -> None:
        svg = to_svg(trace(square()).paths)

        assert svg.startswith("<?xml")
        assert "<svg" in svg
        assert svg.rstrip().endswith("</svg>")

    def test_the_canvas_matches_the_image(self) -> None:
        svg = to_svg(trace(square(size=100)).paths)

        assert 'width="100"' in svg
        assert 'viewBox="0 0 100 100"' in svg

    def test_the_path_is_closed_and_made_of_curves(self) -> None:
        svg = to_svg(trace(square()).paths)

        assert " C " in svg
        assert svg.count("Z") >= 1

    def test_holes_are_left_to_the_fill_rule(self) -> None:
        # Both outlines go into one element, so evenodd can empty the hole.
        svg = to_svg(trace(ring()).paths)

        assert 'fill-rule="evenodd"' in svg
        assert svg.count("<path") == 1
        assert svg.count(" M ") + svg.count('"M ') == 2

    def test_an_empty_tracing_still_gives_a_valid_document(self) -> None:
        blank = RGBAImage(np.zeros((40, 40, 4), dtype=np.uint8))

        svg = to_svg(trace(blank).paths)

        assert "<svg" in svg
        assert "<path" not in svg

    def test_it_can_be_parsed_as_xml(self) -> None:
        # The strongest cheap check that the file is well formed.
        import xml.etree.ElementTree as ElementTree

        svg = to_svg(trace(ring()).paths)

        ElementTree.fromstring(svg)
