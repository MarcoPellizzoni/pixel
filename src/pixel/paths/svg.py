"""Writing traced paths out as SVG.

Single responsibility: turn paths into the text of an SVG file, and nothing else.
Deciding where that text goes belongs to whoever asked for it.

SVG is the format to leave in: Illustrator, Inkscape, Figma and a browser all
read it, and so does Photoshop. The curves map across exactly — a cubic Bézier is
a cubic Bézier — so what is written is the path that was traced, not a picture of
it.

Holes are handled by the fill rule rather than by any arithmetic here. Every
outline goes into one `<path>` element, and `fill-rule="evenodd"` leaves the
regions enclosed an even number of times empty, which is precisely what makes the
inside of a letter O a hole.
"""

from __future__ import annotations

from pixel.paths.geometry import BezierPath, Point, TracedPaths

# How many decimal places to write. Two is finer than any screen or printer will
# resolve, and keeps the file from doubling in size for nothing.
COORDINATE_PRECISION: int = 2


def to_svg(
    traced: TracedPaths,
    stroke: str = "#000000",
    fill: str = "none",
    stroke_width: float = 1.0,
) -> str:
    """Write traced paths as a complete SVG document.

    Args:
        traced: the paths, and the size of the image they came from.
        stroke: colour of the outline, or "none" for no outline.
        fill: colour inside the paths, or "none" to leave them empty. The default
            draws the path the way an editor shows it: a line with nothing in it.
        stroke_width: thickness of the outline, in pixels.

    Returns:
        The SVG document, ready to be written to a file.
    """
    body = (
        _path_element(traced, stroke, fill, stroke_width)
        if not traced.is_empty
        else ""
    )

    return (
        '<?xml version="1.0" encoding="UTF-8"?>\n'
        f'<svg xmlns="http://www.w3.org/2000/svg" '
        f'width="{traced.width}" height="{traced.height}" '
        f'viewBox="0 0 {traced.width} {traced.height}">\n'
        f"{body}"
        "</svg>\n"
    )


def to_path_data(path: BezierPath) -> str:
    """Write one path as the contents of an SVG `d` attribute.

    Args:
        path: the path to write.

    Returns:
        A `d` string: a move, a run of curves, and a close.
    """
    pieces = [f"M {_pair(path.start)}"]

    pieces.extend(
        f"C {_pair(segment.control_start)} "
        f"{_pair(segment.control_end)} {_pair(segment.end)}"
        for segment in path.segments
    )

    # Closing the path is what lets the fill rule decide the holes.
    pieces.append("Z")

    return " ".join(pieces)


def _path_element(
    traced: TracedPaths, stroke: str, fill: str, stroke_width: float
) -> str:
    """Build the single `<path>` element holding every traced path.

    Keeping them in one element is what makes the fill rule apply across them,
    and so what makes holes come out as holes.

    Args:
        traced: the paths to write.
        stroke: outline colour.
        fill: fill colour.
        stroke_width: outline thickness.

    Returns:
        The element, indented ready to sit inside the document.
    """
    data = " ".join(to_path_data(path) for path in traced.paths)

    return (
        f'  <path d="{data}" fill="{fill}" fill-rule="evenodd" '
        f'stroke="{stroke}" stroke-width="{stroke_width}"/>\n'
    )


def _pair(point: Point) -> str:
    """Write one point as a pair of coordinates.

    Args:
        point: the point to write.

    Returns:
        Its coordinates, rounded and separated by a comma.
    """
    x, y = point

    return f"{round(x, COORDINATE_PRECISION)},{round(y, COORDINATE_PRECISION)}"
