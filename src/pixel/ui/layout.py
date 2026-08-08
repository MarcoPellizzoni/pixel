"""How wide the side panels are, and whether they are showing at all.

Single responsibility: hold the arrangement of the window — the two panel widths
and whether each is open — and keep it within sensible bounds.

It is a plain value with no Flet in it, for two reasons. It can be tested by
itself, and it can be written to disk and read back, which is what lets the
editor open next time looking the way it was left.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, cast

from pixel.ui import theme


@dataclass
class PanelLayout:
    """The width and visibility of the two side panels.

    Attributes:
        library_width: width of the step library, on the left.
        pipeline_width: width of the pipeline panel, on the right.
        library_visible: whether the library is showing.
        pipeline_visible: whether the pipeline panel is showing.
    """

    library_width: int = theme.LIBRARY_WIDTH
    pipeline_width: int = theme.PIPELINE_WIDTH
    library_visible: bool = True
    pipeline_visible: bool = True

    def __post_init__(self) -> None:
        """Pull the widths inside their limits, whatever they were set to.

        Doing it here means a layout read from a file written by an older version
        — or edited by hand — cannot produce a panel too narrow to use or wide
        enough to swallow the picture.
        """
        self.library_width = _clamp(
            self.library_width, theme.LIBRARY_MIN_WIDTH, theme.LIBRARY_MAX_WIDTH
        )
        self.pipeline_width = _clamp(
            self.pipeline_width, theme.PIPELINE_MIN_WIDTH, theme.PIPELINE_MAX_WIDTH
        )

    # ------------------------------------------------------------------
    # Resizing
    # ------------------------------------------------------------------

    def resize_library(self, delta: float) -> None:
        """Widen or narrow the library by a dragged amount.

        Args:
            delta: how far the divider moved, in pixels. Positive widens, because
                the library's divider sits on its right-hand edge.
        """
        self.library_width = _clamp(
            round(self.library_width + delta),
            theme.LIBRARY_MIN_WIDTH,
            theme.LIBRARY_MAX_WIDTH,
        )

    def resize_pipeline(self, delta: float) -> None:
        """Widen or narrow the pipeline panel by a dragged amount.

        Args:
            delta: how far the divider moved, in pixels. The sign is flipped
                against the library's, because this divider sits on the panel's
                left-hand edge: dragging left makes the panel wider.
        """
        self.pipeline_width = _clamp(
            round(self.pipeline_width - delta),
            theme.PIPELINE_MIN_WIDTH,
            theme.PIPELINE_MAX_WIDTH,
        )

    # ------------------------------------------------------------------
    # Opening and closing
    # ------------------------------------------------------------------

    def toggle_library(self) -> None:
        """Show the library if it is hidden, hide it if it is showing."""
        self.library_visible = not self.library_visible

    def toggle_pipeline(self) -> None:
        """Show the pipeline panel if it is hidden, hide it if it is showing."""
        self.pipeline_visible = not self.pipeline_visible

    # ------------------------------------------------------------------
    # Writing it down and reading it back
    # ------------------------------------------------------------------

    def to_dict(self) -> dict[str, Any]:
        """Describe the layout in a form that can be written to a file.

        Returns:
            A mapping of plain values, ready for JSON.
        """
        return {
            "library_width": self.library_width,
            "pipeline_width": self.pipeline_width,
            "library_visible": self.library_visible,
            "pipeline_visible": self.pipeline_visible,
        }

    @classmethod
    def from_dict(cls, stored: object) -> PanelLayout:
        """Rebuild a layout from what was written to a file.

        Anything missing or of the wrong type falls back to the default, so a
        truncated or hand-edited file costs the user a preference rather than a
        working editor.

        Args:
            stored: whatever was read back from the file.

        Returns:
            The layout, with every value within its limits.
        """
        values = cast_mapping(stored)

        return cls(
            library_width=_read_int(values, "library_width", theme.LIBRARY_WIDTH),
            pipeline_width=_read_int(values, "pipeline_width", theme.PIPELINE_WIDTH),
            library_visible=_read_bool(values, "library_visible", True),
            pipeline_visible=_read_bool(values, "pipeline_visible", True),
        )


def cast_mapping(stored: object) -> dict[str, object]:
    """Narrow whatever came back from a file into a mapping with string keys.

    The check lives here rather than at each call: a reader hands back `object`,
    and narrowing it in one declared place keeps every caller free of the partial
    types that fall out of `isinstance` on a plain `dict`.

    Args:
        stored: whatever the reader produced.

    Returns:
        The entries as a mapping with string keys, or an empty one if what came
        back was not a mapping at all.
    """
    if not isinstance(stored, dict):
        return {}

    entries = cast(dict[object, object], stored)
    return {str(key): value for key, value in entries.items()}


def _read_int(values: dict[str, object], key: str, fallback: int) -> int:
    """Read a whole number from stored values, falling back when it is not one."""
    value = values.get(key)
    # `bool` is a subclass of `int`, and a stored `true` should not become 1.
    if isinstance(value, int) and not isinstance(value, bool):
        return value
    return fallback


def _read_bool(values: dict[str, object], key: str, fallback: bool) -> bool:
    """Read a true/false value from stored values, falling back when it is not one."""
    value = values.get(key)
    if isinstance(value, bool):
        return value
    return fallback


def _clamp(value: int, lowest: int, highest: int) -> int:
    """Hold a number between two bounds.

    Args:
        value: the number to constrain.
        lowest: the smallest it may be.
        highest: the largest it may be.

    Returns:
        The number, pulled inside the bounds.
    """
    return max(lowest, min(value, highest))
