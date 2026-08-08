"""The draggable divider between a side panel and the canvas.

Single responsibility: report how far the user has dragged it, and look like
something that can be dragged.

It reports a distance rather than a width. Which panel is being resized, and what
its limits are, is not the divider's business — that belongs to the layout, which
is also the only thing that can say whether the drag should have any effect at
all once a limit is reached.

Two details make it usable rather than merely present: the grab area is several
times wider than the line that is drawn, so it does not have to be aimed at; and
the pointer changes over it, which is how anyone finds out it can be dragged in
the first place.
"""

from __future__ import annotations

from collections.abc import Callable

import flet as ft

from pixel.ui import theme
from pixel.ui.updates import refresh


class PanelSplitter:
    """A vertical divider that can be dragged to resize the panel beside it."""

    def __init__(
        self,
        on_drag: Callable[[float], None],
        on_drag_end: Callable[[], None] | None = None,
    ) -> None:
        """Build the divider.

        Args:
            on_drag: called with how far the pointer moved horizontally, in
                pixels, each time it moves while held.
            on_drag_end: called once when the pointer is let go. It exists so
                that the new width can be written down once, rather than on every
                report of a drag still in progress.
        """
        self._on_drag = on_drag
        self._on_drag_end = on_drag_end

        # The line that is actually seen. It brightens under the pointer, which
        # is the other half of saying "this can be dragged".
        self._line = ft.Container(
            width=theme.SPLITTER_WIDTH,
            bgcolor=theme.BORDER,
            expand=True,
        )

        self.control = ft.GestureDetector(
            content=ft.Container(
                content=self._line,
                # The invisible part: a comfortable target around a hairline.
                width=theme.SPLITTER_GRAB_WIDTH,
                alignment=ft.Alignment.CENTER,
                bgcolor=theme.CANVAS_BACKGROUND,
            ),
            mouse_cursor=ft.MouseCursor.RESIZE_LEFT_RIGHT,
            # Horizontal only: a vertical drag here means the user is scrolling
            # something behind, and swallowing it would be surprising.
            on_horizontal_drag_update=self._handle_drag,
            on_horizontal_drag_end=self._handle_drag_end,
            on_enter=self._handle_enter,
            on_exit=self._handle_exit,
        )

    def _handle_drag(self, event: ft.DragUpdateEvent[ft.GestureDetector]) -> None:
        """Report how far the pointer moved.

        Args:
            event: the drag. Its `local_delta` is the movement since the previous
                report, and is absent on the occasional event that carries no
                movement at all.
        """
        if event.local_delta is None:
            return

        self._on_drag(event.local_delta.x)

    def _handle_drag_end(self, event: ft.DragEndEvent[ft.GestureDetector]) -> None:
        """Report that the drag has finished."""
        if self._on_drag_end is not None:
            self._on_drag_end()

    def _handle_enter(self, event: ft.Event[ft.GestureDetector]) -> None:
        """Brighten the line while the pointer is over it."""
        self._line.bgcolor = theme.ACCENT
        refresh(self._line)

    def _handle_exit(self, event: ft.Event[ft.GestureDetector]) -> None:
        """Put the line back to its resting colour."""
        self._line.bgcolor = theme.BORDER
        refresh(self._line)
