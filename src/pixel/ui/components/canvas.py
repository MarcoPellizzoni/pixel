"""The centre of the window: the photo being edited.

Single responsibility: show the current image, or an invitation to open one when
there is none yet.

The component knows nothing about steps or editing. It is handed ready-made PNG
bytes and displays them, which keeps the decision of *what* the picture should
look like entirely inside the session and `preview`.
"""

from __future__ import annotations

import flet as ft

from pixel.ui import theme
from pixel.ui.updates import refresh


class EditorCanvas:
    """The area showing the image under edit."""

    def __init__(self) -> None:
        """Build the canvas, initially empty."""
        # A Flet `Image` cannot exist without a source, so it is created on the
        # first image rather than up front. From then on only its `src` changes:
        # replacing the whole control on every edit would make the picture flicker.
        self._image: ft.Image | None = None

        self._placeholder: ft.Control = ft.Column(
            controls=[
                ft.Icon(
                    ft.Icons.IMAGE_OUTLINED,
                    size=64,
                    color=theme.TEXT_MUTED,
                ),
                ft.Text(
                    value="Open an image to start editing",
                    size=theme.BODY_SIZE,
                    color=theme.TEXT_MUTED,
                ),
            ],
            alignment=ft.MainAxisAlignment.CENTER,
            horizontal_alignment=ft.CrossAxisAlignment.CENTER,
            spacing=theme.SPACING,
        )

        self.control = ft.Container(
            content=self._placeholder,
            expand=True,
            bgcolor=theme.CANVAS_BACKGROUND,
            padding=ft.Padding.all(theme.SPACING * 2),
            alignment=ft.Alignment.CENTER,
        )

    def show_image(self, png_bytes: bytes) -> None:
        """Display an image, replacing whatever was on screen.

        Args:
            png_bytes: the encoded image, as produced by `preview.to_png_bytes`.
        """
        if self._image is None:
            self._image = ft.Image(
                src=png_bytes,
                fit=ft.BoxFit.CONTAIN,
                expand=True,
            )
            self.control.content = self._image
        else:
            # Same control, new pixels: the picture is swapped without the canvas
            # blanking out in between.
            self._image.src = png_bytes

        refresh(self.control)

    def show_placeholder(self) -> None:
        """Go back to the empty state, with no image loaded."""
        self._image = None
        self.control.content = self._placeholder
        refresh(self.control)
