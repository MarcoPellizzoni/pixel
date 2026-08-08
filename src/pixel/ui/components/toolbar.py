"""The top bar: the actions that apply to the picture as a whole.

Single responsibility: offer the actions that apply to the whole picture or the
whole window — open, save, undo, redo, reset, the help, and showing or hiding the
side panels — and say which file is being edited and when the editor is busy.

The bar decides nothing. Each button reports that it was pressed and the
application carries the action out, which is what lets the same bar be driven by
a session in any state.
"""

from __future__ import annotations

from collections.abc import Callable

import flet as ft

from pixel.ui import theme
from pixel.ui.updates import refresh


class EditorToolbar:
    """The row of whole-image actions along the top of the window."""

    def __init__(
        self,
        on_open: Callable[[], None],
        on_save: Callable[[], None],
        on_undo: Callable[[], None],
        on_redo: Callable[[], None],
        on_reset: Callable[[], None],
        on_help: Callable[[], None],
        on_toggle_library: Callable[[], None],
        on_toggle_pipeline: Callable[[], None],
        on_save_session: Callable[[], None],
        on_open_session: Callable[[], None],
    ) -> None:
        """Build the toolbar with every action disabled but "open".

        Args:
            on_open: called to choose an image to edit.
            on_save: called to write the current image to a file.
            on_undo: called to take back the last change.
            on_redo: called to put back a change that was taken back.
            on_reset: called to discard every applied step.
            on_help: called to show the help covering every step.
            on_toggle_library: called to show or hide the step library.
            on_toggle_pipeline: called to show or hide the pipeline panel.
            on_save_session: called to write the pipeline to a named file.
            on_open_session: called to load a pipeline from a named file.
        """
        self._open_button = ft.Button(
            content="Open",
            icon=ft.Icons.FOLDER_OPEN,
            on_click=lambda _: on_open(),
        )
        self._save_button = ft.Button(
            content="Save",
            icon=ft.Icons.SAVE,
            on_click=lambda _: on_save(),
            disabled=True,
        )
        self._undo_button = ft.Button(
            content="Undo",
            icon=ft.Icons.UNDO,
            on_click=lambda _: on_undo(),
            disabled=True,
        )
        self._redo_button = ft.Button(
            content="Redo",
            icon=ft.Icons.REDO,
            on_click=lambda _: on_redo(),
            disabled=True,
        )
        self._reset_button = ft.Button(
            content="Reset",
            icon=ft.Icons.RESTART_ALT,
            on_click=lambda _: on_reset(),
            disabled=True,
            style=ft.ButtonStyle(color=theme.DANGER),
        )

        # Saving and opening a session are kept behind a menu rather than given
        # buttons of their own. They are used far less often than Save, and a bar
        # of nine buttons stops being read at a glance.
        self._session_menu = ft.PopupMenuButton(
            icon=ft.Icons.FOLDER_SPECIAL_OUTLINED,
            tooltip="Sessions",
            items=[
                ft.PopupMenuItem(
                    content="Save session as...",
                    icon=ft.Icons.BOOKMARK_ADD_OUTLINED,
                    on_click=lambda _: on_save_session(),
                ),
                ft.PopupMenuItem(
                    content="Open session...",
                    icon=ft.Icons.BOOKMARK_OUTLINED,
                    on_click=lambda _: on_open_session(),
                ),
            ],
        )

        # The two panel toggles sit at the far left, beside the panels they
        # govern, rather than among the actions that change the picture.
        self._library_toggle = ft.IconButton(
            icon=ft.Icons.VIEW_SIDEBAR_OUTLINED,
            icon_color=theme.TEXT_MUTED,
            tooltip="Show or hide the steps",
            on_click=lambda _: on_toggle_library(),
        )
        self._pipeline_toggle = ft.IconButton(
            icon=ft.Icons.VIEW_SIDEBAR_OUTLINED,
            icon_color=theme.TEXT_MUTED,
            tooltip="Show or hide the pipeline",
            on_click=lambda _: on_toggle_pipeline(),
            # Mirrored, so each button points at the side it opens.
            rtl=True,
        )

        self._help_button = ft.IconButton(
            icon=ft.Icons.HELP_OUTLINE,
            icon_color=theme.TEXT_MUTED,
            tooltip="What the steps do",
            on_click=lambda _: on_help(),
        )

        # Name of the file being edited, and its dimensions.
        self._file_label = ft.Text(
            value="No image open",
            size=theme.BODY_SIZE,
            color=theme.TEXT_MUTED,
        )

        # Only visible while a step is running. Steps such as background removal
        # take seconds, and without this the window would look frozen.
        self._busy_indicator = ft.ProgressRing(
            width=16,
            height=16,
            stroke_width=2,
            color=theme.ACCENT,
            visible=False,
        )

        self.control = ft.Container(
            content=ft.Row(
                controls=[
                    self._library_toggle,
                    ft.VerticalDivider(width=1, color=theme.BORDER),
                    self._open_button,
                    self._save_button,
                    self._session_menu,
                    ft.VerticalDivider(width=1, color=theme.BORDER),
                    self._undo_button,
                    self._redo_button,
                    self._reset_button,
                    ft.Container(width=theme.SPACING),
                    self._busy_indicator,
                    ft.Container(expand=True),
                    self._file_label,
                    self._help_button,
                    self._pipeline_toggle,
                ],
                spacing=theme.SPACING_TIGHT,
                vertical_alignment=ft.CrossAxisAlignment.CENTER,
            ),
            height=theme.TOOLBAR_HEIGHT,
            bgcolor=theme.PANEL_BACKGROUND,
            padding=ft.Padding.symmetric(horizontal=theme.SPACING),
        )

    def set_state(
        self,
        *,
        has_image: bool,
        can_undo: bool,
        can_redo: bool,
        is_modified: bool,
    ) -> None:
        """Enable or disable the buttons to match the session.

        Args:
            has_image: whether an image is currently open.
            can_undo: whether there is a change that could be taken back.
            can_redo: whether a change that was taken back could be put back.
            is_modified: whether any step has been applied.
        """
        self._save_button.disabled = not has_image
        self._undo_button.disabled = not can_undo
        self._redo_button.disabled = not can_redo
        self._reset_button.disabled = not is_modified
        # Opening a session is always available; saving one needs something to
        # save. The menu itself stays reachable so the first item is discoverable.
        self._session_menu.disabled = False
        refresh(self.control)

    def set_file_label(self, text: str) -> None:
        """Show which file is being edited.

        Args:
            text: the caption to display, typically name and dimensions.
        """
        self._file_label.value = text
        refresh(self.control)

    def set_busy(self, busy: bool) -> None:
        """Show or hide the spinner and lock the buttons while working.

        Args:
            busy: True while a step is running.
        """
        self._busy_indicator.visible = busy

        # Every action is locked during the work, so a second step cannot be
        # started on an image that is still being computed.
        self._open_button.disabled = busy
        if busy:
            self._save_button.disabled = True
            self._undo_button.disabled = True
            self._redo_button.disabled = True
            self._reset_button.disabled = True

        refresh(self.control)
