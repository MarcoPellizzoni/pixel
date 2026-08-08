"""The left panel: the catalogue of steps that can be applied.

Single responsibility: show every available step, grouped by family, and offer
the two ways of using one:

- dragging it onto the pipeline panel, to append it to the sequence;
- clicking its "+" button, to apply it to the picture straight away.

The two share the same outcome, so the panel does not decide what happens. It
reports the chosen step's name and lets the application deal with it.

The step list is not written here: it is read from `pixel.registry`, the same
catalogue the command line uses. A step added to the library therefore appears in
this panel with no change to the interface at all.
"""

from __future__ import annotations

from collections.abc import Callable

import flet as ft

from pixel.registry import StepDefinition, list_definitions
from pixel.ui import theme

# Name of the drag group. A `Draggable` is only accepted by a `DragTarget` in the
# same group, which is what stops steps being dropped anywhere else in the window.
DRAG_GROUP: str = "pixel-step"


class StepLibraryPanel:
    """The panel listing every step that can be applied."""

    def __init__(
        self,
        on_apply_now: Callable[[str], None],
        on_help: Callable[[str], None],
    ) -> None:
        """Build the panel from the shared step catalogue.

        Args:
            on_apply_now: called with a step's name when its "+" button is
                clicked, meaning "apply this to the image right now".
            on_help: called with a step's name when its help is asked for.
        """
        self._on_apply_now = on_apply_now
        self._on_help = on_help

        self.control = ft.Container(
            content=ft.Column(
                controls=[
                    theme.panel_title("Steps"),
                    theme.micro("Drag onto the pipeline, or press + to apply."),
                    ft.Container(height=2),
                    ft.Column(
                        controls=self._build_grouped_entries(),
                        spacing=3,
                        scroll=ft.ScrollMode.AUTO,
                        expand=True,
                    ),
                ],
                spacing=theme.SPACING_TIGHT,
                expand=True,
            ),
            width=theme.LIBRARY_WIDTH,
            bgcolor=theme.PANEL_BACKGROUND,
            padding=ft.Padding.all(theme.SPACING),
        )

    # ------------------------------------------------------------------
    # Building the list
    # ------------------------------------------------------------------

    def _build_grouped_entries(self) -> list[ft.Control]:
        """Build the whole list, with a heading before each family.

        Returns:
            The controls making up the scrolling list, in catalogue order.
        """
        controls: list[ft.Control] = []
        current_category: str | None = None

        for definition in list_definitions():
            # The catalogue is already ordered by family, so a heading is needed
            # exactly when the family changes.
            if definition.category.value != current_category:
                current_category = definition.category.value
                controls.append(
                    ft.Container(
                        content=ft.Text(
                            value=current_category.upper(),
                            size=theme.MICRO_SIZE,
                            weight=ft.FontWeight.BOLD,
                            color=theme.TEXT_FAINT,
                        ),
                        padding=ft.Padding.only(top=theme.SPACING, bottom=2),
                    )
                )

            controls.append(self._build_entry(definition))

        return controls

    def _build_entry(self, definition: StepDefinition) -> ft.Control:
        """Build one draggable step card.

        Args:
            definition: the step's catalogue entry.

        Returns:
            A `Draggable` carrying the step name, ready to drop on the pipeline.
        """
        # The list is annotated because Python's lists are invariant: a
        # list[IconButton] is not a list[Control], and Flet asks for the latter.
        row_controls: list[ft.Control] = [
            ft.Icon(ft.Icons.DRAG_INDICATOR, size=15, color=theme.TEXT_FAINT),
            ft.Text(
                value=definition.name,
                size=theme.CAPTION_SIZE,
                color=theme.TEXT,
                expand=True,
            ),
            ft.IconButton(
                icon=ft.Icons.HELP_OUTLINE,
                icon_size=14,
                icon_color=theme.TEXT_FAINT,
                tooltip=f"What does {definition.name} do?",
                # The step name is bound as a default argument so that every
                # button keeps referring to its own step, rather than to
                # whichever one the loop happened to finish on.
                on_click=lambda _, name=definition.name: self._on_help(name),
                style=ft.ButtonStyle(padding=ft.Padding.all(2)),
            ),
            ft.IconButton(
                icon=ft.Icons.ADD,
                icon_size=16,
                icon_color=theme.ACCENT,
                tooltip=f"Apply {definition.name} now",
                on_click=lambda _, name=definition.name: self._on_apply_now(name),
                style=ft.ButtonStyle(padding=ft.Padding.all(2)),
            ),
        ]

        card = ft.Container(
            content=ft.Row(
                controls=row_controls,
                spacing=theme.SPACING_TIGHT,
                tight=True,
            ),
            bgcolor=theme.CARD_BACKGROUND,
            border_radius=theme.RADIUS_SMALL,
            padding=ft.Padding.only(left=theme.SPACING_TIGHT, right=2),
            tooltip=definition.summary,
        )

        return ft.Draggable(
            group=DRAG_GROUP,
            # `data` travels with the drag and is read back by the drop target.
            # The step name alone is enough: the target looks the rest up in the
            # catalogue.
            data=definition.name,
            content=card,
            # A compact label follows the pointer, instead of dragging a full
            # width card across the window.
            content_feedback=ft.Container(
                content=ft.Text(
                    value=definition.name,
                    size=theme.CAPTION_SIZE,
                    color=theme.TEXT,
                ),
                bgcolor=theme.ACCENT,
                border_radius=theme.RADIUS,
                padding=ft.Padding.symmetric(horizontal=theme.SPACING, vertical=6),
                opacity=0.9,
            ),
        )
