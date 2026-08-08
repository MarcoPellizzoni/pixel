"""The right panel: the pipeline, and everything that can be done to it.

Single responsibility: show the steps that make up the pipeline and offer the
ways of changing it — accept a new one dropped from the library, move one up or
down, open one to change its settings, remove one, or read what any of them does.

The list is not a plan waiting to be run: it is what the picture is currently
made of. Every change takes effect immediately, so the panel and the canvas can
never disagree.

Order is meaningful — greying then inverting is not the same as inverting then
greying — which is why moving a step is an edit like any other and not just
tidying.

At the bottom the same pipeline appears written in the command line's syntax,
ready to be copied into a terminal.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence

import flet as ft

from pixel.params import describe_parameters
from pixel.registry import get_definition
from pixel.ui import theme
from pixel.ui.components.library import DRAG_GROUP
from pixel.ui.components.parameters import ParameterEditor
from pixel.ui.session import AppliedStep
from pixel.ui.updates import refresh

# Beyond this many settings, a step's fields get their own scrolling area rather
# than making the card taller than the window.
SETTINGS_SCROLL_THRESHOLD: int = 5

# Height of that scrolling area. Tall enough to show three or four fields with
# their explanations, so scrolling is a nudge rather than a hunt.
SETTINGS_MAX_HEIGHT: int = 320


class PipelinePanel:
    """The panel holding the pipeline and the controls that change it."""

    def __init__(
        self,
        on_step_dropped: Callable[[str], None],
        on_move: Callable[[int, int], None],
        on_remove: Callable[[int], None],
        on_parameters_changed: Callable[[int, dict[str, str]], None],
        on_help: Callable[[str], None],
    ) -> None:
        """Build the panel, initially empty.

        Args:
            on_step_dropped: called with a step's name when one is dropped here.
            on_move: called with a step's position and where it should go.
            on_remove: called with the position of a step to delete.
            on_parameters_changed: called with a step's position and its new
                parameters when the user edits one of its fields.
            on_help: called with a step's name when its help is asked for.
        """
        self._on_step_dropped = on_step_dropped
        self._on_move = on_move
        self._on_remove = on_remove
        self._on_parameters_changed = on_parameters_changed
        self._on_help = on_help

        # Which step is open for editing, by position. Only one at a time: two
        # open panels of fields would leave no room for the list itself.
        self._expanded: int | None = None

        # The pipeline as last drawn. Kept so that opening or closing a step's
        # settings can redraw the panel on its own, without having to ask the
        # application for state it has already been given.
        self._applied: tuple[AppliedStep, ...] = ()
        self._pipeline_source: str = ""

        # A `ListView` rather than a scrolling `Column`: it is built for exactly
        # this job, and it scrolls on the height its parent gives it instead of
        # first having to work out how tall its contents are. A step with a dozen
        # settings makes the list far taller than the panel, and this is what
        # keeps the rest of it reachable.
        self._step_list = ft.ListView(
            controls=[],
            spacing=theme.SPACING_TIGHT,
            expand=True,
        )

        hint_controls: list[ft.Control] = [
            ft.Icon(ft.Icons.ADD_CIRCLE_OUTLINE, size=28, color=theme.TEXT_FAINT),
            theme.caption("Drop a step here"),
            theme.micro("or press + in the library"),
        ]

        self._empty_hint = ft.Container(
            content=ft.Column(
                controls=hint_controls,
                alignment=ft.MainAxisAlignment.CENTER,
                horizontal_alignment=ft.CrossAxisAlignment.CENTER,
                spacing=theme.SPACING_TIGHT,
            ),
            alignment=ft.Alignment.CENTER,
            expand=True,
        )

        # The area the steps are listed in. Its content is swapped between the
        # hint and the list rather than the two being stacked: a `Stack` gives
        # its children no definite height, and a scrolling list without one never
        # works out that it has more to show than fits.
        self._drop_zone = ft.Container(
            content=self._empty_hint,
            expand=True,
            bgcolor=theme.CANVAS_BACKGROUND,
            border=ft.Border.all(1, theme.BORDER),
            border_radius=theme.RADIUS,
            padding=ft.Padding.all(theme.SPACING_TIGHT),
        )

        # Read-only: it mirrors the pipeline rather than being a way to enter it,
        # so that there is only ever one source of truth.
        self._pipeline_text = ft.Text(
            value="",
            size=theme.MICRO_SIZE,
            color=theme.TEXT_FAINT,
            selectable=True,
            font_family="monospace",
        )

        self._pipeline_box = ft.Container(
            content=self._pipeline_text,
            bgcolor=theme.CANVAS_BACKGROUND,
            border_radius=theme.RADIUS_SMALL,
            padding=ft.Padding.all(theme.SPACING_TIGHT + 2),
            visible=False,
        )

        # The panel's visible body. It is kept to hand because the width the user
        # drags the panel to is set on it, the drag target having none of its own.
        self._panel_box = self._build_panel_box()

        # The drag target wraps the whole panel rather than sitting between the
        # column and the list. Two reasons: a step can then be dropped anywhere
        # on the panel, which is a far easier target to hit; and the list's
        # height now comes straight from the window through plain containers,
        # instead of having to survive a trip through the drag target, which does
        # not pass a definite height on to what it wraps.
        self.control = ft.DragTarget(
            group=DRAG_GROUP,
            expand=True,
            on_will_accept=self._handle_will_accept,
            on_leave=self._handle_leave,
            on_accept=self._handle_accept,
            content=self._panel_box,
        )

    def set_width(self, width: int) -> None:
        """Set how wide the panel is drawn.

        The width sits on the container inside the drag target rather than on the
        target itself, which has none of its own.

        Args:
            width: the width in pixels.
        """
        self._panel_box.width = width

    def _build_panel_box(self) -> ft.Container:
        """Build the panel's visible body: heading, step list and pipeline text."""
        return ft.Container(
            content=ft.Column(
                controls=[
                    theme.panel_title("Pipeline"),
                    theme.micro("Applied in order, top to bottom."),
                    ft.Container(height=2),
                    self._drop_zone,
                    self._pipeline_box,
                ],
                spacing=theme.SPACING_TIGHT,
                expand=True,
            ),
            width=theme.PIPELINE_WIDTH,
            bgcolor=theme.PANEL_BACKGROUND,
            padding=ft.Padding.all(theme.SPACING),
        )

    # ------------------------------------------------------------------
    # Drag and drop
    # ------------------------------------------------------------------

    def _handle_will_accept(self, event: ft.DragWillAcceptEvent) -> None:
        """Highlight the drop zone while a step hovers over it."""
        self._drop_zone.border = ft.Border.all(2, theme.ACCENT)
        self._drop_zone.bgcolor = theme.ACCENT_SOFT
        refresh(self._drop_zone)

    def _handle_leave(self, event: ft.DragTargetLeaveEvent) -> None:
        """Remove the highlight when the step is dragged back out."""
        self._reset_drop_zone()

    def _handle_accept(self, event: ft.DragTargetEvent) -> None:
        """Take delivery of a dropped step and report it.

        Args:
            event: the drop event; its source carries the step's name in `data`.
        """
        self._reset_drop_zone()

        # `data` is whatever the `Draggable` was given, which the library sets to
        # the step name. Anything else means a control from another group somehow
        # reached us, and is ignored rather than trusted.
        step_name = event.src.data
        if isinstance(step_name, str):
            self._on_step_dropped(step_name)

    def _reset_drop_zone(self) -> None:
        """Put the drop zone's ordinary appearance back."""
        self._drop_zone.border = ft.Border.all(1, theme.BORDER)
        self._drop_zone.bgcolor = theme.CANVAS_BACKGROUND
        refresh(self._drop_zone)

    # ------------------------------------------------------------------
    # Showing the pipeline
    # ------------------------------------------------------------------

    def show_steps(self, applied: Sequence[AppliedStep], pipeline_text: str) -> None:
        """Redraw the list to match the session.

        Args:
            applied: the steps making up the pipeline, first to last.
            pipeline_text: the same pipeline in command line syntax.
        """
        self._applied = tuple(applied)
        self._pipeline_source = pipeline_text

        # A step may have been removed while its fields were open, so the
        # expanded position is checked against the list that actually exists.
        if self._expanded is not None and self._expanded >= len(applied):
            self._expanded = None

        self._step_list.controls = [
            self._build_card(position, step, len(applied))
            for position, step in enumerate(applied)
        ]

        # The hint and the list swap places: an empty list would otherwise leave
        # the panel looking broken rather than simply empty.
        self._drop_zone.content = self._step_list if applied else self._empty_hint

        self._pipeline_text.value = pipeline_text
        self._pipeline_box.visible = bool(pipeline_text)

        refresh(self.control)

    def _build_card(
        self, position: int, step: AppliedStep, total: int
    ) -> ft.Control:
        """Build one step's card: its header, and its fields when open.

        Args:
            position: the step's place in the pipeline, counting from 0.
            step: the step and the image it produced.
            total: how many steps the pipeline holds, so the move buttons know
                which ends they are at.

        Returns:
            The card control.
        """
        is_open = self._expanded == position
        name = step.invocation.name

        body: list[ft.Control] = [self._build_header(position, step, total, is_open)]

        if is_open:
            body.append(self._build_settings(position, name, step.invocation.parameters))

        return ft.Container(
            content=ft.Column(controls=body, spacing=theme.SPACING_TIGHT, tight=True),
            bgcolor=theme.ACCENT_SOFT if is_open else theme.CARD_BACKGROUND,
            border_radius=theme.RADIUS_SMALL,
            padding=ft.Padding.all(theme.SPACING_TIGHT),
        )

    def _build_header(
        self, position: int, step: AppliedStep, total: int, is_open: bool
    ) -> ft.Control:
        """Build a card's top row: number, name, and the buttons acting on it.

        Args:
            position: the step's place in the pipeline, counting from 0.
            step: the step and the image it produced.
            total: how many steps the pipeline holds.
            is_open: whether this step's fields are currently showing.

        Returns:
            The header row.
        """
        # A summary of the settings that differ from the defaults, so a closed
        # card still says how it is configured.
        parameters = step.invocation.parameters
        summary = (
            ", ".join(f"{key}={value}" for key, value in parameters.items())
            if parameters
            else "default settings"
        )

        controls: list[ft.Control] = [
            ft.Container(
                content=ft.Text(
                    value=str(position + 1),
                    size=theme.MICRO_SIZE,
                    color=theme.TEXT_FAINT,
                ),
                width=16,
            ),
            ft.Column(
                controls=[
                    ft.Text(
                        value=step.invocation.name,
                        size=theme.CAPTION_SIZE,
                        weight=ft.FontWeight.W_500,
                        color=theme.TEXT,
                    ),
                    theme.micro(summary),
                ],
                spacing=0,
                tight=True,
                expand=True,
            ),
            self._icon_button(
                ft.Icons.KEYBOARD_ARROW_UP,
                "Move up",
                disabled=position == 0,
                on_click=lambda _: self._on_move(position, position - 1),
            ),
            self._icon_button(
                ft.Icons.KEYBOARD_ARROW_DOWN,
                "Move down",
                disabled=position == total - 1,
                on_click=lambda _: self._on_move(position, position + 1),
            ),
            self._icon_button(
                ft.Icons.TUNE,
                "Hide settings" if is_open else "Settings",
                colour=theme.ACCENT if is_open else theme.TEXT_MUTED,
                on_click=lambda _: self._toggle(position),
            ),
            self._icon_button(
                ft.Icons.CLOSE,
                "Remove",
                colour=theme.TEXT_FAINT,
                on_click=lambda _: self._on_remove(position),
            ),
        ]

        return ft.Row(controls=controls, spacing=0, tight=True)

    def _build_settings(
        self, position: int, step_name: str, parameters: Mapping[str, str]
    ) -> ft.Control:
        """Build the fields shown when a step is opened.

        Args:
            position: the step's place in the pipeline.
            step_name: the step's catalogue name.
            parameters: the parameters currently set on it.

        Returns:
            The settings area, help button included.
        """
        definition = get_definition(step_name)
        editor = ParameterEditor(
            definition=definition,
            values=parameters,
            on_change=lambda values: self._on_parameters_changed(position, values),
        )

        # A step like `pen-sketch` has fifteen settings, which stacked up come to
        # well over a thousand pixels — taller than the window. Past a certain
        # number the fields get their own scrolling area with a stated height,
        # rather than relying on the panel around them to provide one: a stated
        # height is the only thing that makes a scrolling area work no matter
        # what encloses it.
        field_count = len(describe_parameters(definition.config_class))
        fields: ft.Control = editor.control

        if field_count > SETTINGS_SCROLL_THRESHOLD:
            fields = ft.Container(
                content=ft.Column(
                    controls=[editor.control],
                    scroll=ft.ScrollMode.AUTO,
                    spacing=0,
                ),
                height=SETTINGS_MAX_HEIGHT,
            )

        return ft.Container(
            content=ft.Column(
                controls=[
                    ft.Row(
                        controls=[
                            theme.micro("SETTINGS", theme.TEXT_MUTED),
                            ft.Container(expand=True),
                            ft.TextButton(
                                content=ft.Row(
                                    controls=[
                                        ft.Icon(
                                            ft.Icons.HELP_OUTLINE,
                                            size=13,
                                            color=theme.ACCENT,
                                        ),
                                        theme.micro("What does this do?", theme.ACCENT),
                                    ],
                                    spacing=4,
                                    tight=True,
                                ),
                                on_click=lambda _: self._on_help(step_name),
                            ),
                        ],
                        spacing=0,
                        tight=True,
                    ),
                    fields,
                ],
                spacing=theme.SPACING_TIGHT,
                tight=True,
            ),
            padding=ft.Padding.only(
                left=theme.SPACING_TIGHT,
                right=theme.SPACING_TIGHT,
                bottom=theme.SPACING_TIGHT,
            ),
        )

    def _toggle(self, position: int) -> None:
        """Open a step's settings, or close them if they were already open.

        Opening a step changes nothing about the picture, so the panel redraws
        itself from what it was last given rather than disturbing the session.

        Args:
            position: the step whose settings were asked for.
        """
        self._expanded = None if self._expanded == position else position
        self.show_steps(self._applied, self._pipeline_source)

    @staticmethod
    def _icon_button(
        icon: ft.IconData,
        tooltip: str,
        on_click: Callable[[ft.Event[ft.IconButton]], None],
        colour: str = theme.TEXT_MUTED,
        disabled: bool = False,
    ) -> ft.IconButton:
        """Build one of the small buttons along a step's header.

        Args:
            icon: the glyph to show.
            tooltip: what the button does.
            on_click: what to call when it is pressed.
            colour: the glyph's colour.
            disabled: whether the action is unavailable here.

        Returns:
            The button.
        """
        return ft.IconButton(
            icon=icon,
            icon_size=15,
            icon_color=colour,
            tooltip=tooltip,
            disabled=disabled,
            on_click=on_click,
            style=ft.ButtonStyle(padding=ft.Padding.all(2)),
        )
