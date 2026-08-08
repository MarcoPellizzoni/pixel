"""The help: what each step does, and what its settings mean.

Single responsibility: present the catalogue as something to read rather than
something to use — one step in full, or every step at a glance.

Nothing here is written by hand per step. Both views are built from the same
catalogue that `pixel steps` and `pixel describe` print, so the help cannot fall
out of date with the steps themselves, and the two interfaces cannot disagree
about what a parameter means.
"""

from __future__ import annotations

import flet as ft

from pixel.params import describe_parameters
from pixel.registry import StepDefinition, list_definitions
from pixel.ui import theme

# Width of the help window. Wide enough for an explanation to sit on two or three
# lines, narrow enough that the eye does not lose the start of the next one.
DIALOG_WIDTH: int = 620

# Height of the browsable overview. Fixed, so the window does not resize as
# different steps are opened.
OVERVIEW_HEIGHT: int = 560


def build_step_help(definition: StepDefinition) -> ft.AlertDialog:
    """Build the help for a single step: what it does, and every setting.

    Args:
        definition: the step's catalogue entry.

    Returns:
        A dialog ready to be shown.
    """
    return ft.AlertDialog(
        title=ft.Row(
            controls=[
                ft.Icon(ft.Icons.HELP_OUTLINE, size=18, color=theme.ACCENT),
                ft.Text(value=definition.name, size=16, weight=ft.FontWeight.BOLD),
            ],
            spacing=theme.SPACING_TIGHT,
            tight=True,
        ),
        content=ft.Container(
            content=ft.Column(
                controls=_step_body(definition),
                spacing=theme.SPACING,
                scroll=ft.ScrollMode.AUTO,
                tight=True,
            ),
            width=DIALOG_WIDTH,
        ),
        bgcolor=theme.PANEL_BACKGROUND,
    )


def build_overview() -> ft.AlertDialog:
    """Build the help covering every step, grouped by family.

    Returns:
        A dialog ready to be shown.
    """
    sections: list[ft.Control] = [
        ft.Text(
            value=(
                "Steps are applied in order, top to bottom. Drag one onto the "
                "pipeline or press its + button, then open it in the pipeline to "
                "change its settings."
            ),
            size=theme.CAPTION_SIZE,
            color=theme.TEXT_MUTED,
        ),
    ]

    current_family: str | None = None
    for definition in list_definitions():
        if definition.category.value != current_family:
            current_family = definition.category.value
            sections.append(
                ft.Container(
                    content=ft.Text(
                        value=current_family.upper(),
                        size=theme.MICRO_SIZE,
                        weight=ft.FontWeight.BOLD,
                        color=theme.ACCENT,
                    ),
                    padding=ft.Padding.only(top=theme.SPACING),
                )
            )

        sections.append(_overview_row(definition))

    return ft.AlertDialog(
        title=ft.Text(value="Steps", size=16, weight=ft.FontWeight.BOLD),
        content=ft.Container(
            content=ft.Column(
                controls=sections,
                spacing=theme.SPACING_TIGHT,
                scroll=ft.ScrollMode.AUTO,
                tight=True,
            ),
            width=DIALOG_WIDTH,
            height=OVERVIEW_HEIGHT,
        ),
        bgcolor=theme.PANEL_BACKGROUND,
    )


# ----------------------------------------------------------------------
# The pieces
# ----------------------------------------------------------------------


def _step_body(definition: StepDefinition) -> list[ft.Control]:
    """Build the contents of a single step's help.

    Args:
        definition: the step's catalogue entry.

    Returns:
        The controls making up the dialog body.
    """
    body: list[ft.Control] = [
        ft.Text(
            value=definition.summary,
            size=theme.BODY_SIZE,
            color=theme.TEXT,
        ),
        theme.micro(f"Family: {definition.category.value}"),
    ]

    parameters = describe_parameters(definition.config_class)

    if not parameters:
        body.append(
            ft.Container(
                content=theme.caption("This step has no settings."),
                padding=ft.Padding.only(top=theme.SPACING),
            )
        )
        return body

    body.append(
        ft.Container(
            content=ft.Text(
                value="SETTINGS",
                size=theme.MICRO_SIZE,
                weight=ft.FontWeight.BOLD,
                color=theme.TEXT_MUTED,
            ),
            padding=ft.Padding.only(top=theme.SPACING),
        )
    )

    for parameter in parameters:
        body.append(
            ft.Container(
                content=ft.Column(
                    controls=[
                        ft.Row(
                            controls=[
                                ft.Text(
                                    value=parameter.name,
                                    size=theme.CAPTION_SIZE,
                                    weight=ft.FontWeight.W_500,
                                    color=theme.TEXT,
                                ),
                                theme.micro(parameter.type_label),
                                ft.Container(expand=True),
                                theme.micro(f"default: {parameter.default}"),
                            ],
                            spacing=theme.SPACING_TIGHT,
                        ),
                        ft.Text(
                            value=parameter.description,
                            size=theme.MICRO_SIZE,
                            color=theme.TEXT_MUTED,
                        ),
                    ],
                    spacing=4,
                    tight=True,
                ),
                bgcolor=theme.CARD_BACKGROUND,
                border_radius=theme.RADIUS_SMALL,
                padding=ft.Padding.all(theme.SPACING_TIGHT + 2),
            )
        )

    # The equivalent terminal command, so the help doubles as a way of learning
    # the syntax the pipeline panel shows.
    first = parameters[0]
    body.append(
        ft.Container(
            content=ft.Text(
                value=(
                    f'pixel run photo.jpg "{definition.name}:'
                    f'{first.name}={first.default}"'
                ),
                size=theme.MICRO_SIZE,
                color=theme.TEXT_FAINT,
                selectable=True,
                font_family="monospace",
            ),
            padding=ft.Padding.only(top=theme.SPACING_TIGHT),
        )
    )

    return body


def _overview_row(definition: StepDefinition) -> ft.Control:
    """Build one step's line in the overview.

    Args:
        definition: the step's catalogue entry.

    Returns:
        The row control.
    """
    parameters = describe_parameters(definition.config_class)
    settings = (
        ", ".join(parameter.name for parameter in parameters)
        if parameters
        else "no settings"
    )

    return ft.Container(
        content=ft.Column(
            controls=[
                ft.Text(
                    value=definition.name,
                    size=theme.CAPTION_SIZE,
                    weight=ft.FontWeight.W_500,
                    color=theme.TEXT,
                ),
                theme.caption(definition.summary),
                theme.micro(settings),
            ],
            spacing=2,
            tight=True,
        ),
        bgcolor=theme.CARD_BACKGROUND,
        border_radius=theme.RADIUS_SMALL,
        padding=ft.Padding.all(theme.SPACING_TIGHT + 2),
    )
