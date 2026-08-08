"""The editor's visual language.

Single responsibility: give a name to every colour, size and spacing the
interface uses, so the look can be adjusted in one place instead of hunting for
numbers scattered across the panels.

The palette is a dark, low-saturation one, as in most image editors: a neutral
surround stops the interface from influencing how the colours in the photo are
perceived. Surfaces are layered rather than outlined — the canvas sits deepest,
panels one step up, cards one step above those — which is what modern web
interfaces use in place of borders to show what contains what.

Sizes follow a 4-pixel rhythm, so that everything lines up without anything
having to be measured by eye.
"""

from __future__ import annotations

import flet as ft

# ----------------------------------------------------------------------
# Surfaces, from deepest to nearest
# ----------------------------------------------------------------------

# The area the photo sits on: the darkest surface, so the picture reads as lit.
CANVAS_BACKGROUND: str = "#141417"

# Side panels and toolbar.
PANEL_BACKGROUND: str = "#1B1B20"

# Cards inside a panel: a step in the library, a step in the pipeline.
CARD_BACKGROUND: str = "#232329"

# The same card under the pointer, and the surface of a text field.
CARD_HOVER_BACKGROUND: str = "#2B2B33"
INPUT_BACKGROUND: str = "#141417"

# Hairlines. Used sparingly: layering usually says it better.
BORDER: str = "#2E2E36"

# ----------------------------------------------------------------------
# Accent and meaning
# ----------------------------------------------------------------------

# Highlighted controls, the drop zone under a dragged step, focus rings.
ACCENT: str = "#5B8DEF"

# A tint of the accent, for the background of a selected row.
ACCENT_SOFT: str = "#1E2A44"

# Destructive actions: discarding every edit, removing a step.
DANGER: str = "#EF6B62"

# ----------------------------------------------------------------------
# Text, from most to least prominent
# ----------------------------------------------------------------------

TEXT: str = "#EDEDF2"
TEXT_MUTED: str = "#9494A4"
TEXT_FAINT: str = "#6E6E7E"

# ----------------------------------------------------------------------
# Sizes
# ----------------------------------------------------------------------

# Width of the step library, on the left, and how far it may be dragged. The
# lower bound is where the longest step name stops fitting; the upper one is
# where the panel starts crowding the photo out.
LIBRARY_WIDTH: int = 250
LIBRARY_MIN_WIDTH: int = 190
LIBRARY_MAX_WIDTH: int = 420

# Width of the pipeline panel, on the right, and its limits. Wider than the
# library because it has to hold parameter fields as well as step names, and its
# lower bound is where those fields stop being usable.
PIPELINE_WIDTH: int = 340
PIPELINE_MIN_WIDTH: int = 280
PIPELINE_MAX_WIDTH: int = 560

# Width of the draggable divider between a panel and the canvas. Narrow, but with
# a wider invisible grab area around it so it does not have to be aimed at.
SPLITTER_WIDTH: int = 1
SPLITTER_GRAB_WIDTH: int = 9

# Height of the top toolbar.
TOOLBAR_HEIGHT: int = 56

# The spacing rhythm. Everything is one of these.
SPACING_TIGHT: int = 6
SPACING: int = 12
SPACING_WIDE: int = 20

# Corner radii: the larger one for panels and cards, the smaller for controls
# sitting inside them.
RADIUS: int = 10
RADIUS_SMALL: int = 6

# Initial window size. Wide enough for the three columns to breathe.
WINDOW_WIDTH: int = 1500
WINDOW_HEIGHT: int = 940

# Smallest the window may be dragged to. Below this the two side panels would
# leave the photo no room at all, so the limit is their narrowest combined width
# plus enough canvas to still show something.
WINDOW_MIN_WIDTH: int = LIBRARY_MIN_WIDTH + PIPELINE_MIN_WIDTH + 320
WINDOW_MIN_HEIGHT: int = 560

# ----------------------------------------------------------------------
# Type scale
# ----------------------------------------------------------------------

# Panel headings.
TITLE_SIZE: int = 12

# Ordinary text.
BODY_SIZE: int = 13

# Supporting detail: step descriptions, dimensions, hints.
CAPTION_SIZE: int = 12

# The smallest text: parameter types and explanations.
MICRO_SIZE: int = 11


def panel_title(text: str) -> ft.Text:
    """Build a panel heading, styled the same way everywhere.

    Args:
        text: the heading to show.

    Returns:
        The ready-made text control.
    """
    return ft.Text(
        value=text.upper(),
        size=TITLE_SIZE,
        weight=ft.FontWeight.BOLD,
        color=TEXT_MUTED,
        # Letter-spaced capitals are the usual way of marking a small heading as
        # a heading without making it loud.
        spans=None,
    )


def caption(text: str, color: str = TEXT_MUTED) -> ft.Text:
    """Build a line of supporting detail.

    Args:
        text: the text to show.
        color: its colour, muted by default.

    Returns:
        The ready-made text control.
    """
    return ft.Text(value=text, size=CAPTION_SIZE, color=color)


def micro(text: str, color: str = TEXT_FAINT) -> ft.Text:
    """Build the smallest supporting text, for types and explanations.

    Args:
        text: the text to show.
        color: its colour, faint by default.

    Returns:
        The ready-made text control.
    """
    return ft.Text(value=text, size=MICRO_SIZE, color=color)
