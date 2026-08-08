"""Pushing a changed control back to the screen.

Single responsibility: redraw a control, but only when there is a screen to
redraw it on.

Flet refuses to update a control that has not been added to a page yet, and
raises rather than ignoring the request. That is reasonable on its own, but it
means every panel would otherwise have to know whether it happens to be mounted
before touching any of its own controls. Routing the updates through here keeps
that concern in one place, and lets the panels be built and driven in a test
without a window ever opening.
"""

from __future__ import annotations

import flet as ft


def is_mounted(control: ft.Control) -> bool:
    """Tell whether a control currently belongs to a page.

    Args:
        control: the control to check.

    Returns:
        True if the control is on screen, False if it has only been built.
    """
    # Flet offers no direct test for this: reaching `page` walks up the parents
    # and raises when it finds none. Catching that is the supported way to ask.
    try:
        _ = control.page
    except RuntimeError:
        return False
    return True


def refresh(control: ft.Control) -> None:
    """Redraw a control if it is on screen, and do nothing if it is not.

    Args:
        control: the control whose appearance has changed.
    """
    if is_mounted(control):
        control.update()
