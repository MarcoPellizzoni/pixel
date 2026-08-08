"""The graphical image editor.

Single responsibility of this package: give the `pixel` library a window. It adds
no image processing of its own — every effect comes from `pixel.registry`, the
same catalogue the command line uses.

    session.py     what has been done to the image, and undo. No Flet at all.
    preview.py     turning an image into the PNG bytes the screen can show.
    theme.py       colours, sizes and spacing.
    updates.py     redrawing a control only when it is on screen.
    components/    the panels, each unaware of the others.
    app.py         wires the panels to the session.
    launcher.py    reads the command line and starts Flet.

The dependencies run one way only: `app` -> `components` -> `theme`, and
`app` -> `session` -> the processing library. Nothing points back.
"""

from pixel.ui.app import ImageEditorApp, create_main
from pixel.ui.launcher import run
from pixel.ui.session import AppliedStep, EditingSession

__all__ = ["ImageEditorApp", "create_main", "run", "EditingSession", "AppliedStep"]
