"""The editor's panels, one module each.

Every panel builds its own controls and reports what the user did through
callbacks handed to it at construction. None of them knows about the others, nor
about the editing session: joining them up is `app`'s job alone.

    toolbar.py    open, save, undo, reset, and the busy indicator
    library.py    the catalogue of steps, draggable
    canvas.py     the picture, in the centre
    pipeline.py   the drop zone and the list of applied steps
"""

from pixel.ui.components.canvas import EditorCanvas
from pixel.ui.components.library import StepLibraryPanel
from pixel.ui.components.pipeline import PipelinePanel
from pixel.ui.components.toolbar import EditorToolbar

__all__ = ["EditorCanvas", "StepLibraryPanel", "PipelinePanel", "EditorToolbar"]
