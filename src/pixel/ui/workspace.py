"""Remembering the work between one run and the next.

Single responsibility: write down what the editor was doing when it was last
closed — which photo, which pipeline, how the window was arranged — and read it
back at start-up.

What gets stored is deliberately small: a path, a pipeline written in the command
line's own syntax, and a handful of numbers. No images. Reopening therefore means
loading the original file and running the pipeline again, which produces exactly
what was on screen before, and leaves a file small enough to read and edit by
hand if anyone wants to.

Nothing here may ever stop the editor from starting. A file that is missing,
truncated, from a future version or written by someone's text editor costs the
user their restored session, never their program.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from pixel.ui.layout import PanelLayout, cast_mapping

# Bumped when the stored shape changes in a way older readers cannot make sense
# of. A file from a version we do not know is ignored rather than guessed at.
STATE_VERSION: int = 1

# Where the file lives, following the usual convention for a user's own settings
# on Linux. `XDG_CONFIG_HOME` is honoured when it is set.
APPLICATION_DIRECTORY: str = "pixel"
STATE_FILENAME: str = "workspace.json"


@dataclass
class Workspace:
    """What the editor was doing, in a form that can be written to a file.

    Attributes:
        image_path: the photo that was open, or None if there was none.
        pipeline: the steps applied to it, in command line syntax.
        layout: how the window was arranged.
    """

    image_path: Path | None = None
    pipeline: str = ""
    layout: PanelLayout = field(default_factory=PanelLayout)

    def to_dict(self) -> dict[str, Any]:
        """Describe the workspace in a form that can be written as JSON."""
        return {
            "version": STATE_VERSION,
            "image": str(self.image_path) if self.image_path else None,
            "pipeline": self.pipeline,
            "layout": self.layout.to_dict(),
        }

    @classmethod
    def from_dict(cls, stored: object) -> Workspace:
        """Rebuild a workspace from what was read back.

        Args:
            stored: whatever the file contained.

        Returns:
            The workspace, or an empty one if the contents cannot be used.
        """
        values = cast_mapping(stored)

        # A file from a version we do not understand is ignored outright: reading
        # half of it would be worse than starting fresh.
        if values.get("version") != STATE_VERSION:
            return cls()

        image = values.get("image")
        pipeline = values.get("pipeline")

        return cls(
            image_path=Path(image) if isinstance(image, str) and image else None,
            pipeline=pipeline if isinstance(pipeline, str) else "",
            layout=PanelLayout.from_dict(values.get("layout")),
        )


def state_path() -> Path:
    """Work out where the workspace file belongs.

    Returns:
        The full path, whether or not anything is there yet.
    """
    configured = os.environ.get("XDG_CONFIG_HOME")
    base = Path(configured) if configured else Path.home() / ".config"

    return base / APPLICATION_DIRECTORY / STATE_FILENAME


def save(workspace: Workspace, path: Path | None = None) -> bool:
    """Write the workspace down.

    Args:
        workspace: what to remember.
        path: where to write it; the usual location when not given.

    Returns:
        True if it was written, False if it could not be. A failure is reported
        rather than raised: not being able to remember the session is a small
        loss, and never a reason to interrupt someone's editing.
    """
    destination = path or state_path()

    try:
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(json.dumps(workspace.to_dict(), indent=2))
    except OSError:
        return False

    return True


def load(path: Path | None = None) -> Workspace:
    """Read back the workspace from the last run.

    Args:
        path: where to read from; the usual location when not given.

    Returns:
        What was stored, or an empty workspace if there is nothing usable there.
    """
    source = path or state_path()

    try:
        stored = json.loads(source.read_text())
    except (OSError, ValueError):
        # Missing, unreadable, or not valid JSON. Any of those simply means there
        # is no session to restore.
        return Workspace()

    return Workspace.from_dict(stored)
