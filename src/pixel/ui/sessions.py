"""Session files: a photo and the steps applied to it, saved under a name.

Single responsibility: write a named session to a file the user chose, and read
one back.

This is a different thing from `workspace`, which quietly remembers where the
editor was left so it can open there next time. A session is a document: the user
names it, keeps it, moves it, opens it again next month. The difference shows in
how failure is handled — `workspace` swallows it, because nobody asked for it to
be written, whereas here an error is raised and reported, because someone pressed
Save and is entitled to know whether it worked.

What is stored is the photo's path and the pipeline written in the command line's
own syntax. No image data: opening a session means loading that photo again and
replaying the steps over it, which produces exactly what was on screen and leaves
a file small enough to read, edit or keep in version control.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from pixel.ui.layout import cast_mapping

# Bumped when the stored shape changes in a way older readers cannot make sense
# of. Refusing to read an unknown version beats guessing at its contents.
SESSION_VERSION: int = 1

# Marks the file as one of ours, so opening a JSON file that happens to be
# something else fails with a clear message rather than an empty session.
SESSION_KIND: str = "pixel-session"

# Extension offered in the save and open dialogs.
SESSION_EXTENSION: str = "json"

# Name suggested when saving a session that has no better name to go on.
DEFAULT_SESSION_NAME: str = "session.json"


class SessionFileError(Exception):
    """A session file could not be read, or is not one of ours.

    Given a type of its own so the editor can tell "this file is no good" apart
    from "the disk is full", and say something useful about each.
    """


@dataclass(frozen=True)
class SavedSession:
    """A photo and the pipeline applied to it, as stored in a file.

    Attributes:
        image_path: the photo the steps were applied to.
        pipeline: those steps, in the command line's syntax, which is what
            carries their order and their parameters.
        saved_at: when it was written, for the user's benefit only.
    """

    image_path: Path | None
    pipeline: str
    saved_at: str = field(default="")

    def to_dict(self) -> dict[str, Any]:
        """Describe the session in a form that can be written as JSON.

        The photo's path is written out in full. A session is a file the user
        keeps and may open from anywhere, and a relative path would only find the
        photo when the editor happened to be started from the same directory.
        """
        return {
            "kind": SESSION_KIND,
            "version": SESSION_VERSION,
            "image": str(self.image_path.resolve()) if self.image_path else None,
            "pipeline": self.pipeline,
            "saved_at": self.saved_at or _now(),
        }

    @classmethod
    def from_dict(cls, stored: object) -> SavedSession:
        """Rebuild a session from what a file contained.

        Args:
            stored: whatever was read back.

        Returns:
            The session.

        Raises:
            SessionFileError: if the file is not a pixel session, or is from a
                version this one cannot read.
        """
        values = cast_mapping(stored)

        if values.get("kind") != SESSION_KIND:
            raise SessionFileError("That file is not a pixel session.")

        if values.get("version") != SESSION_VERSION:
            raise SessionFileError(
                "That session was written by a different version of pixel."
            )

        image = values.get("image")
        pipeline = values.get("pipeline")
        saved_at = values.get("saved_at")

        return cls(
            image_path=Path(image) if isinstance(image, str) and image else None,
            pipeline=pipeline if isinstance(pipeline, str) else "",
            saved_at=saved_at if isinstance(saved_at, str) else "",
        )


def save(session: SavedSession, path: Path) -> None:
    """Write a session to a file.

    Args:
        session: what to store.
        path: where to write it.

    Raises:
        OSError: if the file cannot be written. Unlike the quiet remembering in
            `workspace`, this is raised: the user asked for the save and needs to
            know whether it happened.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(session.to_dict(), indent=2))


def load(path: Path) -> SavedSession:
    """Read a session back from a file.

    Args:
        path: the file to read.

    Returns:
        The session it holds.

    Raises:
        OSError: if the file cannot be read.
        SessionFileError: if its contents are not a readable pixel session.
    """
    try:
        stored = json.loads(path.read_text())
    except ValueError as error:
        raise SessionFileError("That file is not readable JSON.") from error

    return SavedSession.from_dict(stored)


def suggested_name(image_path: Path | None) -> str:
    """Work out a file name to offer when saving a session.

    Args:
        image_path: the photo being edited, if any.

    Returns:
        A name based on the photo, so a folder of sessions stays navigable.
    """
    if image_path is None:
        return DEFAULT_SESSION_NAME

    return f"{image_path.stem}.{SESSION_EXTENSION}"


def _now() -> str:
    """Return the current moment, written in a form that sorts and reads well."""
    return datetime.now(UTC).isoformat(timespec="seconds")
