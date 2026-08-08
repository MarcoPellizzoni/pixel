"""Tests for session files: a photo and its pipeline, saved under a name.

A session is a document the user names and keeps, so these check the two things
that matter about a document: that what comes back is what went in, and that a
file which is not one of ours is refused with something worth reading rather than
accepted as an empty session.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from pixel.ui import sessions
from pixel.ui.sessions import SavedSession, SessionFileError


class TestRoundTrip:
    """What is saved must come back exactly."""

    def test_a_session_survives_being_written_and_read(self, tmp_path: Path) -> None:
        path = tmp_path / "holiday.json"
        original = SavedSession(
            image_path=Path("/photos/holiday.jpg"),
            pipeline="resize:scale=0.5 | grayscale | pen-sketch:ink-threshold=0.7",
        )

        sessions.save(original, path)
        restored = sessions.load(path)

        assert restored.image_path == original.image_path
        assert restored.pipeline == original.pipeline

    def test_the_steps_keep_their_order(self, tmp_path: Path) -> None:
        # Order is half of what a pipeline means, so it is the half most worth
        # checking survives a trip through a file.
        path = tmp_path / "s.json"
        sessions.save(SavedSession(None, "invert | grayscale"), path)

        assert sessions.load(path).pipeline == "invert | grayscale"

    def test_the_parameters_come_back_too(self, tmp_path: Path) -> None:
        path = tmp_path / "s.json"
        sessions.save(SavedSession(None, "blur:radius=9 | posterize:levels=3"), path)

        assert sessions.load(path).pipeline == "blur:radius=9 | posterize:levels=3"

    def test_a_session_with_no_image_is_allowed(self, tmp_path: Path) -> None:
        path = tmp_path / "recipe.json"
        sessions.save(SavedSession(None, "grayscale"), path)

        assert sessions.load(path).image_path is None

    def test_missing_directories_are_created(self, tmp_path: Path) -> None:
        path = tmp_path / "one" / "two" / "s.json"

        sessions.save(SavedSession(None, "grayscale"), path)

        assert path.is_file()

    def test_the_time_of_saving_is_recorded(self, tmp_path: Path) -> None:
        path = tmp_path / "s.json"
        sessions.save(SavedSession(None, "grayscale"), path)

        assert sessions.load(path).saved_at


class TestWhatIsStored:
    """The file must stay small and readable by a person."""

    def test_it_holds_the_path_and_the_pipeline_and_no_pixels(
        self, tmp_path: Path
    ) -> None:
        path = tmp_path / "s.json"
        sessions.save(SavedSession(Path("/photos/a.jpg"), "blur"), path)

        stored = json.loads(path.read_text())

        assert stored["image"] == "/photos/a.jpg"
        assert stored["pipeline"] == "blur"
        assert path.stat().st_size < 1000

    def test_the_photo_is_stored_as_a_full_path(self, tmp_path: Path) -> None:
        # A session may be opened from anywhere, and a relative path would only
        # find the photo when the editor happened to start in the right place.
        path = tmp_path / "s.json"
        sessions.save(SavedSession(Path("photo.jpg"), "blur"), path)

        stored = json.loads(path.read_text())

        assert Path(stored["image"]).is_absolute()

    def test_it_says_what_kind_of_file_it_is(self, tmp_path: Path) -> None:
        # Without this, any JSON file at all would be accepted as a session.
        path = tmp_path / "s.json"
        sessions.save(SavedSession(None, "blur"), path)

        assert json.loads(path.read_text())["kind"] == sessions.SESSION_KIND


class TestRefusingBadFiles:
    """A file that cannot be used must say so, not come back empty."""

    def test_json_that_is_not_a_session_is_refused(self, tmp_path: Path) -> None:
        path = tmp_path / "shopping.json"
        path.write_text(json.dumps({"milk": True}))

        with pytest.raises(SessionFileError, match="not a pixel session"):
            sessions.load(path)

    def test_a_file_that_is_not_json_is_refused(self, tmp_path: Path) -> None:
        path = tmp_path / "photo.json"
        path.write_text("this is not json at all")

        with pytest.raises(SessionFileError, match="readable JSON"):
            sessions.load(path)

    def test_a_session_from_another_version_is_refused(self, tmp_path: Path) -> None:
        path = tmp_path / "s.json"
        path.write_text(
            json.dumps({"kind": sessions.SESSION_KIND, "version": 99, "pipeline": "x"})
        )

        with pytest.raises(SessionFileError, match="different version"):
            sessions.load(path)

    def test_a_missing_file_raises_rather_than_returning_nothing(
        self, tmp_path: Path
    ) -> None:
        # Unlike the quiet remembering in `workspace`, someone asked for this
        # file by name and is entitled to be told it is not there.
        with pytest.raises(OSError):
            sessions.load(tmp_path / "absent.json")

    def test_saving_somewhere_unwritable_raises(self, tmp_path: Path) -> None:
        blocked = tmp_path / "file"
        blocked.write_text("in the way")

        with pytest.raises(OSError):
            sessions.save(SavedSession(None, "blur"), blocked / "s.json")


class TestSuggestedName:
    """The name offered when saving should keep a folder of sessions navigable."""

    def test_it_follows_the_photo(self) -> None:
        assert sessions.suggested_name(Path("/photos/holiday.jpg")) == "holiday.json"

    def test_it_falls_back_when_there_is_no_photo(self) -> None:
        assert sessions.suggested_name(None) == sessions.DEFAULT_SESSION_NAME

    def test_it_always_ends_in_the_session_extension(self) -> None:
        for photo in (Path("/a/b.png"), Path("c.tiff"), None):
            assert sessions.suggested_name(photo).endswith(
                f".{sessions.SESSION_EXTENSION}"
            )


class TestSessionsAreCommandLinePipelines:
    """A saved session must hold something the terminal would also accept."""

    def test_the_stored_pipeline_can_be_run_from_the_command_line(
        self, tmp_path: Path
    ) -> None:
        from pixel import build_pipeline

        path = tmp_path / "s.json"
        sessions.save(SavedSession(None, "grayscale | posterize:levels=6"), path)

        rebuilt = build_pipeline(sessions.load(path).pipeline)

        assert list(rebuilt.iter_step_names()) == ["grayscale", "posterize"]
