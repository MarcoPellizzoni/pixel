"""Tests for the window arrangement and for remembering the work.

Both are plain values with no Flet in them, so what they promise — that a panel
cannot be dragged to a useless width, and that a session survives being closed
and reopened — can be checked without a window.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from pixel.ui import theme, workspace
from pixel.ui.layout import PanelLayout
from pixel.ui.workspace import Workspace


class TestResizing:
    """A panel must follow the drag, but never past a usable width."""

    def test_dragging_right_widens_the_library(self) -> None:
        layout = PanelLayout()

        layout.resize_library(40)

        assert layout.library_width == theme.LIBRARY_WIDTH + 40

    def test_dragging_left_narrows_the_library(self) -> None:
        layout = PanelLayout()

        layout.resize_library(-40)

        assert layout.library_width == theme.LIBRARY_WIDTH - 40

    def test_the_library_stops_at_its_widest(self) -> None:
        layout = PanelLayout()

        layout.resize_library(5000)

        assert layout.library_width == theme.LIBRARY_MAX_WIDTH

    def test_the_library_stops_at_its_narrowest(self) -> None:
        layout = PanelLayout()

        layout.resize_library(-5000)

        assert layout.library_width == theme.LIBRARY_MIN_WIDTH

    def test_the_pipeline_divider_works_the_other_way_round(self) -> None:
        # Its divider is on the panel's left edge, so dragging left widens it.
        layout = PanelLayout()

        layout.resize_pipeline(-40)

        assert layout.pipeline_width == theme.PIPELINE_WIDTH + 40

    def test_the_pipeline_stops_at_its_limits(self) -> None:
        layout = PanelLayout()

        layout.resize_pipeline(-5000)
        assert layout.pipeline_width == theme.PIPELINE_MAX_WIDTH

        layout.resize_pipeline(5000)
        assert layout.pipeline_width == theme.PIPELINE_MIN_WIDTH

    def test_the_two_panels_at_their_narrowest_still_fit_the_window(self) -> None:
        # Otherwise the smallest allowed window could not show all three columns.
        assert (
            theme.LIBRARY_MIN_WIDTH + theme.PIPELINE_MIN_WIDTH < theme.WINDOW_MIN_WIDTH
        )


class TestOpeningAndClosing:
    """Each panel must be able to be put away and brought back."""

    def test_the_library_can_be_hidden_and_shown(self) -> None:
        layout = PanelLayout()

        layout.toggle_library()
        assert not layout.library_visible

        layout.toggle_library()
        assert layout.library_visible

    def test_the_pipeline_can_be_hidden_and_shown(self) -> None:
        layout = PanelLayout()

        layout.toggle_pipeline()
        assert not layout.pipeline_visible

        layout.toggle_pipeline()
        assert layout.pipeline_visible

    def test_hiding_one_leaves_the_other_alone(self) -> None:
        layout = PanelLayout()

        layout.toggle_library()

        assert layout.pipeline_visible

    def test_a_hidden_panel_keeps_its_width(self) -> None:
        # So that bringing it back does not also reset how wide it was.
        layout = PanelLayout()
        layout.resize_library(60)

        layout.toggle_library()
        layout.toggle_library()

        assert layout.library_width == theme.LIBRARY_WIDTH + 60


class TestLayoutStorage:
    """A layout must survive being written down and read back."""

    def test_it_round_trips(self) -> None:
        layout = PanelLayout(library_width=300, pipeline_visible=False)

        assert PanelLayout.from_dict(layout.to_dict()) == layout

    def test_a_stored_width_out_of_range_is_pulled_back_in(self) -> None:
        restored = PanelLayout.from_dict({"library_width": 9999})

        assert restored.library_width == theme.LIBRARY_MAX_WIDTH

    def test_nonsense_falls_back_to_the_defaults(self) -> None:
        assert PanelLayout.from_dict("not a layout at all") == PanelLayout()

    def test_a_missing_value_falls_back_to_its_default(self) -> None:
        restored = PanelLayout.from_dict({"library_width": 300})

        assert restored.library_width == 300
        assert restored.pipeline_width == theme.PIPELINE_WIDTH


class TestWorkspaceStorage:
    """The work must survive the editor being closed."""

    def test_it_round_trips(self, tmp_path: Path) -> None:
        path = tmp_path / "workspace.json"
        original = Workspace(
            image_path=Path("/photos/holiday.jpg"),
            pipeline="grayscale | invert",
            layout=PanelLayout(library_width=300),
        )

        assert workspace.save(original, path)

        assert workspace.load(path) == original

    def test_nothing_stored_gives_an_empty_workspace(self, tmp_path: Path) -> None:
        assert workspace.load(tmp_path / "absent.json") == Workspace()

    def test_a_damaged_file_gives_an_empty_workspace(self, tmp_path: Path) -> None:
        # Losing the remembered session is a small matter; failing to start is not.
        path = tmp_path / "workspace.json"
        path.write_text("{ this is not json")

        assert workspace.load(path) == Workspace()

    def test_a_file_from_another_version_is_ignored(self, tmp_path: Path) -> None:
        path = tmp_path / "workspace.json"
        path.write_text(json.dumps({"version": 999, "image": "/somewhere.jpg"}))

        assert workspace.load(path) == Workspace()

    def test_no_image_is_stored_as_no_image(self, tmp_path: Path) -> None:
        path = tmp_path / "workspace.json"
        workspace.save(Workspace(), path)

        assert workspace.load(path).image_path is None

    def test_saving_somewhere_unwritable_is_reported_not_raised(
        self, tmp_path: Path
    ) -> None:
        # A read-only home directory must not take the editor down with it.
        blocked = tmp_path / "file"
        blocked.write_text("in the way")

        assert not workspace.save(Workspace(), blocked / "workspace.json")

    def test_what_is_stored_is_small_and_readable(self, tmp_path: Path) -> None:
        # No images: reopening replays the pipeline over the original file, which
        # is what keeps the file something a person could read or edit.
        path = tmp_path / "workspace.json"
        workspace.save(
            Workspace(image_path=Path("/photos/a.jpg"), pipeline="blur"), path
        )

        stored = json.loads(path.read_text())

        assert stored["image"] == "/photos/a.jpg"
        assert stored["pipeline"] == "blur"
        assert path.stat().st_size < 1000


class TestStatePath:
    """The file must land where a user's own settings belong."""

    def test_it_honours_the_configured_location(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path))

        assert workspace.state_path().parent.parent == tmp_path

    def test_it_falls_back_to_the_usual_place(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.delenv("XDG_CONFIG_HOME", raising=False)

        assert workspace.state_path().parent.parent.name == ".config"
