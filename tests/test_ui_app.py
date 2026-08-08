"""Tests for the application that wires the panels to the session.

These drive the editor through a stand-in for a Flet page: the whole control tree
is built, and every user action is carried out for real, but nothing is drawn.

What this covers is the wiring, which is where the mistakes in an application
class actually live — a button that changes the session but forgets to redraw, a
reset that clears the history but leaves the canvas showing the old picture. What
it cannot cover is whether the result looks right on screen.
"""

from __future__ import annotations

import asyncio
from collections.abc import Callable, Coroutine
from pathlib import Path
from types import SimpleNamespace
from typing import Any, cast

import flet as ft
import numpy as np
import pytest
from conftest import solid_image
from test_ui_components import find_controls, fire

from pixel import paths
from pixel.domain import RGBAImage
from pixel.dsl import parse_pipeline
from pixel.image_io import save_image
from pixel.ui import sessions, theme, workspace
from pixel.ui.app import ImageEditorApp


class FakeWindow:
    """A stand-in for the real window, which records its settings.

    The editor states the window's size and limits at start-up, so a page that
    could not be asked about its window would fail before drawing anything.
    """

    def __init__(self) -> None:
        """Start a window with nothing set on it, as Flet does."""
        self.width: int | None = None
        self.height: int | None = None
        self.min_width: int | None = None
        self.min_height: int | None = None
        self.resizable: bool | None = None
        self.maximizable: bool | None = None


class FakePage:
    """The smallest stand-in for `ft.Page` the editor needs to run.

    It records what the application asks of it instead of drawing anything, and
    runs scheduled coroutines straight away so a test does not have to wait.
    """

    def __init__(self) -> None:
        """Start an empty page."""
        self.window = FakeWindow()
        self.controls: list[ft.Control] = []
        self.services: list[ft.Service] = []
        self.dialogs: list[ft.Control] = []
        self.title: str = ""
        self.theme_mode: ft.ThemeMode | None = None
        self.bgcolor: str | None = None
        self.padding: int | None = None
        self.spacing: int | None = None

    def add(self, *controls: ft.Control) -> None:
        """Take delivery of the top-level controls."""
        self.controls.extend(controls)

    def run_task(
        self,
        handler: Callable[..., Coroutine[Any, Any, Any]],
        *args: object,
    ) -> None:
        """Run a scheduled coroutine to completion immediately.

        The real page hands it to the event loop. Running it here and now keeps
        the tests free of waiting and of arbitrary timeouts.
        """
        asyncio.run(handler(*args))

    def show_dialog(self, dialog: ft.Control) -> None:
        """Record a message instead of showing it."""
        self.dialogs.append(dialog)

    def update(self) -> None:
        """Accept a redraw request and do nothing with it."""


@pytest.fixture(autouse=True)
def isolated_workspace(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Give every test its own workspace file, well away from the real one.

    The editor writes down what it is doing after each change. Without this the
    tests would trample the user's own remembered session, and would see each
    other's leftovers.
    """
    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path / "config"))


def build_app(
    initial: Path | None = None, restore: bool = False
) -> tuple[ImageEditorApp, FakePage]:
    """Build an editor on a fake page.

    Args:
        initial: an image to open at start-up, if any.
        restore: whether to reopen the work from last time. Off by default, so a
            test says explicitly when that is what it is checking.

    Returns:
        The application and the page it was built on.
    """
    page = FakePage()
    app = ImageEditorApp(cast(ft.Page, page), initial, restore=restore)
    return app, page


def drop_step(page: FakePage, step_name: str) -> None:
    """Drop a step onto the pipeline panel, as a user would.

    Args:
        page: the page the editor was built on.
        step_name: the step being dragged.
    """
    target = find_controls(page.controls[0], ft.DragTarget)[0]
    fire(target.on_accept, SimpleNamespace(src=SimpleNamespace(data=step_name)))


def click_step(page: FakePage, step_name: str) -> None:
    """Click a step's "+" button in the library, as a user would.

    Args:
        page: the page the editor was built on.
        step_name: the step to apply.
    """
    for button in find_controls(page.controls[0], ft.IconButton):
        if button.tooltip == f"Apply {step_name} now":
            fire(button.on_click, SimpleNamespace())
            return
    raise AssertionError(f"no button found for step {step_name!r}")


@pytest.fixture
def photo_on_disk(tmp_path: Path) -> Path:
    """Write a small test photo and return its path."""
    path = tmp_path / "photo.png"
    save_image(solid_image((200, 100, 50), width=40, height=30), path)
    return path


class TestStartup:
    """Building the editor must produce a complete, usable window."""

    def test_it_builds_without_an_image(self) -> None:
        _, page = build_app()

        assert page.controls, "the editor added nothing to the page"

    def test_it_registers_the_file_picker(self) -> None:
        # The picker is a service, not a control: forgetting to register it
        # would make the open and save dialogs silently never appear.
        _, page = build_app()

        assert any(isinstance(s, ft.FilePicker) for s in page.services)

    def test_the_window_carries_all_four_areas(self) -> None:
        _, page = build_app()

        root = page.controls[0]
        # Toolbar buttons, the step library, and the pipeline's drop target.
        assert find_controls(root, ft.Button)
        assert find_controls(root, ft.Draggable)
        assert find_controls(root, ft.DragTarget)

    def test_the_window_can_be_resized(self) -> None:
        # A window left with no size at all is handed to the window manager with
        # nothing to go on, and on Linux that can produce one whose edges cannot
        # be dragged.
        _, page = build_app()

        assert page.window.resizable is True
        assert page.window.width == theme.WINDOW_WIDTH
        assert page.window.height == theme.WINDOW_HEIGHT

    def test_the_window_cannot_be_shrunk_past_its_panels(self) -> None:
        _, page = build_app()

        assert page.window.min_width is not None
        assert page.window.min_width >= theme.LIBRARY_WIDTH + theme.PIPELINE_WIDTH
        assert page.window.min_height is not None

    def test_it_opens_the_image_it_is_given(self, photo_on_disk: Path) -> None:
        app, _ = build_app(photo_on_disk)

        assert app.session is not None
        assert (app.session.current.width, app.session.current.height) == (40, 30)

    def test_a_missing_start_image_is_reported_not_fatal(self, tmp_path: Path) -> None:
        app, page = build_app(tmp_path / "absent.png")

        assert app.session is None
        assert page.dialogs, "the failure was not reported to the user"


class TestEditing:
    """Applying, undoing and resetting must keep session and screen in step."""

    def test_applying_a_step_changes_the_image(self, photo_on_disk: Path) -> None:
        app, _ = build_app(photo_on_disk)

        app.apply_step("grayscale")

        assert app.session is not None
        rgb = app.session.current.rgb
        assert np.array_equal(rgb[:, :, 0], rgb[:, :, 2])

    def test_applying_a_step_shows_it_in_the_pipeline(
        self, photo_on_disk: Path
    ) -> None:
        app, _ = build_app(photo_on_disk)

        app.apply_step("grayscale")
        app.apply_step("invert")

        assert app.session is not None
        assert app.session.pipeline_text == "grayscale | invert"

    def test_undo_takes_back_one_step(self, photo_on_disk: Path) -> None:
        app, _ = build_app(photo_on_disk)
        app.apply_step("grayscale")
        app.apply_step("invert")

        app.undo()

        assert app.session is not None
        assert app.session.pipeline_text == "grayscale"

    def test_reset_goes_back_to_the_opened_image(self, photo_on_disk: Path) -> None:
        app, _ = build_app(photo_on_disk)
        app.apply_step("grayscale")

        app.reset()

        assert app.session is not None
        assert not app.session.is_modified
        assert np.array_equal(
            app.session.current.rgb, app.session.source.rgb
        )

    def test_the_canvas_follows_every_edit(self, photo_on_disk: Path) -> None:
        # The picture on screen must change with the session, not lag behind it.
        app, page = build_app(photo_on_disk)

        def displayed() -> object:
            """Read the picture currently on the canvas, through the window."""
            return find_controls(page.controls[0], ft.Image)[0].src

        opened = displayed()
        app.apply_step("invert")
        after_step = displayed()
        app.undo()
        after_undo = displayed()

        assert after_step != opened
        assert after_undo == opened

    def test_a_failing_step_is_reported_and_survivable(
        self, photo_on_disk: Path
    ) -> None:
        app, page = build_app(photo_on_disk)
        messages_before = len(page.dialogs)

        app.apply_step("nonexistent")

        assert len(page.dialogs) > messages_before
        assert app.session is not None
        assert not app.session.is_modified

    def test_editing_without_an_image_is_refused_politely(self) -> None:
        app, page = build_app()

        app.apply_step("grayscale")

        assert page.dialogs, "the user was not told to open an image first"

    def test_undo_and_reset_do_nothing_without_an_image(self) -> None:
        # Both are reachable before anything is open, so neither may crash.
        app, _ = build_app()

        app.undo()
        app.reset()

        assert app.session is None


class TestDroppingAStep:
    """A step dropped on the pipeline must be applied like any other."""

    def test_a_drop_applies_the_step(self, photo_on_disk: Path) -> None:
        app, page = build_app(photo_on_disk)

        drop_step(page, "grayscale")

        assert app.session is not None
        assert app.session.pipeline_text == "grayscale"

    def test_dropping_and_clicking_lead_to_the_same_place(
        self, photo_on_disk: Path
    ) -> None:
        dropped_app, dropped_page = build_app(photo_on_disk)
        clicked_app, clicked_page = build_app(photo_on_disk)

        drop_step(dropped_page, "sepia")
        click_step(clicked_page, "sepia")

        assert dropped_app.session is not None
        assert clicked_app.session is not None
        assert np.array_equal(
            dropped_app.session.current.rgb, clicked_app.session.current.rgb
        )


class TestReorderingAndTuning:
    """Everything the terminal can express must be reachable from the window."""

    def test_a_step_can_be_moved(self, photo_on_disk: Path) -> None:
        app, _ = build_app(photo_on_disk)
        app.apply_step("grayscale")
        app.apply_step("invert")

        app.move_step(1, 0)

        assert app.session is not None
        assert app.session.pipeline_text == "invert | grayscale"

    def test_a_step_in_the_middle_can_be_removed(self, photo_on_disk: Path) -> None:
        app, _ = build_app(photo_on_disk)
        app.apply_step("grayscale")
        app.apply_step("invert")
        app.apply_step("posterize")

        app.remove_step(1)

        assert app.session is not None
        assert app.session.pipeline_text == "grayscale | posterize"

    def test_a_step_can_be_reconfigured(self, photo_on_disk: Path) -> None:
        app, _ = build_app(photo_on_disk)
        app.apply_step("posterize")

        app.set_step_parameters(0, {"levels": "8"})

        assert app.session is not None
        assert app.session.pipeline_text == "posterize:levels=8"

    def test_reconfiguring_redraws_the_canvas(self, photo_on_disk: Path) -> None:
        app, page = build_app(photo_on_disk)
        app.apply_step("posterize")
        before = find_controls(page.controls[0], ft.Image)[0].src

        app.set_step_parameters(0, {"levels": "2"})

        assert find_controls(page.controls[0], ft.Image)[0].src != before

    def test_a_bad_parameter_is_reported_and_survivable(
        self, photo_on_disk: Path
    ) -> None:
        app, page = build_app(photo_on_disk)
        app.apply_step("blur")
        messages_before = len(page.dialogs)

        app.set_step_parameters(0, {"radius": "lots"})

        assert len(page.dialogs) > messages_before
        assert app.session is not None
        assert app.session.pipeline_text == "blur"

    def test_undo_takes_back_a_move(self, photo_on_disk: Path) -> None:
        app, _ = build_app(photo_on_disk)
        app.apply_step("grayscale")
        app.apply_step("invert")
        app.move_step(1, 0)

        app.undo()

        assert app.session is not None
        assert app.session.pipeline_text == "grayscale | invert"

    def test_acting_on_a_step_that_vanished_is_harmless(
        self, photo_on_disk: Path
    ) -> None:
        # The settings panel may still be open on a step that has just been
        # removed, so a stale position must not take the window down.
        app, _ = build_app(photo_on_disk)
        app.apply_step("grayscale")

        app.remove_step(0)
        app.set_step_parameters(0, {"standard": "bt601"})
        app.move_step(4, 0)

        assert app.session is not None
        assert not app.session.is_modified


class TestHelp:
    """The help must be reachable, and must cover every step."""

    def test_the_overview_can_be_shown(self) -> None:
        app, page = build_app()

        app.show_overview()

        assert page.dialogs

    def test_a_single_step_can_be_explained(self) -> None:
        app, page = build_app()

        app.show_step_help("pen-sketch")

        assert page.dialogs

    def test_every_catalogued_step_has_help(self) -> None:
        # The help is generated from the catalogue, so this also guards against a
        # step whose parameters cannot be described.
        from pixel.registry import list_definitions

        app, page = build_app()

        for definition in list_definitions():
            app.show_step_help(definition.name)

        assert len(page.dialogs) == len(list_definitions())

    def test_asking_about_a_step_that_does_not_exist_is_reported(self) -> None:
        app, page = build_app()

        app.show_step_help("nonexistent")

        assert page.dialogs


class TestRedo:
    """A change taken back must be able to be put forward again."""

    def test_redo_puts_a_step_back(self, photo_on_disk: Path) -> None:
        app, _ = build_app(photo_on_disk)
        app.apply_step("grayscale")
        app.undo()

        app.redo()

        assert app.session is not None
        assert app.session.pipeline_text == "grayscale"

    def test_redo_puts_a_move_back(self, photo_on_disk: Path) -> None:
        app, _ = build_app(photo_on_disk)
        app.apply_step("grayscale")
        app.apply_step("invert")
        app.move_step(1, 0)
        app.undo()

        app.redo()

        assert app.session is not None
        assert app.session.pipeline_text == "invert | grayscale"

    def test_a_fresh_change_discards_what_was_undone(
        self, photo_on_disk: Path
    ) -> None:
        # Once you have gone somewhere new, stepping forward into the branch you
        # abandoned would be nothing but confusing.
        app, _ = build_app(photo_on_disk)
        app.apply_step("grayscale")
        app.undo()

        app.apply_step("sepia")
        app.redo()

        assert app.session is not None
        assert app.session.pipeline_text == "sepia"

    def test_redo_with_nothing_to_redo_is_harmless(self, photo_on_disk: Path) -> None:
        app, _ = build_app(photo_on_disk)

        app.redo()

        assert app.session is not None
        assert not app.session.is_modified

    def test_the_canvas_follows_a_redo(self, photo_on_disk: Path) -> None:
        app, page = build_app(photo_on_disk)

        def displayed() -> object:
            return find_controls(page.controls[0], ft.Image)[0].src

        opened = displayed()
        app.apply_step("invert")
        inverted = displayed()
        app.undo()
        app.redo()

        assert displayed() == inverted
        assert displayed() != opened


class TestPanels:
    """The side panels must be resizable and able to be put away."""

    def test_each_panel_can_be_hidden(self) -> None:
        app, _ = build_app()

        app.toggle_library()
        app.toggle_pipeline()

        assert not app.layout.library_visible
        assert not app.layout.pipeline_visible

    def test_a_hidden_panel_takes_its_divider_with_it(self) -> None:
        # A handle for resizing something that is not there would be a trap.
        app, page = build_app()

        app.toggle_library()

        dividers = find_controls(page.controls[0], ft.GestureDetector)
        assert [divider.visible for divider in dividers].count(False) == 1

    def test_a_panel_comes_back_the_width_it_was(self) -> None:
        app, _ = build_app()
        app.resize_library(50)
        widened = app.layout.library_width

        app.toggle_library()
        app.toggle_library()

        assert app.layout.library_width == widened

    def test_dragging_a_divider_resizes_its_panel(self) -> None:
        app, _ = build_app()
        before = app.layout.library_width

        app.resize_library(40)

        assert app.layout.library_width == before + 40

    def test_a_divider_cannot_be_dragged_past_the_limit(self) -> None:
        app, _ = build_app()

        app.resize_library(-9999)

        assert app.layout.library_width == theme.LIBRARY_MIN_WIDTH


class TestRememberingTheWork:
    """Closing and reopening must land the user back where they were."""

    def test_the_work_is_written_down_as_it_goes(
        self, photo_on_disk: Path
    ) -> None:
        app, _ = build_app(photo_on_disk)

        app.apply_step("grayscale")

        stored = workspace.load()
        assert stored.image_path == photo_on_disk
        assert stored.pipeline == "grayscale"

    def test_the_layout_is_written_down_too(self) -> None:
        app, _ = build_app()

        app.toggle_pipeline()

        assert not workspace.load().layout.pipeline_visible

    def test_reopening_restores_photo_and_pipeline(
        self, photo_on_disk: Path
    ) -> None:
        first, _ = build_app(photo_on_disk)
        first.apply_step("grayscale")
        first.apply_step("invert")

        # A second editor, started the way it would be on the next run.
        second, _ = build_app(restore=True)

        assert second.session is not None
        assert second.session.pipeline_text == "grayscale | invert"

    def test_the_restored_picture_is_the_one_that_was_left(
        self, photo_on_disk: Path
    ) -> None:
        first, _ = build_app(photo_on_disk)
        first.apply_step("grayscale")
        first.apply_step("posterize")

        second, _ = build_app(restore=True)

        assert first.session is not None
        assert second.session is not None
        assert np.array_equal(
            first.session.current.data, second.session.current.data
        )

    def test_reopening_restores_the_layout(self) -> None:
        first, _ = build_app()
        first.resize_library(45)
        first.toggle_pipeline()

        second, _ = build_app(restore=True)

        assert second.layout.library_width == first.layout.library_width
        assert not second.layout.pipeline_visible

    def test_a_named_image_wins_over_the_remembered_one(
        self, photo_on_disk: Path, tmp_path: Path
    ) -> None:
        first, _ = build_app(photo_on_disk)
        first.apply_step("grayscale")

        other = tmp_path / "other.png"
        save_image(solid_image((1, 2, 3), width=8, height=8), other)
        second, _ = build_app(other, restore=True)

        assert second.session is not None
        assert not second.session.is_modified

    def test_a_photo_that_has_since_vanished_is_passed_over_quietly(
        self, tmp_path: Path
    ) -> None:
        gone = tmp_path / "gone.png"
        save_image(solid_image((1, 2, 3), width=8, height=8), gone)
        first, _ = build_app(gone)
        first.apply_step("grayscale")
        gone.unlink()

        second, page = build_app(restore=True)

        assert second.session is None
        # The user did not ask for it to be reopened, so its absence is not a
        # failure they need telling about.
        assert page.dialogs == []

    def test_starting_fresh_ignores_what_was_remembered(
        self, photo_on_disk: Path
    ) -> None:
        first, _ = build_app(photo_on_disk)
        first.apply_step("grayscale")

        second, _ = build_app(restore=False)

        assert second.session is None


class TestNamedSessions:
    """Saving a session under a name, and opening it again."""

    def test_saving_writes_the_pipeline(
        self, photo_on_disk: Path, tmp_path: Path
    ) -> None:
        app, _ = build_app(photo_on_disk)
        app.apply_step("grayscale")
        app.apply_step("posterize")
        destination = tmp_path / "mine.json"

        app.save_session_to(destination)

        assert sessions.load(destination).pipeline == "grayscale | posterize"

    def test_saving_records_the_photo_it_was_made_from(
        self, photo_on_disk: Path, tmp_path: Path
    ) -> None:
        app, _ = build_app(photo_on_disk)
        destination = tmp_path / "mine.json"

        app.save_session_to(destination)

        assert sessions.load(destination).image_path == photo_on_disk

    def test_saving_keeps_the_parameters_and_the_order(
        self, photo_on_disk: Path, tmp_path: Path
    ) -> None:
        app, _ = build_app(photo_on_disk)
        app.apply_step("blur")
        app.apply_step("posterize")
        app.set_step_parameters(0, {"radius": "7"})
        app.move_step(1, 0)
        destination = tmp_path / "mine.json"

        app.save_session_to(destination)

        assert sessions.load(destination).pipeline == "posterize | blur:radius=7"

    def test_saving_without_an_image_is_refused_politely(
        self, tmp_path: Path
    ) -> None:
        app, page = build_app()

        app.save_session_to(tmp_path / "mine.json")

        assert page.dialogs
        assert not (tmp_path / "mine.json").exists()

    def test_opening_replays_the_steps(
        self, photo_on_disk: Path, tmp_path: Path
    ) -> None:
        saved = tmp_path / "mine.json"
        first, _ = build_app(photo_on_disk)
        first.apply_step("grayscale")
        first.apply_step("invert")
        first.save_session_to(saved)

        second, _ = build_app()
        second.open_session_from(saved)

        assert second.session is not None
        assert second.session.pipeline_text == "grayscale | invert"

    def test_the_reopened_picture_is_the_one_that_was_saved(
        self, photo_on_disk: Path, tmp_path: Path
    ) -> None:
        saved = tmp_path / "mine.json"
        first, _ = build_app(photo_on_disk)
        first.apply_step("grayscale")
        first.apply_step("posterize")
        first.save_session_to(saved)

        second, _ = build_app()
        second.open_session_from(saved)

        assert first.session is not None
        assert second.session is not None
        assert np.array_equal(
            first.session.current.data, second.session.current.data
        )

    def test_opening_replaces_whatever_was_being_edited(
        self, photo_on_disk: Path, tmp_path: Path
    ) -> None:
        # The steps are applied to the session's own photo, from scratch, not
        # piled on top of the work already in progress.
        saved = tmp_path / "mine.json"
        first, _ = build_app(photo_on_disk)
        first.apply_step("grayscale")
        first.save_session_to(saved)

        second, _ = build_app(photo_on_disk)
        second.apply_step("invert")
        second.apply_step("sepia")
        second.open_session_from(saved)

        assert second.session is not None
        assert second.session.pipeline_text == "grayscale"

    def test_a_session_whose_photo_has_moved_uses_the_open_one(
        self, photo_on_disk: Path, tmp_path: Path
    ) -> None:
        # A saved session doubles as a recipe: refusing to use one because its
        # original photo has been moved would throw that away.
        gone = tmp_path / "gone.png"
        save_image(solid_image((9, 9, 9), width=12, height=12), gone)
        first, _ = build_app(gone)
        first.apply_step("grayscale")
        saved = tmp_path / "mine.json"
        first.save_session_to(saved)
        gone.unlink()

        second, page = build_app(photo_on_disk)
        second.open_session_from(saved)

        assert second.session is not None
        assert second.session.pipeline_text == "grayscale"
        assert page.dialogs

    def test_a_session_with_no_photo_and_nothing_open_is_reported(
        self, tmp_path: Path
    ) -> None:
        saved = tmp_path / "recipe.json"
        sessions.save(sessions.SavedSession(None, "grayscale"), saved)

        app, page = build_app()
        app.open_session_from(saved)

        assert app.session is None
        assert page.dialogs

    def test_a_file_that_is_not_a_session_is_reported(self, tmp_path: Path) -> None:
        intruder = tmp_path / "shopping.json"
        intruder.write_text('{"milk": true}')

        app, page = build_app()
        app.open_session_from(intruder)

        assert page.dialogs
        assert app.session is None

    def test_a_missing_session_file_is_reported(self, tmp_path: Path) -> None:
        app, page = build_app()

        app.open_session_from(tmp_path / "absent.json")

        assert page.dialogs

    def test_a_saved_session_can_be_replayed_by_the_terminal(
        self, photo_on_disk: Path, tmp_path: Path
    ) -> None:
        # The pipeline stored in a session is the same text `pixel run` accepts,
        # so a session doubles as something scriptable.
        from pixel import build_pipeline
        from pixel.image_io import load_image

        saved = tmp_path / "mine.json"
        app, _ = build_app(photo_on_disk)
        app.apply_step("grayscale")
        app.apply_step("posterize")
        app.set_step_parameters(1, {"levels": "5"})
        app.save_session_to(saved)

        stored = sessions.load(saved)
        assert stored.image_path is not None
        straight = build_pipeline(stored.pipeline).run(load_image(stored.image_path))

        assert app.session is not None
        assert np.array_equal(app.session.current.data, straight.final_image.data)


class DeferringPage(FakePage):
    """A page that holds scheduled work back so it can be started all at once.

    The ordinary `FakePage` runs each coroutine the moment it is scheduled, which
    is what keeps the tests simple but also means they never see two pieces of
    work overlapping. This one collects them instead, so a test can set several
    going together and check what happens when they do.
    """

    def __init__(self) -> None:
        """Start with nothing scheduled."""
        super().__init__()
        self.pending: list[Coroutine[Any, Any, Any]] = []

    def run_task(
        self,
        handler: Callable[..., Coroutine[Any, Any, Any]],
        *args: object,
    ) -> None:
        """Hold the coroutine back instead of running it."""
        self.pending.append(handler(*args))

    def run_pending_together(self) -> None:
        """Start everything that was scheduled, all at the same time."""
        held, self.pending = self.pending, []

        async def gather_all() -> None:
            await asyncio.gather(*held)

        asyncio.run(gather_all())


class TestChangesDoNotOverlap:
    """Two edits at once must not leave the session describing itself wrongly."""

    def _app_with_a_step(self, photo: Path) -> tuple[ImageEditorApp, DeferringPage]:
        """Build an editor holding one step, ready for a concurrent test."""
        page = DeferringPage()
        app = ImageEditorApp(cast(ft.Page, page), photo, restore=False)
        page.run_pending_together()
        app.apply_step("gamma")
        page.run_pending_together()
        return app, page

    def test_two_parameter_changes_at_once_leave_the_session_sound(
        self, photo_on_disk: Path
    ) -> None:
        # Pressing Enter in a settings field both submits it and takes the focus
        # away. Before these were serialised, the two changes interleaved inside
        # the session and left it holding more results than steps, and the next
        # redraw died on it.
        app, page = self._app_with_a_step(photo_on_disk)

        app.set_step_parameters(0, {"gamma": "1.6"})
        app.set_step_parameters(0, {"gamma": "1.6"})
        page.run_pending_together()

        assert app.session is not None
        # Reading `applied` is what used to fail: it pairs each step with its
        # result and refuses to do so if the two lists have come apart.
        assert len(app.session.applied) == 1
        assert app.session.pipeline_text == "gamma:gamma=1.6"

    def test_many_changes_at_once_leave_the_session_sound(
        self, photo_on_disk: Path
    ) -> None:
        app, page = self._app_with_a_step(photo_on_disk)

        for value in ("0.5", "0.8", "1.2", "2.0", "2.5"):
            app.set_step_parameters(0, {"gamma": value})
        page.run_pending_together()

        assert app.session is not None
        assert len(app.session.applied) == 1

    def test_adding_and_removing_at_once_leave_the_session_sound(
        self, photo_on_disk: Path
    ) -> None:
        app, page = self._app_with_a_step(photo_on_disk)

        app.apply_step("invert")
        app.remove_step(0)
        app.undo()
        page.run_pending_together()

        assert app.session is not None
        assert len(app.session.applied) == len(
            parse_pipeline(app.session.pipeline_text)
        )


class TestExportingThePath:
    """Tracing the picture and writing the path out."""

    def test_it_writes_an_svg(self, photo_on_disk: Path, tmp_path: Path) -> None:
        app, _ = build_app(photo_on_disk)
        destination = tmp_path / "shape.svg"

        app.export_path_to(destination)

        assert destination.is_file()
        assert destination.read_text().startswith("<?xml")

    def test_the_path_follows_what_is_on_screen(
        self, photo_on_disk: Path, tmp_path: Path
    ) -> None:
        # The picture traced is the edited one, so putting remove-background in
        # the pipeline first is what gets a path around the subject.
        app, _ = build_app(photo_on_disk)
        app.apply_step("invert")
        destination = tmp_path / "shape.svg"

        app.export_path_to(destination)

        assert app.session is not None
        traced = paths.trace(app.session.current)
        assert str(traced.paths.width) in destination.read_text()

    def test_exporting_without_an_image_is_refused_politely(
        self, tmp_path: Path
    ) -> None:
        app, page = build_app()

        app.export_path_to(tmp_path / "shape.svg")

        assert page.dialogs
        assert not (tmp_path / "shape.svg").exists()

    def test_a_picture_with_no_shape_in_it_is_reported(self, tmp_path: Path) -> None:
        # A fully transparent image has no border to follow.
        blank = tmp_path / "blank.png"
        save_image(
            RGBAImage(np.zeros((30, 30, 4), dtype=np.uint8)), blank
        )
        app, page = build_app(blank)

        app.export_path_to(tmp_path / "shape.svg")

        assert page.dialogs
        assert not (tmp_path / "shape.svg").exists()

    def test_somewhere_unwritable_is_reported(self, photo_on_disk: Path, tmp_path: Path) -> None:
        blocked = tmp_path / "file"
        blocked.write_text("in the way")
        app, page = build_app(photo_on_disk)
        messages_before = len(page.dialogs)

        app.export_path_to(blocked / "shape.svg")

        assert len(page.dialogs) > messages_before


class TestSaveName:
    """The suggested file name must not lose the user's work."""

    def test_it_is_derived_from_the_opened_file(self, photo_on_disk: Path) -> None:
        app, _ = build_app(photo_on_disk)

        assert app.suggested_save_name() == "photo-edited.png"

    def test_it_is_always_a_png(self, tmp_path: Path) -> None:
        # A cut-out saved as JPEG would lose its transparency without warning.
        jpeg = tmp_path / "holiday.jpg"
        save_image(solid_image((10, 20, 30), width=8, height=8), jpeg)
        app, _ = build_app(jpeg)

        assert app.suggested_save_name().endswith(".png")
