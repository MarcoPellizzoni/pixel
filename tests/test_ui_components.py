"""Tests for the editor's panels.

These do not open a window. They build each panel's control tree and drive its
callbacks, which is enough to catch the mistakes that matter most here: a Flet
control called with the wrong argument, an icon that does not exist, a drop that
reports the wrong step.

Anything needing a live window — that pixels actually appear — is outside what
can be checked without a display.
"""

from collections.abc import Callable
from types import SimpleNamespace
from typing import cast

import flet as ft
import pytest
from conftest import solid_image

from pixel.registry import list_definitions
from pixel.ui.components.canvas import EditorCanvas
from pixel.ui.components.library import DRAG_GROUP, StepLibraryPanel
from pixel.ui.components.pipeline import SETTINGS_MAX_HEIGHT, PipelinePanel
from pixel.ui.components.toolbar import EditorToolbar
from pixel.ui.preview import to_png_bytes


def fire(handler: object, event: object) -> None:
    """Invoke a Flet event handler with an event.

    Flet types its handlers as "either takes an event or takes nothing", and a
    type checker resolves that union to the no-argument form. The cast states
    which of the two the panels actually install.

    Args:
        handler: the handler read off a control, which must not be None.
        event: the event to hand it.
    """
    assert handler is not None
    cast(Callable[[object], object], handler)(event)


def find_controls[T: ft.Control](root: object, wanted: type[T]) -> list[T]:
    """Collect every control of a given type inside a control tree.

    Flet controls hold their children under several different attribute names,
    so the search walks whichever ones are present rather than assuming a shape.

    Args:
        root: the control to search from.
        wanted: the class to look for.

    Returns:
        Every matching control found, in no particular order.
    """
    found: list[T] = []
    seen: set[int] = set()

    def walk(node: object) -> None:
        if id(node) in seen:
            return
        seen.add(id(node))

        if isinstance(node, wanted):
            found.append(node)

        if isinstance(node, list):
            for item in cast(list[object], node):
                walk(item)
            return

        for attribute in ("content", "controls", "content_feedback"):
            child = getattr(node, attribute, None)
            if child is not None:
                walk(child)

    walk(root)
    return found


def build_pipeline_panel(
    on_step_dropped: Callable[[str], None] = lambda _: None,
    on_move: Callable[[int, int], None] = lambda _a, _b: None,
    on_remove: Callable[[int], None] = lambda _: None,
    on_parameters_changed: Callable[[int, dict[str, str]], None] = lambda _a, _b: None,
    on_help: Callable[[str], None] = lambda _: None,
) -> PipelinePanel:
    """Build a pipeline panel, wiring only the callbacks a test cares about."""
    return PipelinePanel(
        on_step_dropped=on_step_dropped,
        on_move=on_move,
        on_remove=on_remove,
        on_parameters_changed=on_parameters_changed,
        on_help=on_help,
    )


def build_toolbar(
    on_open: Callable[[], None] = lambda: None,
    on_save: Callable[[], None] = lambda: None,
    on_undo: Callable[[], None] = lambda: None,
    on_redo: Callable[[], None] = lambda: None,
    on_reset: Callable[[], None] = lambda: None,
    on_help: Callable[[], None] = lambda: None,
    on_toggle_library: Callable[[], None] = lambda: None,
    on_toggle_pipeline: Callable[[], None] = lambda: None,
    on_save_session: Callable[[], None] = lambda: None,
    on_open_session: Callable[[], None] = lambda: None,
) -> EditorToolbar:
    """Build a toolbar, wiring only the callbacks a test cares about."""
    return EditorToolbar(
        on_open=on_open,
        on_save=on_save,
        on_undo=on_undo,
        on_redo=on_redo,
        on_reset=on_reset,
        on_help=on_help,
        on_toggle_library=on_toggle_library,
        on_toggle_pipeline=on_toggle_pipeline,
        on_save_session=on_save_session,
        on_open_session=on_open_session,
    )


class TestStepLibrary:
    """The library must offer every catalogued step, draggable and clickable."""

    def test_it_builds(self) -> None:
        panel = StepLibraryPanel(on_apply_now=lambda _: None, on_help=lambda _: None)

        assert isinstance(panel.control, ft.Container)

    def test_it_lists_every_step_in_the_catalogue(self) -> None:
        panel = StepLibraryPanel(on_apply_now=lambda _: None, on_help=lambda _: None)

        draggables = find_controls(panel.control, ft.Draggable)

        assert len(draggables) == len(list_definitions())

    def test_each_step_carries_its_name_for_the_drop(self) -> None:
        panel = StepLibraryPanel(on_apply_now=lambda _: None, on_help=lambda _: None)

        carried = {d.data for d in find_controls(panel.control, ft.Draggable)}

        assert carried == {d.name for d in list_definitions()}

    def test_every_step_is_in_the_drop_group(self) -> None:
        # A step in the wrong group would simply refuse to drop, with no error.
        panel = StepLibraryPanel(on_apply_now=lambda _: None, on_help=lambda _: None)

        for draggable in find_controls(panel.control, ft.Draggable):
            assert draggable.group == DRAG_GROUP

    def test_the_plus_button_reports_its_own_step(self) -> None:
        # Each button must report the step it belongs to, not whichever one the
        # building loop happened to finish on.
        applied: list[str] = []
        panel = StepLibraryPanel(on_apply_now=applied.append, on_help=lambda _: None)

        for button in find_controls(panel.control, ft.IconButton):
            fire(button.on_click, SimpleNamespace())

        assert applied == [d.name for d in list_definitions()]


class TestPipelinePanel:
    """The pipeline panel must accept drops and list what was applied."""

    def _drop(self, panel: PipelinePanel, carried: object) -> None:
        """Simulate a drop on the panel's target.

        The event is delivered through the `DragTarget` rather than by calling
        the panel's method, so the test also proves the handler is wired to the
        control the user actually drops on.

        Args:
            panel: the panel under test.
            carried: whatever the dragged control carries in `data`.
        """
        target = find_controls(panel.control, ft.DragTarget)[0]
        fire(target.on_accept, SimpleNamespace(src=SimpleNamespace(data=carried)))

    def test_it_builds(self) -> None:
        panel = build_pipeline_panel()

        assert isinstance(panel.control, ft.DragTarget)

    def test_it_listens_on_the_same_group_the_library_drags_from(self) -> None:
        # If the two groups disagreed, nothing could ever be dropped.
        panel = build_pipeline_panel()

        targets = find_controls(panel.control, ft.DragTarget)

        assert [target.group for target in targets] == [DRAG_GROUP]

    def test_a_drop_reports_the_step_name(self) -> None:
        dropped: list[str] = []
        panel = build_pipeline_panel(on_step_dropped=dropped.append)

        self._drop(panel, "grayscale")

        assert dropped == ["grayscale"]

    def test_a_drop_carrying_something_else_is_ignored(self) -> None:
        dropped: list[str] = []
        panel = build_pipeline_panel(on_step_dropped=dropped.append)

        self._drop(panel, carried=None)

        assert dropped == []


class TestPipelineScrolling:
    """A tall pipeline must stay reachable, however many settings are open."""

    def _panel_with_steps(self, count: int) -> PipelinePanel:
        """Build a panel showing a given number of steps."""
        from pixel.dsl import StepInvocation
        from pixel.ui.session import AppliedStep

        panel = build_pipeline_panel()
        panel.show_steps(
            [
                AppliedStep(
                    invocation=StepInvocation("grayscale"),
                    result=solid_image((10, 20, 30)),
                )
                for _ in range(count)
            ],
            "grayscale",
        )
        return panel

    def test_the_list_scrolls(self) -> None:
        panel = self._panel_with_steps(3)

        lists = find_controls(panel.control, ft.ListView)

        assert lists, "nothing in the pipeline panel can scroll"
        assert lists[0].expand, "the list does not take the height it is given"

    def test_the_scrolling_list_is_not_buried_in_a_stack(self) -> None:
        # A `Stack` gives its children no definite height, and a scrolling list
        # without one never works out that it has more to show than fits: the
        # settings of a step with a dozen parameters simply get cut off.
        panel = self._panel_with_steps(3)

        assert not find_controls(panel.control, ft.Stack)

    def test_the_drag_target_wraps_the_panel_rather_than_the_list(self) -> None:
        # Same failure, different cause: a drag target between the panel and the
        # list does not pass a definite height on to what it wraps. Keeping it at
        # the very top means the list's height comes straight from the window.
        panel = self._panel_with_steps(3)

        targets = find_controls(panel.control, ft.DragTarget)

        assert targets == [panel.control]

    def test_opening_a_step_keeps_the_list_scrollable(self) -> None:
        panel = self._panel_with_steps(2)

        # Pressed the way the user does, through the settings button on the card.
        settings_button = next(
            button
            for button in find_controls(panel.control, ft.IconButton)
            if button.tooltip == "Settings"
        )
        fire(settings_button.on_click, SimpleNamespace())

        assert find_controls(panel.control, ft.ListView)
        # And the settings really did open, so the list is now taller.
        assert find_controls(panel.control, ft.TextField) or find_controls(
            panel.control, ft.Dropdown
        )


class TestSettingsFitInTheWindow:
    """A step with many settings must stay readable to the last field."""

    def _open_settings(self, step_name: str) -> PipelinePanel:
        """Show one step and open its settings, as the user would."""
        from pixel.dsl import StepInvocation
        from pixel.ui.session import AppliedStep

        panel = build_pipeline_panel()
        panel.show_steps(
            [
                AppliedStep(
                    invocation=StepInvocation(step_name),
                    result=solid_image((10, 20, 30)),
                )
            ],
            step_name,
        )
        settings_button = next(
            button
            for button in find_controls(panel.control, ft.IconButton)
            if button.tooltip == "Settings"
        )
        fire(settings_button.on_click, SimpleNamespace())
        return panel

    def test_a_step_with_many_settings_gets_its_own_scrolling_area(self) -> None:
        # `pen-sketch` has fifteen settings. Stacked up they come to more than a
        # window's height, so they need somewhere to scroll that does not depend
        # on whatever encloses the panel.
        panel = self._open_settings("pen-sketch")

        scrollers = [
            column
            for column in find_controls(panel.control, ft.Column)
            if column.scroll is not None
        ]

        assert scrollers, "the settings of a tall step cannot be scrolled"

    def test_that_area_has_a_stated_height(self) -> None:
        # A stated height is the only thing that makes a scrolling area work no
        # matter what encloses it, which is the whole point of having one here.
        panel = self._open_settings("pen-sketch")

        boxed = [
            container
            for container in find_controls(panel.control, ft.Container)
            if container.height == SETTINGS_MAX_HEIGHT
        ]

        assert boxed

    def test_a_step_with_few_settings_is_left_alone(self) -> None:
        # Boxing two fields into a tall scrolling area would only waste space.
        panel = self._open_settings("blur")

        scrollers = [
            column
            for column in find_controls(panel.control, ft.Column)
            if column.scroll is not None
        ]

        assert scrollers == []

    def test_every_field_is_present_however_many_there_are(self) -> None:
        from pixel.params import describe_parameters
        from pixel.registry import get_definition

        panel = self._open_settings("pen-sketch")
        expected = describe_parameters(get_definition("pen-sketch").config_class)

        inputs = (
            find_controls(panel.control, ft.Switch)
            + find_controls(panel.control, ft.Dropdown)
            + find_controls(panel.control, ft.TextField)
        )

        assert len(inputs) == len(expected)


class TestCanvas:
    """The canvas must swap between the placeholder and an image."""

    def test_it_starts_on_the_placeholder(self) -> None:
        canvas = EditorCanvas()

        assert find_controls(canvas.control, ft.Image) == []

    def test_showing_an_image_puts_one_on_screen(self) -> None:
        canvas = EditorCanvas()
        png = to_png_bytes(solid_image((10, 20, 30)))

        canvas.show_image(png)

        images = find_controls(canvas.control, ft.Image)
        assert len(images) == 1
        assert images[0].src == png

    def test_a_second_image_reuses_the_same_control(self) -> None:
        # Reusing it is what stops the picture blanking out between edits.
        canvas = EditorCanvas()
        first = to_png_bytes(solid_image((10, 20, 30)))
        second = to_png_bytes(solid_image((200, 100, 50)))

        canvas.show_image(first)
        control_after_first = find_controls(canvas.control, ft.Image)[0]
        canvas.show_image(second)
        control_after_second = find_controls(canvas.control, ft.Image)[0]

        assert control_after_first is control_after_second
        assert control_after_second.src == second

    def test_it_can_go_back_to_the_placeholder(self) -> None:
        canvas = EditorCanvas()
        canvas.show_image(to_png_bytes(solid_image((10, 20, 30))))

        canvas.show_placeholder()

        assert find_controls(canvas.control, ft.Image) == []


class TestToolbar:
    """The toolbar must reflect what the session allows."""

    def test_it_builds_with_everything_but_open_disabled(self) -> None:
        toolbar = build_toolbar()

        buttons = find_controls(toolbar.control, ft.Button)
        labels = {str(b.content): b.disabled for b in buttons}

        assert labels["Open"] is False
        assert labels["Save"] is True
        assert labels["Undo"] is True
        assert labels["Redo"] is True
        assert labels["Reset"] is True

    def test_each_button_reports_its_own_action(self) -> None:
        fired: list[str] = []
        toolbar = build_toolbar(
            on_open=lambda: fired.append("open"),
            on_save=lambda: fired.append("save"),
            on_undo=lambda: fired.append("undo"),
            on_redo=lambda: fired.append("redo"),
            on_reset=lambda: fired.append("reset"),
        )

        for button in find_controls(toolbar.control, ft.Button):
            fire(button.on_click, SimpleNamespace())

        assert fired == ["open", "save", "undo", "redo", "reset"]


@pytest.mark.parametrize("group", [DRAG_GROUP])
def test_the_two_panels_agree_on_the_drag_group(group: str) -> None:
    """The library drags and the pipeline listens on one and the same group."""
    library = StepLibraryPanel(on_apply_now=lambda _: None, on_help=lambda _: None)
    pipeline = build_pipeline_panel()

    dragged_groups = {d.group for d in find_controls(library.control, ft.Draggable)}
    target_groups = {t.group for t in find_controls(pipeline.control, ft.DragTarget)}

    assert dragged_groups == target_groups == {group}
