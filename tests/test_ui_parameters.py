"""Tests for the parameter fields and the help built from the catalogue.

Between them these cover the promise that anything expressible on the command
line is reachable from the window: every parameter of every step gets a control
it can be set with, and an explanation of what moving it does.
"""

from __future__ import annotations

from types import SimpleNamespace

import flet as ft
import pytest
from test_ui_components import find_controls, fire

from pixel.params import ParameterKind, describe_parameters
from pixel.registry import get_definition, list_definitions
from pixel.ui.components.help import build_overview, build_step_help
from pixel.ui.components.parameters import ParameterEditor


def build_editor(
    step_name: str, values: dict[str, str] | None = None
) -> tuple[ParameterEditor, list[dict[str, str]]]:
    """Build a parameter editor for a step, and a log of what it reports.

    Args:
        step_name: the step whose parameters to edit.
        values: parameters already set on it, if any.

    Returns:
        The editor and the list its reports are appended to.
    """
    reported: list[dict[str, str]] = []
    editor = ParameterEditor(
        definition=get_definition(step_name),
        values=values or {},
        on_change=reported.append,
    )
    return editor, reported


class TestEveryStepIsEditable:
    """Every catalogued parameter must get a control it can be set with."""

    @pytest.mark.parametrize(
        "step_name", [definition.name for definition in list_definitions()]
    )
    def test_a_field_exists_for_every_parameter(self, step_name: str) -> None:
        editor, _ = build_editor(step_name)
        expected = describe_parameters(get_definition(step_name).config_class)

        inputs = (
            find_controls(editor.control, ft.Switch)
            + find_controls(editor.control, ft.Dropdown)
            + find_controls(editor.control, ft.TextField)
        )

        assert len(inputs) == len(expected)

    @pytest.mark.parametrize(
        "step_name", [definition.name for definition in list_definitions()]
    )
    def test_reading_the_fields_back_gives_a_valid_configuration(
        self, step_name: str
    ) -> None:
        # Whatever the editor reports must be something the step accepts,
        # otherwise the window could build a pipeline the library refuses.
        from pixel.dsl import StepInvocation
        from pixel.registry import build_step

        editor, _ = build_editor(step_name)

        build_step(StepInvocation(step_name, editor.collect()))


class TestControlChoice:
    """The control offered must suit the kind of value the parameter takes."""

    def test_a_yes_no_parameter_gets_a_switch(self) -> None:
        editor, _ = build_editor("remove-background")

        assert find_controls(editor.control, ft.Switch)

    def test_a_fixed_set_of_choices_gets_a_menu(self) -> None:
        editor, _ = build_editor("grayscale")

        menus = find_controls(editor.control, ft.Dropdown)
        assert len(menus) == 1
        assert [option.key for option in menus[0].options or []] == ["bt601", "bt709"]

    def test_a_number_gets_a_text_box(self) -> None:
        editor, _ = build_editor("blur")

        assert find_controls(editor.control, ft.TextField)

    def test_a_step_with_no_parameters_says_so(self) -> None:
        editor, _ = build_editor("invert")

        assert not find_controls(editor.control, ft.TextField)
        assert find_controls(editor.control, ft.Text)


class TestReporting:
    """What the editor reports must be the parameters, and nothing more."""

    def test_untouched_fields_report_nothing(self) -> None:
        # A step left at its defaults must stay written `blur`, not
        # `blur:radius=3.0`, so the pipeline reads the way one would type it.
        editor, _ = build_editor("blur")

        assert editor.collect() == {}

    def test_a_changed_field_is_reported(self) -> None:
        editor, reported = build_editor("blur")
        box = find_controls(editor.control, ft.TextField)[0]

        box.value = "9"
        fire(box.on_blur, SimpleNamespace())

        assert reported == [{"radius": "9"}]

    def test_a_field_put_back_to_its_default_drops_out(self) -> None:
        editor, reported = build_editor("blur", {"radius": "9"})
        box = find_controls(editor.control, ft.TextField)[0]

        box.value = "3.0"
        fire(box.on_blur, SimpleNamespace())

        assert reported == [{}]

    def test_a_switch_reports_the_moment_it_is_flipped(self) -> None:
        editor, reported = build_editor("remove-background")
        switch = find_controls(editor.control, ft.Switch)[0]

        switch.value = False
        fire(switch.on_change, SimpleNamespace())

        assert reported and "alpha-matting" in reported[-1]

    def test_a_menu_reports_the_moment_it_is_chosen(self) -> None:
        editor, reported = build_editor("grayscale")
        menu = find_controls(editor.control, ft.Dropdown)[0]

        menu.value = "bt601"
        fire(menu.on_select, SimpleNamespace())

        assert reported == [{"standard": "bt601"}]

    def test_values_already_set_are_shown(self) -> None:
        editor, _ = build_editor("blur", {"radius": "7.5"})

        assert find_controls(editor.control, ft.TextField)[0].value == "7.5"


class TestNotRepeatingItself:
    """The same values must not be handed over twice."""

    def test_reporting_the_same_value_twice_only_says_it_once(self) -> None:
        # Pressing Enter in a field both submits it and takes the focus away, so
        # both handlers fire. Running the pipeline twice for one edit would make
        # a slow step take twice as long for nothing.
        editor, reported = build_editor("gamma")
        box = find_controls(editor.control, ft.TextField)[0]

        box.value = "1.6"
        fire(box.on_submit, SimpleNamespace())
        fire(box.on_blur, SimpleNamespace())

        assert reported == [{"gamma": "1.6"}]

    def test_leaving_a_field_untouched_says_nothing(self) -> None:
        editor, reported = build_editor("gamma")
        box = find_controls(editor.control, ft.TextField)[0]

        fire(box.on_blur, SimpleNamespace())

        assert reported == []

    def test_a_further_change_is_still_reported(self) -> None:
        editor, reported = build_editor("gamma")
        box = find_controls(editor.control, ft.TextField)[0]

        box.value = "1.6"
        fire(box.on_blur, SimpleNamespace())
        box.value = "2.2"
        fire(box.on_blur, SimpleNamespace())

        assert reported == [{"gamma": "1.6"}, {"gamma": "2.2"}]

    def test_going_back_to_the_starting_value_is_reported(self) -> None:
        # It is a real change from what the step is currently set to.
        editor, reported = build_editor("gamma", {"gamma": "1.6"})
        box = find_controls(editor.control, ft.TextField)[0]

        box.value = "1.0"
        fire(box.on_blur, SimpleNamespace())

        assert reported == [{}]


class TestHelpText:
    """Every parameter must come with an explanation, in both interfaces."""

    def test_every_parameter_of_every_step_is_explained(self) -> None:
        # This is what makes the help worth opening: a parameter with no
        # explanation would leave the user guessing which way to move it.
        missing: list[str] = []

        for definition in list_definitions():
            for parameter in describe_parameters(definition.config_class):
                if not parameter.description.strip():
                    missing.append(f"{definition.name}.{parameter.name}")

        assert missing == []

    def test_the_explanation_is_shown_beside_the_field(self) -> None:
        editor, _ = build_editor("blur")
        expected = describe_parameters(get_definition("blur").config_class)[0]

        shown = [text.value for text in find_controls(editor.control, ft.Text)]

        assert expected.description in shown

    @pytest.mark.parametrize(
        "step_name", [definition.name for definition in list_definitions()]
    )
    def test_a_help_dialog_can_be_built_for_every_step(self, step_name: str) -> None:
        dialog = build_step_help(get_definition(step_name))

        assert isinstance(dialog, ft.AlertDialog)

    def test_the_overview_covers_every_step(self) -> None:
        dialog = build_overview()

        shown = {text.value for text in find_controls(dialog, ft.Text)}
        for definition in list_definitions():
            assert definition.name in shown


class TestKindsAreExhaustive:
    """Every parameter in the catalogue must map to a kind an interface handles."""

    def test_every_parameter_has_a_known_kind(self) -> None:
        for definition in list_definitions():
            for parameter in describe_parameters(definition.config_class):
                assert parameter.kind in set(ParameterKind)

    def test_choices_are_listed_for_menus_and_only_for_menus(self) -> None:
        for definition in list_definitions():
            for parameter in describe_parameters(definition.config_class):
                if parameter.kind is ParameterKind.CHOICE:
                    assert parameter.choices
                else:
                    assert parameter.choices == ()
