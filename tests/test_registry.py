"""Tests for the step catalogue."""

import pytest

from pixel.dsl import StepInvocation, parse_pipeline
from pixel.errors import (
    InvalidParameterValueError,
    UnknownParameterError,
    UnknownStepError,
)
from pixel.params import describe_parameters
from pixel.registry import build_step, build_steps, get_definition, list_definitions


class TestCatalogue:
    """The catalogue must be consistent and complete."""

    def test_it_contains_steps(self) -> None:
        assert len(list_definitions()) > 0

    def test_the_names_are_unique(self) -> None:
        names = [definition.name for definition in list_definitions()]

        assert len(names) == len(set(names))

    def test_the_names_are_lowercase_and_hyphenated(self) -> None:
        for definition in list_definitions():
            assert definition.name == definition.name.lower()
            assert "_" not in definition.name

    def test_every_entry_has_a_description(self) -> None:
        for definition in list_definitions():
            assert definition.summary.strip()

    def test_every_step_reports_the_name_it_has_in_the_catalogue(self) -> None:
        # If the two names diverged, the intermediate files would be called one
        # thing and the pipeline would be written another way.
        for definition in list_definitions():
            step = definition.factory(definition.config_class())

            assert step.name == definition.name

    def test_every_step_builds_with_its_default_values(self) -> None:
        for definition in list_definitions():
            build_step(StepInvocation(definition.name))

    def test_every_documented_parameter_is_accepted(self) -> None:
        # Cross-check between the built-in help and the actual conversion: what
        # `describe` shows must be exactly what `run` accepts.
        for definition in list_definitions():
            for parameter in describe_parameters(definition.config_class):
                build_step(
                    StepInvocation(
                        definition.name, {parameter.name: parameter.default}
                    )
                )


class TestLookup:
    """Looking up by name must be forgiving, and helpful when it fails."""

    def test_finds_an_existing_step(self) -> None:
        assert get_definition("grayscale").name == "grayscale"

    def test_ignores_whitespace_and_case(self) -> None:
        assert get_definition("  GrayScale ").name == "grayscale"

    def test_a_nonexistent_step_is_an_error(self) -> None:
        with pytest.raises(UnknownStepError, match="Unknown step"):
            get_definition("nonexistent")

    def test_an_incomplete_name_gets_a_suggestion(self) -> None:
        with pytest.raises(UnknownStepError, match="grayscale"):
            get_definition("gray")


class TestBuilding:
    """Parameter errors must say which step they refer to."""

    def test_builds_a_step_with_its_parameters(self) -> None:
        step = build_step(StepInvocation("blur", {"radius": "9"}))

        assert step.name == "blur"

    def test_an_unknown_parameter_names_the_step(self) -> None:
        with pytest.raises(UnknownParameterError, match="In step 'blur'"):
            build_step(StepInvocation("blur", {"radiuz": "9"}))

    def test_a_wrong_value_names_the_step(self) -> None:
        with pytest.raises(InvalidParameterValueError, match="In step 'blur'"):
            build_step(StepInvocation("blur", {"radius": "lots"}))

    def test_builds_a_whole_pipeline(self) -> None:
        steps = build_steps(parse_pipeline("blur:radius=2 | grayscale | edges"))

        assert [step.name for step in steps] == ["blur", "grayscale", "edges"]

    def test_an_error_midway_through_the_pipeline_is_reported(self) -> None:
        # Every step is built before the processing starts: an error in the last
        # one must not surface after minutes of computation.
        with pytest.raises(UnknownStepError):
            build_steps(parse_pipeline("blur | nonexistent | grayscale"))
