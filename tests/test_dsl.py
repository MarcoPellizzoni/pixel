"""Tests for the language used to describe a pipeline."""

import pytest

from pixel.dsl import StepInvocation, format_pipeline, parse_pipeline
from pixel.errors import PipelineSyntaxError


class TestSimpleSteps:
    """A step is written with its name alone."""

    def test_a_single_step(self) -> None:
        invocations = parse_pipeline("grayscale")

        assert len(invocations) == 1
        assert invocations[0].name == "grayscale"
        assert invocations[0].parameters == {}

    def test_several_steps_separated_by_the_pipe(self) -> None:
        invocations = parse_pipeline("blur | grayscale | edges")

        assert [step.name for step in invocations] == ["blur", "grayscale", "edges"]

    def test_whitespace_around_the_separators_is_free(self) -> None:
        without = parse_pipeline("blur|grayscale")
        with_spaces = parse_pipeline("   blur   |   grayscale   ")

        assert [s.name for s in without] == [s.name for s in with_spaces]

    def test_hyphenated_names_are_accepted(self) -> None:
        invocations = parse_pipeline("pen-sketch | remove-background")

        assert [step.name for step in invocations] == ["pen-sketch", "remove-background"]


class TestParameters:
    """Parameters open with a colon and are separated by commas."""

    def test_a_single_parameter(self) -> None:
        invocations = parse_pipeline("blur:radius=5")

        assert invocations[0].parameters == {"radius": "5"}

    def test_several_parameters(self) -> None:
        invocations = parse_pipeline("resize:width=800,height=600,fit=cover")

        assert invocations[0].parameters == {
            "width": "800",
            "height": "600",
            "fit": "cover",
        }

    def test_whitespace_inside_the_parameters_is_ignored(self) -> None:
        invocations = parse_pipeline("resize: width = 800 , height = 600 ")

        assert invocations[0].parameters == {"width": "800", "height": "600"}

    def test_parameters_of_different_steps_stay_separate(self) -> None:
        invocations = parse_pipeline("blur:radius=2 | posterize:levels=8")

        assert invocations[0].parameters == {"radius": "2"}
        assert invocations[1].parameters == {"levels": "8"}

    def test_a_value_may_contain_a_dot(self) -> None:
        invocations = parse_pipeline("pen-sketch:ink-threshold=0.65")

        assert invocations[0].parameters == {"ink-threshold": "0.65"}

    def test_a_value_may_contain_a_hash(self) -> None:
        # This is the case of hexadecimal colours.
        invocations = parse_pipeline("remove-background:fill=#ff0000")

        assert invocations[0].parameters == {"fill": "#ff0000"}

    def test_a_trailing_colon_with_no_parameters_is_tolerated(self) -> None:
        invocations = parse_pipeline("grayscale:")

        assert invocations[0].parameters == {}


class TestSyntaxErrors:
    """Every malformed shape must produce a message pointing at the problem."""

    def test_an_empty_pipeline_is_an_error(self) -> None:
        with pytest.raises(PipelineSyntaxError, match="empty"):
            parse_pipeline("   ")

    def test_one_pipe_too_many_is_an_error(self) -> None:
        with pytest.raises(PipelineSyntaxError, match="position 2"):
            parse_pipeline("blur | | grayscale")

    def test_a_trailing_pipe_is_an_error(self) -> None:
        with pytest.raises(PipelineSyntaxError, match="empty"):
            parse_pipeline("blur |")

    def test_a_parameter_with_no_value_is_an_error(self) -> None:
        with pytest.raises(PipelineSyntaxError, match="has no value"):
            parse_pipeline("blur:radius")

    def test_a_parameter_with_no_name_is_an_error(self) -> None:
        with pytest.raises(PipelineSyntaxError, match="with no name"):
            parse_pipeline("blur:=5")

    def test_a_step_with_no_name_is_an_error(self) -> None:
        with pytest.raises(PipelineSyntaxError, match="has no name"):
            parse_pipeline(":radius=5")

    def test_the_same_parameter_twice_is_an_error(self) -> None:
        with pytest.raises(PipelineSyntaxError, match="more than once"):
            parse_pipeline("blur:radius=2,radius=5")

    def test_one_comma_too_many_is_an_error(self) -> None:
        with pytest.raises(PipelineSyntaxError, match="Empty parameter"):
            parse_pipeline("blur:radius=2,,")


class TestFormatPipeline:
    """Rewriting must produce text that can be read back."""

    def test_rewrites_steps_without_parameters(self) -> None:
        text = format_pipeline((StepInvocation("blur"), StepInvocation("grayscale")))

        assert text == "blur | grayscale"

    def test_rewrites_the_parameters_too(self) -> None:
        text = format_pipeline((StepInvocation("blur", {"radius": "5"}),))

        assert text == "blur:radius=5"

    def test_the_result_can_be_read_back(self) -> None:
        original = "resize:width=800,fit=cover | grayscale | pen-sketch:sharpness=3.0"

        rewritten = format_pipeline(parse_pipeline(original))

        assert parse_pipeline(rewritten) == parse_pipeline(original)
