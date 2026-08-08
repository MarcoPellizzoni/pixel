"""Tests for converting textual parameters into typed configurations."""

from dataclasses import dataclass
from enum import StrEnum

import pytest

from pixel.errors import InvalidParameterValueError, UnknownParameterError
from pixel.params import build_config, describe_parameters


class Mode(StrEnum):
    """Test choices, used to check the conversion of enumerations."""

    SLOW = "slow"
    FAST = "fast"


@dataclass(frozen=True)
class SampleConfig:
    """A configuration using every kind of parameter that is handled."""

    whole_number: int = 7
    decimal_number: float = 1.5
    switch: bool = True
    text: str = "hello"
    mode: Mode = Mode.SLOW
    colour: tuple[int, int, int] = (255, 255, 255)


class TestDefaultValues:
    """Whatever is not given must keep its default value."""

    def test_with_no_parameters_every_default_is_used(self) -> None:
        config = build_config(SampleConfig, {})

        assert config == SampleConfig()

    def test_the_parameters_not_given_do_not_change(self) -> None:
        config = build_config(SampleConfig, {"whole-number": "99"})

        assert config.whole_number == 99
        assert config.decimal_number == 1.5
        assert config.text == "hello"


class TestConversions:
    """Every declared type must be derived correctly from the text."""

    def test_whole_number(self) -> None:
        assert build_config(SampleConfig, {"whole-number": "-3"}).whole_number == -3

    def test_decimal_number(self) -> None:
        config = build_config(SampleConfig, {"decimal-number": "0.25"})

        assert config.decimal_number == pytest.approx(0.25)

    def test_text(self) -> None:
        assert build_config(SampleConfig, {"text": "hello world"}).text == "hello world"

    def test_enumeration(self) -> None:
        assert build_config(SampleConfig, {"mode": "fast"}).mode is Mode.FAST

    def test_enumeration_ignoring_case(self) -> None:
        assert build_config(SampleConfig, {"mode": "FAST"}).mode is Mode.FAST

    @pytest.mark.parametrize("written", ["true", "yes", "on", "1", "TRUE"])
    def test_the_true_values(self, written: str) -> None:
        assert build_config(SampleConfig, {"switch": written}).switch

    @pytest.mark.parametrize("written", ["false", "no", "off", "0", "FALSE"])
    def test_the_false_values(self, written: str) -> None:
        assert not build_config(SampleConfig, {"switch": written}).switch

    def test_hexadecimal_colour_with_a_hash(self) -> None:
        config = build_config(SampleConfig, {"colour": "#ff8800"})

        assert config.colour == (255, 136, 0)

    def test_hexadecimal_colour_without_a_hash(self) -> None:
        config = build_config(SampleConfig, {"colour": "0a0b0c"})

        assert config.colour == (10, 11, 12)


class TestParameterNames:
    """Names must be writable with either a hyphen or an underscore."""

    def test_the_hyphen_is_accepted(self) -> None:
        assert build_config(SampleConfig, {"whole-number": "5"}).whole_number == 5

    def test_the_underscore_is_accepted(self) -> None:
        assert build_config(SampleConfig, {"whole_number": "5"}).whole_number == 5


class TestErrors:
    """Every unacceptable value must say what is wrong with it."""

    def test_an_unknown_parameter_lists_the_valid_ones(self) -> None:
        with pytest.raises(UnknownParameterError, match="Accepted parameters"):
            build_config(SampleConfig, {"nonexistent": "1"})

    def test_a_non_numeric_integer(self) -> None:
        with pytest.raises(InvalidParameterValueError, match="whole number"):
            build_config(SampleConfig, {"whole-number": "many"})

    def test_a_non_numeric_decimal(self) -> None:
        with pytest.raises(InvalidParameterValueError, match="expects a number"):
            build_config(SampleConfig, {"decimal-number": "lots"})

    def test_an_incomprehensible_boolean(self) -> None:
        with pytest.raises(InvalidParameterValueError, match="true/false"):
            build_config(SampleConfig, {"switch": "maybe"})

    def test_a_value_outside_the_choices_lists_them(self) -> None:
        with pytest.raises(InvalidParameterValueError, match="slow, fast"):
            build_config(SampleConfig, {"mode": "sluggish"})

    def test_a_colour_that_is_too_short(self) -> None:
        with pytest.raises(InvalidParameterValueError, match="hexadecimal"):
            build_config(SampleConfig, {"colour": "#fff"})

    def test_a_non_hexadecimal_colour(self) -> None:
        with pytest.raises(InvalidParameterValueError, match="hexadecimal"):
            build_config(SampleConfig, {"colour": "reddish"})


class TestDescribeParameters:
    """The description feeds the built-in help and must be readable."""

    def test_lists_every_parameter(self) -> None:
        described = describe_parameters(SampleConfig)

        assert len(described) == 6

    def test_the_names_are_shown_hyphenated(self) -> None:
        names = [p.name for p in describe_parameters(SampleConfig)]

        assert "whole-number" in names
        assert "whole_number" not in names

    def test_enumerations_show_the_choices(self) -> None:
        types = {p.name: p.type_label for p in describe_parameters(SampleConfig)}

        assert types["mode"] == "slow | fast"

    def test_colours_are_shown_in_hexadecimal(self) -> None:
        defaults = {p.name: p.default for p in describe_parameters(SampleConfig)}

        assert defaults["colour"] == "#ffffff"

    def test_booleans_are_shown_the_way_they_are_written(self) -> None:
        defaults = {p.name: p.default for p in describe_parameters(SampleConfig)}

        assert defaults["switch"] == "true"

    def test_every_shown_value_can_be_read_back(self) -> None:
        # The default shown to the user must be accepted if they type it back:
        # otherwise the built-in help would suggest commands that do not work.
        for parameter in describe_parameters(SampleConfig):
            build_config(SampleConfig, {parameter.name: parameter.default})
