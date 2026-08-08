"""The language used to describe a pipeline on the command line.

Single responsibility: read a string and return the sequence of steps requested.
It does not know which steps exist (`registry` does) nor how they run (the steps
themselves do): it deals only with the form.

The syntax mirrors the shell pipe, because it describes the same thing: the
output of one step feeds into the next.

    resize:width=800 | grayscale | pen-sketch:ink-threshold=0.7

Rules:
- steps are separated by `|`;
- a step's parameters open with `:` and are separated by `,`;
- each parameter is written `name=value`;
- whitespace around the separators is free.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from types import MappingProxyType

from pixel.errors import PipelineSyntaxError

# Empty read-only mapping, used as the default value for steps invoked without
# any parameters.
EMPTY_PARAMETERS: Mapping[str, str] = MappingProxyType({})

# Separates one step from the next.
STEP_SEPARATOR: str = "|"

# Separates a step's name from its parameters.
PARAMETERS_SEPARATOR: str = ":"

# Separates one parameter from the next.
PARAMETER_SEPARATOR: str = ","

# Separates a parameter's name from its value.
ASSIGNMENT_SEPARATOR: str = "="


@dataclass(frozen=True)
class StepInvocation:
    """A request to run a step with certain parameters.

    This is the result of reading the text alone: at this point we do not yet
    know whether the step exists or whether the parameters are valid. Separating
    form from meaning lets us report syntax errors with different messages from
    the ones about wrong names.
    """

    # The step name, exactly as the user wrote it.
    name: str

    # The parameters, still as text: converting them to the right types happens
    # later, once we know which configuration they belong to.
    # The default value can be shared by every instance, instead of being rebuilt
    # by a `default_factory`, precisely because it is read-only: there is no risk
    # of anyone modifying it.
    parameters: Mapping[str, str] = EMPTY_PARAMETERS


def parse_pipeline(text: str) -> tuple[StepInvocation, ...]:
    """Read a pipeline description and return its steps.

    Args:
        text: the string written by the user.

    Returns:
        The requested steps, in the order they should run.

    Raises:
        PipelineSyntaxError: if the string does not follow the syntax.
    """
    if not text.strip():
        raise PipelineSyntaxError(
            "The pipeline is empty: name at least one step, "
            f"for example 'grayscale {STEP_SEPARATOR} pen-sketch'."
        )

    invocations = [
        _parse_invocation(segment, position)
        for position, segment in enumerate(text.split(STEP_SEPARATOR), start=1)
    ]

    return tuple(invocations)


def format_pipeline(invocations: tuple[StepInvocation, ...]) -> str:
    """Rewrite a sequence of steps in the language's syntax.

    Used to show the user the pipeline that actually ran, including the default
    one they did not write themselves.

    Args:
        invocations: the steps to rewrite.

    Returns:
        The corresponding string.
    """
    segments: list[str] = []

    for invocation in invocations:
        if invocation.parameters:
            assignments = PARAMETER_SEPARATOR.join(
                f"{name}{ASSIGNMENT_SEPARATOR}{value}"
                for name, value in invocation.parameters.items()
            )
            segments.append(f"{invocation.name}{PARAMETERS_SEPARATOR}{assignments}")
        else:
            segments.append(invocation.name)

    return f" {STEP_SEPARATOR} ".join(segments)


# ----------------------------------------------------------------------
# Reading the individual pieces
# ----------------------------------------------------------------------


def _parse_invocation(segment: str, position: int) -> StepInvocation:
    """Read a single step, with any parameters.

    Args:
        segment: the text between two pipes.
        position: the step's position, used in error messages.

    Returns:
        The corresponding request.

    Raises:
        PipelineSyntaxError: if the segment is empty or malformed.
    """
    cleaned = segment.strip()

    if not cleaned:
        raise PipelineSyntaxError(
            f"The step at position {position} is empty: there is probably one "
            f"'{STEP_SEPARATOR}' too many."
        )

    # `partition` isolates the name: any further colons stay in the parameter
    # part, where they are harmless.
    name, _, parameters_text = cleaned.partition(PARAMETERS_SEPARATOR)

    step_name = name.strip()
    if not step_name:
        raise PipelineSyntaxError(
            f"The step at position {position} has no name before "
            f"'{PARAMETERS_SEPARATOR}'."
        )

    return StepInvocation(
        name=step_name,
        # `MappingProxyType` makes the mapping read-only, consistently with the
        # dataclass being frozen.
        parameters=MappingProxyType(_parse_parameters(parameters_text, step_name)),
    )


def _parse_parameters(text: str, step_name: str) -> dict[str, str]:
    """Read a step's parameter section.

    Args:
        text: the text after the colon; it may be empty.
        step_name: the step's name, used in error messages.

    Returns:
        The parameters as textual name/value pairs.

    Raises:
        PipelineSyntaxError: if a parameter is not of the form `name=value`.
    """
    if not text.strip():
        # No parameters: the step will use all its default values.
        return {}

    parameters: dict[str, str] = {}

    for chunk in text.split(PARAMETER_SEPARATOR):
        assignment = chunk.strip()
        if not assignment:
            raise PipelineSyntaxError(
                f"Empty parameter in step '{step_name}': there is probably one "
                f"'{PARAMETER_SEPARATOR}' too many."
            )

        name, separator, value = assignment.partition(ASSIGNMENT_SEPARATOR)
        if not separator:
            raise PipelineSyntaxError(
                f"Parameter '{assignment}' of step '{step_name}' has no value: "
                f"the expected form is 'name{ASSIGNMENT_SEPARATOR}value'."
            )

        parameter_name = name.strip()
        if not parameter_name:
            raise PipelineSyntaxError(
                f"Parameter with no name in step '{step_name}': '{assignment}'."
            )

        if parameter_name in parameters:
            raise PipelineSyntaxError(
                f"Parameter '{parameter_name}' is given more than once in step "
                f"'{step_name}'."
            )

        parameters[parameter_name] = value.strip()

    return parameters
