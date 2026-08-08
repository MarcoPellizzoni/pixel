"""Converting textual parameters into typed configurations, and describing them.

Single responsibility: bridge the world of strings (the command line, a text
field in a window) and the world of types (the steps' configuration dataclasses).

The module knows nothing about any particular step: it derives everything by
introspection from the dataclass it is handed. That is what makes it possible to
add a new step without writing a single line of parsing code, and without
touching either interface: its dataclass and one line in the catalogue are enough.

A parameter's explanation is part of that dataclass too, written as an
`Annotated` note beside its type:

    radius: Annotated[float, "How far the blur reaches, in pixels."] = 3.0

Keeping the explanation there rather than in a comment is what lets both `pixel
describe` and the editor's help show it, always saying the same thing as the
code it describes.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, fields, is_dataclass
from enum import Enum, StrEnum
from typing import Annotated, cast, get_args, get_origin, get_type_hints

from pixel.errors import InvalidParameterValueError, UnknownParameterError

# How boolean values are written on the command line. Several forms are accepted
# so the user does not have to remember which one the program prefers.
TRUE_LITERALS: frozenset[str] = frozenset({"true", "yes", "y", "on", "1"})
FALSE_LITERALS: frozenset[str] = frozenset({"false", "no", "n", "off", "0"})

# Number of components in a colour: red, green, blue.
COLOR_CHANNELS: int = 3

# A colour is written in hexadecimal, as on the web: `#ff8800` or `ff8800`.
# Two digits per component.
COLOR_DIGITS: int = COLOR_CHANNELS * 2

# The type of the colour triples used in the configurations.
COLOR_ANNOTATION = tuple[int, int, int]


class ParameterKind(StrEnum):
    """What sort of value a parameter takes.

    An interface uses this to decide how to ask for the value: a switch for a
    yes/no, a menu for a fixed set of choices, a plain box for a number. It says
    nothing about how the value is written down, which is the same everywhere.
    """

    BOOLEAN = "boolean"
    WHOLE_NUMBER = "whole number"
    NUMBER = "number"
    TEXT = "text"
    CHOICE = "choice"
    COLOUR = "colour"


@dataclass(frozen=True)
class ParameterInfo:
    """Everything an interface needs in order to show and edit one parameter."""

    # The name as it is written by the user, hyphenated.
    name: str

    # The name of the dataclass field behind it, with underscores.
    field_name: str

    # What sort of value it takes, so an interface can pick the right control.
    kind: ParameterKind

    # The default value, written the way the user would write it.
    default: str

    # A sentence or two on what the parameter does and which way to move it.
    # Empty when the dataclass carries no explanation.
    description: str

    # The permitted values, for `CHOICE` parameters. Empty for every other kind.
    choices: tuple[str, ...] = ()

    @property
    def type_label(self) -> str:
        """A readable name for the type, for listings and help text."""
        if self.kind is ParameterKind.CHOICE:
            # For a fixed set of choices the most useful "type" is the list itself.
            return " | ".join(self.choices)
        if self.kind is ParameterKind.COLOUR:
            return "colour (#rrggbb)"
        if self.kind is ParameterKind.BOOLEAN:
            return "true/false"
        return str(self.kind.value)


def describe_parameters(config_class: type) -> tuple[ParameterInfo, ...]:
    """List a configuration's parameters, ready to show to the user.

    Args:
        config_class: a step's configuration dataclass.

    Returns:
        One `ParameterInfo` per field, in declaration order.
    """
    # Two passes over the annotations: one keeping the `Annotated` notes, to read
    # the explanations, and one without them, to reason about the bare types.
    annotated_hints = get_type_hints(config_class, include_extras=True)
    bare_hints = get_type_hints(config_class)

    described: list[ParameterInfo] = []
    for field in fields(config_class):
        annotation = bare_hints[field.name]
        enum_class = as_enum_class(annotation)

        described.append(
            ParameterInfo(
                # On the command line names are written with a hyphen, which is
                # easier to type than an underscore.
                name=field.name.replace("_", "-"),
                field_name=field.name,
                kind=_classify(annotation),
                default=_describe_default(config_class, field.name, annotation),
                description=_read_description(annotated_hints[field.name]),
                choices=(
                    tuple(str(member.value) for member in enum_class)
                    if enum_class is not None
                    else ()
                ),
            )
        )

    return tuple(described)


def build_config[ConfigT](
    config_class: type[ConfigT], raw_parameters: Mapping[str, str]
) -> ConfigT:
    """Build a configuration from hand-written parameters.

    Parameters that are not given keep their default value.

    The returned type is the same one passed in: calling
    `build_config(BlurConfig, ...)` yields a `BlurConfig`, not a generic object.
    That is what lets the type checker verify the code using the result too.

    Args:
        config_class: the configuration dataclass to build.
        raw_parameters: the parameters as textual name/value pairs.

    Returns:
        An instance of `config_class`.

    Raises:
        TypeError: if the given class is not a dataclass.
        UnknownParameterError: if a parameter does not belong to the configuration.
        InvalidParameterValueError: if a value cannot be converted.
    """
    if not is_dataclass(config_class):
        raise TypeError(f"{config_class.__name__} is not a dataclass.")

    hints = get_type_hints(config_class)
    known_names = {field.name for field in fields(config_class)}

    arguments: dict[str, object] = {}
    for raw_name, raw_value in raw_parameters.items():
        # Names are accepted with either a hyphen or an underscore.
        name = raw_name.replace("-", "_")

        if name not in known_names:
            raise UnknownParameterError(
                f"Unknown parameter '{raw_name}'. "
                f"Accepted parameters: {_format_names(known_names)}."
            )

        arguments[name] = coerce_value(hints[name], raw_value, raw_name)

    return config_class(**arguments)


def as_enum_class(annotation: object) -> type[Enum] | None:
    """Tell whether an annotation is an enumeration.

    The check lives in its own function because `isinstance(x, type)` leaves
    behind a type the static checker cannot complete. Isolating it here, where
    the returned value has a declared type, keeps every function that needs it
    clean.

    Args:
        annotation: the declared type of a parameter.

    Returns:
        The enumeration, or None if the annotation is of another kind.
    """
    if isinstance(annotation, type) and issubclass(annotation, Enum):
        return annotation
    return None


def coerce_value(annotation: object, raw_value: str, parameter_name: str) -> object:
    """Convert a textual value to the type declared by the configuration.

    Args:
        annotation: the declared type of the parameter.
        raw_value: the value exactly as the user wrote it.
        parameter_name: the parameter's name, used in error messages.

    Returns:
        The converted value.

    Raises:
        InvalidParameterValueError: if the conversion is not possible.
    """
    text = raw_value.strip()

    # Booleans must be checked before integers: in Python `bool` is a subclass
    # of `int`, and the wrong order would send them down the numeric branch,
    # where "true" would mean nothing.
    if annotation is bool:
        return _coerce_bool(text, parameter_name)

    if annotation is int:
        return _coerce_int(text, parameter_name)

    if annotation is float:
        return _coerce_float(text, parameter_name)

    if annotation is str:
        return text

    enum_class = as_enum_class(annotation)
    if enum_class is not None:
        return _coerce_enum(enum_class, text, parameter_name)

    if annotation == COLOR_ANNOTATION:
        return _coerce_color(text, parameter_name)

    # Guards against adding a parameter of a type the module does not yet know
    # how to translate: an explicit error beats a conversion made up on the spot.
    raise InvalidParameterValueError(
        f"Parameter '{parameter_name}' has an unsupported type ({annotation!r})."
    )


# ----------------------------------------------------------------------
# Per-type conversions
# ----------------------------------------------------------------------


def _coerce_bool(text: str, parameter_name: str) -> bool:
    """Convert a textual value to a boolean."""
    lowered = text.lower()

    if lowered in TRUE_LITERALS:
        return True
    if lowered in FALSE_LITERALS:
        return False

    raise InvalidParameterValueError(
        f"Parameter '{parameter_name}' expects a true/false value, "
        f"got '{text}'. Accepted values: true, false."
    )


def _coerce_int(text: str, parameter_name: str) -> int:
    """Convert a textual value to an integer."""
    try:
        return int(text)
    except ValueError as error:
        raise InvalidParameterValueError(
            f"Parameter '{parameter_name}' expects a whole number, got '{text}'."
        ) from error


def _coerce_float(text: str, parameter_name: str) -> float:
    """Convert a textual value to a decimal number."""
    try:
        return float(text)
    except ValueError as error:
        raise InvalidParameterValueError(
            f"Parameter '{parameter_name}' expects a number, got '{text}'."
        ) from error


def _coerce_enum(enum_class: type[Enum], text: str, parameter_name: str) -> Enum:
    """Convert a textual value to one of the allowed choices."""
    try:
        return enum_class(text.lower())
    except ValueError as error:
        alternatives = ", ".join(str(member.value) for member in enum_class)
        raise InvalidParameterValueError(
            f"Value '{text}' is not valid for '{parameter_name}'. "
            f"Choices: {alternatives}."
        ) from error


def _coerce_color(text: str, parameter_name: str) -> tuple[int, int, int]:
    """Convert a hexadecimal colour to an (R, G, B) triple."""
    digits = text.removeprefix("#")

    if len(digits) != COLOR_DIGITS:
        raise InvalidParameterValueError(
            f"Parameter '{parameter_name}' expects a {COLOR_DIGITS}-digit "
            f"hexadecimal colour, for example '#ffffff'; got '{text}'."
        )

    try:
        # Two hexadecimal digits per channel, in red, green, blue order.
        return (
            int(digits[0:2], 16),
            int(digits[2:4], 16),
            int(digits[4:6], 16),
        )
    except ValueError as error:
        raise InvalidParameterValueError(
            f"Colour '{text}' passed to '{parameter_name}' is not valid hexadecimal."
        ) from error


# ----------------------------------------------------------------------
# Reading a parameter's shape and explanation
# ----------------------------------------------------------------------


def _classify(annotation: object) -> ParameterKind:
    """Work out which sort of value a parameter takes.

    Args:
        annotation: the declared type, with any `Annotated` note removed.

    Returns:
        The matching kind.

    Raises:
        InvalidParameterValueError: if the type is one no interface knows how to
            ask for. Failing here means a new parameter type cannot be added
            without also teaching the interfaces to display it.
    """
    # Booleans come first, for the same reason as in `coerce_value`.
    if annotation is bool:
        return ParameterKind.BOOLEAN
    if annotation is int:
        return ParameterKind.WHOLE_NUMBER
    if annotation is float:
        return ParameterKind.NUMBER
    if annotation is str:
        return ParameterKind.TEXT
    if as_enum_class(annotation) is not None:
        return ParameterKind.CHOICE
    if annotation == COLOR_ANNOTATION:
        return ParameterKind.COLOUR

    raise InvalidParameterValueError(
        f"No interface knows how to ask for a value of type {annotation!r}."
    )


def _read_description(annotated: object) -> str:
    """Pull a parameter's explanation out of its `Annotated` note.

    Args:
        annotated: the declared type, `Annotated` note included.

    Returns:
        The explanation, or an empty string when there is none.
    """
    if get_origin(annotated) is not Annotated:
        return ""

    # The first argument is the type itself; the notes follow. The first note
    # that is a string is taken as the explanation.
    for note in get_args(annotated)[1:]:
        if isinstance(note, str):
            return note

    return ""


def _describe_default(config_class: type, field_name: str, annotation: object) -> str:
    """Return a field's default value in readable form.

    Args:
        config_class: the dataclass the field belongs to.
        field_name: the field's name.
        annotation: the declared type of that field. It is needed to format the
            value the way the user will have to write it: from the runtime value
            alone a colour would be indistinguishable from any other triple.

    Returns:
        The default value, in the same form in which it is written.
    """
    # The most reliable way to know a default value, `default_factory` included,
    # is to build the configuration and read it. The value read can be of any
    # type: `object` states that explicitly, and the checks below narrow it down.
    value: object = getattr(config_class(), field_name)

    # Booleans have a form of their own: `str(True)` would give "True", whereas
    # on the command line one writes "true".
    if isinstance(value, bool):
        return "true" if value else "false"

    if isinstance(value, Enum):
        return str(value.value)

    if annotation == COLOR_ANNOTATION:
        # Colours are shown in the same form in which they are written. It is the
        # annotation that guarantees they are three integers: the type checker
        # cannot deduce that from a value read at runtime, so we declare it here,
        # at the one place where the information is available.
        channels = cast(tuple[int, int, int], value)
        return "#" + "".join(f"{channel:02x}" for channel in channels)

    return f"{value}"


def _format_names(names: set[str]) -> str:
    """Format a list of parameter names, sorted and hyphenated."""
    return ", ".join(sorted(name.replace("_", "-") for name in names))
