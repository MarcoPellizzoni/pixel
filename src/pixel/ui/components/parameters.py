"""The controls for setting one step's parameters.

Single responsibility: turn a step's parameter list into editable fields, and
report the values back when the user changes one.

Nothing here is written per step. The fields are derived from `ParameterInfo`,
which the catalogue produces by introspection, so a new parameter on any step
appears in the editor with no change to this module — the same property that
lets it appear in `pixel describe`.

Each field carries its own explanation, so the answer to "what does this do?" is
next to the control rather than in a manual.

When a change is reported matters as much as what is reported. A switch or a menu
reports immediately, because the choice is complete the moment it is made. A
typed value reports when the field is left or Enter is pressed: re-running the
pipeline on every keystroke would mean recomputing the picture five times while
someone types "12.5".
"""

from __future__ import annotations

from collections.abc import Callable, Mapping

import flet as ft

from pixel.params import ParameterInfo, ParameterKind, describe_parameters
from pixel.registry import StepDefinition
from pixel.ui import theme


class ParameterEditor:
    """The editable fields for one step's parameters."""

    def __init__(
        self,
        definition: StepDefinition,
        values: Mapping[str, str],
        on_change: Callable[[dict[str, str]], None],
    ) -> None:
        """Build a field for every parameter the step accepts.

        Args:
            definition: the step's catalogue entry.
            values: the parameters currently set on this step, by hyphenated
                name. Anything absent is shown at its default.
            on_change: called with the full set of non-default parameters
                whenever the user changes one.
        """
        self._definition = definition
        self._on_change = on_change
        self._parameters = describe_parameters(definition.config_class)

        # What was last handed over. Pressing Enter in a field both submits it
        # and takes the focus away, so the same values would otherwise be
        # reported twice, and the pipeline recomputed twice for nothing.
        self._last_reported: dict[str, str] | None = None

        # The control holding each parameter's value, so they can all be read
        # back together when any one of them changes.
        self._fields: dict[str, ft.Control] = {}

        rows: list[ft.Control] = [
            self._build_field(parameter, values.get(parameter.name, parameter.default))
            for parameter in self._parameters
        ]

        # The values the editor opened with count as already reported: they are
        # what the step is set to, so handing them back would be a no-op change.
        self._last_reported = self.collect()

        if not rows:
            rows = [theme.caption("This step has no settings.")]

        self.control = ft.Column(controls=rows, spacing=theme.SPACING, tight=True)

    # ------------------------------------------------------------------
    # Reading the fields back
    # ------------------------------------------------------------------

    def collect(self) -> dict[str, str]:
        """Read every field and return the parameters that are not at default.

        Leaving the defaults out keeps the pipeline readable: a step at its
        standard settings is written `grayscale`, not
        `grayscale:standard=bt709`, which is also how someone would type it into
        a terminal.

        Returns:
            The parameters to set on the step, by hyphenated name.
        """
        collected: dict[str, str] = {}

        for parameter in self._parameters:
            value = self._read(parameter)
            if value != parameter.default:
                collected[parameter.name] = value

        return collected

    def _read(self, parameter: ParameterInfo) -> str:
        """Read one field's current value as text.

        Args:
            parameter: the parameter to read.

        Returns:
            The value, written the way the user would write it.
        """
        control = self._fields[parameter.name]

        if isinstance(control, ft.Switch):
            return "true" if control.value else "false"
        if isinstance(control, ft.Dropdown):
            return str(control.value)
        if isinstance(control, ft.TextField):
            return (control.value or "").strip()

        # Unreachable while `_build_field` is the only thing filling `_fields`,
        # but stating it beats returning something misleading.
        raise TypeError(f"Cannot read a value from {type(control).__name__}.")

    def _report(self) -> None:
        """Hand the current parameters over, unless they are the ones already sent.

        Staying silent when nothing has changed is not merely an optimisation: a
        step that takes seconds would otherwise be run a second time for the same
        result, while the user waits.
        """
        values = self.collect()

        if values == self._last_reported:
            return

        self._last_reported = values
        self._on_change(values)

    # ------------------------------------------------------------------
    # Building the fields
    # ------------------------------------------------------------------

    def _build_field(self, parameter: ParameterInfo, value: str) -> ft.Control:
        """Build the row for one parameter: label, help, and its control.

        Args:
            parameter: the parameter to build a field for.
            value: its current value, as text.

        Returns:
            The row, ready to place in the panel.
        """
        control = self._build_input(parameter, value)
        self._fields[parameter.name] = control

        header: list[ft.Control] = [
            ft.Text(
                value=parameter.name,
                size=theme.CAPTION_SIZE,
                weight=ft.FontWeight.W_500,
                color=theme.TEXT,
            ),
            ft.Text(
                value=parameter.type_label,
                size=theme.MICRO_SIZE,
                color=theme.TEXT_FAINT,
                expand=True,
            ),
        ]

        body: list[ft.Control] = [
            ft.Row(controls=header, spacing=theme.SPACING_TIGHT, tight=True),
            control,
        ]

        if parameter.description:
            # The explanation sits under the field rather than in a tooltip: it
            # is the answer to "which way do I move this?", which is worth
            # reading before touching the control, not after hovering it.
            body.append(
                ft.Text(
                    value=parameter.description,
                    size=theme.MICRO_SIZE,
                    color=theme.TEXT_FAINT,
                )
            )

        return ft.Column(controls=body, spacing=4, tight=True)

    def _build_input(self, parameter: ParameterInfo, value: str) -> ft.Control:
        """Build the control that actually holds a parameter's value.

        Args:
            parameter: the parameter to build a control for.
            value: its current value, as text.

        Returns:
            A switch, a menu or a text box, according to the parameter's kind.
        """
        if parameter.kind is ParameterKind.BOOLEAN:
            return ft.Switch(
                value=value.lower() in {"true", "yes", "on", "1"},
                active_color=theme.ACCENT,
                # A switch is complete the moment it is flipped, so it reports
                # straight away.
                on_change=lambda _: self._report(),
            )

        if parameter.kind is ParameterKind.CHOICE:
            return ft.Dropdown(
                value=value,
                options=[ft.DropdownOption(key=choice) for choice in parameter.choices],
                dense=True,
                filled=True,
                bgcolor=theme.INPUT_BACKGROUND,
                border_color=theme.BORDER,
                focused_border_color=theme.ACCENT,
                text_size=theme.CAPTION_SIZE,
                content_padding=ft.Padding.symmetric(horizontal=10, vertical=6),
                # A menu reports through `on_select`, not `on_change`: the latter
                # belongs to the editable variety of dropdown.
                on_select=lambda _: self._report(),
            )

        return ft.TextField(
            value=value,
            dense=True,
            filled=True,
            bgcolor=theme.INPUT_BACKGROUND,
            border_color=theme.BORDER,
            focused_border_color=theme.ACCENT,
            text_size=theme.CAPTION_SIZE,
            color=theme.TEXT,
            content_padding=ft.Padding.symmetric(horizontal=10, vertical=8),
            # Typed values report on leaving the field or pressing Enter, never
            # per keystroke: each report re-runs the pipeline.
            on_blur=lambda _: self._report(),
            on_submit=lambda _: self._report(),
        )
