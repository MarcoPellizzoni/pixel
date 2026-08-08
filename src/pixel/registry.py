"""The catalogue of available steps.

Single responsibility: map each public name to the step that implements it and
to the configuration it accepts.

This is the one place in the program that knows about all the steps together.
Adding one means writing its class (with its configuration dataclass) and adding
a line to `STEP_DEFINITIONS`: the command line, the built-in help and the
parameter conversion all adapt by themselves.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable
from dataclasses import dataclass
from enum import StrEnum
from typing import Any

from pixel.dsl import StepInvocation
from pixel.errors import PipelineDefinitionError, UnknownStepError
from pixel.params import build_config
from pixel.steps.artistic import (
    CartoonConfig,
    CartoonStep,
    PencilSketchConfig,
    PencilSketchStep,
    VignetteConfig,
    VignetteStep,
)
from pixel.steps.background import RemoveBackgroundConfig, RemoveBackgroundStep
from pixel.steps.base import ProcessingStep
from pixel.steps.color import (
    GrayscaleConfig,
    GrayscaleStep,
    InvertConfig,
    InvertStep,
    PosterizeConfig,
    PosterizeStep,
    SaturationConfig,
    SaturationStep,
    SepiaConfig,
    SepiaStep,
)
from pixel.steps.filters import (
    BlurConfig,
    BlurStep,
    DenoiseConfig,
    DenoiseStep,
    EdgesConfig,
    EdgesStep,
    SharpenConfig,
    SharpenStep,
)
from pixel.steps.geometry import (
    CropConfig,
    CropStep,
    FlipConfig,
    FlipStep,
    ResizeConfig,
    ResizeStep,
    RotateConfig,
    RotateStep,
)
from pixel.steps.pen_sketch import PenSketchConfig, PenSketchStep
from pixel.steps.tone import (
    AutoContrastConfig,
    AutoContrastStep,
    BrightnessContrastConfig,
    BrightnessContrastStep,
    GammaConfig,
    GammaStep,
    ThresholdConfig,
    ThresholdStep,
)
from pixel.steps.trace import TraceStep, TraceStepConfig


class StepCategory(StrEnum):
    """Families the steps are grouped into, to find one's way in the catalogue."""

    GEOMETRY = "geometry"
    COLOR = "colour"
    TONE = "tone"
    FILTER = "filters"
    SEGMENTATION = "segmentation"
    ARTISTIC = "artistic"


@dataclass(frozen=True)
class StepDefinition:
    """A single step's catalogue entry."""

    # The name used to call the step up in a pipeline.
    name: str

    # The family it belongs to, used only to group the listing.
    category: StepCategory

    # A one-line description, shown in the step listing.
    summary: str

    # The dataclass describing the parameters accepted. Names, types and default
    # values are derived from it by introspection.
    # It is `type[Any]` rather than `type` because the catalogue deliberately
    # gathers configurations of different classes: which one it is is known only
    # at runtime, and stating that openly beats letting it be inferred as
    # "unknown".
    config_class: type[Any]

    # How to build the step from its configuration.
    factory: Callable[[Any], ProcessingStep]


# The catalogue. The order is the one in which the steps are listed to the user:
# the basic operations first, then the increasingly specialised ones.
STEP_DEFINITIONS: tuple[StepDefinition, ...] = (
    # --- Geometry ---
    StepDefinition(
        name="resize",
        category=StepCategory.GEOMETRY,
        summary="Resize the image, by scale or by requested dimensions.",
        config_class=ResizeConfig,
        factory=ResizeStep,
    ),
    StepDefinition(
        name="crop",
        category=StepCategory.GEOMETRY,
        summary="Crop a rectangular region.",
        config_class=CropConfig,
        factory=CropStep,
    ),
    StepDefinition(
        name="rotate",
        category=StepCategory.GEOMETRY,
        summary="Rotate by any angle, widening the canvas if needed.",
        config_class=RotateConfig,
        factory=RotateStep,
    ),
    StepDefinition(
        name="flip",
        category=StepCategory.GEOMETRY,
        summary="Flip the image horizontally or vertically.",
        config_class=FlipConfig,
        factory=FlipStep,
    ),
    # --- Colour ---
    StepDefinition(
        name="grayscale",
        category=StepCategory.COLOR,
        summary="Convert to greyscale using a perceptual luminance.",
        config_class=GrayscaleConfig,
        factory=GrayscaleStep,
    ),
    StepDefinition(
        name="sepia",
        category=StepCategory.COLOR,
        summary="Tone towards the brown of vintage prints.",
        config_class=SepiaConfig,
        factory=SepiaStep,
    ),
    StepDefinition(
        name="invert",
        category=StepCategory.COLOR,
        summary="Invert the colours, like a film negative.",
        config_class=InvertConfig,
        factory=InvertStep,
    ),
    StepDefinition(
        name="saturation",
        category=StepCategory.COLOR,
        summary="Make the hues more vivid or more muted.",
        config_class=SaturationConfig,
        factory=SaturationStep,
    ),
    StepDefinition(
        name="posterize",
        category=StepCategory.COLOR,
        summary="Reduce the image to a few flat colours, like a screen print.",
        config_class=PosterizeConfig,
        factory=PosterizeStep,
    ),
    # --- Tone ---
    StepDefinition(
        name="brightness-contrast",
        category=StepCategory.TONE,
        summary="Adjust brightness and contrast.",
        config_class=BrightnessContrastConfig,
        factory=BrightnessContrastStep,
    ),
    StepDefinition(
        name="gamma",
        category=StepCategory.TONE,
        summary="Lift or deepen the midtones, leaving blacks and whites alone.",
        config_class=GammaConfig,
        factory=GammaStep,
    ),
    StepDefinition(
        name="auto-contrast",
        category=StepCategory.TONE,
        summary="Recover shadow detail with a local equalisation (CLAHE).",
        config_class=AutoContrastConfig,
        factory=AutoContrastStep,
    ),
    StepDefinition(
        name="threshold",
        category=StepCategory.TONE,
        summary="Binarise to pure black and white (fixed, Otsu or adaptive).",
        config_class=ThresholdConfig,
        factory=ThresholdStep,
    ),
    # --- Filters ---
    StepDefinition(
        name="blur",
        category=StepCategory.FILTER,
        summary="Blur with a Gaussian filter.",
        config_class=BlurConfig,
        factory=BlurStep,
    ),
    StepDefinition(
        name="sharpen",
        category=StepCategory.FILTER,
        summary="Enhance detail with an unsharp mask.",
        config_class=SharpenConfig,
        factory=SharpenStep,
    ),
    StepDefinition(
        name="denoise",
        category=StepCategory.FILTER,
        summary="Attenuate noise while keeping contours crisp.",
        config_class=DenoiseConfig,
        factory=DenoiseStep,
    ),
    StepDefinition(
        name="edges",
        category=StepCategory.FILTER,
        summary="Reduce the image to its contours (Canny's algorithm).",
        config_class=EdgesConfig,
        factory=EdgesStep,
    ),
    # --- Segmentation ---
    StepDefinition(
        name="remove-background",
        category=StepCategory.SEGMENTATION,
        summary="Isolate the subject and make the background transparent (neural net).",
        config_class=RemoveBackgroundConfig,
        factory=RemoveBackgroundStep,
    ),
    StepDefinition(
        name="trace",
        category=StepCategory.SEGMENTATION,
        summary="Trace the shape's outline as an editable Bezier path.",
        config_class=TraceStepConfig,
        factory=TraceStep,
    ),
    # --- Artistic ---
    StepDefinition(
        name="pen-sketch",
        category=StepCategory.ARTISTIC,
        summary="Pen drawing: ink contours and hatched shadows.",
        config_class=PenSketchConfig,
        factory=PenSketchStep,
    ),
    StepDefinition(
        name="pencil-sketch",
        category=StepCategory.ARTISTIC,
        summary="Pencil sketch, with a soft stroke.",
        config_class=PencilSketchConfig,
        factory=PencilSketchStep,
    ),
    StepDefinition(
        name="cartoon",
        category=StepCategory.ARTISTIC,
        summary="Comic book: flat colours and heavy black contours.",
        config_class=CartoonConfig,
        factory=CartoonStep,
    ),
    StepDefinition(
        name="vignette",
        category=StepCategory.ARTISTIC,
        summary="Progressively darken the edges, like an old lens.",
        config_class=VignetteConfig,
        factory=VignetteStep,
    ),
)

# Index by name, built once: a step is looked up on every invocation and there is
# no reason to scan the whole catalogue each time.
_DEFINITIONS_BY_NAME: dict[str, StepDefinition] = {
    definition.name: definition for definition in STEP_DEFINITIONS
}


def list_definitions() -> tuple[StepDefinition, ...]:
    """Return the whole catalogue, in presentation order."""
    return STEP_DEFINITIONS


def get_definition(name: str) -> StepDefinition:
    """Look up the step with the given name in the catalogue.

    Args:
        name: the name written by the user.

    Returns:
        The matching catalogue entry.

    Raises:
        UnknownStepError: if the name does not exist, with a suggestion.
    """
    definition = _DEFINITIONS_BY_NAME.get(name.strip().lower())

    if definition is None:
        raise UnknownStepError(
            f"Unknown step: '{name}'.{_suggest_similar(name)}\n"
            "The 'steps' command shows the full list."
        )

    return definition


def build_step(invocation: StepInvocation) -> ProcessingStep:
    """Build the step matching a request.

    Args:
        invocation: name and textual parameters exactly as read by the DSL.

    Returns:
        The step, ready to run.

    Raises:
        PipelineDefinitionError: if the step does not exist or the parameters are
            wrong.
    """
    definition = get_definition(invocation.name)

    try:
        config = build_config(definition.config_class, invocation.parameters)
    except PipelineDefinitionError as error:
        # The original message talks about the parameter alone: we add the step,
        # without which the user would not know where to look.
        raise type(error)(f"In step '{definition.name}': {error}") from error

    return definition.factory(config)


def build_steps(invocations: Iterable[StepInvocation]) -> tuple[ProcessingStep, ...]:
    """Build every step of a pipeline.

    Args:
        invocations: the requests read by the DSL, in order.

    Returns:
        The steps, ready to run.

    Raises:
        PipelineDefinitionError: at the first step that cannot be built.
    """
    # Every step is built before the processing begins: this way a typo is
    # reported immediately, rather than after a minute of computation spent on
    # the preceding steps.
    return tuple(build_step(invocation) for invocation in invocations)


def _suggest_similar(name: str) -> str:
    """Suggest the catalogue names resembling the wrong one.

    Args:
        name: the name that was not found.

    Returns:
        A suggestion sentence, or an empty string if nothing resembles it.
    """
    normalized = name.strip().lower()

    # A substring comparison is enough and predictable: it catches the frequent
    # cases ("gray" for "grayscale", "sketch" for "pen-sketch") without proposing
    # arbitrary matches.
    similar = [
        definition.name
        for definition in STEP_DEFINITIONS
        if normalized
        and (normalized in definition.name or definition.name in normalized)
    ]

    if not similar:
        return ""

    return f" Did you mean: {', '.join(similar)}?"
