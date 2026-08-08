"""`pixel`: composable, step-based image processing.

This file defines the package's public API: what another program can import
without having to know how the modules are organised internally.

Example of library use, composing the steps by hand:

    from pathlib import Path

    from pixel import GrayscaleConfig, GrayscaleStep, ImagePipeline
    from pixel import PenSketchConfig, PenSketchStep, load_image, save_image

    image = load_image(Path("photo.jpg"))
    pipeline = ImagePipeline([
        GrayscaleStep(GrayscaleConfig()),
        PenSketchStep(PenSketchConfig(ink_threshold=0.7)),
    ])
    save_image(pipeline.run(image).final_image, Path("drawing.png"))

Or by describing the pipeline with the same syntax used on the command line:

    from pixel import build_pipeline

    pipeline = build_pipeline("resize:width=800 | grayscale | pen-sketch")
"""

from pixel.domain import RGBAImage, to_uint8
from pixel.dsl import StepInvocation, format_pipeline, parse_pipeline
from pixel.errors import (
    InvalidParameterValueError,
    PipelineDefinitionError,
    PipelineSyntaxError,
    UnknownParameterError,
    UnknownStepError,
)
from pixel.image_io import load_image, save_image
from pixel.params import build_config, describe_parameters
from pixel.pipeline import ImagePipeline, PipelineResult, StepResult, save_results
from pixel.registry import (
    StepCategory,
    StepDefinition,
    build_step,
    build_steps,
    get_definition,
    list_definitions,
)
from pixel.steps.artistic import (
    CartoonConfig,
    CartoonStep,
    PencilSketchConfig,
    PencilSketchStep,
    VignetteConfig,
    VignetteStep,
)
from pixel.steps.background import (
    RemoveBackgroundConfig,
    RemoveBackgroundStep,
    SegmentationModel,
)
from pixel.steps.base import ProcessingStep
from pixel.steps.color import (
    GrayscaleConfig,
    GrayscaleStep,
    InvertConfig,
    InvertStep,
    LuminanceStandard,
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
    DenoiseMethod,
    DenoiseStep,
    EdgesConfig,
    EdgesStep,
    SharpenConfig,
    SharpenStep,
)
from pixel.steps.geometry import (
    CropConfig,
    CropStep,
    FitMode,
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
    ThresholdMethod,
    ThresholdStep,
)

__version__ = "0.2.0"


def build_pipeline(pipeline_text: str) -> ImagePipeline:
    """Build a pipeline from its textual description.

    This is the same syntax the command line accepts, for anyone who wants to
    define the processing in a configuration file rather than in code.

    Args:
        pipeline_text: for example ``"resize:width=800 | grayscale"``.

    Returns:
        The pipeline, ready to run.

    Raises:
        PipelineDefinitionError: if the description is not valid.
    """
    return ImagePipeline(build_steps(parse_pipeline(pipeline_text)))


# Explicit list of the exported names: it makes the boundary between public
# API and internal details obvious, and drives `from pixel import *`.
__all__ = [
    # Domain model
    "RGBAImage",
    "to_uint8",
    # Input/output
    "load_image",
    "save_image",
    # Pipeline composition
    "ImagePipeline",
    "PipelineResult",
    "StepResult",
    "save_results",
    "build_pipeline",
    "ProcessingStep",
    # Description language and catalogue
    "parse_pipeline",
    "format_pipeline",
    "StepInvocation",
    "StepDefinition",
    "StepCategory",
    "list_definitions",
    "get_definition",
    "build_step",
    "build_steps",
    "build_config",
    "describe_parameters",
    # Errors
    "PipelineDefinitionError",
    "PipelineSyntaxError",
    "UnknownStepError",
    "UnknownParameterError",
    "InvalidParameterValueError",
    # Geometry steps
    "ResizeStep",
    "ResizeConfig",
    "FitMode",
    "CropStep",
    "CropConfig",
    "RotateStep",
    "RotateConfig",
    "FlipStep",
    "FlipConfig",
    # Colour steps
    "GrayscaleStep",
    "GrayscaleConfig",
    "LuminanceStandard",
    "SepiaStep",
    "SepiaConfig",
    "InvertStep",
    "InvertConfig",
    "SaturationStep",
    "SaturationConfig",
    "PosterizeStep",
    "PosterizeConfig",
    # Tonal steps
    "BrightnessContrastStep",
    "BrightnessContrastConfig",
    "GammaStep",
    "GammaConfig",
    "AutoContrastStep",
    "AutoContrastConfig",
    "ThresholdStep",
    "ThresholdConfig",
    "ThresholdMethod",
    # Filtering steps
    "BlurStep",
    "BlurConfig",
    "SharpenStep",
    "SharpenConfig",
    "DenoiseStep",
    "DenoiseConfig",
    "DenoiseMethod",
    "EdgesStep",
    "EdgesConfig",
    # Segmentation steps
    "RemoveBackgroundStep",
    "RemoveBackgroundConfig",
    "SegmentationModel",
    # Artistic steps
    "PenSketchStep",
    "PenSketchConfig",
    "PencilSketchStep",
    "PencilSketchConfig",
    "CartoonStep",
    "CartoonConfig",
    "VignetteStep",
    "VignetteConfig",
    "__version__",
]
