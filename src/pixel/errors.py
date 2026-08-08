"""The errors a user can cause by writing an invalid pipeline.

Single responsibility: give these errors a type, so the CLI can tell them apart
from a genuine program fault and show them as a clear message rather than a
stack trace.
"""

from __future__ import annotations


class PipelineDefinitionError(Exception):
    """The requested pipeline is not valid.

    Common root of every error made while writing the step sequence, so the CLI
    only has to catch this one class.
    """


class PipelineSyntaxError(PipelineDefinitionError):
    """The pipeline description does not follow the expected syntax."""


class UnknownStepError(PipelineDefinitionError):
    """A step was requested that does not exist in the catalogue."""


class UnknownParameterError(PipelineDefinitionError):
    """A parameter was passed to a step it does not belong to."""


class InvalidParameterValueError(PipelineDefinitionError):
    """A parameter value cannot be converted to the required type."""
