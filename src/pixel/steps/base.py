"""The contract shared by every processing step.

Single responsibility: define the interface each step must honour, so the
pipeline can compose them without knowing what they actually do.

A `Protocol` is used instead of an abstract base class: steps inherit nothing
and share no code, they only need to *have the right shape* (structural typing).
The check is static, carried out by the type checker.
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable

from pixel.domain import RGBAImage


@runtime_checkable
class ProcessingStep(Protocol):
    """A single transformation: image in, image out.

    Every step must be a pure function on the image: it receives an `RGBAImage`
    and returns a new one, without modifying the original.
    """

    @property
    def name(self) -> str:
        """Short, readable name of the step, used in logs and intermediate files."""
        ...

    def apply(self, image: RGBAImage) -> RGBAImage:
        """Run the transformation.

        Args:
            image: the input image, which must not be modified.

        Returns:
            A new, transformed image.
        """
        ...
