"""Orchestration: runs a sequence of steps and keeps the results.

Single responsibility: pass the image through the steps in the order received.
It contains no pixel arithmetic at all and does not know which steps exist: it
receives them already built from whoever uses it.

That ignorance is exactly what makes the pipeline generic: any sequence of
objects honouring the `ProcessingStep` protocol will work.
"""

from __future__ import annotations

from collections.abc import Callable, Iterator, Sequence
from dataclasses import dataclass
from pathlib import Path

from pixel.domain import RGBAImage
from pixel.image_io import save_image
from pixel.steps.base import ProcessingStep


@dataclass(frozen=True)
class StepResult:
    """The result of a single step, with the information needed to track it."""

    # Position of the step in the sequence, starting from 1.
    order: int

    # Step name (e.g. "grayscale").
    name: str

    # The image the step produced.
    image: RGBAImage


@dataclass(frozen=True)
class PipelineResult:
    """The complete outcome of a run."""

    # The starting image, as it was loaded.
    source: RGBAImage

    # The results of all the steps, in the order they ran.
    steps: tuple[StepResult, ...]

    @property
    def final_image(self) -> RGBAImage:
        """The image produced by the last step."""
        if not self.steps:
            # Empty pipeline: the result is the input itself.
            return self.source
        return self.steps[-1].image


class ImagePipeline:
    """Applies a series of steps to an image, in sequence."""

    def __init__(self, steps: Sequence[ProcessingStep]) -> None:
        """Build the pipeline.

        Args:
            steps: the steps to run, already configured, in the wanted order.
        """
        self._steps: tuple[ProcessingStep, ...] = tuple(steps)

    def run(
        self,
        source: RGBAImage,
        on_step_start: Callable[[int, str], None] | None = None,
    ) -> PipelineResult:
        """Run every step on the starting image.

        Args:
            source: the image loaded from disk.
            on_step_start: optional callback invoked before each step, with its
                position and its name. It exists for whoever wants to display
                progress without the pipeline having to know how.

        Returns:
            The complete outcome, including the intermediate results.
        """
        results: list[StepResult] = []

        # The "current" image flows from one step to the next.
        current_image = source

        for order, step in enumerate(self._steps, start=1):
            if on_step_start is not None:
                on_step_start(order, step.name)

            current_image = step.apply(current_image)
            results.append(StepResult(order=order, name=step.name, image=current_image))

        return PipelineResult(source=source, steps=tuple(results))

    def iter_step_names(self) -> Iterator[str]:
        """List the configured step names, in the order they run."""
        for step in self._steps:
            yield step.name

    def __len__(self) -> int:
        """Number of steps making up the pipeline."""
        return len(self._steps)


def save_results(
    result: PipelineResult,
    output_directory: Path,
    final_filename: str,
    save_intermediate_steps: bool,
) -> list[Path]:
    """Write the final result, and optionally the intermediate ones, to disk.

    This function lives outside the class because saving is not part of the
    processing: the pipeline stays pure and remains reusable by anyone who wants
    to keep the images in memory (a web service, for instance).

    Args:
        result: the pipeline's outcome.
        output_directory: destination directory.
        final_filename: file name for the final image.
        save_intermediate_steps: if True, also save every intermediate result.

    Returns:
        The list of paths written, in the order they were written.
    """
    written_paths: list[Path] = []

    if save_intermediate_steps:
        for step_result in result.steps:
            # The numeric prefix keeps the files sorted alphabetically in the
            # same sequence in which they were produced.
            intermediate_path = (
                output_directory / f"{step_result.order:02d}_{step_result.name}.png"
            )
            save_image(step_result.image, intermediate_path)
            written_paths.append(intermediate_path)

    final_path = output_directory / final_filename
    save_image(result.final_image, final_path)
    written_paths.append(final_path)

    return written_paths
