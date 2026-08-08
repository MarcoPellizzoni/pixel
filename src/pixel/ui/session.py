"""The editing session: the pipeline being built, and the picture it produces.

Single responsibility: hold the list of steps the user has assembled, keep the
resulting image up to date, and let any of it be changed — a step added, removed,
moved or reconfigured — with undo behind all of it.

This module deliberately imports nothing from Flet. The user interface reads and
drives a session, but the session knows nothing about buttons or panels, which is
what makes the whole editing behaviour testable without ever opening a window.

Two ideas carry the design:

**The pipeline is the state.** The image is not edited in place; it is whatever
the current list of steps produces from the opened file. So reordering steps or
retyping a parameter is an ordinary change to that list, not a special case.

**Results are cached per position.** `_results[i]` is the image after step `i`.
Changing something at position `i` only invalidates the cache from there on, so
moving the last step of a long pipeline re-runs one step, not all of them. Undo
snapshots cost nothing to keep, because a snapshot is just a tuple of step names
and parameters.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

from pixel.domain import RGBAImage
from pixel.dsl import StepInvocation, format_pipeline
from pixel.registry import build_step

# How many undo snapshots to keep. Each one is a handful of strings, so the limit
# exists only to stop a long session growing without bound.
UNDO_DEPTH: int = 100


@dataclass(frozen=True)
class AppliedStep:
    """One step in the pipeline, and the image it produced.

    Attributes:
        invocation: the step as it was requested, name and parameters. Keeping it
            means the session can describe itself back to the user in exactly the
            syntax the command line accepts.
        result: the image as it looked immediately after this step ran.
    """

    invocation: StepInvocation
    result: RGBAImage


class EditingSession:
    """The pipeline under construction, and the image it currently produces."""

    def __init__(self, source: RGBAImage) -> None:
        """Start a session on a freshly opened image.

        Args:
            source: the image as it was loaded, which `reset` will return to.
        """
        self._source = source

        # The pipeline itself: the single source of truth for what the picture
        # looks like.
        self._steps: list[StepInvocation] = []

        # `_results[i]` is the image after `_steps[i]`. Always the same length as
        # `_steps`, and always in step with it.
        self._results: list[RGBAImage] = []

        # Past states of the pipeline, most recent last. Undo restores one.
        self._history: list[tuple[StepInvocation, ...]] = []

        # States undone and not yet redone, most recently undone last. Making a
        # fresh change clears it, which is what everyone expects: once you have
        # gone somewhere new, the branch you abandoned is gone.
        self._future: list[tuple[StepInvocation, ...]] = []

    # ------------------------------------------------------------------
    # Reading the state
    # ------------------------------------------------------------------

    @property
    def source(self) -> RGBAImage:
        """The untouched image the session started from."""
        return self._source

    @property
    def current(self) -> RGBAImage:
        """The image as it looks right now, after every step in the pipeline."""
        if not self._results:
            return self._source
        return self._results[-1]

    @property
    def applied(self) -> tuple[AppliedStep, ...]:
        """The steps in the pipeline, first to last, each with its result."""
        return tuple(
            AppliedStep(invocation=invocation, result=result)
            for invocation, result in zip(self._steps, self._results, strict=True)
        )

    @property
    def can_undo(self) -> bool:
        """Whether there is a previous state to go back to."""
        return bool(self._history)

    @property
    def can_redo(self) -> bool:
        """Whether an undone change can be put back."""
        return bool(self._future)

    @property
    def is_modified(self) -> bool:
        """Whether the pipeline holds any step at all."""
        return bool(self._steps)

    @property
    def pipeline_text(self) -> str:
        """The pipeline written in the command line's own syntax.

        This is what makes the editor and the `pixel run` command two views of
        the same thing: whatever is built by dragging can be copied straight into
        a terminal, and vice versa.

        Returns:
            For example ``"remove-background | grayscale | pen-sketch"``, or an
            empty string while the pipeline is empty.
        """
        if not self._steps:
            return ""
        return format_pipeline(tuple(self._steps))

    def step_at(self, index: int) -> StepInvocation:
        """Return the step at a given position.

        Args:
            index: position in the pipeline, counting from 0.

        Returns:
            The step as it is currently configured.

        Raises:
            IndexError: if there is no step at that position.
        """
        return self._steps[index]

    # ------------------------------------------------------------------
    # Changing the pipeline
    #
    # Every one of these records the previous pipeline for undo, then rebuilds
    # only the part of the image that the change can possibly have affected.
    # ------------------------------------------------------------------

    def append(self, invocation: StepInvocation) -> None:
        """Add a step to the end of the pipeline and run it.

        Args:
            invocation: the step to add, with its parameters.

        Raises:
            PipelineDefinitionError: if the step does not exist or its parameters
                are wrong. The pipeline is left untouched in that case.
        """
        # The step is built before anything is recorded, so a bad request cannot
        # leave the session half-changed.
        step = build_step(invocation)
        result = step.apply(self.current)

        self._remember()
        self._steps.append(invocation)
        self._results.append(result)

    def remove_at(self, index: int) -> None:
        """Remove one step, wherever it sits in the pipeline.

        Args:
            index: position of the step to remove, counting from 0.

        Raises:
            IndexError: if there is no step at that position.
        """
        self._require_index(index)

        self._remember()
        del self._steps[index]
        self._rebuild_from(index)

    def move(self, index: int, destination: int) -> None:
        """Move a step to another position in the pipeline.

        Order matters: greying an image and then inverting it is not the same as
        inverting it and then greying it, so moving a step really does change the
        result.

        Args:
            index: position of the step to move.
            destination: position it should end up at. Values outside the
                pipeline are pulled back to its ends, so that nudging the first
                step up simply does nothing.

        Raises:
            IndexError: if there is no step at `index`.
        """
        self._require_index(index)

        target = max(0, min(destination, len(self._steps) - 1))
        if target == index:
            # Nothing to do, and recording an undo step for it would leave the
            # user pressing undo with no visible effect.
            return

        self._remember()
        self._steps.insert(target, self._steps.pop(index))

        # Everything from the earlier of the two positions onwards may now sit on
        # a different input, so that is where the rebuild starts.
        self._rebuild_from(min(index, target))

    def replace_at(self, index: int, invocation: StepInvocation) -> None:
        """Reconfigure a step already in the pipeline.

        Args:
            index: position of the step to change.
            invocation: the step as it should now be, usually the same name with
                different parameters.

        Raises:
            IndexError: if there is no step at that position.
            PipelineDefinitionError: if the new parameters are not valid. The
                pipeline is left untouched in that case.
        """
        self._require_index(index)

        # Built first, so that a mistyped parameter cannot destroy the pipeline.
        build_step(invocation)

        self._remember()
        self._steps[index] = invocation
        self._rebuild_from(index)

    def undo(self) -> bool:
        """Go back to the pipeline as it was before the last change.

        Undo covers every kind of change, not just the last step added: removing
        a step, reordering, and retyping a parameter can all be taken back.

        Returns:
            True if a change was undone, False if there was nothing to undo.
        """
        if not self._history:
            return False

        # Where we are now becomes the state redo can return to.
        self._future.append(tuple(self._steps))
        self._restore(self._history.pop())

        return True

    def redo(self) -> bool:
        """Put back a change that was undone.

        Returns:
            True if a change was redone, False if there was nothing to redo.
        """
        if not self._future:
            return False

        # The mirror image of undo: where we are now goes back onto the past.
        self._history.append(tuple(self._steps))
        self._restore(self._future.pop())

        return True

    def reset(self) -> None:
        """Empty the pipeline and go back to the opened image."""
        if not self._steps:
            return

        self._remember()
        self._steps.clear()
        self._results.clear()

    # ------------------------------------------------------------------
    # Keeping the results in step with the pipeline
    # ------------------------------------------------------------------

    def _restore(self, pipeline: tuple[StepInvocation, ...]) -> None:
        """Make a remembered pipeline the current one, and rebuild the picture.

        Args:
            pipeline: the pipeline to go back to.
        """
        # The rebuild starts at the first position where the two differ, so
        # stepping back over a change to the last step does not re-run the whole
        # pipeline.
        divergence = _first_difference(self._steps, pipeline)
        self._steps = list(pipeline)
        self._rebuild_from(divergence)

    def _rebuild_from(self, index: int) -> None:
        """Re-run the pipeline from a position onwards.

        Everything before `index` is untouched and its cached results stay valid,
        which is what keeps a small change to a long pipeline quick.

        Args:
            index: the first position whose result may have changed.
        """
        # Drop the results that can no longer be trusted.
        del self._results[index:]

        for invocation in self._steps[index:]:
            step = build_step(invocation)
            self._results.append(step.apply(self.current))

    def _remember(self) -> None:
        """Record the current pipeline so the change about to happen can be undone.

        Anything waiting to be redone is discarded at the same time: a new change
        starts a new branch, and the abandoned one would only be confusing to
        step forward into.
        """
        self._history.append(tuple(self._steps))
        self._future.clear()

        if len(self._history) > UNDO_DEPTH:
            # Oldest first: the far past is the least likely to be wanted back.
            del self._history[0]

    def _require_index(self, index: int) -> None:
        """Check that a position exists, and say so clearly when it does not.

        Args:
            index: the position to check.

        Raises:
            IndexError: if the position is outside the pipeline.
        """
        if not 0 <= index < len(self._steps):
            raise IndexError(
                f"No step at position {index}: the pipeline holds "
                f"{len(self._steps)}."
            )


def _first_difference(
    current: Sequence[StepInvocation], previous: Sequence[StepInvocation]
) -> int:
    """Find the first position at which two pipelines differ.

    Args:
        current: the pipeline as it is now.
        previous: the pipeline being restored.

    Returns:
        The index of the first difference, or the length of the shorter pipeline
        when one is simply a prefix of the other.
    """
    for index, (left, right) in enumerate(zip(current, previous, strict=False)):
        if left != right:
            return index

    return min(len(current), len(previous))
