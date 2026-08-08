"""Tests for the orchestration of the steps."""

from pathlib import Path

import numpy as np

from pixel import build_pipeline
from pixel.cli import DEFAULT_PIPELINE
from pixel.domain import RGBAImage
from pixel.pipeline import ImagePipeline, PipelineResult, StepResult, save_results


class FakeStep:
    """A test step that brightens the image by a fixed amount.

    It exists to check the orchestration without running the neural networks:
    the pipeline does not need to know what the steps do, so it can be tested
    with any steps at all.
    """

    def __init__(self, name: str, increment: int) -> None:
        """Store the step's name and how much it should brighten."""
        self._name = name
        self._increment = increment

    @property
    def name(self) -> str:
        """Name of the fake step."""
        return self._name

    def apply(self, image: RGBAImage) -> RGBAImage:
        """Add the increment to every colour channel."""
        brightened = np.clip(
            image.rgb.astype(np.int16) + self._increment, 0, 255
        ).astype(np.uint8)
        return image.with_rgb(brightened)


def black_image(side: int = 4) -> RGBAImage:
    """Create an opaque black image."""
    data = np.zeros((side, side, 4), dtype=np.uint8)
    data[:, :, 3] = 255
    return RGBAImage(data)


class TestStepComposition:
    """The pipeline must apply the steps in the right order."""

    def test_the_steps_are_applied_in_sequence(self) -> None:
        pipeline = ImagePipeline((FakeStep("first", 10), FakeStep("second", 5)))

        result = pipeline.run(black_image())

        assert np.all(result.final_image.rgb == 15)

    def test_every_step_leaves_its_own_intermediate_result(self) -> None:
        pipeline = ImagePipeline((FakeStep("first", 10), FakeStep("second", 5)))

        result = pipeline.run(black_image())

        assert len(result.steps) == 2
        assert np.all(result.steps[0].image.rgb == 10)
        assert np.all(result.steps[1].image.rgb == 15)

    def test_the_results_are_numbered_from_one(self) -> None:
        pipeline = ImagePipeline((FakeStep("first", 1), FakeStep("second", 1)))

        result = pipeline.run(black_image())

        assert [step.order for step in result.steps] == [1, 2]

    def test_the_starting_image_stays_intact(self) -> None:
        original = black_image()
        pipeline = ImagePipeline((FakeStep("first", 50),))

        pipeline.run(original)

        assert np.all(original.rgb == 0)


class TestDefaultPipeline:
    """The pipeline the command line offers must be valid."""

    def test_the_default_pipeline_builds(self) -> None:
        # It is also the first example the user sees: if it did not work, the
        # program would not even start with no arguments.
        pipeline = build_pipeline(DEFAULT_PIPELINE)

        assert list(pipeline.iter_step_names()) == [
            "remove-background",
            "grayscale",
            "pen-sketch",
        ]


class TestProgress:
    """Whoever runs the pipeline must be able to follow its progress."""

    def test_the_callback_receives_every_step_in_order(self) -> None:
        announced: list[tuple[int, str]] = []
        pipeline = ImagePipeline((FakeStep("first", 1), FakeStep("second", 1)))

        pipeline.run(
            black_image(),
            on_step_start=lambda order, name: announced.append((order, name)),
        )

        assert announced == [(1, "first"), (2, "second")]

    def test_without_a_callback_the_run_proceeds_anyway(self) -> None:
        pipeline = ImagePipeline((FakeStep("first", 7),))

        assert np.all(pipeline.run(black_image()).final_image.rgb == 7)

    def test_the_length_is_the_number_of_steps(self) -> None:
        pipeline = ImagePipeline((FakeStep("a", 1), FakeStep("b", 1)))

        assert len(pipeline) == 2


class TestResult:
    """`PipelineResult` must expose the final image correctly."""

    def test_the_final_image_is_the_last_step_s(self) -> None:
        source = black_image()
        last = FakeStep("last", 99).apply(source)
        result = PipelineResult(
            source=source,
            steps=(StepResult(order=1, name="last", image=last),),
        )

        assert np.all(result.final_image.rgb == 99)

    def test_with_no_steps_the_final_image_is_the_source(self) -> None:
        source = black_image()

        result = PipelineResult(source=source, steps=())

        assert result.final_image is source


class TestSavingTheResults:
    """Writing to disk must honour the intermediate-files option."""

    def _sample_result(self) -> PipelineResult:
        source = black_image()
        return PipelineResult(
            source=source,
            steps=(
                StepResult(order=1, name="first", image=source),
                StepResult(order=2, name="second", image=source),
            ),
        )

    def test_saves_the_intermediate_steps_when_asked(self, tmp_path: Path) -> None:
        paths = save_results(
            result=self._sample_result(),
            output_directory=tmp_path,
            final_filename="final.png",
            save_intermediate_steps=True,
        )

        assert len(paths) == 3
        assert (tmp_path / "01_first.png").is_file()
        assert (tmp_path / "02_second.png").is_file()
        assert (tmp_path / "final.png").is_file()

    def test_saves_only_the_final_one_when_not_asked(self, tmp_path: Path) -> None:
        paths = save_results(
            result=self._sample_result(),
            output_directory=tmp_path,
            final_filename="final.png",
            save_intermediate_steps=False,
        )

        assert paths == [tmp_path / "final.png"]
        assert not (tmp_path / "01_first.png").exists()

    def test_the_intermediate_names_sort_alphabetically(self, tmp_path: Path) -> None:
        save_results(
            result=self._sample_result(),
            output_directory=tmp_path,
            final_filename="final.png",
            save_intermediate_steps=True,
        )

        intermediate = sorted(p.name for p in tmp_path.glob("0*.png"))
        assert intermediate == ["01_first.png", "02_second.png"]
