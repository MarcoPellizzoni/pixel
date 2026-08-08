"""Command-line interface.

Single responsibility: read what the user wrote, start the processing and report
what happened.

No processing logic lives here: if this file were deleted, the library would go
on working in full.

Three commands:
    run       process an image with the requested pipeline;
    steps     list the available steps;
    describe  show the parameters of a single step.
"""

from __future__ import annotations

import sys
import textwrap
from pathlib import Path
from typing import Annotated

import typer

from pixel import paths
from pixel.dsl import format_pipeline, parse_pipeline
from pixel.errors import PipelineDefinitionError
from pixel.image_io import STDIO_PATH, load_image, save_image
from pixel.params import describe_parameters
from pixel.paths import TraceSource
from pixel.pipeline import ImagePipeline, save_results
from pixel.registry import build_steps, get_definition, list_definitions

# The name the program is installed under (see `[project.scripts]` in
# pyproject.toml). It lives in a constant because it appears in several help
# messages: writing it out by hand each time would mean, at the first rename,
# leaving behind examples that no longer work.
PROGRAM_NAME = "pixel"

# Pipeline used when the user does not name one: isolate the subject, take it to
# greyscale and render it in pen. It doubles as an example of how they are written.
DEFAULT_PIPELINE = "remove-background | grayscale | pen-sketch"

# Directory the results end up in.
DEFAULT_OUTPUT_DIRECTORY = Path("output")

# Name of the final file. PNG because it preserves any transparency.
DEFAULT_FINAL_FILENAME = "result.png"

application = typer.Typer(
    add_completion=False,
    help="Process images by composing steps with the pipe: 'blur | grayscale | edges'.",
    # Without this option Typer does not show the list of commands when the
    # program is launched with no arguments, which is exactly when it is needed.
    no_args_is_help=True,
)


def _report(message: str) -> None:
    """Write an informational message to standard error.

    It always goes to stderr, never to stdout: when the image is written to
    standard output (the `-` path), any line of text mixed in with the bytes
    would make it unreadable.
    """
    typer.echo(message, err=True)


@application.command()
def run(
    input_path: Annotated[
        Path,
        typer.Argument(
            metavar="IMAGE",
            help="Image to process, or '-' to read it from standard input.",
        ),
    ],
    pipeline_text: Annotated[
        str,
        typer.Argument(
            metavar="PIPELINE",
            help=(
                "Sequence of steps separated by '|', for example "
                '"resize:width=800 | grayscale | pen-sketch".'
            ),
        ),
    ] = DEFAULT_PIPELINE,
    output_directory: Annotated[
        Path,
        typer.Option("--output", "-o", help="Directory to save the results in."),
    ] = DEFAULT_OUTPUT_DIRECTORY,
    output_name: Annotated[
        str,
        typer.Option("--name", "-n", help="Name of the final file."),
    ] = DEFAULT_FINAL_FILENAME,
    to_stdout: Annotated[
        bool,
        typer.Option(
            "--stdout",
            help=(
                "Write the final PNG to standard output instead of to a file, "
                "so several commands can be chained with shell pipes."
            ),
        ),
    ] = False,
    save_steps: Annotated[
        bool,
        typer.Option(
            "--save-steps/--no-save-steps",
            help="Also save the result of every individual step.",
        ),
    ] = False,
) -> None:
    """Process an image by applying the requested pipeline."""
    # Build everything first, process afterwards: a typo in the pipeline must be
    # reported immediately, not after loading a huge image.
    try:
        invocations = parse_pipeline(pipeline_text)
        steps_to_run = build_steps(invocations)
    except PipelineDefinitionError as error:
        typer.secho(str(error), fg=typer.colors.RED, err=True)
        raise typer.Exit(code=2) from error

    pipeline = ImagePipeline(steps_to_run)

    _report(f"Pipeline: {format_pipeline(invocations)}")

    try:
        source_image = load_image(input_path)
    except FileNotFoundError as error:
        # A predictable user error: a clear message is worth more than a stack
        # trace.
        typer.secho(str(error), fg=typer.colors.RED, err=True)
        raise typer.Exit(code=1) from error

    _report(f"Image: {source_image.width}x{source_image.height} pixels")

    def announce(order: int, name: str) -> None:
        """Announce the start of a step as the pipeline advances."""
        _report(f"  [{order}/{len(pipeline)}] {name}")

    result = pipeline.run(source_image, on_step_start=announce)

    if to_stdout:
        # Only the image goes to standard output: the intermediate results would
        # have no way of being told apart within the stream.
        save_image(result.final_image, STDIO_PATH)
        _report("Final image written to standard output.")
        return

    written_paths = save_results(
        result=result,
        output_directory=output_directory,
        final_filename=output_name,
        save_intermediate_steps=save_steps,
    )

    typer.secho("Done. Files written:", fg=typer.colors.GREEN, err=True)
    for path in written_paths:
        _report(f"  - {path}")


@application.command()
def trace(
    input_path: Annotated[
        Path,
        typer.Argument(metavar="IMAGE", help="Image to trace."),
    ],
    output_path: Annotated[
        Path,
        typer.Argument(metavar="OUTPUT.SVG", help="Where to write the SVG."),
    ],
    pipeline_text: Annotated[
        str,
        typer.Option(
            "--pipeline",
            "-p",
            help=(
                "Steps to run before tracing, in the usual syntax. "
                'Typically "remove-background", so the subject is traced.'
            ),
        ),
    ] = "",
    source: Annotated[
        TraceSource,
        typer.Option("--source", help="Which part of the image marks the shape."),
    ] = TraceSource.ALPHA,
    tolerance: Annotated[
        float,
        typer.Option("--tolerance", help="How far the path may stray, in pixels."),
    ] = 2.0,
    smoothness: Annotated[
        float,
        typer.Option("--smoothness", help="How rounded the corners are; 0 keeps them sharp."),
    ] = 1.0 / 3.0,
    threshold: Annotated[
        int,
        typer.Option("--threshold", help="Level separating inside from outside."),
    ] = 128,
) -> None:
    """Trace an image's outline and write it out as an SVG path."""
    try:
        source_image = load_image(input_path)
    except FileNotFoundError as error:
        typer.secho(str(error), fg=typer.colors.RED, err=True)
        raise typer.Exit(code=1) from error

    if pipeline_text:
        try:
            prepared = ImagePipeline(build_steps(parse_pipeline(pipeline_text)))
            source_image = prepared.run(source_image).final_image
        except PipelineDefinitionError as error:
            typer.secho(str(error), fg=typer.colors.RED, err=True)
            raise typer.Exit(code=2) from error

    report = paths.trace(
        source_image,
        config=paths.TraceConfig(source=source, threshold=threshold),
        style=paths.PathStyle(tolerance=tolerance, smoothness=smoothness),
    )

    if report.paths.is_empty:
        typer.secho(
            "Nothing to trace: no shape was found at that threshold.",
            fg=typer.colors.RED,
            err=True,
        )
        raise typer.Exit(code=1)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(paths.to_svg(report.paths))

    _report(
        f"Traced {report.outlines_found} outline(s): "
        f"{report.points_before} points reduced to {report.points_after} "
        f"({report.reduction:.0%} fewer)."
    )
    typer.secho(f"Written to {output_path}", fg=typer.colors.GREEN, err=True)


@application.command()
def steps() -> None:
    """List every available step, grouped by family."""
    current_category: str | None = None

    for definition in list_definitions():
        if definition.category.value != current_category:
            current_category = definition.category.value
            typer.secho(f"\n{current_category.upper()}", fg=typer.colors.CYAN, bold=True)

        # Fixed-width name, so the descriptions stay in a column.
        typer.echo(f"  {definition.name:<22} {definition.summary}")

    typer.echo("")
    typer.secho(
        f"Details of a step:  {PROGRAM_NAME} describe NAME",
        fg=typer.colors.BRIGHT_BLACK,
    )


@application.command()
def describe(
    step_name: Annotated[
        str,
        typer.Argument(metavar="NAME", help="Name of the step to describe."),
    ],
) -> None:
    """Show the parameters a step accepts, with their default values."""
    try:
        definition = get_definition(step_name)
    except PipelineDefinitionError as error:
        typer.secho(str(error), fg=typer.colors.RED, err=True)
        raise typer.Exit(code=2) from error

    typer.secho(definition.name, fg=typer.colors.CYAN, bold=True)
    typer.echo(f"  {definition.summary}")
    typer.echo(f"  Family: {definition.category.value}")

    parameters = describe_parameters(definition.config_class)

    if not parameters:
        typer.echo("\n  No parameters: this step always behaves the same way.")
        return

    typer.secho("\n  PARAMETERS", bold=True)
    for parameter in parameters:
        typer.echo("")
        typer.secho(f"    {parameter.name}", fg=typer.colors.CYAN)
        typer.echo(
            f"      {parameter.type_label:<34} default: {parameter.default}"
        )
        # The explanation is wrapped by hand rather than left to the terminal,
        # so it stays lined up under the name it belongs to.
        for line in textwrap.wrap(parameter.description, width=66):
            typer.secho(f"      {line}", fg=typer.colors.BRIGHT_BLACK)

    # A concrete example is worth more than abstract syntax: it is built from the
    # step's first parameter, so it is always valid.
    first = parameters[0]
    typer.echo("")
    typer.secho(
        f'  Example:  {PROGRAM_NAME} run photo.jpg "{definition.name}:'
        f'{first.name}={first.default}"',
        fg=typer.colors.BRIGHT_BLACK,
    )


def main() -> None:
    """Entry point registered as a command in pyproject.toml."""
    # The expected exceptions are already handled by the commands; here we catch
    # the keyboard interrupt, which would otherwise print a stack trace for a
    # perfectly normal action.
    try:
        application()
    except KeyboardInterrupt:
        typer.secho("\nInterrupted.", fg=typer.colors.YELLOW, err=True)
        sys.exit(130)
