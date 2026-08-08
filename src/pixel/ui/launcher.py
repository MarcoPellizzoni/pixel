"""How the editor gets started.

Single responsibility: read what the user typed on the command line and hand the
application to Flet in the requested form — a desktop window, or a page served to
a browser.

The two modes exist because a desktop window needs the system's graphical
libraries, which a headless Linux box or a plain WSL install does not have.
Serving the same interface over HTTP needs none of them.
"""

from __future__ import annotations

from pathlib import Path
from typing import Annotated

import flet as ft
import typer

from pixel.ui.app import create_main

# Port used when serving to a browser and none was asked for. 0 would let the
# operating system choose, but then the address would change on every run.
DEFAULT_PORT: int = 8550

application = typer.Typer(
    add_completion=False,
    help="Open the pixel image editor.",
)


@application.command()
def launch(
    image: Annotated[
        Path | None,
        typer.Argument(
            metavar="IMAGE",
            help="Image to open at start-up. Optional.",
        ),
    ] = None,
    web: Annotated[
        bool,
        typer.Option(
            "--web",
            help=(
                "Serve the editor to a browser instead of opening a window. "
                "Use this where the system's graphical libraries are missing, "
                "as on a headless server or a plain WSL install."
            ),
        ),
    ] = False,
    port: Annotated[
        int,
        typer.Option("--port", help="Port to serve on, with --web."),
    ] = DEFAULT_PORT,
    fresh: Annotated[
        bool,
        typer.Option(
            "--fresh",
            help=(
                "Start with an empty editor instead of reopening the photo and "
                "pipeline from last time."
            ),
        ),
    ] = False,
) -> None:
    """Open the editor, optionally on a given image."""
    if image is not None and not image.is_file():
        typer.secho(f"Image not found: {image}", fg=typer.colors.RED, err=True)
        raise typer.Exit(code=1)

    handler = create_main(image, restore=not fresh)

    if web:
        typer.echo(f"Editor available at http://127.0.0.1:{port}", err=True)
        # `ft.run` and its view modes carry incomplete annotations in Flet
        # itself, which the strict type checker reports as unknown types. The
        # suppressions stay at this one border with the framework.
        ft.run(  # pyright: ignore[reportUnknownMemberType]
            handler,
            view=ft.AppView.WEB_BROWSER,
            port=port,
            host="127.0.0.1",
        )
    else:
        ft.run(handler)  # pyright: ignore[reportUnknownMemberType]


def run() -> None:
    """Entry point registered as the `pixel-editor` command."""
    application()
