"""Interfaccia a riga di comando.

Responsabilita' unica: tradurre gli argomenti scritti dall'utente in una
`PipelineConfig`, avviare l'elaborazione e riferire cosa e' successo.

Nessuna logica di elaborazione vive qui: se questo file venisse cancellato, la
libreria continuerebbe a funzionare per intero.
"""

from __future__ import annotations

from pathlib import Path
from typing import Annotated

import typer

from gatto.config import (
    BackgroundRemovalConfig,
    GrayscaleConfig,
    LuminanceStandard,
    PenSketchConfig,
    PipelineConfig,
    SegmentationModel,
)
from gatto.image_io import load_image
from gatto.pipeline import CatSketchPipeline, save_results

# Immagine elaborata quando l'utente non ne indica una: e' la foto del gatto
# che si trova nella radice del progetto.
DEFAULT_INPUT_PATH = Path("g1.jpeg")

# Cartella in cui finiscono i risultati.
DEFAULT_OUTPUT_DIRECTORY = Path("output")

# Nome del file finale. PNG perche' conserva la trasparenza attorno al gatto.
DEFAULT_FINAL_FILENAME = "gatto_disegno_a_penna.png"

application = typer.Typer(
    add_completion=False,
    help="Trasforma la foto di un gatto in un disegno a penna in bianco e nero.",
)


@application.command()
def process(
    input_path: Annotated[
        Path,
        typer.Argument(help="Percorso dell'immagine da elaborare."),
    ] = DEFAULT_INPUT_PATH,
    output_directory: Annotated[
        Path,
        typer.Option("--output", "-o", help="Cartella dove salvare i risultati."),
    ] = DEFAULT_OUTPUT_DIRECTORY,
    model: Annotated[
        SegmentationModel,
        typer.Option(
            "--model", "-m", help="Rete neurale per la rimozione dello sfondo."
        ),
    ] = SegmentationModel.ISNET_GENERAL,
    luminance: Annotated[
        LuminanceStandard,
        typer.Option("--luminance", "-l", help="Formula di conversione in grigio."),
    ] = LuminanceStandard.BT709,
    alpha_matting: Annotated[
        bool,
        typer.Option(
            "--alpha-matting/--no-alpha-matting",
            help="Rifinisce il bordo per recuperare baffi e pelo (piu' lento).",
        ),
    ] = True,
    hatching: Annotated[
        bool,
        typer.Option(
            "--hatching/--no-hatching",
            help="Riempie le ombre con un tratteggio a penna.",
        ),
    ] = True,
    ink_threshold: Annotated[
        float,
        typer.Option(
            "--ink-threshold",
            help="Soglia carta/inchiostro; alzala per un tratto piu' rado.",
        ),
    ] = 0.55,
    sharpness: Annotated[
        float,
        typer.Option(
            "--sharpness",
            help="Accentuazione dei contorni; alzala per tratti piu' marcati.",
        ),
    ] = 2.0,
    save_steps: Annotated[
        bool,
        typer.Option(
            "--save-steps/--no-save-steps",
            help="Salva anche il risultato di ogni singola fase.",
        ),
    ] = True,
) -> None:
    """Rimuove lo sfondo, converte in bianco e nero e applica l'effetto penna."""
    # Costruzione della configurazione a partire dalle scelte dell'utente.
    # Tutti i parametri non esposti dalla CLI restano ai valori predefiniti.
    config = PipelineConfig(
        background_removal=BackgroundRemovalConfig(
            model=model,
            use_alpha_matting=alpha_matting,
        ),
        grayscale=GrayscaleConfig(standard=luminance),
        pen_sketch=PenSketchConfig(
            sharpness=sharpness,
            ink_threshold=ink_threshold,
            enable_hatching=hatching,
        ),
        save_intermediate_steps=save_steps,
    )

    pipeline = CatSketchPipeline(config)

    typer.echo(f"Carico l'immagine: {input_path}")
    try:
        source_image = load_image(input_path)
    except FileNotFoundError as error:
        # Errore prevedibile dell'utente: un messaggio chiaro vale piu' di una
        # traccia di stack.
        typer.secho(str(error), fg=typer.colors.RED, err=True)
        raise typer.Exit(code=1) from error

    typer.echo(f"Dimensioni: {source_image.width}x{source_image.height} pixel")
    typer.echo("Fasi previste: " + " -> ".join(pipeline.iter_step_names()))
    typer.echo("Elaborazione in corso (il primo avvio scarica il modello)...")

    result = pipeline.run(source_image)

    written_paths = save_results(
        result=result,
        output_directory=output_directory,
        final_filename=DEFAULT_FINAL_FILENAME,
        save_intermediate_steps=config.save_intermediate_steps,
    )

    typer.secho("\nFatto. File scritti:", fg=typer.colors.GREEN)
    for path in written_paths:
        typer.echo(f"  - {path}")


def main() -> None:
    """Punto di ingresso registrato come comando `gatto` in pyproject.toml."""
    application()
