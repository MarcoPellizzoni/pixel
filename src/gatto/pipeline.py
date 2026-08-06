"""Orchestrazione: mette in fila le tre fasi e gestisce i risultati.

Responsabilita' unica: decidere *l'ordine* delle fasi e cosa salvare. Non
contiene un solo calcolo su pixel: quelli stanno nel package `steps`.

L'ordine non e' arbitrario:
1. lo sfondo va tolto per primo, mentre l'immagine ha ancora i colori originali
   su cui la rete di segmentazione e' stata addestrata;
2. il bianco e nero viene dopo, perche' l'effetto penna ragiona su luminosita';
3. l'effetto penna e' l'ultimo, perche' lavora su un'immagine gia' pulita e
   contrastata.
"""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path

from gatto.config import PipelineConfig
from gatto.domain import RGBAImage
from gatto.image_io import save_image
from gatto.steps.background import BackgroundRemover
from gatto.steps.base import ProcessingStep
from gatto.steps.grayscale import GrayscaleConverter
from gatto.steps.pen_sketch import PenSketchRenderer


@dataclass(frozen=True)
class StepResult:
    """Il risultato di una singola fase, con le informazioni per tracciarlo."""

    # Posizione della fase nella sequenza, a partire da 1.
    order: int

    # Nome leggibile della fase (es. "rimozione-sfondo").
    name: str

    # L'immagine prodotta dalla fase.
    image: RGBAImage


@dataclass(frozen=True)
class PipelineResult:
    """L'esito completo di un'elaborazione."""

    # L'immagine di partenza, come e' stata caricata.
    source: RGBAImage

    # I risultati di tutte le fasi, in ordine di esecuzione.
    steps: tuple[StepResult, ...]

    @property
    def final_image(self) -> RGBAImage:
        """L'immagine prodotta dall'ultima fase."""
        if not self.steps:
            # Pipeline vuota: il risultato coincide con l'ingresso.
            return self.source
        return self.steps[-1].image


class CatSketchPipeline:
    """Applica in sequenza le tre elaborazioni richieste."""

    def __init__(
        self,
        config: PipelineConfig,
        steps: tuple[ProcessingStep, ...] | None = None,
    ) -> None:
        """Costruisce la pipeline a partire dalla configurazione.

        Ogni fase riceve solo la porzione di configurazione che la riguarda:
        `PenSketchRenderer` non sa nemmeno che esista un modello di segmentazione.

        Args:
            config: la configurazione completa.
            steps: sequenza di fasi alternativa a quella standard. Serve a chi
                vuole comporre un'elaborazione diversa (e ai test, che cosi'
                verificano l'orchestrazione senza far girare le reti neurali).
        """
        self._config = config

        # Le fasi sono una semplice tupla ordinata di oggetti che rispettano il
        # protocollo `ProcessingStep`. Aggiungerne una nuova (per esempio un
        # effetto "carta invecchiata") significa scrivere una classe e
        # inserirla qui: nient'altro nel modulo cambia.
        self._steps: tuple[ProcessingStep, ...] = steps or self._build_default_steps(
            config
        )

    @staticmethod
    def _build_default_steps(config: PipelineConfig) -> tuple[ProcessingStep, ...]:
        """Crea le tre fasi richieste, nell'ordine in cui vanno eseguite."""
        return (
            BackgroundRemover(config.background_removal),
            GrayscaleConverter(config.grayscale),
            PenSketchRenderer(config.pen_sketch),
        )

    def run(self, source: RGBAImage) -> PipelineResult:
        """Esegue tutte le fasi sull'immagine di partenza.

        Args:
            source: l'immagine caricata dal disco.

        Returns:
            L'esito completo, con i risultati intermedi.
        """
        results: list[StepResult] = []

        # L'immagine "corrente" scorre da una fase all'altra.
        current_image = source

        for order, step in enumerate(self._steps, start=1):
            current_image = step.apply(current_image)
            results.append(
                StepResult(order=order, name=step.name, image=current_image)
            )

        return PipelineResult(source=source, steps=tuple(results))

    def iter_step_names(self) -> Iterator[str]:
        """Elenca i nomi delle fasi configurate, in ordine di esecuzione."""
        for step in self._steps:
            yield step.name


def save_results(
    result: PipelineResult,
    output_directory: Path,
    final_filename: str,
    save_intermediate_steps: bool,
) -> list[Path]:
    """Scrive su disco il risultato finale ed, eventualmente, quelli intermedi.

    Questa funzione sta fuori dalla classe perche' salvare non e' parte
    dell'elaborazione: la pipeline resta pura e riutilizzabile anche da chi
    vuole tenere le immagini in memoria (per esempio un servizio web).

    Args:
        result: l'esito della pipeline.
        output_directory: cartella di destinazione.
        final_filename: nome del file per l'immagine finale.
        save_intermediate_steps: se True salva anche ogni fase intermedia.

    Returns:
        L'elenco dei percorsi scritti, in ordine di scrittura.
    """
    written_paths: list[Path] = []

    if save_intermediate_steps:
        for step_result in result.steps:
            # Il prefisso numerico mantiene i file ordinati alfabeticamente
            # nella stessa sequenza in cui sono stati prodotti.
            intermediate_path = (
                output_directory / f"{step_result.order:02d}_{step_result.name}.png"
            )
            save_image(step_result.image, intermediate_path)
            written_paths.append(intermediate_path)

    final_path = output_directory / final_filename
    save_image(result.final_image, final_path)
    written_paths.append(final_path)

    return written_paths
