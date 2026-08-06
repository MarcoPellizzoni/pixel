"""`gatto`: elaborazione di una foto di gatto in disegno a penna.

Questo file definisce l'API pubblica del pacchetto: cio' che un altro programma
puo' importare senza dover conoscere l'organizzazione interna dei moduli.

Esempio d'uso come libreria:

    from pathlib import Path
    from gatto import CatSketchPipeline, PipelineConfig, load_image, save_image

    immagine = load_image(Path("g1.jpeg"))
    risultato = CatSketchPipeline(PipelineConfig()).run(immagine)
    save_image(risultato.final_image, Path("disegno.png"))
"""

from gatto.config import (
    BackgroundRemovalConfig,
    GrayscaleConfig,
    LuminanceStandard,
    PenSketchConfig,
    PipelineConfig,
    SegmentationModel,
)
from gatto.domain import RGBAImage
from gatto.image_io import load_image, save_image
from gatto.pipeline import CatSketchPipeline, PipelineResult, StepResult, save_results
from gatto.steps.background import BackgroundRemover
from gatto.steps.base import ProcessingStep
from gatto.steps.grayscale import GrayscaleConverter
from gatto.steps.pen_sketch import PenSketchRenderer

__version__ = "0.1.0"

# Elenco esplicito dei nomi esportati: rende evidente il confine tra API
# pubblica e dettagli interni, e guida `from gatto import *`.
__all__ = [
    # Modello di dominio
    "RGBAImage",
    # Ingresso/uscita
    "load_image",
    "save_image",
    # Configurazione
    "PipelineConfig",
    "BackgroundRemovalConfig",
    "GrayscaleConfig",
    "PenSketchConfig",
    "SegmentationModel",
    "LuminanceStandard",
    # Fasi di elaborazione
    "ProcessingStep",
    "BackgroundRemover",
    "GrayscaleConverter",
    "PenSketchRenderer",
    # Orchestrazione
    "CatSketchPipeline",
    "PipelineResult",
    "StepResult",
    "save_results",
    "__version__",
]
