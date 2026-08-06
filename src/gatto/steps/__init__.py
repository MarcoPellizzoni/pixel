"""Le singole fasi di elaborazione dell'immagine.

Ogni modulo di questo package contiene esattamente un algoritmo e rispetta il
protocollo `ProcessingStep` definito in `base`: riceve un'immagine, ne
restituisce una nuova, e non sa nulla delle altre fasi ne' del filesystem.
"""

from gatto.steps.background import BackgroundRemover
from gatto.steps.base import ProcessingStep
from gatto.steps.grayscale import GrayscaleConverter
from gatto.steps.pen_sketch import PenSketchRenderer

__all__ = [
    "ProcessingStep",
    "BackgroundRemover",
    "GrayscaleConverter",
    "PenSketchRenderer",
]
