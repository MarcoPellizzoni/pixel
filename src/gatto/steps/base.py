"""Il contratto comune a tutte le fasi di elaborazione.

Responsabilita' unica: definire l'interfaccia che ogni fase deve rispettare,
cosi' che la pipeline possa comporle senza sapere cosa facciano davvero.

Si usa un `Protocol` invece di una classe base astratta: le fasi non ereditano
nulla e non condividono codice, devono solo *avere la forma giusta* (structural
typing). Il controllo e' statico, a carico del type checker.
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable

from gatto.domain import RGBAImage


@runtime_checkable
class ProcessingStep(Protocol):
    """Una singola trasformazione: immagine in ingresso, immagine in uscita.

    Ogni fase deve essere una funzione pura sull'immagine: riceve un
    `RGBAImage` e ne restituisce uno nuovo, senza modificare l'originale.
    """

    @property
    def name(self) -> str:
        """Nome breve e leggibile della fase, usato nei log e nei file intermedi."""
        ...

    def apply(self, image: RGBAImage) -> RGBAImage:
        """Esegue la trasformazione.

        Args:
            image: l'immagine in ingresso, che non deve essere modificata.

        Returns:
            Una nuova immagine trasformata.
        """
        ...
