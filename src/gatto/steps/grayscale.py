"""Fase 2: conversione in bianco e nero.

Responsabilita' unica: trasformare i canali di colore in una scala di grigi
percettivamente corretta, lasciando intatto il canale alfa calcolato dalla
fase precedente.

Due passaggi distinti:
1. la luminanza pesata, che e' *la* conversione in bianco e nero;
2. un'equalizzazione locale del contrasto (CLAHE), che recupera i dettagli
   schiacciati nelle ombre della foto originale, scattata in penombra.
"""

from __future__ import annotations

import cv2
import numpy as np

from gatto.config import LUMINANCE_WEIGHTS, GrayscaleConfig
from gatto.domain import RGBAImage


class GrayscaleConverter:
    """Converte i colori dell'immagine in scala di grigi."""

    def __init__(self, config: GrayscaleConfig) -> None:
        """Prepara la fase.

        Args:
            config: standard di luminanza e parametri di contrasto.
        """
        self._config = config

    @property
    def name(self) -> str:
        """Nome della fase."""
        return "bianco-e-nero"

    def apply(self, image: RGBAImage) -> RGBAImage:
        """Sostituisce i canali di colore con il loro equivalente in grigio.

        Args:
            image: immagine a colori (con o senza trasparenza).

        Returns:
            La stessa immagine in bianco e nero, con l'alfa invariato.
        """
        luminance = self._compute_luminance(image.rgb)
        enhanced_luminance = self._enhance_local_contrast(luminance)

        # Un'immagine in bianco e nero resta tecnicamente RGB: replichiamo lo
        # stesso valore sui tre canali. Cosi' il tipo di dato non cambia e le
        # fasi successive non devono gestire due formati diversi.
        gray_as_rgb = np.dstack([enhanced_luminance] * 3)

        return image.with_rgb(gray_as_rgb)

    # ------------------------------------------------------------------
    # Passaggi interni
    # ------------------------------------------------------------------

    def _compute_luminance(self, rgb: np.ndarray) -> np.ndarray:
        """Calcola la luminosita' percepita come media pesata di R, G e B.

        Args:
            rgb: array (altezza, larghezza, 3) uint8.

        Returns:
            Array (altezza, larghezza) uint8 con i livelli di grigio.
        """
        red_weight, green_weight, blue_weight = LUMINANCE_WEIGHTS[self._config.standard]

        # Si lavora in virgola mobile per non perdere precisione nella somma
        # pesata; solo il risultato finale torna a 8 bit.
        rgb_float = rgb.astype(np.float32)

        luminance = (
            rgb_float[:, :, 0] * red_weight
            + rgb_float[:, :, 1] * green_weight
            + rgb_float[:, :, 2] * blue_weight
        )

        return np.clip(luminance, 0, 255).astype(np.uint8)

    def _enhance_local_contrast(self, luminance: np.ndarray) -> np.ndarray:
        """Applica CLAHE: equalizzazione dell'istogramma a contrasto limitato.

        Un'equalizzazione globale schiarirebbe tutta l'immagine in blocco. CLAHE
        invece divide l'immagine in tessere e ridistribuisce i toni dentro
        ciascuna, facendo emergere il pelo nelle zone in ombra senza bruciare
        quelle gia' illuminate. Il "limite di taglio" impedisce di amplificare
        troppo il rumore nelle aree uniformi.

        Args:
            luminance: immagine in scala di grigi.

        Returns:
            L'immagine con il contrasto locale migliorato.
        """
        if self._config.clahe_clip_limit <= 0:
            # Equalizzazione disattivata.
            return luminance

        clahe = cv2.createCLAHE(
            clipLimit=self._config.clahe_clip_limit,
            tileGridSize=(self._config.clahe_tile_size, self._config.clahe_tile_size),
        )

        return clahe.apply(luminance)
