"""Modello di dominio: il tipo di dato che attraversa tutta la pipeline.

Questo modulo ha una sola responsabilita': definire *cos'e'* un'immagine per
questo programma, e garantire che sia sempre in una forma valida e prevedibile
(RGBA, 8 bit per canale, contigua in memoria).

Non conosce ne' i file su disco (vedi `image_io`) ne' gli algoritmi di
elaborazione (vedi il package `steps`).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

# Numero di canali di un'immagine RGBA: rosso, verde, blu, alfa (opacita').
RGBA_CHANNELS: int = 4

# Numero di canali di un'immagine RGB (senza trasparenza).
RGB_CHANNELS: int = 3

# Valore massimo di un canale a 8 bit: alfa 255 = pixel completamente opaco.
MAX_CHANNEL_VALUE: int = 255


@dataclass(frozen=True)
class RGBAImage:
    """Un'immagine RGBA immutabile.

    Ogni fase della pipeline riceve un `RGBAImage` e ne restituisce uno nuovo:
    l'immutabilita' (`frozen=True`) evita che una fase modifichi per errore
    l'input di un'altra, e rende banale conservare i risultati intermedi.

    Attributi:
        data: array NumPy di forma (altezza, larghezza, 4) e tipo `uint8`.
              I primi tre canali sono il colore, il quarto e' l'opacita'.
    """

    data: np.ndarray

    # ------------------------------------------------------------------
    # Validazione
    # ------------------------------------------------------------------

    def __post_init__(self) -> None:
        """Verifica gli invarianti del tipo subito dopo la costruzione.

        Fallire qui, il prima possibile, e' molto piu' chiaro che vedere un
        errore incomprensibile di NumPy tre funzioni piu' avanti.
        """
        if self.data.ndim != 3:
            raise ValueError(
                f"Attese 3 dimensioni (altezza, larghezza, canali), "
                f"ricevute {self.data.ndim}."
            )
        if self.data.shape[2] != RGBA_CHANNELS:
            raise ValueError(
                f"Attesi {RGBA_CHANNELS} canali (RGBA), "
                f"ricevuti {self.data.shape[2]}."
            )
        if self.data.dtype != np.uint8:
            raise ValueError(
                f"Atteso tipo di dato uint8 (0-255), ricevuto {self.data.dtype}."
            )

    # ------------------------------------------------------------------
    # Costruttori alternativi
    # ------------------------------------------------------------------

    @classmethod
    def from_rgb(cls, rgb: np.ndarray) -> RGBAImage:
        """Costruisce un'immagine RGBA da un array RGB, rendendola opaca.

        Usato quando si carica un JPEG, che per definizione non ha trasparenza.

        Args:
            rgb: array (altezza, larghezza, 3) di tipo uint8.

        Returns:
            La stessa immagine con un canale alfa aggiunto, tutto a 255.
        """
        if rgb.ndim != 3 or rgb.shape[2] != RGB_CHANNELS:
            raise ValueError(
                f"Atteso un array RGB (altezza, larghezza, 3), "
                f"ricevuta forma {rgb.shape}."
            )

        height, width = rgb.shape[:2]
        # Canale alfa completamente opaco: nessun pixel e' trasparente.
        opaque_alpha = np.full((height, width), MAX_CHANNEL_VALUE, dtype=np.uint8)

        # `dstack` impila i canali lungo l'ultima dimensione: (H, W, 3) + (H, W) -> (H, W, 4).
        return cls(np.dstack([rgb.astype(np.uint8), opaque_alpha]))

    # ------------------------------------------------------------------
    # Accesso in sola lettura ai componenti
    # ------------------------------------------------------------------

    @property
    def rgb(self) -> np.ndarray:
        """I soli canali di colore, come copia (altezza, larghezza, 3) uint8."""
        # La copia protegge l'immutabilita': chi la riceve puo' modificarla
        # liberamente senza corrompere questa istanza.
        return self.data[:, :, :RGB_CHANNELS].copy()

    @property
    def alpha(self) -> np.ndarray:
        """Il solo canale di opacita', come copia (altezza, larghezza) uint8."""
        return self.data[:, :, 3].copy()

    @property
    def height(self) -> int:
        """Altezza dell'immagine in pixel."""
        return int(self.data.shape[0])

    @property
    def width(self) -> int:
        """Larghezza dell'immagine in pixel."""
        return int(self.data.shape[1])

    # ------------------------------------------------------------------
    # Trasformazioni (restituiscono sempre una nuova istanza)
    # ------------------------------------------------------------------

    def with_rgb(self, rgb: np.ndarray) -> RGBAImage:
        """Restituisce una copia con nuovi canali di colore e lo stesso alfa.

        E' l'operazione tipica delle fasi che cambiano l'aspetto (bianco e nero,
        effetto penna) senza toccare la sagoma ritagliata.
        """
        if rgb.shape[:2] != (self.height, self.width):
            raise ValueError(
                f"Le dimensioni del nuovo RGB {rgb.shape[:2]} non corrispondono "
                f"a quelle dell'immagine ({self.height}, {self.width})."
            )
        return RGBAImage(np.dstack([rgb.astype(np.uint8), self.alpha]))

    def with_alpha(self, alpha: np.ndarray) -> RGBAImage:
        """Restituisce una copia con un nuovo canale alfa e lo stesso colore.

        E' l'operazione tipica della fase di rimozione dello sfondo, che decide
        quali pixel appartengono al soggetto e quali no.
        """
        if alpha.shape != (self.height, self.width):
            raise ValueError(
                f"Le dimensioni del nuovo alfa {alpha.shape} non corrispondono "
                f"a quelle dell'immagine ({self.height}, {self.width})."
            )
        return RGBAImage(np.dstack([self.rgb, alpha.astype(np.uint8)]))

    def composite_over(self, background: tuple[int, int, int]) -> np.ndarray:
        """Fonde l'immagine su uno sfondo a tinta unita, eliminando la trasparenza.

        Serve agli algoritmi che ragionano su immagini opache: se dessimo loro
        i pixel trasparenti cosi' come sono, il colore "sotto" la trasparenza
        (spesso nero) creerebbe bordi e contorni falsi.

        La formula e' il classico "alpha compositing" su sfondo opaco:
            risultato = primo_piano * alfa + sfondo * (1 - alfa)

        Args:
            background: colore di sfondo come tripla (R, G, B) in 0-255.

        Returns:
            Array RGB (altezza, larghezza, 3) uint8, senza trasparenza.
        """
        # Portiamo l'alfa in [0.0, 1.0] e gli diamo una terza dimensione, cosi'
        # NumPy lo propaga automaticamente sui tre canali di colore.
        alpha_ratio = (self.alpha.astype(np.float32) / MAX_CHANNEL_VALUE)[:, :, np.newaxis]

        foreground = self.rgb.astype(np.float32)
        background_plane = np.array(background, dtype=np.float32)

        blended = foreground * alpha_ratio + background_plane * (1.0 - alpha_ratio)

        # `clip` protegge da errori di arrotondamento in virgola mobile.
        return np.clip(blended, 0, MAX_CHANNEL_VALUE).astype(np.uint8)
