"""Ingresso/uscita: l'unico modulo che tocca il filesystem.

Responsabilita' unica: tradurre tra file su disco e `RGBAImage`. Isolare qui la
lettura e la scrittura significa che gli algoritmi del package `steps` sono
testabili con array in memoria, senza mai creare un file temporaneo.

Si appoggia a Pillow, che gestisce correttamente i formati, i profili colore e
soprattutto l'orientamento EXIF delle foto scattate da smartphone.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image, ImageOps

from gatto.domain import RGBAImage

# Formati che sanno conservare il canale alfa. Salvare un ritaglio in JPEG
# perderebbe silenziosamente la trasparenza, quindi lo segnaliamo.
FORMATS_WITH_TRANSPARENCY: frozenset[str] = frozenset({".png", ".webp", ".tif", ".tiff"})

# Colore usato per appiattire la trasparenza quando il formato non la supporta:
# bianco, coerente con l'idea di "disegno su foglio".
DEFAULT_FLATTEN_COLOR: tuple[int, int, int] = (255, 255, 255)


def load_image(path: Path) -> RGBAImage:
    """Carica un'immagine dal disco e la normalizza in RGBA.

    Args:
        path: percorso del file da leggere.

    Returns:
        L'immagine come `RGBAImage`.

    Raises:
        FileNotFoundError: se il percorso non esiste.
    """
    if not path.is_file():
        raise FileNotFoundError(f"Immagine non trovata: {path}")

    with Image.open(path) as opened_image:
        # Le foto da telefono memorizzano la rotazione nei metadati EXIF invece
        # di ruotare davvero i pixel: senza questa chiamata il gatto potrebbe
        # risultare coricato. `exif_transpose` applica la rotazione ai pixel.
        upright_image = ImageOps.exif_transpose(opened_image)

        # Convertiamo subito in RGBA: cosi' il resto del programma tratta
        # allo stesso modo JPEG (opachi), PNG con trasparenza e immagini in
        # scala di grigi o con palette.
        rgba_image = upright_image.convert("RGBA")

        return RGBAImage(np.array(rgba_image, dtype=np.uint8))


def save_image(image: RGBAImage, path: Path) -> None:
    """Salva un'immagine sul disco, creando le cartelle mancanti.

    Se il formato di destinazione non supporta la trasparenza, l'immagine viene
    fusa su sfondo bianco invece di perdere il canale alfa senza avvisare.

    Args:
        image: l'immagine da salvare.
        path: percorso di destinazione; l'estensione determina il formato.
    """
    # `parents=True` crea l'intero albero di cartelle, `exist_ok=True` non
    # protesta se esiste gia'.
    path.parent.mkdir(parents=True, exist_ok=True)

    if path.suffix.lower() in FORMATS_WITH_TRANSPARENCY:
        # Il formato regge l'alfa: salviamo i quattro canali come sono.
        Image.fromarray(image.data, mode="RGBA").save(path)
    else:
        # Formato senza alfa (es. JPEG): appiattiamo esplicitamente su bianco.
        flattened_rgb = image.composite_over(DEFAULT_FLATTEN_COLOR)
        Image.fromarray(flattened_rgb, mode="RGB").save(path, quality=95)
