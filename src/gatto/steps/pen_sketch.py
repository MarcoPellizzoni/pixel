"""Fase 3: effetto disegno a penna.

Responsabilita' unica: trasformare un'immagine in scala di grigi nel suo
equivalente "disegnato a penna", cioe' tratti di inchiostro nero su carta bianca.

L'algoritmo di base e' XDoG (eXtended Difference of Gaussians, Winnemoeller 2012),
lo standard di riferimento per la stilizzazione a inchiostro. L'idea:

1. si sfoca l'immagine due volte, con raggi diversi;
2. la differenza tra le due sfocature e' quasi nulla nelle zone uniformi e
   grande dove c'e' un bordo, perche' una sfocatura piu' ampia "spalma" il
   contrasto piu' dell'altra;
3. quella differenza, amplificata e passata attraverso una soglia morbida,
   diventa il tratto di penna.

A XDoG si aggiunge un tratteggio (hatching) delle ombre: una penna non produce
grigi, riempie le zone scure con linee piu' o meno fitte. Nelle ombre piu'
profonde le linee si incrociano (cross-hatching), esattamente come in un
disegno a china.
"""

from __future__ import annotations

import cv2
import numpy as np

from gatto.config import PenSketchConfig
from gatto.domain import RGBAImage

# Fattore di conversione tra il dominio 8 bit (0-255) e quello normalizzato
# (0.0-1.0) su cui lavorano le formule.
BYTE_RANGE: float = 255.0

# Percentile usato come "bordo piu' marcato dell'immagine" per rendere le
# soglie indipendenti dal contrasto dello scatto. Non si usa il massimo
# assoluto perche' sarebbe deciso da un singolo pixel di rumore.
STRENGTH_PERCENTILE: float = 99.0


class PenSketchRenderer:
    """Rende un'immagine come se fosse disegnata a penna su carta bianca."""

    def __init__(self, config: PenSketchConfig) -> None:
        """Prepara la fase.

        Args:
            config: parametri del tratto e del tratteggio.
        """
        self._config = config

    @property
    def name(self) -> str:
        """Nome della fase."""
        return "disegno-a-penna"

    def apply(self, image: RGBAImage) -> RGBAImage:
        """Trasforma l'immagine in un disegno a penna.

        Args:
            image: immagine in ingresso, tipicamente gia' in bianco e nero.

        Returns:
            Il disegno, con il canale alfa invariato.
        """
        # Si lavora su un solo canale: se l'immagine e' gia' grigia i tre canali
        # sono identici, altrimenti questa e' una conversione di ripiego.
        luminance = cv2.cvtColor(image.rgb, cv2.COLOR_RGB2GRAY)

        # 1. Attenuazione del rumore, preservando i bordi.
        smoothed = self._reduce_noise(luminance)

        # 2. Estrazione del tratto vero e proprio.
        ink = self._extract_ink_strokes(smoothed)

        # 3. Rimozione dei puntini isolati lasciati dal passaggio precedente.
        cleaned_ink = self._remove_speckles(ink)

        # 4. Tratteggio delle zone in ombra.
        drawing = self._add_shadow_hatching(cleaned_ink, smoothed)

        # Il disegno e' monocromatico: lo replichiamo sui tre canali RGB.
        return image.with_rgb(np.dstack([drawing] * 3))

    # ------------------------------------------------------------------
    # 1. Attenuazione del rumore
    # ------------------------------------------------------------------

    def _reduce_noise(self, luminance: np.ndarray) -> np.ndarray:
        """Leviga le superfici uniformi senza smussare i bordi.

        Il filtro bilaterale media un pixel solo con i vicini che gli somigliano
        anche come colore. Risultato: il rumore di compressione JPEG e la grana
        del pelo spariscono, mentre il profilo delle orecchie o dei baffi resta
        netto. E' un passaggio essenziale: senza, XDoG disegnerebbe con la
        stessa enfasi i contorni veri e i difetti dell'immagine.

        Args:
            luminance: immagine in scala di grigi.

        Returns:
            L'immagine levigata.
        """
        return cv2.bilateralFilter(
            luminance,
            d=self._config.bilateral_diameter,
            sigmaColor=self._config.bilateral_sigma_color,
            sigmaSpace=self._config.bilateral_sigma_space,
        )

    # ------------------------------------------------------------------
    # 2. Estrazione del tratto (XDoG)
    # ------------------------------------------------------------------

    def _extract_ink_strokes(self, luminance: np.ndarray) -> np.ndarray:
        """Calcola i tratti di penna con l'algoritmo XDoG.

        Args:
            luminance: immagine in scala di grigi, gia' levigata.

        Returns:
            Immagine (altezza, larghezza) uint8: 255 = carta, 0 = inchiostro.
        """
        # Si lavora in virgola mobile normalizzata: le formule seguenti
        # sommano e sottraggono valori che escono dall'intervallo 0-255.
        normalized = luminance.astype(np.float32) / BYTE_RANGE

        narrow_blur = cv2.GaussianBlur(
            normalized, ksize=(0, 0), sigmaX=self._config.dog_sigma
        )
        wide_blur = cv2.GaussianBlur(
            normalized,
            ksize=(0, 0),
            sigmaX=self._config.dog_sigma * self._config.dog_sigma_ratio,
        )

        # Differenza di gaussiane: vale circa zero nelle zone uniformi (dove le
        # due sfocature coincidono) e si allontana da zero attorno ai bordi.
        difference_of_gaussians = narrow_blur - wide_blur

        # Ogni bordo produce una differenza negativa sul lato scuro e positiva
        # su quello chiaro. Teniamo solo il lato scuro: e' li' che una penna
        # appoggerebbe il tratto, e prendere entrambi i lati raddoppierebbe
        # ogni linea. Sottrarre il tono di base (che XDoG userebbe per le
        # campiture piene) e' cio' che distingue un disegno a linee da una
        # silhouette a macchie nere: le ombre le rendera' il tratteggio.
        stroke_response = np.maximum(-difference_of_gaussians, 0.0)

        # La forza dei bordi dipende dal contrasto della foto. La rapportiamo
        # ai bordi piu' marcati presenti nell'immagine, cosi' le soglie qui
        # sotto hanno lo stesso significato su qualsiasi scatto. Si usa un
        # percentile alto invece del massimo assoluto perche' un singolo pixel
        # di rumore non deve dettare la scala dell'intera immagine.
        reference_strength = float(np.percentile(stroke_response, STRENGTH_PERCENTILE))
        if reference_strength < 1e-6:
            # Immagine senza alcun bordo (tinta unita): il foglio resta bianco.
            return np.full(luminance.shape, 255, dtype=np.uint8)

        relative_strength = np.clip(stroke_response / reference_strength, 0.0, 1.0)

        # L'accentuazione comprime la scala verso l'alto: piu' e' alta, piu'
        # anche i bordi deboli (la trama del pelo) raggiungono la forza dei
        # bordi netti, e piu' il disegno risulta fitto e marcato.
        boosted_strength = np.tanh(self._config.sharpness * relative_strength)

        # Soglia morbida con tangente iperbolica: sotto la soglia resta carta
        # bianca, sopra si passa all'inchiostro tanto piu' bruscamente quanto
        # e' alta la durezza del tratto.
        ink_amount = 0.5 * (
            1.0
            + np.tanh(
                self._config.ink_softness
                * (boosted_strength - self._config.ink_threshold)
            )
        )

        # `ink_amount` vale 1 sull'inchiostro pieno: il foglio e' il suo negativo.
        paper = 1.0 - ink_amount

        return (np.clip(paper, 0.0, 1.0) * BYTE_RANGE).astype(np.uint8)

    # ------------------------------------------------------------------
    # 3. Pulizia
    # ------------------------------------------------------------------

    def _remove_speckles(self, ink: np.ndarray) -> np.ndarray:
        """Elimina i puntini neri isolati.

        Il filtro mediano sostituisce ogni pixel con il valore mediano dei
        vicini: un punto nero circondato da bianco sparisce, mentre una linea
        continua sopravvive perche' i suoi vicini sono anch'essi neri. E' la
        scelta giusta qui, dove una sfocatura ammorbidirebbe anche i tratti.

        Args:
            ink: il disegno grezzo.

        Returns:
            Il disegno ripulito.
        """
        size = self._config.despeckle_size
        if size <= 1:
            # Pulizia disattivata.
            return ink

        # OpenCV richiede una finestra di lato dispari.
        odd_size = size if size % 2 == 1 else size + 1

        return cv2.medianBlur(ink, odd_size)

    # ------------------------------------------------------------------
    # 4. Tratteggio delle ombre
    # ------------------------------------------------------------------

    def _add_shadow_hatching(
        self, ink: np.ndarray, luminance: np.ndarray
    ) -> np.ndarray:
        """Riempie le zone scure con un tratteggio a penna.

        Args:
            ink: il disegno a contorni prodotto da XDoG.
            luminance: l'immagine in scala di grigi, usata per sapere dove sono
                le ombre.

        Returns:
            Il disegno con il tratteggio sovrapposto.
        """
        if not self._config.enable_hatching:
            return ink

        # Quanto e' in ombra ogni pixel, da 0.0 (luce piena) a 1.0 (nero).
        shadow_amount = self._compute_shadow_amount(luminance)

        # Linee diagonali su tutta l'immagine, nella direzione principale.
        primary_lines = self._build_line_pattern(
            shape=ink.shape, angle_degrees=self._config.hatching_angle_degrees
        )

        # Secondo tratteggio, perpendicolare al primo: incrociandosi crea il
        # classico "cross-hatching" che rende le ombre piu' profonde.
        crossing_lines = self._build_line_pattern(
            shape=ink.shape, angle_degrees=self._config.hatching_angle_degrees + 90.0
        )

        # Il tratteggio semplice copre tutte le ombre; quello incrociato entra
        # in gioco solo nelle ombre piu' fitte, dove `shadow_amount` e' alto.
        crosshatch_zone = shadow_amount >= self._config.crosshatch_shadow_ratio
        hatch_pattern = primary_lines | (crossing_lines & crosshatch_zone)

        # Opacita' del tratteggio pixel per pixel: proporzionale alla profondita'
        # dell'ombra, cosi' le zone appena scure ricevono un accenno di linee e
        # quelle molto scure un tratto pieno.
        hatch_opacity = np.where(
            hatch_pattern, shadow_amount * self._config.hatching_strength, 0.0
        )

        # Sovrapposizione del tratteggio (nero) sul disegno esistente:
        #   risultato = disegno * (1 - opacita')
        # Il tratteggio puo' solo scurire, mai schiarire: cosi' i contorni gia'
        # tracciati da XDoG restano intatti.
        hatched = ink.astype(np.float32) * (1.0 - hatch_opacity)

        return np.clip(hatched, 0, BYTE_RANGE).astype(np.uint8)

    def _compute_shadow_amount(self, luminance: np.ndarray) -> np.ndarray:
        """Misura quanto ogni pixel e' in ombra, in scala 0.0-1.0.

        Args:
            luminance: immagine in scala di grigi.

        Returns:
            Array float32 (altezza, larghezza): 0.0 dove c'e' luce, 1.0 al nero.
        """
        threshold = float(self._config.hatching_shadow_threshold)
        if threshold <= 0:
            # Nessuna zona e' considerata in ombra.
            return np.zeros(luminance.shape, dtype=np.float32)

        # Le ombre si valutano sulla forma generale, non sui singoli peli:
        # una sfocatura ampia evita che il tratteggio si accenda e si spenga
        # pixel per pixel producendo un effetto sporco.
        shadow_field = cv2.GaussianBlur(
            luminance.astype(np.float32), ksize=(0, 0), sigmaX=4.0
        )

        # Mappatura lineare: alla soglia vale 0.0, al nero assoluto vale 1.0.
        amount = (threshold - shadow_field) / threshold

        return np.clip(amount, 0.0, 1.0).astype(np.float32)

    def _build_line_pattern(
        self, shape: tuple[int, ...], angle_degrees: float
    ) -> np.ndarray:
        """Genera un reticolo di linee parallele inclinate.

        Ogni pixel viene proiettato sulla direzione perpendicolare alle linee;
        il resto della divisione di quella proiezione per il passo dice a che
        punto del ciclo "linea/spazio vuoto" ci troviamo.

        Args:
            shape: dimensioni (altezza, larghezza) del reticolo da generare.
            angle_degrees: inclinazione delle linee in gradi.

        Returns:
            Maschera booleana (altezza, larghezza): True dove passa una linea.
        """
        height, width = shape[0], shape[1]

        # Coordinate di ogni pixel. `indexing="ij"` mantiene l'ordine
        # (riga, colonna) usato dalle immagini NumPy.
        rows, columns = np.meshgrid(
            np.arange(height, dtype=np.float32),
            np.arange(width, dtype=np.float32),
            indexing="ij",
        )

        angle_radians = np.deg2rad(angle_degrees)

        # Proiezione sulla normale alle linee: pixel con la stessa proiezione
        # stanno sulla stessa linea.
        projection = rows * np.cos(angle_radians) + columns * np.sin(angle_radians)

        spacing = float(max(self._config.hatching_spacing, 1))
        position_in_cycle = np.mod(projection, spacing)

        return position_in_cycle < float(self._config.hatching_line_width)
