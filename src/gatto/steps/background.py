"""Fase 1: rimozione dello sfondo, per lasciare solo il gatto.

Responsabilita' unica: decidere quali pixel appartengono al soggetto e
scriverlo nel canale alfa dell'immagine.

Il lavoro pesante lo fa `rembg`, che esegue una rete neurale di segmentazione
(famiglia U^2-Net / IS-Net) addestrata a separare l'oggetto in primo piano dal
resto della scena. E' l'approccio corretto qui: lo sfondo della foto (termosifone,
coperte, cuscini) ha colori e trame simili al pelo del gatto, quindi nessuna
tecnica classica basata su soglie di colore funzionerebbe.

La rifinitura geometrica della maschera e' delegata a `mask_cleanup`: qui
restano solo la chiamata alla rete e la composizione del risultato.
"""

from __future__ import annotations

import numpy as np
from rembg import new_session, remove
from rembg.sessions.base import BaseSession

from gatto.config import BackgroundRemovalConfig
from gatto.domain import RGBAImage
from gatto.steps.mask_cleanup import feather_mask_edges, keep_largest_region


class BackgroundRemover:
    """Isola il soggetto dell'immagine sostituendone il canale alfa."""

    def __init__(self, config: BackgroundRemovalConfig) -> None:
        """Prepara la fase.

        Args:
            config: parametri di segmentazione e rifinitura del bordo.
        """
        self._config = config

        # La sessione (modello caricato in memoria) viene creata pigramente al
        # primo uso: costruire un `BackgroundRemover` non deve scaricare
        # centinaia di megabyte ne' occupare RAM finche' non serve davvero.
        self._session: BaseSession | None = None

    @property
    def name(self) -> str:
        """Nome della fase."""
        return "rimozione-sfondo"

    def apply(self, image: RGBAImage) -> RGBAImage:
        """Isola il soggetto: lo rende trasparente attorno e ne pulisce i colori.

        Args:
            image: l'immagine originale, tipicamente opaca.

        Returns:
            Il solo gatto, su sfondo trasparente e con i colori del vecchio
            sfondo sostituiti dal colore della carta.
        """
        subject_mask = self._predict_subject_mask(image)
        clean_mask = self._clean_up_mask(subject_mask)

        # Il ritaglio vero e proprio: lo sfondo diventa trasparente.
        cut_out = image.with_alpha(clean_mask)

        # I colori sotto la trasparenza vanno sostituiti, non solo nascosti:
        # vedi la nota su `replaced_background_color` in `config`.
        cleaned_rgb = cut_out.composite_over(self._config.replaced_background_color)

        return cut_out.with_rgb(cleaned_rgb)

    # ------------------------------------------------------------------
    # Passaggi interni
    # ------------------------------------------------------------------

    def _predict_subject_mask(self, image: RGBAImage) -> np.ndarray:
        """Chiede alla rete neurale quali pixel appartengono al soggetto.

        Args:
            image: l'immagine da segmentare.

        Returns:
            Maschera (altezza, larghezza) uint8: 255 = soggetto, 0 = sfondo,
            valori intermedi = bordo semitrasparente.
        """
        # `only_mask=True` fa restituire la sola maschera invece dell'immagine
        # gia' ritagliata: cosi' i colori originali restano intatti e siamo noi
        # a decidere come combinarli, mantenendo separate le responsabilita'.
        mask_image = remove(
            image.rgb,
            session=self._get_session(),
            only_mask=True,
            alpha_matting=self._config.use_alpha_matting,
            alpha_matting_foreground_threshold=self._config.alpha_matting_foreground_threshold,
            alpha_matting_background_threshold=self._config.alpha_matting_background_threshold,
            alpha_matting_erode_size=self._config.alpha_matting_erode_size,
        )

        mask = np.asarray(mask_image, dtype=np.uint8)

        # A seconda delle opzioni, `rembg` puo' restituire una maschera con una
        # dimensione di canale superflua: la appiattiamo a due dimensioni.
        if mask.ndim == 3:
            mask = mask[:, :, 0]

        return mask

    def _clean_up_mask(self, mask: np.ndarray) -> np.ndarray:
        """Applica alla maschera le rifiniture previste dalla configurazione.

        Args:
            mask: la maschera grezza prodotta dalla rete.

        Returns:
            La maschera pronta per essere usata come canale alfa.
        """
        if self._config.keep_largest_component:
            mask = keep_largest_region(
                mask, alpha_threshold=self._config.connectivity_alpha_threshold
            )

        return feather_mask_edges(mask, radius=self._config.mask_feather_radius)

    def _get_session(self) -> BaseSession:
        """Restituisce la sessione del modello, creandola al primo utilizzo.

        Tenerla in un attributo evita di ricaricare la rete a ogni immagine,
        operazione che dominerebbe il tempo di esecuzione su piu' file.
        """
        if self._session is None:
            self._session = new_session(self._config.model.value)
        return self._session
