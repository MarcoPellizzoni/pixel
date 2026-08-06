"""Rifinitura della maschera di ritaglio.

Responsabilita' unica: correggere i difetti tipici di una maschera prodotta da
una rete di segmentazione. Sono operazioni puramente geometriche su un array di
opacita': non sanno da quale rete arrivi la maschera ne' a quale immagine
appartenga, e per questo si provano e si riusano da sole.
"""

from __future__ import annotations

import cv2
import numpy as np


def keep_largest_region(mask: np.ndarray, alpha_threshold: int) -> np.ndarray:
    """Scarta dalla maschera tutto cio' che non tocca il soggetto principale.

    Il gatto e' un corpo unico: qualunque macchia isolata che la rete ha marcato
    come primo piano (un lembo di cuscino, un oggetto in penombra) e' un errore.
    Etichettando le regioni connesse e tenendo solo la piu' estesa, quegli
    scarti spariscono.

    Args:
        mask: maschera di opacita' (altezza, larghezza) uint8.
        alpha_threshold: opacita' minima perche' un pixel conti come soggetto
            nel calcolo delle regioni. Va tenuta bassa: baffi e ciuffi di pelo
            sono poco opachi e con una soglia alta risulterebbero staccati dal
            corpo, finendo cancellati insieme allo scarto.

    Returns:
        La maschera con le sole regioni collegate al soggetto principale. I
        valori sfumati dei pixel superstiti restano invariati.
    """
    # Versione binaria della maschera, usata solo per decidere cosa e'
    # collegato a cosa; i valori originali vengono conservati piu' sotto.
    is_subject = (mask > alpha_threshold).astype(np.uint8)

    # `connectivity=8` considera vicini anche i pixel in diagonale: un tratto
    # sottile e obliquo come un baffo resta cosi' un'unica regione.
    region_count, labels, stats, _ = cv2.connectedComponentsWithStats(
        is_subject, connectivity=8
    )

    # L'etichetta 0 e' sempre lo sfondo: se non c'e' nient'altro, la rete non ha
    # trovato alcun soggetto e non c'e' niente da filtrare.
    if region_count <= 1:
        return mask

    # `stats` riporta per ogni etichetta area, posizione e dimensioni: cerchiamo
    # l'area maggiore tra le etichette diverse dallo sfondo.
    areas = stats[1:, cv2.CC_STAT_AREA]
    largest_label = int(np.argmax(areas)) + 1

    # Azzeriamo l'opacita' ovunque tranne che nella regione vincente.
    return np.where(labels == largest_label, mask, 0).astype(np.uint8)


def feather_mask_edges(mask: np.ndarray, radius: int) -> np.ndarray:
    """Ammorbidisce il bordo della maschera.

    La rete lavora a risoluzione ridotta e poi riscala il risultato: il bordo
    puo' presentare una leggera scalettatura. Una sfocatura gaussiana minima la
    trasforma in una transizione graduale.

    Args:
        mask: maschera di opacita' (altezza, larghezza) uint8.
        radius: raggio della sfocatura in pixel; 0 o negativo la disattiva.

    Returns:
        La maschera con i bordi rifiniti.
    """
    if radius <= 0:
        # Rifinitura disattivata: restituiamo la maschera com'e'.
        return mask

    # OpenCV richiede una dimensione del kernel dispari e positiva.
    kernel_size = radius * 2 + 1

    # `sigmaX=0` lascia che OpenCV deduca la deviazione standard dal kernel.
    return cv2.GaussianBlur(mask, (kernel_size, kernel_size), sigmaX=0)
