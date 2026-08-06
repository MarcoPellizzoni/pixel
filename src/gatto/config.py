"""Configurazione: tutti i parametri regolabili della pipeline, in un posto solo.

Responsabilita' unica: dare un nome e un valore predefinito a ogni "manopola"
degli algoritmi. Nessun modulo di elaborazione contiene numeri magici scritti
a mano: li riceve da qui.

Le dataclass sono `frozen` perche' una configurazione, una volta decisa, non
deve cambiare mentre la pipeline gira.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum


class SegmentationModel(StrEnum):
    """Reti neurali disponibili per separare il soggetto dallo sfondo.

    Sono i modelli pre-addestrati distribuiti da `rembg`; vengono scaricati
    automaticamente al primo utilizzo e poi tenuti in cache locale.
    """

    # Il modello storico, robusto e ben collaudato su soggetti generici.
    U2NET = "u2net"

    # Piu' recente: bordi piu' netti e miglior resa su pelo e dettagli sottili.
    ISNET_GENERAL = "isnet-general-use"

    # Specializzato sulle persone: inutile per un gatto, ma incluso per completezza.
    U2NET_HUMAN = "u2net_human_seg"


class LuminanceStandard(StrEnum):
    """Formule standard per calcolare la luminosita' percepita di un colore.

    L'occhio umano non e' ugualmente sensibile ai tre colori primari: il verde
    pesa molto piu' del blu. Una media aritmetica (R+G+B)/3 darebbe un grigio
    innaturale, quindi si usa una media pesata definita da uno standard.
    """

    # ITU-R BT.601: lo standard della televisione analogica, ancora usato da
    # OpenCV. Rende i toni caldi (il pelo rosso del gatto) piu' chiari.
    BT601 = "bt601"

    # ITU-R BT.709: lo standard HD/sRGB, il piu' corretto per foto digitali.
    BT709 = "bt709"


# Coefficienti (rosso, verde, blu) di ciascuno standard di luminanza.
# La somma di ogni tripla e' 1.0, quindi il grigio risultante resta in 0-255.
LUMINANCE_WEIGHTS: dict[LuminanceStandard, tuple[float, float, float]] = {
    LuminanceStandard.BT601: (0.299, 0.587, 0.114),
    LuminanceStandard.BT709: (0.2126, 0.7152, 0.0722),
}


@dataclass(frozen=True)
class BackgroundRemovalConfig:
    """Parametri della fase 1: isolamento del gatto dallo sfondo."""

    # Quale rete neurale usare per la segmentazione.
    model: SegmentationModel = SegmentationModel.ISNET_GENERAL

    # L'"alpha matting" e' un raffinamento del bordo eseguito dopo la
    # segmentazione: recupera dettagli semitrasparenti come baffi e ciuffi di
    # pelo, che una maschera binaria taglierebbe di netto. Costa tempo di
    # calcolo, ma su un gatto la differenza si vede.
    use_alpha_matting: bool = True

    # Soglia sopra la quale un pixel della maschera e' considerato "sicuramente
    # soggetto" durante l'alpha matting (0-255).
    alpha_matting_foreground_threshold: int = 240

    # Soglia sotto la quale un pixel e' considerato "sicuramente sfondo" (0-255).
    # Tutto cio' che sta tra le due soglie e' la zona incerta da ricostruire.
    alpha_matting_background_threshold: int = 15

    # Ampiezza in pixel della zona incerta attorno al bordo: piu' e' larga,
    # piu' pelo sfumato viene recuperato (e piu' lento diventa il calcolo).
    alpha_matting_erode_size: int = 12

    # Se attivo, della maschera viene tenuta solo la regione connessa piu'
    # estesa. La rete a volte promuove a "soggetto" anche qualche brandello di
    # sfondo staccato (un lembo di cuscino, un oggetto in penombra): il gatto e'
    # un corpo unico, quindi tutto cio' che non lo tocca e' certamente scarto.
    keep_largest_component: bool = True

    # Opacita' minima (0-255) perche' un pixel conti come parte del soggetto nel
    # calcolo delle regioni connesse. Va tenuta bassa: baffi e ciuffi di pelo
    # hanno un'opacita' debole e con una soglia alta risulterebbero staccati dal
    # corpo, finendo cancellati insieme allo scarto.
    connectivity_alpha_threshold: int = 8

    # Raggio della sfocatura applicata al solo canale alfa, per ammorbidire i
    # gradini della maschera. 0 disattiva la rifinitura.
    mask_feather_radius: int = 1

    # Colore con cui vengono riempiti i pixel diventati trasparenti. Il canale
    # alfa da solo non basta: sotto la trasparenza resterebbero i colori del
    # vecchio sfondo, e le fasi successive li vedrebbero comunque, disegnando
    # contorni di cuscini e termosifone invisibili all'occhio ma ben presenti
    # nel calcolo. Riempiendo di bianco, il profilo del gatto diventa inoltre
    # un bordo netto, che l'effetto penna trasforma in un contorno pulito.
    replaced_background_color: tuple[int, int, int] = (255, 255, 255)


@dataclass(frozen=True)
class GrayscaleConfig:
    """Parametri della fase 2: conversione in bianco e nero."""

    # Formula di luminanza da applicare.
    standard: LuminanceStandard = LuminanceStandard.BT709

    # Forza dell'equalizzazione locale del contrasto (CLAHE). La foto originale
    # e' scattata in penombra: senza questo passaggio molti dettagli del pelo
    # resterebbero schiacciati sui toni scuri e il disegno a penna li perderebbe.
    # 0.0 disattiva l'equalizzazione.
    clahe_clip_limit: float = 1.0

    # Lato in pixel delle tessere su cui CLAHE calcola l'istogramma locale.
    clahe_tile_size: int = 8


@dataclass(frozen=True)
class PenSketchConfig:
    """Parametri della fase 3: effetto disegno a penna.

    L'algoritmo di base e' XDoG (eXtended Difference of Gaussians), la tecnica
    classica per la stilizzazione a inchiostro: confronta due versioni sfocate
    dell'immagine per isolare i contorni e li rende come tratti di penna.
    """

    # --- Attenuazione del rumore, prima di cercare i contorni ---

    # Diametro del filtro bilaterale, che leva il rumore JPEG mantenendo i bordi
    # nitidi. Senza, l'effetto penna disegnerebbe anche i difetti di compressione.
    bilateral_diameter: int = 9

    # Quanto due colori possono differire ed essere comunque mediati insieme:
    # valori alti appiattiscono le sfumature del pelo in campiture uniformi.
    bilateral_sigma_color: float = 60.0

    # Quanto lontano si estende il filtro nello spazio dell'immagine.
    bilateral_sigma_space: float = 60.0

    # --- Nucleo XDoG ---

    # Deviazione standard della sfocatura gaussiana "stretta": governa lo
    # spessore del tratto. Valori piccoli = linee sottili e nervose.
    dog_sigma: float = 1.3

    # Rapporto tra la sfocatura "larga" e quella "stretta". 1.6 e' il valore
    # che meglio approssima il Laplaciano di gaussiana, standard in letteratura.
    dog_sigma_ratio: float = 1.6

    # Accentuazione dei contorni. Agisce sulla forza dei bordi gia' rapportata
    # al bordo piu' marcato dell'immagine: alzandolo, anche i bordi deboli
    # (la trama del pelo) raggiungono la forza di quelli netti e il disegno si
    # infittisce. Valori utili tra 1 e 4; oltre, compare "sporco" nelle zone piatte.
    sharpness: float = 2.0

    # Soglia che separa la carta bianca dall'inchiostro, sulla scala 0.0-1.0
    # della forza dei bordi. Alzarla lascia piu' bianco, abbassarla annerisce
    # il disegno.
    ink_threshold: float = 0.55

    # Pendenza della transizione bianco/nero attorno alla soglia: valori alti
    # danno un tratto deciso e quasi binario, come una penna a sfera; valori
    # bassi danno sfumature morbide, come una matita.
    ink_softness: float = 10.0

    # --- Rifinitura del tratto ---

    # Lato della finestra del filtro mediano che elimina i puntini isolati
    # (deve essere dispari). 0 disattiva la pulizia.
    despeckle_size: int = 3

    # --- Tratteggio delle ombre ---

    # Se attivo, riempie le zone in ombra con un tratteggio diagonale, come
    # farebbe una penna che non puo' produrre grigi ma solo linee piu' o meno fitte.
    enable_hatching: bool = True

    # Distanza in pixel tra due linee del tratteggio.
    hatching_spacing: int = 6

    # Spessore in pixel di ciascuna linea. Deve restare minore del passo,
    # altrimenti le linee si toccano e l'ombra diventa una campitura piena.
    hatching_line_width: int = 1

    # Inclinazione delle linee in gradi.
    hatching_angle_degrees: float = 45.0

    # Livello di grigio (0-255) sotto il quale una zona e' considerata in ombra
    # e quindi da tratteggiare.
    hatching_shadow_threshold: int = 95

    # Profondita' d'ombra (0.0-1.0) oltre la quale al tratteggio se ne
    # sovrappone un secondo perpendicolare: e' il "cross-hatching" con cui si
    # rendono i neri profondi in un disegno a china.
    crosshatch_shadow_ratio: float = 0.55

    # Opacita' del tratteggio, da 0.0 (invisibile) a 1.0 (nero pieno).
    hatching_strength: float = 0.55


@dataclass(frozen=True)
class PipelineConfig:
    """Configurazione completa: raggruppa i parametri di tutte le fasi.

    E' l'unico oggetto che la CLI deve costruire e passare alla pipeline.
    """

    background_removal: BackgroundRemovalConfig = field(
        default_factory=BackgroundRemovalConfig
    )
    grayscale: GrayscaleConfig = field(default_factory=GrayscaleConfig)
    pen_sketch: PenSketchConfig = field(default_factory=PenSketchConfig)

    # Colore del "foglio" su cui viene appoggiato il disegno finale quando si
    # salva in un formato senza trasparenza.
    paper_color: tuple[int, int, int] = (255, 255, 255)

    # Se attivo, salva anche il risultato di ogni singola fase, utile per
    # capire quale passaggio va ritoccato.
    save_intermediate_steps: bool = True
