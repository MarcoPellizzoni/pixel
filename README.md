# gatto

Trasforma la foto di un gatto in un disegno a penna in bianco e nero.

La pipeline esegue tre elaborazioni, in quest'ordine:

1. **Rimozione dello sfondo** — isola il gatto con una rete di segmentazione.
2. **Conversione in bianco e nero** — luminanza percettiva più equalizzazione locale del contrasto.
3. **Effetto disegno a penna** — contorni a inchiostro (XDoG) e tratteggio delle ombre.

## Requisiti

- Python 3.12 o superiore
- [uv](https://docs.astral.sh/uv/)

## Installazione

```bash
uv sync
```

Al primo avvio viene scaricato automaticamente il modello di segmentazione
(~180 MB), che resta poi in cache in `~/.u2net/`.

## Uso

```bash
# Elabora g1.jpeg e scrive i risultati in output/
uv run gatto

# Su un'altra immagine, in un'altra cartella
uv run gatto foto.jpg --output risultati/

# Tratto più rado e più marcato, senza tratteggio
uv run gatto --ink-threshold 0.7 --sharpness 3.0 --no-hatching

# Elenco completo delle opzioni
uv run gatto --help
```

In `output/` finiscono il disegno finale e il risultato di ogni singola fase
(`01_rimozione-sfondo.png`, `02_bianco-e-nero.png`, `03_disegno-a-penna.png`),
utili per capire quale passaggio ritoccare.

## Uso come libreria

```python
from pathlib import Path

from gatto import CatSketchPipeline, PipelineConfig, load_image, save_image

immagine = load_image(Path("g1.jpeg"))
risultato = CatSketchPipeline(PipelineConfig()).run(immagine)
save_image(risultato.final_image, Path("disegno.png"))
```

Ogni parametro è regolabile tramite `PipelineConfig` e le sue sotto-configurazioni:

```python
from gatto import GrayscaleConfig, PenSketchConfig, PipelineConfig

config = PipelineConfig(
    grayscale=GrayscaleConfig(clahe_clip_limit=2.0),
    pen_sketch=PenSketchConfig(dog_sigma=0.9, ink_threshold=0.65),
)
```

## Organizzazione del codice

Layout `src/`, con una responsabilità per modulo:

```
src/gatto/
├── domain.py          Il tipo `RGBAImage` che attraversa tutta la pipeline
├── config.py          Tutti i parametri regolabili, con i valori predefiniti
├── image_io.py        L'unico modulo che tocca il filesystem
├── pipeline.py        Ordine delle fasi e salvataggio dei risultati
├── cli.py             Interfaccia a riga di comando
└── steps/
    ├── base.py          Il protocollo `ProcessingStep` comune alle fasi
    ├── background.py    Fase 1 — rimozione dello sfondo
    ├── mask_cleanup.py  Rifiniture geometriche della maschera di ritaglio
    ├── grayscale.py     Fase 2 — conversione in bianco e nero
    └── pen_sketch.py    Fase 3 — effetto disegno a penna
```

Le direzioni di dipendenza sono a senso unico: `cli` → `pipeline` → `steps` →
`domain`. Le fasi non conoscono né il filesystem né le altre fasi, quindi si
provano con array in memoria e si ricombinano liberamente.

## Pacchetti usati e perché

| Pacchetto | Ruolo |
| --- | --- |
| **rembg** (U²-Net / IS-Net) | Segmentazione del soggetto. Lo sfondo della foto ha colori e trame simili al pelo: nessuna tecnica basata su soglie di colore funzionerebbe. |
| **OpenCV** | Filtro bilaterale, gaussiane, CLAHE, filtro mediano, componenti connesse. |
| **Pillow** | Lettura/scrittura dei file e gestione dell'orientamento EXIF. |
| **NumPy** | Rappresentazione e calcolo vettoriale sulle matrici di pixel. |
| **Typer** | Interfaccia a riga di comando. |

## Sviluppo

```bash
uv run pytest       # 64 test
uv run ruff check . # lint
```
