"""Test di lettura e scrittura dei file immagine."""

from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from gatto.domain import RGBAImage
from gatto.image_io import load_image, save_image


@pytest.fixture
def immagine_di_prova() -> RGBAImage:
    """Un'immagine 4x4 con un pixel opaco, uno trasparente e il resto a meta'."""
    data = np.full((4, 4, 4), 128, dtype=np.uint8)
    data[0, 0] = [255, 0, 0, 255]
    data[1, 1] = [0, 255, 0, 0]
    return RGBAImage(data)


class TestCaricamento:
    """La lettura deve normalizzare qualunque formato in RGBA."""

    def test_un_jpeg_viene_caricato_completamente_opaco(self, tmp_path: Path) -> None:
        percorso = tmp_path / "foto.jpg"
        Image.new("RGB", (5, 3), (10, 20, 30)).save(percorso)

        immagine = load_image(percorso)

        assert np.all(immagine.alpha == 255)
        assert (immagine.height, immagine.width) == (3, 5)

    def test_un_png_in_scala_di_grigi_diventa_rgba(self, tmp_path: Path) -> None:
        percorso = tmp_path / "grigio.png"
        Image.new("L", (5, 3), 90).save(percorso)

        immagine = load_image(percorso)

        assert immagine.data.shape == (3, 5, 4)

    def test_un_file_inesistente_solleva_un_errore_chiaro(self, tmp_path: Path) -> None:
        with pytest.raises(FileNotFoundError, match="non trovata"):
            load_image(tmp_path / "questo_file_non_esiste.png")


class TestSalvataggio:
    """La scrittura deve conservare la trasparenza quando il formato lo permette."""

    def test_il_png_conserva_il_canale_alfa(
        self, tmp_path: Path, immagine_di_prova: RGBAImage
    ) -> None:
        percorso = tmp_path / "risultato.png"

        save_image(immagine_di_prova, percorso)
        riletta = load_image(percorso)

        assert np.array_equal(riletta.data, immagine_di_prova.data)

    def test_il_jpeg_appiattisce_su_bianco_invece_di_perdere_l_alfa(
        self, tmp_path: Path
    ) -> None:
        # Meta' immagine opaca e nera, meta' completamente trasparente. Le zone
        # sono ampie perche' il JPEG e' un formato con perdita: su blocchi di
        # pochi pixel gli artefatti di compressione falserebbero il confronto.
        data = np.zeros((32, 32, 4), dtype=np.uint8)
        data[:, :16, 3] = 255
        percorso = tmp_path / "risultato.jpg"

        save_image(RGBAImage(data), percorso)
        riletta = load_image(percorso)

        # Il JPEG non ha trasparenza: tutto deve risultare opaco...
        assert np.all(riletta.alpha == 255)
        # ...la meta' opaca deve essere rimasta nera...
        assert riletta.rgb[:, :8].max() < 15
        # ...e quella trasparente deve essere diventata bianca.
        assert riletta.rgb[:, 24:].min() > 240

    def test_le_cartelle_mancanti_vengono_create(
        self, tmp_path: Path, immagine_di_prova: RGBAImage
    ) -> None:
        percorso = tmp_path / "una" / "due" / "tre" / "risultato.png"

        save_image(immagine_di_prova, percorso)

        assert percorso.is_file()


class TestAndataERitorno:
    """Salvare e ricaricare non deve alterare l'immagine."""

    def test_il_png_e_senza_perdita(self, tmp_path: Path) -> None:
        casuale = np.random.default_rng(seed=0).integers(
            0, 256, size=(9, 7, 4), dtype=np.uint8
        )
        originale = RGBAImage(casuale)
        percorso = tmp_path / "andata_e_ritorno.png"

        save_image(originale, percorso)
        riletta = load_image(percorso)

        assert np.array_equal(riletta.data, originale.data)
