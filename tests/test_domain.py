"""Test del modello di dominio `RGBAImage`."""

import numpy as np
import pytest

from gatto.domain import RGBAImage


def make_rgba(height: int = 4, width: int = 6, fill: int = 128) -> np.ndarray:
    """Crea un array RGBA valido, utile come punto di partenza dei test."""
    return np.full((height, width, 4), fill, dtype=np.uint8)


class TestValidation:
    """La costruzione deve rifiutare subito gli array malformati."""

    def test_accetta_un_array_rgba_valido(self) -> None:
        image = RGBAImage(make_rgba())
        assert image.height == 4
        assert image.width == 6

    def test_rifiuta_un_array_bidimensionale(self) -> None:
        with pytest.raises(ValueError, match="3 dimensioni"):
            RGBAImage(np.zeros((4, 6), dtype=np.uint8))

    def test_rifiuta_un_numero_di_canali_sbagliato(self) -> None:
        with pytest.raises(ValueError, match="canali"):
            RGBAImage(np.zeros((4, 6, 3), dtype=np.uint8))

    def test_rifiuta_un_tipo_di_dato_non_uint8(self) -> None:
        with pytest.raises(ValueError, match="uint8"):
            RGBAImage(np.zeros((4, 6, 4), dtype=np.float32))


class TestFromRgb:
    """Un'immagine senza trasparenza deve diventare completamente opaca."""

    def test_aggiunge_un_canale_alfa_opaco(self) -> None:
        rgb = np.full((3, 3, 3), 200, dtype=np.uint8)

        image = RGBAImage.from_rgb(rgb)

        assert np.all(image.alpha == 255)
        assert np.array_equal(image.rgb, rgb)

    def test_rifiuta_un_array_non_rgb(self) -> None:
        with pytest.raises(ValueError, match="RGB"):
            RGBAImage.from_rgb(np.zeros((3, 3, 4), dtype=np.uint8))


class TestImmutabilita:
    """Modificare cio' che si ottiene da un'immagine non deve alterarla."""

    def test_modificare_la_copia_rgb_non_tocca_l_originale(self) -> None:
        image = RGBAImage(make_rgba(fill=100))

        estratto = image.rgb
        estratto[:] = 0

        assert np.all(image.rgb == 100)

    def test_modificare_la_copia_alfa_non_tocca_l_originale(self) -> None:
        image = RGBAImage(make_rgba(fill=100))

        estratto = image.alpha
        estratto[:] = 0

        assert np.all(image.alpha == 100)


class TestTrasformazioni:
    """`with_rgb` e `with_alpha` cambiano una parte sola, lasciando l'altra."""

    def test_with_rgb_conserva_il_canale_alfa(self) -> None:
        image = RGBAImage(make_rgba(fill=50))
        nuovo_rgb = np.full((4, 6, 3), 210, dtype=np.uint8)

        risultato = image.with_rgb(nuovo_rgb)

        assert np.array_equal(risultato.rgb, nuovo_rgb)
        assert np.all(risultato.alpha == 50)

    def test_with_alpha_conserva_i_colori(self) -> None:
        image = RGBAImage(make_rgba(fill=50))
        nuovo_alfa = np.full((4, 6), 255, dtype=np.uint8)

        risultato = image.with_alpha(nuovo_alfa)

        assert np.all(risultato.rgb == 50)
        assert np.all(risultato.alpha == 255)

    def test_with_rgb_rifiuta_dimensioni_diverse(self) -> None:
        image = RGBAImage(make_rgba(height=4, width=6))

        with pytest.raises(ValueError, match="dimensioni"):
            image.with_rgb(np.zeros((9, 9, 3), dtype=np.uint8))

    def test_with_alpha_rifiuta_dimensioni_diverse(self) -> None:
        image = RGBAImage(make_rgba(height=4, width=6))

        with pytest.raises(ValueError, match="dimensioni"):
            image.with_alpha(np.zeros((9, 9), dtype=np.uint8))


class TestCompositeOver:
    """La fusione su sfondo opaco deve seguire la formula dell'alpha blending."""

    def test_un_pixel_opaco_conserva_il_proprio_colore(self) -> None:
        data = np.zeros((1, 1, 4), dtype=np.uint8)
        data[0, 0] = [10, 20, 30, 255]

        risultato = RGBAImage(data).composite_over((255, 255, 255))

        assert list(risultato[0, 0]) == [10, 20, 30]

    def test_un_pixel_trasparente_assume_il_colore_di_sfondo(self) -> None:
        data = np.zeros((1, 1, 4), dtype=np.uint8)
        data[0, 0] = [10, 20, 30, 0]

        risultato = RGBAImage(data).composite_over((255, 255, 255))

        assert list(risultato[0, 0]) == [255, 255, 255]

    def test_un_pixel_a_meta_mescola_i_due_colori(self) -> None:
        data = np.zeros((1, 1, 4), dtype=np.uint8)
        # Alfa 128/255 ~= 0.502: il risultato deve stare a meta' tra 0 e 255.
        data[0, 0] = [0, 0, 0, 128]

        risultato = RGBAImage(data).composite_over((255, 255, 255))

        assert risultato[0, 0, 0] == pytest.approx(127, abs=1)

    def test_il_risultato_non_ha_piu_il_canale_alfa(self) -> None:
        risultato = RGBAImage(make_rgba()).composite_over((0, 0, 0))

        assert risultato.shape == (4, 6, 3)
