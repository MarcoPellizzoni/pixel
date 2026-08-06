"""Test della fase di conversione in bianco e nero."""

import numpy as np
import pytest

from gatto.config import GrayscaleConfig, LuminanceStandard
from gatto.domain import RGBAImage
from gatto.steps.grayscale import GrayscaleConverter


def immagine_a_tinta_unita(colore: tuple[int, int, int], alfa: int = 255) -> RGBAImage:
    """Crea un'immagine 8x8 di un solo colore, con l'opacita' indicata."""
    data = np.zeros((8, 8, 4), dtype=np.uint8)
    data[:, :, :3] = colore
    data[:, :, 3] = alfa
    return RGBAImage(data)


# CLAHE va disattivato quando si verifica la formula di luminanza, altrimenti
# altererebbe i valori attesi.
SENZA_EQUALIZZAZIONE = GrayscaleConfig(clahe_clip_limit=0.0)


class TestLuminanza:
    """La conversione deve applicare i pesi dello standard scelto."""

    def test_i_tre_canali_diventano_uguali(self) -> None:
        immagine = immagine_a_tinta_unita((200, 100, 50))

        risultato = GrayscaleConverter(SENZA_EQUALIZZAZIONE).apply(immagine)

        rgb = risultato.rgb
        assert np.array_equal(rgb[:, :, 0], rgb[:, :, 1])
        assert np.array_equal(rgb[:, :, 1], rgb[:, :, 2])

    def test_bt709_usa_i_propri_coefficienti(self) -> None:
        immagine = immagine_a_tinta_unita((200, 100, 50))
        config = GrayscaleConfig(
            standard=LuminanceStandard.BT709, clahe_clip_limit=0.0
        )

        risultato = GrayscaleConverter(config).apply(immagine)

        atteso = 200 * 0.2126 + 100 * 0.7152 + 50 * 0.0722
        assert risultato.rgb[0, 0, 0] == pytest.approx(atteso, abs=1)

    def test_bt601_usa_i_propri_coefficienti(self) -> None:
        immagine = immagine_a_tinta_unita((200, 100, 50))
        config = GrayscaleConfig(
            standard=LuminanceStandard.BT601, clahe_clip_limit=0.0
        )

        risultato = GrayscaleConverter(config).apply(immagine)

        atteso = 200 * 0.299 + 100 * 0.587 + 50 * 0.114
        assert risultato.rgb[0, 0, 0] == pytest.approx(atteso, abs=1)

    def test_i_due_standard_danno_risultati_diversi_sui_toni_caldi(self) -> None:
        immagine = immagine_a_tinta_unita((230, 140, 60))

        bt601 = GrayscaleConverter(
            GrayscaleConfig(standard=LuminanceStandard.BT601, clahe_clip_limit=0.0)
        ).apply(immagine)
        bt709 = GrayscaleConverter(
            GrayscaleConfig(standard=LuminanceStandard.BT709, clahe_clip_limit=0.0)
        ).apply(immagine)

        # BT.601 pesa di piu' il rosso, quindi su un arancione schiarisce.
        assert bt601.rgb[0, 0, 0] > bt709.rgb[0, 0, 0]

    def test_il_bianco_resta_bianco_e_il_nero_resta_nero(self) -> None:
        convertitore = GrayscaleConverter(SENZA_EQUALIZZAZIONE)

        bianco = convertitore.apply(immagine_a_tinta_unita((255, 255, 255)))
        nero = convertitore.apply(immagine_a_tinta_unita((0, 0, 0)))

        assert np.all(bianco.rgb == 255)
        assert np.all(nero.rgb == 0)


class TestCanaleAlfa:
    """La conversione non deve toccare la sagoma ritagliata."""

    def test_l_alfa_resta_invariato(self) -> None:
        immagine = immagine_a_tinta_unita((200, 100, 50), alfa=77)

        risultato = GrayscaleConverter(SENZA_EQUALIZZAZIONE).apply(immagine)

        assert np.all(risultato.alpha == 77)


class TestEqualizzazione:
    """CLAHE deve essere disattivabile e non deve alterare le dimensioni."""

    def test_clip_limit_a_zero_disattiva_l_equalizzazione(self) -> None:
        # Un gradiente: se CLAHE agisse, i valori cambierebbero.
        data = np.zeros((16, 16, 4), dtype=np.uint8)
        data[:, :, :3] = np.linspace(0, 255, 16, dtype=np.uint8)[None, :, None]
        data[:, :, 3] = 255
        immagine = RGBAImage(data)

        senza = GrayscaleConverter(SENZA_EQUALIZZAZIONE).apply(immagine)
        con = GrayscaleConverter(GrayscaleConfig(clahe_clip_limit=4.0)).apply(immagine)

        assert not np.array_equal(senza.rgb, con.rgb)

    def test_le_dimensioni_restano_invariate(self) -> None:
        immagine = immagine_a_tinta_unita((120, 120, 120))

        risultato = GrayscaleConverter(GrayscaleConfig()).apply(immagine)

        assert (risultato.height, risultato.width) == (immagine.height, immagine.width)
