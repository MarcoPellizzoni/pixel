"""Test della fase di resa a penna."""

import numpy as np

from gatto.config import PenSketchConfig
from gatto.domain import RGBAImage
from gatto.steps.pen_sketch import PenSketchRenderer

# Il tratteggio va disattivato quando si verifica il solo comportamento del
# tratto, altrimenti aggiungerebbe linee non prodotte da XDoG.
SENZA_TRATTEGGIO = PenSketchConfig(enable_hatching=False)


def immagine_grigia(livello: int, lato: int = 64) -> RGBAImage:
    """Crea un'immagine quadrata di un solo livello di grigio, opaca."""
    data = np.zeros((lato, lato, 4), dtype=np.uint8)
    data[:, :, :3] = livello
    data[:, :, 3] = 255
    return RGBAImage(data)


def immagine_con_bordo(lato: int = 64) -> RGBAImage:
    """Crea un'immagine divisa a meta': sinistra chiara, destra scura."""
    data = np.zeros((lato, lato, 4), dtype=np.uint8)
    data[:, : lato // 2, :3] = 230
    data[:, lato // 2 :, :3] = 25
    data[:, :, 3] = 255
    return RGBAImage(data)


class TestSuperficiUniformi:
    """Dove non c'e' nulla da disegnare, il foglio deve restare bianco."""

    def test_una_tinta_unita_non_produce_inchiostro(self) -> None:
        risultato = PenSketchRenderer(SENZA_TRATTEGGIO).apply(immagine_grigia(140))

        assert np.all(risultato.rgb == 255)

    def test_anche_una_tinta_unita_scura_resta_bianca(self) -> None:
        # Verifica che il tono di base non venga confuso con un contorno: e' la
        # differenza tra un disegno a linee e una silhouette annerita.
        risultato = PenSketchRenderer(SENZA_TRATTEGGIO).apply(immagine_grigia(20))

        assert np.all(risultato.rgb == 255)


class TestContorni:
    """Un bordo netto deve diventare un tratto di penna."""

    def test_un_bordo_produce_inchiostro(self) -> None:
        risultato = PenSketchRenderer(SENZA_TRATTEGGIO).apply(immagine_con_bordo())

        assert risultato.rgb.min() < 128

    def test_l_inchiostro_si_concentra_sul_bordo(self) -> None:
        lato = 64
        risultato = PenSketchRenderer(SENZA_TRATTEGGIO).apply(immagine_con_bordo(lato))

        grigio = risultato.rgb[:, :, 0]
        # Fascia stretta attorno alla linea di separazione.
        sul_bordo = grigio[:, lato // 2 - 3 : lato // 2 + 3]
        # Zona lontana dal bordo, dove il colore e' uniforme.
        lontano = grigio[:, : lato // 4]

        assert sul_bordo.min() < lontano.min()

    def test_una_soglia_piu_alta_lascia_piu_bianco(self) -> None:
        immagine = immagine_con_bordo()

        rado = PenSketchRenderer(
            PenSketchConfig(ink_threshold=0.85, enable_hatching=False)
        ).apply(immagine)
        fitto = PenSketchRenderer(
            PenSketchConfig(ink_threshold=0.25, enable_hatching=False)
        ).apply(immagine)

        assert rado.rgb.mean() > fitto.rgb.mean()


class TestTratteggio:
    """Il tratteggio deve scurire le ombre e solo quelle."""

    def test_il_tratteggio_scurisce_una_zona_in_ombra(self) -> None:
        # Grigio scuro uniforme: senza tratteggio resterebbe foglio bianco.
        immagine = immagine_grigia(20)

        senza = PenSketchRenderer(SENZA_TRATTEGGIO).apply(immagine)
        con = PenSketchRenderer(PenSketchConfig(enable_hatching=True)).apply(immagine)

        assert con.rgb.mean() < senza.rgb.mean()

    def test_il_tratteggio_non_tocca_le_zone_chiare(self) -> None:
        # Ben sopra la soglia d'ombra predefinita (95).
        immagine = immagine_grigia(240)

        risultato = PenSketchRenderer(PenSketchConfig(enable_hatching=True)).apply(
            immagine
        )

        assert np.all(risultato.rgb == 255)

    def test_il_tratteggio_lascia_spazi_bianchi_tra_le_linee(self) -> None:
        # Un tratteggio e' fatto di linee separate: se annerisse tutto sarebbe
        # una campitura piena, non un disegno a penna.
        risultato = PenSketchRenderer(PenSketchConfig(enable_hatching=True)).apply(
            immagine_grigia(10)
        )

        grigio = risultato.rgb[:, :, 0]
        assert grigio.max() == 255
        assert grigio.min() < 255


class TestInvarianti:
    """Proprieta' che la fase deve rispettare in ogni caso."""

    def test_l_alfa_resta_invariato(self) -> None:
        data = np.zeros((32, 32, 4), dtype=np.uint8)
        data[:, :, :3] = 120
        data[:, :, 3] = 42
        immagine = RGBAImage(data)

        risultato = PenSketchRenderer(PenSketchConfig()).apply(immagine)

        assert np.all(risultato.alpha == 42)

    def test_il_risultato_e_monocromatico(self) -> None:
        risultato = PenSketchRenderer(PenSketchConfig()).apply(immagine_con_bordo())

        rgb = risultato.rgb
        assert np.array_equal(rgb[:, :, 0], rgb[:, :, 1])
        assert np.array_equal(rgb[:, :, 1], rgb[:, :, 2])

    def test_le_dimensioni_restano_invariate(self) -> None:
        immagine = immagine_con_bordo(lato=48)

        risultato = PenSketchRenderer(PenSketchConfig()).apply(immagine)

        assert (risultato.height, risultato.width) == (48, 48)
