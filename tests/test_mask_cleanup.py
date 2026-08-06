"""Test delle rifiniture applicate alla maschera di ritaglio."""

import numpy as np

from gatto.steps.mask_cleanup import feather_mask_edges, keep_largest_region


class TestKeepLargestRegion:
    """Deve sopravvivere solo la regione connessa piu' estesa."""

    def test_la_macchia_isolata_piu_piccola_viene_eliminata(self) -> None:
        mask = np.zeros((40, 40), dtype=np.uint8)
        mask[5:25, 5:25] = 255  # soggetto: 400 pixel
        mask[32:36, 32:36] = 255  # scarto isolato: 16 pixel

        risultato = keep_largest_region(mask, alpha_threshold=8)

        assert np.all(risultato[5:25, 5:25] == 255)
        assert np.all(risultato[32:36, 32:36] == 0)

    def test_un_appendice_sottile_collegata_sopravvive(self) -> None:
        # Riproduce il caso dei baffi: un tratto sottile ma attaccato al corpo.
        mask = np.zeros((40, 40), dtype=np.uint8)
        mask[10:30, 10:30] = 255
        mask[19, 30:38] = 255

        risultato = keep_largest_region(mask, alpha_threshold=8)

        assert np.all(risultato[19, 30:38] == 255)

    def test_un_appendice_semitrasparente_collegata_sopravvive(self) -> None:
        # I baffi hanno un'opacita' debole: la soglia bassa deve considerarli
        # comunque parte del corpo, invece di staccarli e cancellarli.
        mask = np.zeros((40, 40), dtype=np.uint8)
        mask[10:30, 10:30] = 255
        mask[19, 30:38] = 20

        risultato = keep_largest_region(mask, alpha_threshold=8)

        assert np.all(risultato[19, 30:38] == 20)

    def test_una_soglia_troppo_alta_stacca_le_parti_deboli(self) -> None:
        # Documenta il motivo per cui la soglia va tenuta bassa.
        mask = np.zeros((40, 40), dtype=np.uint8)
        mask[10:30, 10:30] = 255
        mask[19, 30:38] = 20

        risultato = keep_largest_region(mask, alpha_threshold=100)

        assert np.all(risultato[19, 30:38] == 0)

    def test_i_valori_sfumati_dei_pixel_superstiti_non_cambiano(self) -> None:
        mask = np.zeros((20, 20), dtype=np.uint8)
        mask[5:15, 5:15] = 200
        mask[5, 5] = 37

        risultato = keep_largest_region(mask, alpha_threshold=8)

        assert risultato[5, 5] == 37
        assert risultato[10, 10] == 200

    def test_una_maschera_vuota_resta_vuota(self) -> None:
        mask = np.zeros((10, 10), dtype=np.uint8)

        risultato = keep_largest_region(mask, alpha_threshold=8)

        assert np.all(risultato == 0)

    def test_una_maschera_piena_resta_piena(self) -> None:
        mask = np.full((10, 10), 255, dtype=np.uint8)

        risultato = keep_largest_region(mask, alpha_threshold=8)

        assert np.all(risultato == 255)

    def test_le_regioni_collegate_in_diagonale_contano_come_una_sola(self) -> None:
        mask = np.zeros((20, 20), dtype=np.uint8)
        mask[2:10, 2:10] = 255
        # Tocca la precedente solo per uno spigolo.
        mask[10:18, 10:18] = 255

        risultato = keep_largest_region(mask, alpha_threshold=8)

        assert np.all(risultato[2:10, 2:10] == 255)
        assert np.all(risultato[10:18, 10:18] == 255)


class TestFeatherMaskEdges:
    """La sfumatura deve ammorbidire il bordo senza spostarlo."""

    def test_raggio_zero_lascia_la_maschera_invariata(self) -> None:
        mask = np.zeros((20, 20), dtype=np.uint8)
        mask[5:15, 5:15] = 255

        risultato = feather_mask_edges(mask, radius=0)

        assert np.array_equal(risultato, mask)

    def test_la_sfumatura_crea_valori_intermedi_sul_bordo(self) -> None:
        mask = np.zeros((20, 20), dtype=np.uint8)
        mask[5:15, 5:15] = 255

        risultato = feather_mask_edges(mask, radius=2)

        intermedi = (risultato > 0) & (risultato < 255)
        assert intermedi.any()

    def test_il_centro_e_l_esterno_restano_saturi(self) -> None:
        mask = np.zeros((30, 30), dtype=np.uint8)
        mask[10:20, 10:20] = 255

        risultato = feather_mask_edges(mask, radius=1)

        assert risultato[15, 15] == 255
        assert risultato[0, 0] == 0

    def test_le_dimensioni_restano_invariate(self) -> None:
        mask = np.zeros((13, 17), dtype=np.uint8)

        risultato = feather_mask_edges(mask, radius=3)

        assert risultato.shape == (13, 17)
