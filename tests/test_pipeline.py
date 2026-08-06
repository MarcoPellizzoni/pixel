"""Test dell'orchestrazione delle fasi."""

from pathlib import Path

import numpy as np

from gatto.config import PipelineConfig
from gatto.domain import RGBAImage
from gatto.pipeline import CatSketchPipeline, PipelineResult, StepResult, save_results


class FaseFinta:
    """Una fase di prova che schiarisce l'immagine di una quantita' fissa.

    Serve a verificare l'orchestrazione senza far girare le reti neurali: la
    pipeline non deve sapere cosa fanno le fasi, quindi puo' essere provata con
    fasi qualsiasi.
    """

    def __init__(self, nome: str, incremento: int) -> None:
        """Memorizza il nome della fase e di quanto deve schiarire."""
        self._nome = nome
        self._incremento = incremento

    @property
    def name(self) -> str:
        """Nome della fase finta."""
        return self._nome

    def apply(self, image: RGBAImage) -> RGBAImage:
        """Somma l'incremento a ogni canale di colore."""
        schiarita = np.clip(
            image.rgb.astype(np.int16) + self._incremento, 0, 255
        ).astype(np.uint8)
        return image.with_rgb(schiarita)


def immagine_nera(lato: int = 4) -> RGBAImage:
    """Crea un'immagine nera opaca."""
    data = np.zeros((lato, lato, 4), dtype=np.uint8)
    data[:, :, 3] = 255
    return RGBAImage(data)


class TestComposizioneDelleFasi:
    """La pipeline deve applicare le fasi nell'ordine giusto."""

    def test_le_fasi_vengono_applicate_in_sequenza(self) -> None:
        pipeline = CatSketchPipeline(
            PipelineConfig(), steps=(FaseFinta("prima", 10), FaseFinta("seconda", 5))
        )

        risultato = pipeline.run(immagine_nera())

        assert np.all(risultato.final_image.rgb == 15)

    def test_ogni_fase_lascia_il_proprio_risultato_intermedio(self) -> None:
        pipeline = CatSketchPipeline(
            PipelineConfig(), steps=(FaseFinta("prima", 10), FaseFinta("seconda", 5))
        )

        risultato = pipeline.run(immagine_nera())

        assert len(risultato.steps) == 2
        assert np.all(risultato.steps[0].image.rgb == 10)
        assert np.all(risultato.steps[1].image.rgb == 15)

    def test_i_risultati_sono_numerati_a_partire_da_uno(self) -> None:
        pipeline = CatSketchPipeline(
            PipelineConfig(), steps=(FaseFinta("prima", 1), FaseFinta("seconda", 1))
        )

        risultato = pipeline.run(immagine_nera())

        assert [passo.order for passo in risultato.steps] == [1, 2]

    def test_l_immagine_di_partenza_resta_intatta(self) -> None:
        originale = immagine_nera()
        pipeline = CatSketchPipeline(PipelineConfig(), steps=(FaseFinta("prima", 50),))

        pipeline.run(originale)

        assert np.all(originale.rgb == 0)


class TestFasiPredefinite:
    """La configurazione standard deve produrre le tre fasi richieste, in ordine."""

    def test_le_tre_fasi_sono_nell_ordine_atteso(self) -> None:
        pipeline = CatSketchPipeline(PipelineConfig())

        assert list(pipeline.iter_step_names()) == [
            "rimozione-sfondo",
            "bianco-e-nero",
            "disegno-a-penna",
        ]


class TestRisultato:
    """`PipelineResult` deve esporre correttamente l'immagine finale."""

    def test_l_immagine_finale_e_quella_dell_ultima_fase(self) -> None:
        sorgente = immagine_nera()
        ultima = FaseFinta("ultima", 99).apply(sorgente)
        risultato = PipelineResult(
            source=sorgente,
            steps=(StepResult(order=1, name="ultima", image=ultima),),
        )

        assert np.all(risultato.final_image.rgb == 99)

    def test_senza_fasi_l_immagine_finale_e_la_sorgente(self) -> None:
        sorgente = immagine_nera()

        risultato = PipelineResult(source=sorgente, steps=())

        assert risultato.final_image is sorgente


class TestSalvataggioDeiRisultati:
    """La scrittura su disco deve rispettare l'opzione sui file intermedi."""

    def _risultato_di_prova(self) -> PipelineResult:
        sorgente = immagine_nera()
        return PipelineResult(
            source=sorgente,
            steps=(
                StepResult(order=1, name="prima", image=sorgente),
                StepResult(order=2, name="seconda", image=sorgente),
            ),
        )

    def test_salva_le_fasi_intermedie_quando_richiesto(self, tmp_path: Path) -> None:
        percorsi = save_results(
            result=self._risultato_di_prova(),
            output_directory=tmp_path,
            final_filename="finale.png",
            save_intermediate_steps=True,
        )

        assert len(percorsi) == 3
        assert (tmp_path / "01_prima.png").is_file()
        assert (tmp_path / "02_seconda.png").is_file()
        assert (tmp_path / "finale.png").is_file()

    def test_salva_solo_il_finale_quando_non_richiesto(self, tmp_path: Path) -> None:
        percorsi = save_results(
            result=self._risultato_di_prova(),
            output_directory=tmp_path,
            final_filename="finale.png",
            save_intermediate_steps=False,
        )

        assert percorsi == [tmp_path / "finale.png"]
        assert not (tmp_path / "01_prima.png").exists()

    def test_i_nomi_intermedi_sono_ordinati_alfabeticamente(
        self, tmp_path: Path
    ) -> None:
        save_results(
            result=self._risultato_di_prova(),
            output_directory=tmp_path,
            final_filename="finale.png",
            save_intermediate_steps=True,
        )

        intermedi = sorted(p.name for p in tmp_path.glob("0*.png"))
        assert intermedi == ["01_prima.png", "02_seconda.png"]
