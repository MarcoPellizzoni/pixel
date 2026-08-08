"""Tests for reading and writing image files."""

import sys
from io import BytesIO
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
from PIL import Image

from pixel.domain import RGBAImage
from pixel.image_io import STDIO_PATH, load_image, save_image


@pytest.fixture
def sample_image() -> RGBAImage:
    """A 4x4 image with one opaque pixel, one transparent, the rest half-way."""
    data = np.full((4, 4, 4), 128, dtype=np.uint8)
    data[0, 0] = [255, 0, 0, 255]
    data[1, 1] = [0, 255, 0, 0]
    return RGBAImage(data)


class TestLoading:
    """Reading must normalise any format to RGBA."""

    def test_a_jpeg_is_loaded_fully_opaque(self, tmp_path: Path) -> None:
        path = tmp_path / "photo.jpg"
        Image.new("RGB", (5, 3), (10, 20, 30)).save(path)

        image = load_image(path)

        assert np.all(image.alpha == 255)
        assert (image.height, image.width) == (3, 5)

    def test_a_greyscale_png_becomes_rgba(self, tmp_path: Path) -> None:
        path = tmp_path / "grey.png"
        Image.new("L", (5, 3), 90).save(path)

        image = load_image(path)

        assert image.data.shape == (3, 5, 4)

    def test_a_missing_file_raises_a_clear_error(self, tmp_path: Path) -> None:
        with pytest.raises(FileNotFoundError, match="not found"):
            load_image(tmp_path / "this_file_does_not_exist.png")


class TestSaving:
    """Writing must preserve transparency whenever the format allows it."""

    def test_png_preserves_the_alpha_channel(
        self, tmp_path: Path, sample_image: RGBAImage
    ) -> None:
        path = tmp_path / "result.png"

        save_image(sample_image, path)
        reloaded = load_image(path)

        assert np.array_equal(reloaded.data, sample_image.data)

    def test_jpeg_flattens_onto_white_instead_of_losing_the_alpha(
        self, tmp_path: Path
    ) -> None:
        # Half the image opaque and black, half fully transparent. The areas are
        # large because JPEG is a lossy format: on blocks of a few pixels the
        # compression artefacts would skew the comparison.
        data = np.zeros((32, 32, 4), dtype=np.uint8)
        data[:, :16, 3] = 255
        path = tmp_path / "result.jpg"

        save_image(RGBAImage(data), path)
        reloaded = load_image(path)

        # JPEG has no transparency: everything must come back opaque...
        assert np.all(reloaded.alpha == 255)
        # ...the opaque half must have stayed black...
        assert reloaded.rgb[:, :8].max() < 15
        # ...and the transparent one must have become white.
        assert reloaded.rgb[:, 24:].min() > 240

    def test_missing_directories_are_created(
        self, tmp_path: Path, sample_image: RGBAImage
    ) -> None:
        path = tmp_path / "one" / "two" / "three" / "result.png"

        save_image(sample_image, path)

        assert path.is_file()


class TestStandardInputOutput:
    """The `-` path must read from and write to the streams, not to files."""

    def test_reads_the_image_from_standard_input(
        self, monkeypatch: pytest.MonkeyPatch, sample_image: RGBAImage
    ) -> None:
        # Prepare a PNG in memory and offer it as standard input.
        buffer = BytesIO()
        Image.fromarray(sample_image.data, mode="RGBA").save(buffer, format="PNG")
        monkeypatch.setattr(
            sys, "stdin", SimpleNamespace(buffer=BytesIO(buffer.getvalue()))
        )

        reloaded = load_image(STDIO_PATH)

        assert np.array_equal(reloaded.data, sample_image.data)

    def test_writes_the_image_to_standard_output(
        self, monkeypatch: pytest.MonkeyPatch, sample_image: RGBAImage
    ) -> None:
        buffer = BytesIO()
        monkeypatch.setattr(sys, "stdout", SimpleNamespace(buffer=buffer))

        save_image(sample_image, STDIO_PATH)

        # The bytes written must be a readable PNG identical to the original.
        reloaded = Image.open(BytesIO(buffer.getvalue()))
        assert reloaded.format == "PNG"
        assert np.array_equal(np.array(reloaded.convert("RGBA")), sample_image.data)

    def test_the_stream_preserves_transparency(
        self, monkeypatch: pytest.MonkeyPatch, sample_image: RGBAImage
    ) -> None:
        # This is what makes it possible to chain several commands without losing
        # the cut-out made by an earlier step.
        buffer = BytesIO()
        monkeypatch.setattr(sys, "stdout", SimpleNamespace(buffer=buffer))
        save_image(sample_image, STDIO_PATH)

        monkeypatch.setattr(
            sys, "stdin", SimpleNamespace(buffer=BytesIO(buffer.getvalue()))
        )
        reloaded = load_image(STDIO_PATH)

        assert reloaded.alpha[1, 1] == 0


class TestRoundTrip:
    """Saving and reloading must not alter the image."""

    def test_png_is_lossless(self, tmp_path: Path) -> None:
        random_pixels = np.random.default_rng(seed=0).integers(
            0, 256, size=(9, 7, 4), dtype=np.uint8
        )
        original = RGBAImage(random_pixels)
        path = tmp_path / "round_trip.png"

        save_image(original, path)
        reloaded = load_image(path)

        assert np.array_equal(reloaded.data, original.data)
