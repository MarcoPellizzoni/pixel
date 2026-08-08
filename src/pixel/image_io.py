"""Input/output: the only module that touches the filesystem.

Single responsibility: translate between bytes (on disk or on a stream) and
`RGBAImage`. Isolating reading and writing here means the algorithms in the
`steps` package are testable with in-memory arrays, without ever creating a
temporary file.

The special path `-` means standard input or standard output, so the program
can be chained with real shell pipes.

It relies on Pillow, which correctly handles formats, colour profiles and above
all the EXIF orientation of photos taken on smartphones.
"""

from __future__ import annotations

import sys
from io import BytesIO
from pathlib import Path
from typing import IO

import numpy as np
from PIL import Image, ImageOps

from pixel.domain import RGBAImage

# Conventional path meaning standard input or standard output. This is the
# convention used by almost every command-line tool.
STDIO_PATH: Path = Path("-")

# Formats that can preserve the alpha channel. Saving a cut-out as JPEG would
# silently lose the transparency, so we handle that case explicitly.
FORMATS_WITH_TRANSPARENCY: frozenset[str] = frozenset({".png", ".webp", ".tif", ".tiff"})

# Colour used to flatten transparency when the format does not support it.
DEFAULT_FLATTEN_COLOR: tuple[int, int, int] = (255, 255, 255)

# Format used when writing to a stream, where there is no extension to infer it
# from: PNG because it is lossless and preserves transparency.
STREAM_FORMAT: str = "PNG"


def load_image(path: Path) -> RGBAImage:
    """Load an image and normalise it to RGBA.

    Args:
        path: path of the file to read, or `-` for standard input.

    Returns:
        The image as an `RGBAImage`.

    Raises:
        FileNotFoundError: if the path does not exist.
    """
    if path == STDIO_PATH:
        # `buffer` gives access to the raw bytes: the textual standard input
        # would corrupt an image's binary data.
        return _decode_image(BytesIO(sys.stdin.buffer.read()))

    if not path.is_file():
        raise FileNotFoundError(f"Image not found: {path}")

    with path.open("rb") as stream:
        return _decode_image(stream)


def save_image(image: RGBAImage, path: Path) -> None:
    """Save an image, creating any missing directories.

    If the target format does not support transparency, the image is flattened
    onto a white background rather than silently losing the alpha channel.

    Args:
        image: the image to save.
        path: destination path (the extension determines the format), or `-`
            for standard output.
    """
    if path == STDIO_PATH:
        Image.fromarray(image.data, mode="RGBA").save(
            sys.stdout.buffer, format=STREAM_FORMAT
        )
        # Without the explicit flush the bytes could stay in the buffer when the
        # output is a pipe, and the next program would hang.
        sys.stdout.buffer.flush()
        return

    # `parents=True` creates the whole directory tree, `exist_ok=True` does not
    # complain if it already exists.
    path.parent.mkdir(parents=True, exist_ok=True)

    if path.suffix.lower() in FORMATS_WITH_TRANSPARENCY:
        # The format supports alpha: save all four channels as they are.
        Image.fromarray(image.data, mode="RGBA").save(path)
    else:
        # Format without alpha (e.g. JPEG): flatten onto white explicitly.
        flattened_rgb = image.composite_over(DEFAULT_FLATTEN_COLOR)
        Image.fromarray(flattened_rgb, mode="RGB").save(path, quality=95)


def _decode_image(stream: IO[bytes]) -> RGBAImage:
    """Decode an image's bytes from a stream open for reading.

    Args:
        stream: a binary stream positioned at the start of the data. It can be
            an open file, standard input or an in-memory buffer: all Pillow
            needs is something that can read bytes.

    Returns:
        The image as an `RGBAImage`.
    """
    with Image.open(stream) as opened_image:
        # Phone cameras store the rotation in the EXIF metadata instead of
        # actually rotating the pixels: without this call the subject could come
        # out lying on its side. `exif_transpose` applies the rotation to the
        # pixels themselves.
        upright_image = ImageOps.exif_transpose(opened_image)

        # Convert to RGBA immediately: this way the rest of the program treats
        # JPEGs (opaque), PNGs with transparency, and greyscale or palette
        # images all the same way.
        rgba_image = upright_image.convert("RGBA")

        return RGBAImage(np.array(rgba_image, dtype=np.uint8))
