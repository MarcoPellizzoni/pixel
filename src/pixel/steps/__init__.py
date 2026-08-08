"""The processing steps, grouped by family.

Each module in this package holds one family of related algorithms, each with its
own configuration dataclass alongside it. All of them honour the
`ProcessingStep` protocol defined in `base`: they take an image, return a new
one, and know nothing about the other steps or about the filesystem.

    geometry     shape and size            (resize, crop, rotate, flip)
    color        colour interpretation     (grayscale, sepia, invert, ...)
    tone         tone curve                (brightness-contrast, gamma, ...)
    filters      spatial filters           (blur, sharpen, denoise, edges)
    background   subject isolation
    artistic     stylisations              (pencil-sketch, cartoon, vignette)
    pen_sketch   pen drawing, the most elaborate effect
"""

from pixel.steps.base import ProcessingStep

__all__ = ["ProcessingStep"]
