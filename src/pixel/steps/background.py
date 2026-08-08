"""Background removal: keeps only the subject in the foreground.

Single responsibility: decide which pixels belong to the subject and write that
into the image's alpha channel.

The heavy lifting is done by `rembg`, which runs a segmentation neural network
(the U^2-Net / IS-Net family) trained to separate the foreground object from the
rest of the scene. It is the only workable approach on an arbitrary photo: the
background may have exactly the same colours and textures as the subject, and no
technique based on colour thresholds would hold up.

Geometric refinement of the mask is delegated to `mask_cleanup`: only the call
to the network and the assembly of the result remain here.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
from typing import Annotated

import numpy as np

# `rembg` ships no type annotations (it has no `py.typed` marker), so as far as
# the static checker is concerned everything coming out of it has an unknown
# type. The suppressions stay confined to these two lines, that is to the border
# with the library: from here on the module is fully checked again, because the
# values coming from it are immediately assigned to names with a declared type.
from rembg import (  # pyright: ignore[reportMissingTypeStubs]
    new_session,  # pyright: ignore[reportUnknownVariableType]
    remove,
)
from rembg.sessions.base import (  # pyright: ignore[reportMissingTypeStubs]
    BaseSession,
)

from pixel.domain import RGBAImage
from pixel.steps.mask_cleanup import feather_mask_edges, keep_largest_region


class SegmentationModel(StrEnum):
    """Neural networks available for separating subject from background.

    These are the pre-trained models distributed by `rembg`; they are downloaded
    automatically on first use and then kept in a local cache.
    """

    # The original model, robust and well proven on generic subjects.
    U2NET = "u2net"

    # More recent: crisper edges and better results on fur, hair and fine detail.
    # This is the default choice.
    ISNET_GENERAL = "isnet-general-use"

    # Specialised on people: more accurate than U2NET on portraits, useless on
    # any other subject.
    U2NET_HUMAN = "u2net_human_seg"

    # A reduced version of U2NET: much lighter to download and run, at the cost
    # of less precise edges.
    U2NETP = "u2netp"


@dataclass(frozen=True)
class RemoveBackgroundConfig:
    """Parameters for isolating the subject."""

    model: Annotated[
        SegmentationModel,
        "Which neural network to use for segmentation.",
    ] = SegmentationModel.ISNET_GENERAL

    alpha_matting: Annotated[
        bool,
        "Alpha matting is an edge refinement performed after segmentation: it "
        "recovers semi-transparent detail such as hair, whiskers and wisps of "
        "fur that a binary mask would slice straight through. It costs "
        "computation time, but on a furry subject the difference shows.",
    ] = True

    foreground_threshold: Annotated[
        int,
        "Threshold above which a mask pixel counts as 'definitely subject' "
        "during alpha matting (0-255).",
    ] = 240

    background_threshold: Annotated[
        int,
        "Threshold below which a pixel counts as 'definitely background' "
        "(0-255). Everything between the two thresholds is the uncertain band "
        "to reconstruct.",
    ] = 15

    erode_size: Annotated[
        int,
        "Width in pixels of the uncertain band around the edge: the wider it "
        "is, the more soft detail is recovered (and the slower the "
        "computation).",
    ] = 12

    keep_largest: Annotated[
        bool,
        "When enabled, only the largest connected region of the mask is kept. "
        "The network sometimes promotes a detached scrap of background to "
        "'subject' (a corner of a cushion, an object in shadow): when the "
        "subject is a single body, anything not touching it is certainly a "
        "leftover. Turn this off if the photo contains several separate "
        "subjects.",
    ] = True

    connectivity_threshold: Annotated[
        int,
        "Minimum opacity (0-255) for a pixel to count as part of the subject "
        "when computing the connected regions. It should be kept low: hair "
        "and wisps of fur are only faintly opaque, and with a high threshold "
        "they would end up detached from the body and deleted along with the "
        "leftovers.",
    ] = 8

    feather: Annotated[
        int,
        "Radius of the blur applied to the alpha channel alone, to soften the "
        "mask's stair-stepping. 0 disables the refinement.",
    ] = 1

    fill: Annotated[
        tuple[int, int, int],
        "Colour used to fill the pixels that have become transparent. The "
        "alpha channel alone is not enough: the old background's colours "
        "would still sit underneath the transparency, and later steps would "
        "still see them, processing detail invisible to the eye but very much "
        "present in the computation. Filling with white also turns the "
        "subject's outline into a crisp edge, which the stylisation effects "
        "render as a clean contour.",
    ] = (255, 255, 255)


class RemoveBackgroundStep:
    """Isolates the image's subject by replacing its alpha channel."""

    def __init__(self, config: RemoveBackgroundConfig) -> None:
        """Prepare the step.

        Args:
            config: segmentation and edge refinement parameters.
        """
        self._config = config

        # The session (the model loaded into memory) is created lazily on first
        # use: constructing a `RemoveBackgroundStep` must not download hundreds
        # of megabytes nor take up RAM until it is genuinely needed. It is also
        # what makes it possible to list the step catalogue, or to test the
        # pipeline, without ever touching the network.
        self._session: BaseSession | None = None

    @property
    def name(self) -> str:
        """Step name."""
        return "remove-background"

    def apply(self, image: RGBAImage) -> RGBAImage:
        """Isolate the subject: make the surroundings transparent and clean its colours.

        Args:
            image: the original image, typically opaque.

        Returns:
            The subject alone, on a transparent background and with the old
            background's colours replaced by the fill colour.
        """
        subject_mask = self._predict_subject_mask(image)
        clean_mask = self._clean_up_mask(subject_mask)

        # The cut-out proper: the background becomes transparent.
        cut_out = image.with_alpha(clean_mask)

        # The colours under the transparency must be replaced, not merely hidden:
        # see the note on `fill` in the configuration.
        cleaned_rgb = cut_out.composite_over(self._config.fill)

        return cut_out.with_rgb(cleaned_rgb)

    # ------------------------------------------------------------------
    # Internal stages
    # ------------------------------------------------------------------

    def _predict_subject_mask(self, image: RGBAImage) -> np.ndarray:
        """Ask the neural network which pixels belong to the subject.

        Args:
            image: the image to segment.

        Returns:
            A (height, width) uint8 mask: 255 = subject, 0 = background,
            intermediate values = semi-transparent edge.
        """
        # `only_mask=True` returns the mask alone instead of the already cut-out
        # image: this way the original colours stay intact and we are the ones
        # deciding how to combine them, keeping the responsibilities separate.
        mask_image = remove(
            image.rgb,
            session=self._get_session(),
            only_mask=True,
            alpha_matting=self._config.alpha_matting,
            alpha_matting_foreground_threshold=self._config.foreground_threshold,
            alpha_matting_background_threshold=self._config.background_threshold,
            alpha_matting_erode_size=self._config.erode_size,
        )

        mask = np.asarray(mask_image, dtype=np.uint8)

        # Depending on the options, `rembg` may return a mask with a superfluous
        # channel axis: flatten it to two dimensions.
        if mask.ndim == 3:
            mask = mask[:, :, 0]

        return mask

    def _clean_up_mask(self, mask: np.ndarray) -> np.ndarray:
        """Apply the refinements the configuration calls for to the mask.

        Args:
            mask: the raw mask produced by the network.

        Returns:
            The mask, ready to be used as an alpha channel.
        """
        if self._config.keep_largest:
            mask = keep_largest_region(
                mask, alpha_threshold=self._config.connectivity_threshold
            )

        return feather_mask_edges(mask, radius=self._config.feather)

    def _get_session(self) -> BaseSession:
        """Return the model's session, creating it on first use.

        Keeping it in an attribute avoids reloading the network for every image,
        which would otherwise dominate the running time across multiple files.
        """
        if self._session is None:
            self._session = new_session(self._config.model.value)
        return self._session
