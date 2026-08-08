"""Refining the cut-out mask.

Single responsibility: correct the typical defects of a mask produced by a
segmentation network. These are purely geometric operations on an array of
opacity values: they do not know which network the mask came from, nor which
image it belongs to, and that is what makes them testable and reusable on their
own.
"""

from __future__ import annotations

import cv2
import numpy as np


def keep_largest_region(mask: np.ndarray, alpha_threshold: int) -> np.ndarray:
    """Discard everything in the mask that does not touch the main subject.

    The subject is a single body: any isolated blob the network marked as
    foreground (a corner of a cushion, an object in shadow) is a mistake.
    Labelling the connected regions and keeping only the largest makes those
    leftovers disappear.

    Args:
        mask: opacity mask of shape (height, width), dtype uint8.
        alpha_threshold: minimum opacity for a pixel to count as subject when
            computing the regions. It should be kept low: whiskers and wisps of
            fur are only faintly opaque, and with a high threshold they would end
            up detached from the body and deleted along with the leftovers.

    Returns:
        The mask with only the regions connected to the main subject. The
        surviving pixels keep their original soft values.
    """
    # Binary version of the mask, used only to decide what is connected to what;
    # the original values are preserved further down.
    is_subject = (mask > alpha_threshold).astype(np.uint8)

    # `connectivity=8` treats diagonal pixels as neighbours too: a thin, slanted
    # stroke such as a whisker therefore stays a single region.
    region_count, labels, stats, _ = cv2.connectedComponentsWithStats(
        is_subject, connectivity=8
    )

    # Label 0 is always the background: if there is nothing else, the network
    # found no subject and there is nothing to filter.
    if region_count <= 1:
        return mask

    # `stats` reports area, position and size for each label: we look for the
    # largest area among the labels other than the background.
    areas = stats[1:, cv2.CC_STAT_AREA]
    largest_label = int(np.argmax(areas)) + 1

    # Zero the opacity everywhere except in the winning region.
    return np.where(labels == largest_label, mask, 0).astype(np.uint8)


def feather_mask_edges(mask: np.ndarray, radius: int) -> np.ndarray:
    """Soften the mask's edge.

    The network works at reduced resolution and then scales the result back up,
    so the edge can show slight stair-stepping. A minimal Gaussian blur turns it
    into a gradual transition.

    Args:
        mask: opacity mask of shape (height, width), dtype uint8.
        radius: blur radius in pixels; 0 or negative disables it.

    Returns:
        The mask with refined edges.
    """
    if radius <= 0:
        # Refinement disabled: return the mask as it is.
        return mask

    # OpenCV requires an odd, positive kernel size.
    kernel_size = radius * 2 + 1

    # `sigmaX=0` lets OpenCV derive the standard deviation from the kernel.
    return cv2.GaussianBlur(mask, (kernel_size, kernel_size), sigmaX=0)
