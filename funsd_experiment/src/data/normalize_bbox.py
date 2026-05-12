"""
Bounding-box normalisation so every coordinate lives in [0, scale],
making layout features independent of the original image resolution.
"""

import logging
from typing import Optional

logger = logging.getLogger(__name__)


def normalize_bbox(
    bbox: list[int],
    width: int,
    height: int,
    scale: int = 1000,
) -> Optional[list[int]]:
    """Scale absolute-pixel *bbox* to the [0, *scale*] range.

    Args:
        bbox: ``[x0, y0, x1, y1]`` in absolute pixel coordinates.
        width: Image width in pixels.
        height: Image height in pixels.
        scale: Target coordinate range (default 1000).

    Returns:
        Normalised ``[x0, y0, x1, y1]`` with each value in [0, scale],
        or *None* when the input is invalid.
    """
    if width <= 0 or height <= 0:
        logger.warning("Invalid image dimensions: %dx%d", width, height)
        return None

    if not isinstance(bbox, (list, tuple)) or len(bbox) != 4:
        logger.warning("Invalid bbox format: %s", bbox)
        return None

    x0, y0, x1, y1 = bbox

    # Clamp negative coordinates to 0
    x0, y0 = max(0, x0), max(0, y0)
    x1, y1 = max(0, x1), max(0, y1)

    # Ensure x1 >= x0, y1 >= y0
    if x1 < x0 or y1 < y0:
        logger.debug("Degenerate bbox (x1<x0 or y1<y0): %s", bbox)
        return None

    # Scale
    x_ratio = scale / width
    y_ratio = scale / height

    nx0 = round(x0 * x_ratio)
    ny0 = round(y0 * y_ratio)
    nx1 = round(x1 * x_ratio)
    ny1 = round(y1 * y_ratio)

    # Clamp to scale range
    nx0 = max(0, min(scale, nx0))
    ny0 = max(0, min(scale, ny0))
    nx1 = max(0, min(scale, nx1))
    ny1 = max(0, min(scale, ny1))

    # After rounding and clamping, re-check validity
    if nx1 < nx0:
        nx1 = nx0
    if ny1 < ny0:
        ny1 = ny0

    return [nx0, ny0, nx1, ny1]
