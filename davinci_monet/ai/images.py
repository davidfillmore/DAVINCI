"""Encode plot images for the Claude vision prompt.

Loads a PNG, downscales its long edge to keep vision token cost predictable,
and returns base64-encoded PNG bytes ready for an Anthropic image content block.
"""

from __future__ import annotations

import base64
import io
from dataclasses import dataclass
from pathlib import Path


@dataclass
class EncodedImage:
    """A base64-encoded image ready for an Anthropic image block."""

    media_type: str
    data: str


def encode_image(path: str | Path, max_edge: int = 1568) -> EncodedImage:
    """Load a PNG, downscale to ``max_edge`` on its long side, return base64.

    Parameters
    ----------
    path
        Path to a PNG file.
    max_edge
        Maximum length (px) of the longer image edge. Larger images are
        downscaled preserving aspect ratio. Anthropic recommends <=1568px.

    Returns
    -------
    EncodedImage
        media_type ``"image/png"`` and base64-encoded PNG data.
    """
    from PIL import Image

    with Image.open(path) as opened:
        img = opened.convert("RGB")
        longest = max(img.size)
        if longest > max_edge:
            scale = max_edge / longest
            new_size = (round(img.size[0] * scale), round(img.size[1] * scale))
            img = img.resize(new_size, Image.Resampling.LANCZOS)
        buf = io.BytesIO()
        img.save(buf, format="PNG")

    data = base64.b64encode(buf.getvalue()).decode("ascii")
    return EncodedImage(media_type="image/png", data=data)
