"""Unit tests for ai.images.encode_image."""

from __future__ import annotations

import base64
from pathlib import Path

import numpy as np
from PIL import Image

from davinci_monet.ai.images import EncodedImage, encode_image


def _write_png(path: Path, width: int, height: int) -> None:
    arr = (np.random.default_rng(0).random((height, width, 3)) * 255).astype("uint8")
    Image.fromarray(arr).save(path)


def test_encode_image_returns_base64_png(tmp_path: Path) -> None:
    p = tmp_path / "small.png"
    _write_png(p, 100, 80)
    enc = encode_image(p)
    assert isinstance(enc, EncodedImage)
    assert enc.media_type == "image/png"
    # data must be valid base64
    decoded = base64.b64decode(enc.data)
    assert decoded[:8] == b"\x89PNG\r\n\x1a\n"


def test_encode_image_downscales_large(tmp_path: Path) -> None:
    p = tmp_path / "big.png"
    _write_png(p, 4000, 2000)
    enc = encode_image(p, max_edge=1568)
    img = Image.open(__import__("io").BytesIO(base64.b64decode(enc.data)))
    assert max(img.size) <= 1568


def test_encode_image_keeps_small_unscaled(tmp_path: Path) -> None:
    p = tmp_path / "ok.png"
    _write_png(p, 800, 600)
    enc = encode_image(p, max_edge=1568)
    img = Image.open(__import__("io").BytesIO(base64.b64decode(enc.data)))
    assert img.size == (800, 600)
