"""Unit tests for the crop utility used by the refine stage."""

from __future__ import annotations

import base64
import io

from PIL import Image

from local_deepl.utils.image import (
    crop_box_to_base64,
    crop_for_ocr,
    crop_for_ocr_from_image,
)


def _make_image_b64(size=(800, 1000)) -> str:
    img = Image.new("RGB", size, "white")
    # Paint a visible patch so we can verify the crop roughly captured it.
    for y in range(400, 600):
        for x in range(100, 300):
            img.putpixel((x, y), (255, 0, 0))
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode()


def _decode_b64_image(b64: str) -> Image.Image:
    return Image.open(io.BytesIO(base64.b64decode(b64))).convert("RGB")


def test_crop_returns_valid_image():
    page = _make_image_b64()
    bbox = [0.1, 0.4, 0.4, 0.6]
    crop_b64 = crop_box_to_base64(page, bbox)
    out = _decode_b64_image(crop_b64)
    assert out.width > 0 and out.height > 0


def test_crop_upscales_tiny_regions():
    page = _make_image_b64(size=(1000, 1000))
    tiny_bbox = [0.0, 0.0, 0.02, 0.02]  # 20x20 pixels raw
    crop_b64 = crop_box_to_base64(page, tiny_bbox, min_dim=256)
    out = _decode_b64_image(crop_b64)
    # The helper should upscale so the VLM can read glyphs.
    assert out.width >= 256 or out.height >= 256


def test_crop_captures_painted_region():
    page = _make_image_b64()  # red patch in (100..300, 400..600)
    bbox = [0.1, 0.4, 0.4, 0.6]  # same region normalized
    crop_b64 = crop_box_to_base64(page, bbox, padding=0.0)
    out = _decode_b64_image(crop_b64)
    # Center pixel should be red-ish (JPEG compression is forgiving).
    cx, cy = out.width // 2, out.height // 2
    r, g, b = out.getpixel((cx, cy))
    assert r > 150 and g < 100 and b < 100, f"expected red-ish, got {(r,g,b)}"


def test_crop_clamps_out_of_range_bbox():
    page = _make_image_b64()
    # Negative + >1 coords: helper must clamp without crashing.
    crop_b64 = crop_box_to_base64(page, [-0.1, -0.1, 1.2, 1.2])
    out = _decode_b64_image(crop_b64)
    assert out.width > 0 and out.height > 0


# --- Tests for crop_for_ocr_from_image (⚡ performance optimization) ---


def _make_pil_image(size=(800, 1000)) -> Image.Image:
    """Create a test PIL Image with a visible red patch."""
    img = Image.new("RGB", size, "white")
    for y in range(400, 600):
        for x in range(100, 300):
            img.putpixel((x, y), (255, 0, 0))
    return img


def test_crop_for_ocr_from_image_returns_valid_jpeg():
    """Pre-decoded image path returns a valid base64 JPEG."""
    img = _make_pil_image()
    bbox = [0.1, 0.4, 0.4, 0.6]
    crop_b64 = crop_for_ocr_from_image(img, bbox)
    assert crop_b64 is not None
    out = _decode_b64_image(crop_b64)
    assert out.width > 0 and out.height > 0


def test_crop_for_ocr_from_image_matches_base64_version():
    """Pre-decoded image path produces identical output to base64 version."""
    pil_img = _make_pil_image()
    # Encode the same image to base64 for comparison
    buf = io.BytesIO()
    pil_img.save(buf, format="PNG")
    img_b64 = base64.b64encode(buf.getvalue()).decode()

    bbox = [0.1, 0.4, 0.4, 0.6]
    # Both should produce the same crop (modulo JPEG compression noise)
    crop_from_b64 = crop_for_ocr(img_b64, bbox, padding=0.0)
    crop_from_pil = crop_for_ocr_from_image(pil_img, bbox, padding=0.0)

    assert crop_from_b64 is not None
    assert crop_from_pil is not None
    # Decode both and check dimensions match
    out_b64 = _decode_b64_image(crop_from_b64)
    out_pil = _decode_b64_image(crop_from_pil)
    assert out_b64.size == out_pil.size


def test_crop_for_ocr_from_image_blank_region_returns_none():
    """Blank/uniform regions return None (skip LLM call optimization)."""
    # Create an all-white image (no visible content)
    blank_img = Image.new("RGB", (800, 1000), "white")
    bbox = [0.1, 0.1, 0.4, 0.3]
    result = crop_for_ocr_from_image(blank_img, bbox)
    assert result is None


def test_crop_for_ocr_from_image_reuses_same_image():
    """
    ⚡ Performance test: verify the same PIL Image can be reused across
    multiple crop calls without issues (the optimization's core behavior).
    """
    img = _make_pil_image()  # red patch at (100..300, 400..600) = (0.125..0.375, 0.4..0.6)
    bboxes = [
        [0.1, 0.4, 0.4, 0.6],  # red patch region - has content
        [0.5, 0.1, 0.8, 0.3],  # blank region (no red pixels)
        [0.12, 0.42, 0.16, 0.46],  # small region INSIDE red patch - will upscale
    ]
    # Call multiple times with the same image - should not corrupt or mutate
    results = [crop_for_ocr_from_image(img, bbox) for bbox in bboxes]
    # Red patch should succeed, blank should be None, small red region should upscale
    assert results[0] is not None  # has content (large red patch)
    assert results[1] is None  # blank (white region)
    assert results[2] is not None  # upscaled (small but has red content)
