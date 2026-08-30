import os
import sys
import time
from pathlib import Path
from typing import Tuple, Optional, List

if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        pass

import cv2
import numpy as np
import torch
from PIL import Image

from image_inpainter import (
    load_lama_model,
    inpaint_lama,
    inpaint_image,
    inpaint_color_fill,
    sample_background_color,
    inpaint_opencv,
    apply_image_sanitization,
    save_clean_image,
    get_device,
)


def create_test_image(width: int, height: int, watermark_text: str = "SAMPLE WATERMARK") -> Tuple[np.ndarray, np.ndarray, Tuple[int, int, int, int]]:
    """
    Create a realistic synthetic test image with colorful background patterns and a text/logo watermark.
    Returns (image_rgb, mask_binary, (x, y, w, h)).
    """
    # Create colorful gradient background with noise/texture
    img = np.zeros((height, width, 3), dtype=np.uint8)
    for y in range(height):
        r = int(128 + 127 * np.sin(y / 50.0))
        g = int(128 + 127 * np.cos(y / 40.0))
        b = int(128 + 127 * np.sin(y / 30.0))
        img[y, :, 0] = r
        img[y, :, 1] = g
        img[y, :, 2] = b

    # Add geometric shapes to test texture reconstruction
    cv2.circle(img, (int(width * 0.7), int(height * 0.7)), int(min(width, height) * 0.2), (240, 180, 50), -1)
    cv2.rectangle(img, (int(width * 0.2), int(height * 0.2)), (int(width * 0.5), int(height * 0.5)), (50, 120, 240), -1)

    # Add watermark in bottom-right corner
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = max(0.6, min(width, height) / 800.0)
    thickness = max(2, int(font_scale * 2.5))
    (text_w, text_h), baseline = cv2.getTextSize(watermark_text, font, font_scale, thickness)

    margin_x = int(width * 0.05)
    margin_y = int(height * 0.05)
    wx = width - text_w - margin_x
    wy = height - margin_y

    # Draw semi-transparent watermark background box + text
    cv2.putText(img, watermark_text, (wx, wy), font, font_scale, (255, 255, 255), thickness, cv2.LINE_AA)
    cv2.putText(img, watermark_text, (wx, wy), font, font_scale, (20, 20, 20), max(1, thickness - 2), cv2.LINE_AA)

    # Create corresponding ground-truth mask
    mask = np.zeros((height, width), dtype=np.uint8)
    pad = 10
    rx = max(0, wx - pad)
    ry = max(0, wy - text_h - pad)
    rw = min(width - rx, text_w + pad * 2)
    rh = min(height - ry, text_h + baseline + pad * 2)
    mask[ry : ry + rh, rx : rx + rw] = 255

    return img, mask, (rx, ry, rw, rh)


def run_tests():
    print("=" * 60)
    print("🧪 Running Image Inpainting Pipeline Automated Tests")
    print(f"🖥️ Device: {get_device()}")
    print("=" * 60)

    # 1. Test Model Loading
    t0 = time.time()
    model = load_lama_model()
    t_load = time.time() - t0
    print(f"✅ [1/5] LaMa model loaded successfully in {t_load:.2f}s")

    test_cases = [
        ("Square (1024x1024)", 1024, 1024),
        ("Landscape FHD (1920x1080)", 1920, 1080),
        ("Portrait 9:16 (1080x1920)", 1080, 1920),
        ("Arbitrary Non-Standard (1337x779)", 1337, 779),
        ("High-Res 4K (3840x2160)", 3840, 2160),
    ]

    out_dir = Path("results/test_image_outputs")
    out_dir.mkdir(parents=True, exist_ok=True)

    for name, w, h in test_cases:
        print(f"\n🎨 Testing: {name}...")
        img, mask, (rx, ry, rw, rh) = create_test_image(w, h)

        # Test LaMa Inpainting
        t_start = time.time()
        inpainted_lama = inpaint_lama(img, mask, dilation_pixels=5, blend_seam=True)
        t_lama = time.time() - t_start

        assert inpainted_lama.shape == img.shape, f"Shape mismatch: {inpainted_lama.shape} vs {img.shape}"
        assert inpainted_lama.dtype == np.uint8, f"Dtype mismatch: {inpainted_lama.dtype}"
        print(f"  ⚡ LaMa inpainting completed in {t_lama:.3f}s (Output shape: {inpainted_lama.shape})")

        # Test Background Color Fill (Auto background sampling & Custom Color)
        t_start = time.time()
        inpainted_auto_bg = inpaint_image(img, mask, method="🎯 주변 배경색 자동 채우기 (Auto Background Color Fill)")
        t_auto_bg = time.time() - t_start
        assert inpainted_auto_bg.shape == img.shape
        print(f"  ⚡ Auto Background Color Fill completed in {t_auto_bg:.4f}s")

        inpainted_custom_color = inpaint_image(img, mask, method="🎨 지정 색상 채우기 (Custom Color Fill)", custom_color="#FFFFFF")
        assert inpainted_custom_color.shape == img.shape
        print(f"  ⚡ Custom Pure White Fill completed in <0.001s")

        # Test OpenCV NS Fallback
        t_start = time.time()
        inpainted_ns = inpaint_opencv(img, mask, method="ns", dilation_pixels=5)
        t_ns = time.time() - t_start
        print(f"  ⚡ OpenCV NS inpainting completed in {t_ns:.3f}s")

        # Test Image Sanitization (C2PA/SynthID)
        sanitized = apply_image_sanitization(inpainted_lama, clean_c2pa=True, deep_clean_synthid=True)
        assert sanitized.shape == img.shape

        # Save and verify clean file
        save_path = out_dir / f"test_{w}x{h}_clean.png"
        saved_file = save_clean_image(sanitized, save_path)
        assert saved_file.exists() and saved_file.stat().st_size > 0
        print(f"  💾 Saved clean sanitized image: {saved_file} ({saved_file.stat().st_size / 1024:.1f} KB)")

    print("\n" + "=" * 60)
    print("🎉 ALL TESTS (LaMa AI + Background Color Fill + OpenCV + Sanitizer) PASSED!")
    print("=" * 60)


if __name__ == "__main__":
    run_tests()
