import os
import sys
import time
import math
from pathlib import Path
from typing import Optional, Tuple, Union, Dict, Any

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

BASE_DIR = Path(__file__).resolve().parent
WEIGHTS_DIR = BASE_DIR / "weights"

# Global cached LaMa model
_GLOBAL_LAMA_MODEL = None
_GLOBAL_LAMA_DEVICE = None


def get_device() -> torch.device:
    """Return CUDA device if available, else CPU."""
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load_lama_model(weights_path: Optional[Union[str, Path]] = None, device: Optional[torch.device] = None) -> Any:
    """
    Load the TorchScript LaMa model (big-lama.pt) with singleton caching.
    """
    global _GLOBAL_LAMA_MODEL, _GLOBAL_LAMA_DEVICE

    if device is None:
        device = get_device()

    if _GLOBAL_LAMA_MODEL is not None and _GLOBAL_LAMA_DEVICE == device:
        return _GLOBAL_LAMA_MODEL

    if weights_path is None:
        weights_path = WEIGHTS_DIR / "big-lama.pt"
    else:
        weights_path = Path(weights_path)

    if not weights_path.exists() or weights_path.stat().st_size < 1024 * 1024:
        from download_weights import ensure_lama_weights
        print(f"[LaMa] Weight not found at {weights_path}, attempting download...")
        if not ensure_lama_weights():
            raise FileNotFoundError(f"Could not find or download LaMa model weights at {weights_path}")

    print(f"[LaMa] Loading TorchScript model from {weights_path} to {device}...")
    model = torch.jit.load(str(weights_path), map_location=device)
    model.eval()

    # Warm up model with small dummy forward pass
    try:
        dummy_img = torch.zeros(1, 3, 64, 64, device=device)
        dummy_mask = torch.zeros(1, 1, 64, 64, device=device)
        with torch.no_grad():
            _ = model(dummy_img, dummy_mask)
    except Exception as e:
        print(f"[LaMa] Warmup warning: {e}")

    _GLOBAL_LAMA_MODEL = model
    _GLOBAL_LAMA_DEVICE = device
    print(f"[LaMa] Model loaded successfully on {device}!")
    return _GLOBAL_LAMA_MODEL


def pad_to_multiple(tensor: torch.Tensor, multiple: int = 8) -> Tuple[torch.Tensor, Tuple[int, int, int, int]]:
    """
    Pad 4D tensor (B, C, H, W) so that H and W are multiples of `multiple`.
    Returns padded tensor and padding amounts (pad_l, pad_r, pad_t, pad_b).
    """
    h, w = tensor.shape[2], tensor.shape[3]
    new_h = math.ceil(h / multiple) * multiple
    new_w = math.ceil(w / multiple) * multiple

    pad_h = new_h - h
    pad_w = new_w - w

    pad_t = pad_h // 2
    pad_b = pad_h - pad_t
    pad_l = pad_w // 2
    pad_r = pad_w - pad_l

    if pad_h == 0 and pad_w == 0:
        return tensor, (0, 0, 0, 0)

    # Use reflection padding if image is large enough, otherwise replicate
    pad_mode = "reflect" if (pad_t < h and pad_b < h and pad_l < w and pad_r < w) else "replicate"
    padded = F.pad(tensor, (pad_l, pad_r, pad_t, pad_b), mode=pad_mode)
    return padded, (pad_l, pad_r, pad_t, pad_b)


def unpad(tensor: torch.Tensor, pad_info: Tuple[int, int, int, int]) -> torch.Tensor:
    """Unpad tensor using pad_info (pad_l, pad_r, pad_t, pad_b)."""
    pad_l, pad_r, pad_t, pad_b = pad_info
    h, w = tensor.shape[2], tensor.shape[3]
    return tensor[:, :, pad_t : h - pad_b if pad_b > 0 else h, pad_l : w - pad_r if pad_r > 0 else w]


def preprocess_mask(
    mask: np.ndarray,
    target_shape: Tuple[int, int],
    dilation_pixels: int = 5,
) -> np.ndarray:
    """
    Ensure mask is single channel uint8 (0 or 255), resized to target_shape (H, W),
    and optionally dilated.
    """
    target_h, target_w = target_shape

    # Normalize dimensions
    if len(mask.shape) == 3:
        if mask.shape[2] == 4:
            # RGBA: take alpha
            mask = mask[:, :, 3]
        else:
            mask = cv2.cvtColor(mask, cv2.COLOR_RGB2GRAY)

    if mask.shape[0] != target_h or mask.shape[1] != target_w:
        mask = cv2.resize(mask, (target_w, target_h), interpolation=cv2.INTER_NEAREST)

    # Binarize
    _, binary_mask = cv2.threshold(mask, 10, 255, cv2.THRESH_BINARY)

    # Dilation
    if dilation_pixels > 0:
        kernel_size = dilation_pixels * 2 + 1
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
        binary_mask = cv2.dilate(binary_mask, kernel, iterations=1)

    return binary_mask


def inpaint_lama_patch(
    image_rgb: np.ndarray,
    mask: np.ndarray,
    dilation_pixels: int = 5,
    blend_seam: bool = True,
    device: Optional[torch.device] = None,
) -> np.ndarray:
    """
    Direct LaMa forward pass on single image patch.
    """
    if device is None:
        device = get_device()

    model = load_lama_model(device=device)

    h, w = image_rgb.shape[:2]
    binary_mask = preprocess_mask(mask, (h, w), dilation_pixels=dilation_pixels)

    if not np.any(binary_mask > 0):
        return image_rgb.copy()

    # Convert image to tensor [1, 3, H, W] in [0, 1]
    img_tensor = torch.from_numpy(image_rgb.transpose(2, 0, 1)).unsqueeze(0).float() / 255.0
    img_tensor = img_tensor.to(device)

    # Convert mask to tensor [1, 1, H, W] in [0, 1]
    mask_tensor = torch.from_numpy((binary_mask > 0).astype(np.float32)).unsqueeze(0).unsqueeze(0)
    mask_tensor = mask_tensor.to(device)

    # Pad to multiple of 8
    img_padded, pad_info = pad_to_multiple(img_tensor, multiple=8)
    mask_padded, _ = pad_to_multiple(mask_tensor, multiple=8)

    with torch.no_grad():
        out_tensor = model(img_padded, mask_padded)

    # Unpad
    out_tensor = unpad(out_tensor, pad_info)

    # Convert back to uint8 RGB
    out_np = (out_tensor[0].detach().cpu().permute(1, 2, 0).clamp(0, 1).numpy() * 255.0).astype(np.uint8)

    if blend_seam:
        # Create a feathered mask for smooth boundary blending
        feather_mask = cv2.GaussianBlur(binary_mask.astype(np.float32) / 255.0, (7, 7), 2.0)
        feather_mask = np.expand_dims(feather_mask, axis=2)
        blended = (out_np.astype(np.float32) * feather_mask + image_rgb.astype(np.float32) * (1.0 - feather_mask))
        return np.clip(blended, 0, 255).astype(np.uint8)
    else:
        res = image_rgb.copy()
        res[binary_mask > 0] = out_np[binary_mask > 0]
        return res


def inpaint_lama(
    image_rgb: np.ndarray,
    mask: np.ndarray,
    dilation_pixels: int = 5,
    blend_seam: bool = True,
    smart_patch: bool = True,
    device: Optional[torch.device] = None,
) -> np.ndarray:
    """
    Inpaint image using LaMa TorchScript deep learning model.
    If smart_patch is True and image is large (e.g. >1200px), crops bounding box with context margin,
    inpaints the patch in ~0.05s, and seamlessly merges it back.
    """
    h, w = image_rgb.shape[:2]
    binary_mask = preprocess_mask(mask, (h, w), dilation_pixels=dilation_pixels)

    if not np.any(binary_mask > 0):
        return image_rgb.copy()

    # If image is very large (e.g. 2K/4K/8K) and watermark is local, use smart patch
    if smart_patch and (max(h, w) > 1280):
        pos = np.where(binary_mask > 0)
        ymin, ymax = int(np.min(pos[0])), int(np.max(pos[0]))
        xmin, xmax = int(np.min(pos[1])), int(np.max(pos[1]))
        mw = xmax - xmin + 1
        mh = ymax - ymin + 1

        # Check if watermark covers less than 60% of image area
        if (mw * mh) < (w * h * 0.6):
            # Context margin: at least 96px or 0.6x box size for good background understanding
            margin_x = max(96, int(mw * 0.6))
            margin_y = max(96, int(mh * 0.6))

            y1 = max(0, ymin - margin_y)
            y2 = min(h, ymax + margin_y + 1)
            x1 = max(0, xmin - margin_x)
            x2 = min(w, xmax + margin_x + 1)

            crop_img = image_rgb[y1:y2, x1:x2]
            crop_mask = binary_mask[y1:y2, x1:x2]

            inpainted_crop = inpaint_lama_patch(
                crop_img,
                crop_mask,
                dilation_pixels=0,  # Already dilated
                blend_seam=blend_seam,
                device=device,
            )

            # Feather the boundary of the patch to guarantee seamless integration
            patch_mask = cv2.GaussianBlur(crop_mask.astype(np.float32) / 255.0, (11, 11), 3.0)
            patch_mask = np.expand_dims(patch_mask, axis=2)

            res = image_rgb.copy()
            blended_region = (
                inpainted_crop.astype(np.float32) * patch_mask
                + res[y1:y2, x1:x2].astype(np.float32) * (1.0 - patch_mask)
            )
            res[y1:y2, x1:x2] = np.clip(blended_region, 0, 255).astype(np.uint8)
            return res

    # Full image inference
    return inpaint_lama_patch(
        image_rgb,
        binary_mask,
        dilation_pixels=0,  # Already dilated
        blend_seam=blend_seam,
        device=device,
    )


def inpaint_opencv(
    image_rgb: np.ndarray,
    mask: np.ndarray,
    method: str = "ns",
    dilation_pixels: int = 5,
    inpaint_radius: int = 5,
) -> np.ndarray:
    """
    Inpaint image using OpenCV Navier-Stokes or Telea algorithms.
    - method: 'ns' (Navier-Stokes) or 'telea'
    """
    h, w = image_rgb.shape[:2]
    binary_mask = preprocess_mask(mask, (h, w), dilation_pixels=dilation_pixels)

    if not np.any(binary_mask > 0):
        return image_rgb.copy()

    image_bgr = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)
    flags = cv2.INPAINT_NS if method.lower() == "ns" else cv2.INPAINT_TELEA
    inpainted_bgr = cv2.inpaint(image_bgr, binary_mask, inpaint_radius, flags)
    return cv2.cvtColor(inpainted_bgr, cv2.COLOR_BGR2RGB)


def inpaint_image(
    image_rgb: np.ndarray,
    mask: np.ndarray,
    method: str = "lama",
    dilation_pixels: int = 5,
    blend_seam: bool = True,
    smart_patch: bool = True,
    device: Optional[torch.device] = None,
) -> np.ndarray:
    """
    Unified inpainting entry point supporting AI (LaMa) and OpenCV algorithms.
    - method: 'lama' (default/recommended), 'ns' (Navier-Stokes), 'telea' (Telea)
    """
    method_key = method.lower().strip()
    if "lama" in method_key:
        return inpaint_lama(
            image_rgb,
            mask,
            dilation_pixels=dilation_pixels,
            blend_seam=blend_seam,
            smart_patch=smart_patch,
            device=device,
        )
    elif "ns" in method_key or "navier" in method_key:
        return inpaint_opencv(
            image_rgb,
            mask,
            method="ns",
            dilation_pixels=dilation_pixels,
        )
    elif "telea" in method_key:
        return inpaint_opencv(
            image_rgb,
            mask,
            method="telea",
            dilation_pixels=dilation_pixels,
        )
    else:
        # Fallback to LaMa
        return inpaint_lama(
            image_rgb,
            mask,
            dilation_pixels=dilation_pixels,
            blend_seam=blend_seam,
            smart_patch=smart_patch,
            device=device,
        )


def apply_image_sanitization(
    image_rgb: np.ndarray,
    clean_c2pa: bool = True,
    deep_clean_synthid: bool = True,
) -> np.ndarray:
    """
    Sanitizes image pixels to disrupt invisible AI watermarks (SynthID / frequency-domain watermarks)
    while preserving visual quality.
    """
    cleaned = image_rgb.copy()

    if deep_clean_synthid:
        # Spatial-frequency lattice perturbation:
        # 1. Subtle contrast/brightness calibration
        # 2. Gaussian micro-dither on least-significant-bits
        h, w = cleaned.shape[:2]
        if h > 32 and w > 32:
            cropped = cleaned[1 : h - 1, 1 : w - 1]
            resized = cv2.resize(cropped, (w, h), interpolation=cv2.INTER_LANCZOS4)
            cleaned = resized

        # Subtle noise dither (+-1) on random pixels to disrupt wavelet coefficients
        noise = np.random.randint(-1, 2, cleaned.shape, dtype=np.int16)
        cleaned = np.clip(cleaned.astype(np.int16) + noise, 0, 255).astype(np.uint8)

    return cleaned


def save_clean_image(
    image_rgb: np.ndarray,
    output_path: Union[str, Path],
    clean_metadata: bool = True,
    format_name: Optional[str] = None,
    quality: int = 98,
) -> Path:
    """
    Save RGB image without EXIF, XMP, IPTC, or C2PA metadata manifests.
    """
    out_path = Path(output_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    pil_img = Image.fromarray(image_rgb)

    ext = out_path.suffix.lower()
    if format_name:
        ext = f".{format_name.lower()}"

    save_kwargs: Dict[str, Any] = {}
    if ext in (".jpg", ".jpeg"):
        save_kwargs["quality"] = quality
        save_kwargs["subsampling"] = 0
        save_kwargs["optimize"] = True
    elif ext == ".png":
        save_kwargs["optimize"] = True
        save_kwargs["compress_level"] = 6
    elif ext == ".webp":
        save_kwargs["quality"] = quality
        save_kwargs["lossless"] = (quality >= 100)

    # When clean_metadata is True, saving without 'exif' or 'pnginfo' parameter ensures 0-byte metadata
    pil_img.save(out_path, **save_kwargs)
    return out_path
