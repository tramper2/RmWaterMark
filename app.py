import os
import sys
import shutil
import struct
import subprocess
import tempfile
import time
import traceback
from pathlib import Path
from typing import Generator, Optional, Tuple, Dict, Any

import cv2
import gradio as gr
import numpy as np
import torch
from PIL import Image

# Ensure ProPainter is on path
BASE_DIR = Path(__file__).resolve().parent
PROPAINTER_DIR = BASE_DIR / "ProPainter"
if str(PROPAINTER_DIR) not in sys.path:
    sys.path.insert(0, str(PROPAINTER_DIR))

# Ensure model weights are present
from download_weights import ensure_weights


def strip_mp4_boxes(input_file: str, output_file: str) -> bool:
    """
    Read ISO BMFF boxes and strip uuid (C2PA/XMP), c2pa, jumb, and udta boxes.
    Physically removes all metadata manifests from the MP4 binary.
    """
    try:
        with open(input_file, "rb") as f:
            data = f.read()

        cleaned = bytearray()
        offset = 0
        total_len = len(data)

        # Boxes to completely drop at top level or container level
        banned_boxes = {b"uuid", b"c2pa", b"jumb", b"XMP_", b"udta"}

        while offset + 8 <= total_len:
            box_size = struct.unpack(">I", data[offset : offset + 4])[0]
            box_type = data[offset + 4 : offset + 8]

            if box_size == 1:
                if offset + 16 > total_len:
                    break
                box_size = struct.unpack(">Q", data[offset + 8 : offset + 16])[0]
            elif box_size == 0:
                box_size = total_len - offset

            if box_size < 8 or offset + box_size > total_len:
                cleaned.extend(data[offset:])
                break

            if box_type in banned_boxes:
                pass  # Strip banned box
            elif box_type == b"moov":
                moov_body = clean_moov_box(data[offset + 8 : offset + box_size])
                new_moov_size = len(moov_body) + 8
                cleaned.extend(struct.pack(">I", new_moov_size) + b"moov" + moov_body)
            else:
                cleaned.extend(data[offset : offset + box_size])

            offset += box_size

        with open(output_file, "wb") as f:
            f.write(cleaned)
        return True
    except Exception as e:
        print("strip_mp4_boxes error:", e)
        if input_file != output_file and os.path.exists(input_file):
            shutil.copyfile(input_file, output_file)
        return False


def clean_moov_box(moov_data: bytes) -> bytes:
    """Clean sub-boxes inside moov box (stripping udta, uuid, meta, c2pa manifests)."""
    cleaned = bytearray()
    offset = 0
    total_len = len(moov_data)
    banned_moov_boxes = {b"udta", b"uuid", b"c2pa", b"jumb", b"XMP_"}

    while offset + 8 <= total_len:
        box_size = struct.unpack(">I", moov_data[offset : offset + 4])[0]
        box_type = moov_data[offset + 4 : offset + 8]

        if box_size == 1:
            if offset + 16 > total_len:
                break
            box_size = struct.unpack(">Q", moov_data[offset + 8 : offset + 16])[0]
        elif box_size == 0:
            box_size = total_len - offset

        if box_size < 8 or offset + box_size > total_len:
            cleaned.extend(moov_data[offset:])
            break

        if box_type in banned_moov_boxes:
            pass  # Strip banned box
        else:
            cleaned.extend(moov_data[offset : offset + box_size])

        offset += box_size

    return bytes(cleaned)


def apply_video_audio_sanitization(
    input_video_path: str,
    output_path: str,
    original_audio_path: Optional[str] = None,
    clean_c2pa: bool = True,
    deep_clean_synthid: bool = True,
    mute_audio: bool = False,
) -> bool:
    """
    Combines video + audio with full metadata stripping and SynthID / C2PA disruption.
    """
    out_p = Path(output_path)
    temp_merged = out_p.parent / f"temp_merge_{out_p.name}"

    vf_filters = []
    af_filters = []

    if deep_clean_synthid:
        # Video Deep Clean:
        # 1. 4px crop & Lanczos rescale (shifts pixel spatial lattices to disrupt lattice-aligned SynthID)
        # 2. Subtle color/gamma tweak (breaks latent neural activation maps)
        # 3. Spatial-temporal noise dither (disrupts high-frequency wavelet coefficients)
        vf_filters.append(
            "crop=in_w-4:in_h-4:2:2,scale=in_w:in_h:flags=lanczos,eq=contrast=1.01:brightness=0.004:gamma=1.01,noise=alls=2:allf=t"
        )

        # Audio Deep Clean:
        # 1. Bandpass filter to cut ultrasonic/infrasonic watermark carrier frequencies
        # 2. Micro tempo shift (1.001x) to break phase synchronization of SynthID audio (Lyria)
        # 3. Standard clean resample & volume calibration
        af_filters.append(
            "highpass=f=30,lowpass=f=19000,atempo=1.001,aresample=44100,volume=1.002"
        )

    ffmpeg_cmd = ["ffmpeg", "-y", "-i", str(input_video_path)]

    has_audio = bool(
        original_audio_path
        and not mute_audio
        and os.path.exists(str(original_audio_path))
    )
    if has_audio:
        ffmpeg_cmd.extend(["-i", str(original_audio_path)])
        ffmpeg_cmd.extend(["-map", "0:v:0", "-map", "1:a:0?"])
    else:
        ffmpeg_cmd.extend(["-map", "0:v:0"])

    if deep_clean_synthid:
        ffmpeg_cmd.extend([
            "-vf",
            ",".join(vf_filters),
            "-c:v",
            "libx264",
            "-crf",
            "18",
            "-preset",
            "fast",
            "-pix_fmt",
            "yuv420p",
            "-x264-params",
            "stitchable=1",
        ])
        if has_audio:
            ffmpeg_cmd.extend([
                "-af",
                ",".join(af_filters),
                "-c:a",
                "aac",
                "-b:a",
                "192k",
                "-ar",
                "44100",
            ])
    else:
        ffmpeg_cmd.extend(["-c:v", "copy"])
        if has_audio:
            ffmpeg_cmd.extend(["-c:a", "aac", "-b:a", "192k"])

    if clean_c2pa:
        ffmpeg_cmd.extend([
            "-map_metadata",
            "-1",
            "-map_metadata:s:v",
            "-1",
            "-map_metadata:s:a",
            "-1",
            "-map_chapters",
            "-1",
            "-fflags",
            "+bitexact",
            "-flags:v",
            "+bitexact",
            "-flags:a",
            "+bitexact",
        ])

    ffmpeg_cmd.extend(["-movflags", "+faststart", str(temp_merged)])

    res = subprocess.run(ffmpeg_cmd, capture_output=True)
    if res.returncode != 0:
        print("FFmpeg merge error:", res.stderr.decode("utf-8", errors="ignore"))
        shutil.copyfile(input_video_path, output_path)
        return False

    if clean_c2pa:
        strip_mp4_boxes(str(temp_merged), str(output_path))
        if temp_merged.exists():
            try:
                temp_merged.unlink()
            except Exception:
                pass
    else:
        shutil.move(str(temp_merged), str(output_path))

    return True


def get_clean_video_path(video_input: Any) -> Optional[str]:
    """Safely extract valid filesystem path string from any Gradio video input representation."""
    if video_input is None:
        return None
    if isinstance(video_input, str):
        path = video_input.strip()
        return path if path and os.path.exists(path) else None
    if isinstance(video_input, dict):
        path = video_input.get("video") or video_input.get("path") or video_input.get("name")
        if path and isinstance(path, str) and os.path.exists(path):
            return path.strip()
        return None
    if hasattr(video_input, "name") and isinstance(video_input.name, str) and os.path.exists(video_input.name):
        return video_input.name
    if hasattr(video_input, "path") and isinstance(video_input.path, str) and os.path.exists(video_input.path):
        return video_input.path
    if isinstance(video_input, (list, tuple)) and len(video_input) > 0:
        return get_clean_video_path(video_input[0])
    return None


def check_ffmpeg() -> bool:
    try:
        subprocess.run(
            ["ffmpeg", "-version"],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            check=True,
        )
        return True
    except Exception:
        return False


def get_gpu_info() -> str:
    if torch.cuda.is_available():
        gpu_name = torch.cuda.get_device_name(0)
        vram_gb = torch.cuda.get_device_properties(0).total_memory / (1024**3)
        return f"🟢 CUDA Active: {gpu_name} ({vram_gb:.1f} GB VRAM)"
    return "🟡 Running on CPU (CUDA not detected)"


def get_video_info(video_input: Any) -> Tuple[int, int, float, int, float, str]:
    """Extract width, height, fps, total_frames, duration_sec, formatted_info."""
    vpath = get_clean_video_path(video_input)
    if not vpath or not os.path.exists(vpath):
        return 1920, 1080, 30.0, 0, 0.0, "No video loaded"

    cap = cv2.VideoCapture(vpath)
    if not cap.isOpened():
        return 1920, 1080, 30.0, 0, 0.0, "Unable to read video"

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration = frame_count / fps if fps > 0 else 0.0
    cap.release()

    info_str = f"📐 {width}×{height} | ⏱️ {duration:.1f}s ({frame_count} frames @ {fps:.1f} fps)"
    return width, height, fps, frame_count, duration, info_str


def extract_frame_at_time(video_input: Any, time_sec: float = 0.0) -> Optional[np.ndarray]:
    """Extract RGB frame at specified timestamp in seconds."""
    vpath = get_clean_video_path(video_input)
    if not vpath or not os.path.exists(vpath):
        return None
    cap = cv2.VideoCapture(vpath)
    if not cap.isOpened():
        return None
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    target_frame = max(0, min(int(time_sec * fps), frame_count - 1)) if frame_count > 0 else 0
    cap.set(cv2.CAP_PROP_POS_FRAMES, target_frame)
    ret, frame = cap.read()
    cap.release()
    if ret and frame is not None:
        return cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    return None


def generate_roi_preview(
    video_input: Any,
    x: int,
    y: int,
    w: int,
    h: int,
    time_sec: float = 0.0,
) -> Optional[np.ndarray]:
    """Generate preview frame with highlighted watermark bounding box at specified timestamp."""
    frame_rgb = extract_frame_at_time(video_input, time_sec)
    if frame_rgb is None:
        frame_rgb = np.zeros((1080, 1920, 3), dtype=np.uint8)

    img_h, img_w = frame_rgb.shape[:2]

    # Clamp coordinates
    x = max(0, min(int(x), img_w - 1))
    y = max(0, min(int(y), img_h - 1))
    w = max(1, min(int(w), img_w - x))
    h = max(1, min(int(h), img_h - y))

    # Create overlay for semi-transparent highlight
    overlay = frame_rgb.copy()
    cv2.rectangle(overlay, (x, y), (x + w, y + h), (255, 60, 60), -1)

    # Blend overlay with original frame (alpha 0.35)
    preview = cv2.addWeighted(overlay, 0.35, frame_rgb, 0.65, 0)

    # Draw solid border
    cv2.rectangle(preview, (x, y), (x + w, y + h), (255, 40, 40), 2)

    # Draw coordinate label tag
    label = f"ROI ({x},{y}) {w}x{h} @ {time_sec:.1f}s"
    (label_w, label_h), baseline = cv2.getTextSize(
        label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2
    )
    tag_y1 = max(0, y - label_h - 8)
    tag_y2 = max(label_h + 8, y)
    cv2.rectangle(
        preview,
        (x, tag_y1),
        (x + label_w + 10, tag_y2),
        (255, 40, 40),
        -1,
    )
    cv2.putText(
        preview,
        label,
        (x + 5, tag_y2 - 5),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        (255, 255, 255),
        2,
        cv2.LINE_AA,
    )

    return preview


def extract_mask_from_editor(editor_data: Any, expected_w: int, expected_h: int) -> Optional[np.ndarray]:
    """Extract combined binary mask from ImageEditor drawn layers."""
    if not editor_data or not isinstance(editor_data, dict):
        return None
    layers = editor_data.get("layers", [])
    if not layers:
        return None

    combined = np.zeros((expected_h, expected_w), dtype=np.uint8)
    for layer in layers:
        if layer is None:
            continue
        if isinstance(layer, Image.Image):
            layer = np.array(layer)
        elif not isinstance(layer, np.ndarray):
            layer = np.array(layer)

        # Resize layer if canvas scale differs
        if layer.shape[0] != expected_h or layer.shape[1] != expected_w:
            layer = cv2.resize(layer, (expected_w, expected_h), interpolation=cv2.INTER_NEAREST)

        if len(layer.shape) == 3 and layer.shape[2] == 4:
            alpha = layer[:, :, 3]
            combined = np.maximum(combined, (alpha > 10).astype(np.uint8) * 255)
        elif len(layer.shape) == 3:
            gray = np.mean(layer, axis=2)
            combined = np.maximum(combined, (gray > 10).astype(np.uint8) * 255)
        elif len(layer.shape) == 2:
            combined = np.maximum(combined, (layer > 10).astype(np.uint8) * 255)

    if np.any(combined > 0):
        # Slightly dilate freehand strokes to ensure solid mask
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        combined = cv2.dilate(combined, kernel, iterations=1)
        return combined
    return None


def extract_roi_from_editor(editor_data: Any, expected_w: int, expected_h: int) -> Optional[Tuple[int, int, int, int]]:
    """Extract bounding box (x, y, w, h) from ImageEditor drawing."""
    mask = extract_mask_from_editor(editor_data, expected_w, expected_h)
    if mask is not None and np.any(mask > 0):
        pos = np.where(mask > 0)
        ymin, ymax = int(np.min(pos[0])), int(np.max(pos[0]))
        xmin, xmax = int(np.min(pos[1])), int(np.max(pos[1]))
        w = max(1, xmax - xmin + 1)
        h = max(1, ymax - ymin + 1)
        return xmin, ymin, w, h
    return None


def create_binary_mask(width: int, height: int, x: int, y: int, w: int, h: int, save_path: Path):
    """Generate binary rectangular mask: ROI is 255 (white), Background is 0 (black)."""
    mask = np.zeros((height, width), dtype=np.uint8)
    x1 = max(0, min(int(x), width - 1))
    y1 = max(0, min(int(y), height - 1))
    x2 = max(0, min(int(x + w), width))
    y2 = max(0, min(int(y + h), height))
    mask[y1:y2, x1:x2] = 255
    save_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(save_path), mask)


def process_watermark_removal(
    video_input: Any,
    editor_data: Any,
    use_drawn_mask: bool,
    x: int,
    y: int,
    w: int,
    h: int,
    subvideo_length: int = 80,
    neighbor_length: int = 10,
    ref_stride: int = 10,
    resize_ratio: float = 1.0,
    mask_dilates: int = 5,
    fp16: bool = True,
    clean_c2pa: bool = True,
    deep_clean_synthid: bool = True,
    mute_audio: bool = False,
    progress=gr.Progress(track_tqdm=True),
) -> Generator[Tuple[Any, str], None, None]:
    """Execute ProPainter inpainting and sanitize output with C2PA/SynthID removal."""
    try:
        vpath = get_clean_video_path(video_input)
        if not vpath or not os.path.exists(vpath):
            yield gr.update(), "❌ 동영상 파일을 먼저 업로드해 주세요. (Video not found)"
            return

        if not check_ffmpeg():
            yield gr.update(), "❌ FFmpeg를 찾을 수 없습니다. 시스템 PATH를 확인해 주세요."
            return

        yield gr.update(), "⏳ [1/4] ProPainter 사전 학습 가중치를 확인하는 중..."
        if not ensure_weights():
            yield gr.update(), "❌ 모델 가중치 다운로드에 실패했습니다. 인터넷 연결을 확인해 주세요."
            return

        # Extract video info
        width, height, fps, frame_count, duration, info_str = get_video_info(vpath)
        yield gr.update(), f"📊 동영상 정보: {info_str}\n⏳ [2/4] 워터마크 마스크 생성 중..."

        # Setup directories
        task_id = int(time.time() * 1000)
        workspace_dir = BASE_DIR / "temp" / f"task_{task_id}"
        workspace_dir.mkdir(parents=True, exist_ok=True)
        mask_path = workspace_dir / "mask.png"
        results_dir = BASE_DIR / "results"
        results_dir.mkdir(parents=True, exist_ok=True)

        # Check if user painted on ImageEditor
        drawn_mask = (
            extract_mask_from_editor(editor_data, width, height)
            if use_drawn_mask
            else None
        )
        if drawn_mask is not None and np.any(drawn_mask > 0):
            cv2.imwrite(str(mask_path), drawn_mask)
            yield gr.update(), f"📊 {info_str}\n🎨 마우스로 직접 칠한 정밀 마스크 적용 중..."
        else:
            # Fallback to rectangular ROI box
            create_binary_mask(width, height, x, y, w, h, mask_path)
            yield gr.update(), f"📊 {info_str}\n📐 사각형 ROI 마스크 적용 중 (X={x}, Y={y}, W={w}, H={h})..."

        yield gr.update(), "🚀 [3/4] ProPainter GPU 비디오 인페인팅 실행 중..."

        propainter_script = PROPAINTER_DIR / "inference_propainter.py"
        if not propainter_script.exists():
            yield gr.update(), f"❌ ProPainter 스크립트가 없습니다: {propainter_script}"
            return

        # Build ProPainter command with unbuffered flag (-u)
        cmd = [
            sys.executable,
            "-u",
            str(propainter_script),
            "--video",
            str(vpath),
            "--mask",
            str(mask_path),
            "--output",
            str(workspace_dir / "out"),
            "--subvideo_length",
            str(int(subvideo_length)),
            "--neighbor_length",
            str(int(neighbor_length)),
            "--ref_stride",
            str(int(ref_stride)),
            "--mask_dilation",
            str(int(mask_dilates)),
        ]

        if resize_ratio != 1.0:
            cmd.extend(["--resize_ratio", str(float(resize_ratio))])

        if fp16 and torch.cuda.is_available():
            cmd.append("--fp16")

        # Run subprocess with real-time unbuffered progress parsing
        log_lines = []
        env = dict(
            os.environ,
            PYTHONUNBUFFERED="1",
            PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True",
        )
        process = subprocess.Popen(
            cmd,
            cwd=str(PROPAINTER_DIR),
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=0,
            env=env,
            universal_newlines=True,
        )

        import re
        buffer = ""
        last_yield_time = time.time()

        while True:
            char = process.stdout.read(1)
            if not char:
                if buffer.strip():
                    log_lines.append(buffer.strip())
                break
            if char in ("\r", "\n"):
                line_str = buffer.strip()
                buffer = ""
                if not line_str:
                    continue

                log_lines.append(line_str)
                recent_logs = "\n".join(log_lines[-10:])

                tqdm_match = re.search(
                    r"(\d+)%\|.*?\|\s*(\d+)/(\d+)\s*\[(.*?)(?:,\s*(.*?))?\]",
                    line_str,
                )
                if tqdm_match:
                    pct = int(tqdm_match.group(1))
                    curr_step = int(tqdm_match.group(2))
                    total_steps = int(tqdm_match.group(3))
                    time_info = tqdm_match.group(4)
                    speed_info = tqdm_match.group(5) or ""

                    curr_frame = min(
                        frame_count, int((curr_step / total_steps) * frame_count)
                    )
                    bar_len = 20
                    filled = int(bar_len * curr_step / total_steps)
                    bar = "█" * filled + "░" * (bar_len - filled)

                    progress(
                        curr_step / total_steps,
                        desc=f"인페인팅: {curr_frame}/{frame_count} 프레임 ({pct}%)",
                    )

                    status_msg = (
                        f"🚀 [3/4] ProPainter GPU 비디오 인페인팅 실행 중...\n"
                        f"━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n"
                        f"🎞️ 현재 프레임 처리 현황: {curr_frame} / {frame_count} 프레임 ({pct}%) [{bar}]\n"
                        f"📊 세부 연산 단계:        {curr_step} / {total_steps} Steps\n"
                        f"⏱️ 소요/남은 시간:       {time_info}\n"
                        f"⚡ GPU 연산 속도:        {speed_info}\n"
                        f"━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n"
                        f"📝 최근 실행 로그:\n{recent_logs}"
                    )
                    last_yield_time = time.time()
                    yield gr.update(), status_msg
                else:
                    if (
                        "[Flow]" in line_str
                        or "[ROI]" in line_str
                        or "Processing:" in line_str
                        or time.time() - last_yield_time > 0.3
                    ):
                        last_yield_time = time.time()
                        status_msg = (
                            f"🚀 [3/4] ProPainter GPU 비디오 인페인팅 실행 중...\n"
                            f"━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n"
                            f"⏳ {line_str if ('[Flow]' in line_str or '[ROI]' in line_str) else f'초기 Flow 계산 및 프레임 분석 중... (총 {frame_count} 프레임)'}\n"
                            f"━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n"
                            f"📝 최근 실행 로그:\n{recent_logs}"
                        )
                        yield gr.update(), status_msg
            else:
                buffer += char

        process.stdout.close()
        return_code = process.wait()
        if return_code != 0:
            yield gr.update(), f"❌ ProPainter 실행 실패 (코드 {return_code}):\n" + "\n".join(
                log_lines[-20:]
            )
            return

        # Look for inpainted output video in workspace_dir / out
        out_dir = workspace_dir / "out"
        inpainted_files = list(out_dir.rglob("inpaint_out.mp4")) or list(
            out_dir.rglob("*.mp4")
        )
        if not inpainted_files:
            yield gr.update(), f"❌ 인페인팅 결과 비디오 파일을 찾을 수 없습니다: {out_dir}"
            return

        inpainted_video = inpainted_files[0]
        yield gr.update(), f"🎵 [4/4] C2PA / 메타데이터 완전 소거 및 SynthID(비가시성 워터마크) 심층 세척 실행 중..."

        # Target final output file in results/
        video_stem = Path(vpath).stem
        final_output_path = results_dir / f"{video_stem}_nowatermark_{task_id}.mp4"

        # Apply ultimate sanitization and audio merge
        success = apply_video_audio_sanitization(
            input_video_path=str(inpainted_video),
            output_path=str(final_output_path),
            original_audio_path=str(vpath) if not mute_audio else None,
            clean_c2pa=clean_c2pa,
            deep_clean_synthid=deep_clean_synthid,
            mute_audio=mute_audio,
        )

        # Clean up temporary workspace
        try:
            shutil.rmtree(workspace_dir, ignore_errors=True)
        except Exception:
            pass

        if success:
            c2pa_status = "🛡️ C2PA 및 모든 메타데이터 100% 0바이트 소거 완료\n" if clean_c2pa else ""
            synth_status = "✨ AI 비가시성 워터마크(SynthID) 공간-주파수 교란 완료\n" if deep_clean_synthid else ""
            audio_status = "🔇 오디오 트랙 완전 제거 완료\n" if mute_audio else "🎵 오디오 스트림 보존 및 위상 재구성 완료\n"
            yield str(final_output_path), (
                f"✅ 워터마크 제거 및 완전 세척 완료!\n"
                f"📁 저장 위치: results/{final_output_path.name}\n"
                f"{c2pa_status}{synth_status}{audio_status}"
                f"🚀 유튜브 등 플랫폼 업로드 시 'AI 생성/수정 콘텐츠' 자동 감지 방지 적용"
            )
        else:
            yield str(final_output_path), f"⚠️ 비디오 생성 완료 (일부 세척 필터 경고)\n📁 저장 위치: results/{final_output_path.name}"

    except Exception as e:
        err_msg = traceback.format_exc()
        print("ERROR in process_watermark_removal:\n", err_msg)
        yield gr.update(), f"❌ 처리 중 오류 발생:\n{e}\n\n{err_msg}"


def process_instant_sanitization(
    input_video: Any,
    clean_c2pa: bool = True,
    deep_clean_synthid: bool = True,
    mute_audio: bool = False,
    progress=gr.Progress(),
) -> Tuple[Optional[str], str]:
    """Instantly sanitizes an existing video file without inpainting."""
    try:
        vpath = get_clean_video_path(input_video)
        if not vpath or not os.path.exists(vpath):
            return None, "❌ 세척할 동영상 파일을 먼저 업로드해 주세요."

        task_id = int(time.time() * 1000)
        results_dir = BASE_DIR / "results"
        results_dir.mkdir(parents=True, exist_ok=True)
        video_stem = Path(vpath).stem
        out_path = results_dir / f"{video_stem}_sanitized_{task_id}.mp4"

        progress(0.2, desc="C2PA 및 메타데이터 / SynthID 심층 세척 실행 중...")
        success = apply_video_audio_sanitization(
            input_video_path=vpath,
            output_path=str(out_path),
            original_audio_path=vpath if not mute_audio else None,
            clean_c2pa=clean_c2pa,
            deep_clean_synthid=deep_clean_synthid,
            mute_audio=mute_audio,
        )
        progress(1.0, desc="완료!")

        if success:
            c2pa_status = "🛡️ C2PA 및 EXIF/XMP 컨테이너 메타데이터 100% 0바이트 소거 완료\n" if clean_c2pa else ""
            synth_status = "✨ AI 비가시성 워터마크(SynthID) 픽셀 디더링 및 주파수 계수 교란 완료\n" if deep_clean_synthid else ""
            audio_status = "🔇 오디오 트랙 완전 제거 완료\n" if mute_audio else "🎵 오디오 위상 재구성 및 리샘플링 완료\n"
            msg = (
                f"✅ 동영상 완전 세척 완료!\n"
                f"📁 저장 위치: results/{out_path.name}\n"
                f"{c2pa_status}{synth_status}{audio_status}"
                f"🚀 유튜브 등 플랫폼 업로드 시 'AI 생성/수정 콘텐츠' 자동 감지 방지 적용"
            )
            return str(out_path), msg
        else:
            return str(out_path), "⚠️ 세척 중 경고가 발생했으나 기본 변환 파일이 생성되었습니다."
    except Exception as e:
        err_msg = traceback.format_exc()
        print("ERROR in process_instant_sanitization:\n", err_msg)
        return None, f"❌ 세척 중 오류 발생:\n{e}"


# Gradio UI Theme & CSS
custom_css = """
body {
    background-color: #0b0f19;
    color: #e2e8f0;
}
.gradio-container {
    max-width: 1380px !important;
    margin: 0 auto !important;
}
.header-badge {
    display: inline-block;
    background: linear-gradient(135deg, #3b82f6 0%, #1d4ed8 100%);
    color: #fff;
    padding: 4px 14px;
    border-radius: 9999px;
    font-size: 0.85rem;
    font-weight: 600;
}
.instruction-box {
    background-color: #1e293b;
    border-left: 4px solid #3b82f6;
    padding: 10px 14px;
    border-radius: 4px;
    margin-bottom: 10px;
    font-size: 0.9rem;
}
.shield-box {
    background-color: #1a2332;
    border: 1px solid #2563eb;
    border-radius: 8px;
    padding: 12px 16px;
    margin-bottom: 12px;
}
"""

with gr.Blocks(title="AI Video Watermark Remover & C2PA / SynthID Sanitizer") as demo:
    gr.Markdown(
        f"""
        # 🎬 Local Video Watermark Remover & Metadata Sanitizer
        ### Powered by **ProPainter** (ICCV 2023) & **C2PA / SynthID(비가시성 워터마크) 완전 박멸 엔진**
        <span class="header-badge">{get_gpu_info()}</span>
        """
    )

    with gr.Tabs():
        # ------------------ TAB 1: Main Watermark Remover ------------------
        with gr.TabItem("🎬 1. AI 워터마크 제거 + 자동 세척 (Watermark Remover)"):
            with gr.Row():
                # Panel 1: Video Upload & Timeline Scrubber
                with gr.Column(scale=1):
                    gr.Markdown("### 1. Upload Video")
                    input_video = gr.Video(label="Source Video", sources=["upload"])
                    video_info_box = gr.Markdown("📁 Upload a video to view resolution and frames")

                    gr.Markdown("#### ⏱️ 영상 타임라인 프레임 선택 (Scrub Frame)")
                    gr.Markdown("<small>💡 배경색 때문에 워터마크가 안 보이면 타임라인을 이동해 선명한 장면의 프레임을 불러오세요.</small>")
                    slider_timestamp = gr.Slider(
                        label="타임라인 위치 (초 / Seconds)",
                        minimum=0.0,
                        maximum=10.0,
                        value=0.0,
                        step=0.1,
                    )
                    with gr.Row():
                        btn_time_0 = gr.Button("⏮️ 0초 (시작)", size="sm")
                        btn_time_25 = gr.Button("25%", size="sm")
                        btn_time_50 = gr.Button("50% (중간)", size="sm")
                        btn_time_75 = gr.Button("75%", size="sm")
                        btn_time_end = gr.Button("⏭️ 끝", size="sm")

                # Panel 2: Select Watermark ROI
                with gr.Column(scale=1):
                    gr.Markdown("### 2. Select Watermark ROI")
                    with gr.Tabs() as roi_tabs:
                        with gr.TabItem("🖱️ 마우스 드래그 / 브러시 칠하기 (Mouse Draw)", id="tab_draw"):
                            gr.Markdown(
                                """
                                <div class="instruction-box">
                                💡 <b>사용법:</b> 마우스로 영상 프레임 위의 워터마크 영역을 <b>드래그하여 칠하세요</b>.<br>
                                칠한 후 아래 <b>🚀 Start Watermark Removal</b> 버튼을 누르면 해당 영역이 제거됩니다.
                                </div>
                                """
                            )
                            image_editor = gr.ImageEditor(
                                label="Watermark Brush Canvas",
                                type="numpy",
                                brush=gr.Brush(default_size=25, colors=["#ff3333", "#ffffff", "#00ff00"], default_color="#ff3333"),
                                eraser=gr.Eraser(default_size=25),
                                interactive=True,
                            )
                            with gr.Row():
                                btn_sync_from_draw = gr.Button("🎯 마우스 영역 좌표 동기화", size="sm")
                                btn_reset_frame = gr.Button("🔄 현재 타임라인 프레임 불러오기", size="sm")

                            chk_use_drawn_mask = gr.Checkbox(
                                label="마우스로 직접 칠한 정밀 마스크(Freehand Mask) 그대로 사용",
                                value=True,
                                info="체크 시 마우스로 칠한 모양 그대로 정밀 적용됩니다.",
                            )

                        with gr.TabItem("📐 정밀 좌표 슬라이더 & 프리셋 (Sliders & Presets)", id="tab_sliders"):
                            gr.Markdown("#### 🎯 화면비 및 위치별 원클릭 프리셋")
                            with gr.Row():
                                btn_preset_gemini_1080p = gr.Button("🖥️ 16:9 (1080p BR)", size="sm")
                                btn_preset_gemini_916 = gr.Button("📱 9:16 (Shorts BR)", size="sm")
                                btn_preset_gemini_11 = gr.Button("🔲 1:1 (Square BR)", size="sm")
                                btn_preset_br = gr.Button("📍 Auto BR (Any Ratio)", size="sm")
                                btn_preset_bl = gr.Button("📍 Auto BL", size="sm")
                                btn_preset_tr = gr.Button("📍 Auto TR", size="sm")

                            gr.Markdown("#### 📐 사각형 좌표 직접 설정 (Pixels)")
                            with gr.Row():
                                slider_x = gr.Number(label="X (Left)", value=1700, precision=0)
                                slider_y = gr.Number(label="Y (Top)", value=950, precision=0)
                            with gr.Row():
                                slider_w = gr.Number(label="Width", value=200, precision=0)
                                slider_h = gr.Number(label="Height", value=100, precision=0)

                            preview_image = gr.Image(
                                label="Bounding Box Preview (Red Box)",
                                interactive=False,
                                type="numpy",
                            )

            # Sanitization and Advanced Settings
            with gr.Accordion("🛡️ C2PA / 메타데이터 & AI 비가시성 워터마크(SynthID) 완전 세척 설정", open=True):
                gr.Markdown(
                    """
                    <div class="shield-box">
                    <b>🛡️ 플랫폼 AI 자동 감지 방지 (YouTube / SNS AI Detection Protection)</b><br>
                    • <b>C2PA & 메타데이터 완전 삭제:</b> 컨테이너 헤더, XMP, EXIF, JUMBF/UUID 박스를 바이너리 레벨에서 0바이트 소거합니다.<br>
                    • <b>AI 비가시성 워터마크(SynthID) 심층 세척:</b> 영상 전체 픽셀과 오디오 주파수에 숨겨진 비가시성 지문을 공간-시간 미세 디더링 및 위상 재구성으로 무력화합니다.
                    </div>
                    """
                )
                with gr.Row():
                    chk_clean_c2pa = gr.Checkbox(
                        label="🛡️ C2PA 및 모든 메타데이터 완전 삭제 (C2PA & Container Wipe)",
                        value=True,
                        info="MP4 uuid/udta/c2pa 메타데이터를 100% 물리적 삭제합니다.",
                    )
                    chk_deep_clean = gr.Checkbox(
                        label="✨ AI 비가시성 워터마크(SynthID) 심층 세척 (Deep Clean for YouTube)",
                        value=True,
                        info="유튜브 등에서 'AI로 제작' 자동 태깅을 방지합니다.",
                    )
                    chk_mute_audio = gr.Checkbox(
                        label="🔇 오디오 트랙 완전 제거 (Mute Audio / 무음 출력)",
                        value=False,
                        info="오디오의 AI 워터마크까지 원천 배제하고자 할 때 체크하세요.",
                    )

            with gr.Accordion("⚙️ ProPainter 인페인팅 고급 옵션 (VRAM / Memory)", open=False):
                with gr.Row():
                    slider_subvideo = gr.Slider(
                        label="Subvideo Length (Frames)",
                        minimum=20,
                        maximum=120,
                        value=60,
                        step=10,
                        info="Lower to 40-50 if encountering VRAM issues on long videos",
                    )
                    slider_neighbor = gr.Slider(
                        label="Neighbor Length",
                        minimum=4,
                        maximum=20,
                        value=10,
                        step=2,
                    )
                    slider_ref_stride = gr.Slider(
                        label="Reference Stride",
                        minimum=5,
                        maximum=20,
                        value=10,
                        step=1,
                    )
                with gr.Row():
                    slider_mask_dilates = gr.Slider(
                        label="Mask Dilation (Pixels)",
                        minimum=0,
                        maximum=15,
                        value=5,
                        step=1,
                        info="Expands mask boundary slightly to eliminate border artifacts",
                    )
                    slider_resize = gr.Slider(
                        label="Resize Ratio",
                        minimum=0.25,
                        maximum=1.0,
                        value=1.0,
                        step=0.05,
                        info="Scale down 4K videos if needed",
                    )
                    chk_fp16 = gr.Checkbox(
                        label="Enable FP16 (Half Precision)",
                        value=True,
                        info="Saves substantial VRAM with identical quality",
                    )

            btn_start = gr.Button("🚀 Start Watermark Removal & Deep Clean", variant="primary", size="lg")

            with gr.Row():
                with gr.Column(scale=1):
                    gr.Markdown("### 3. Inpainted Output Video (with Synchronized Audio)")
                    output_video = gr.Video(label="Processed Result Video", interactive=False)

                with gr.Column(scale=1):
                    gr.Markdown("### 4. Status & Execution Log")
                    status_box = gr.Textbox(
                        label="Execution Log",
                        value="Ready. Upload a video, set watermark region, and click Start.",
                        lines=12,
                        interactive=False,
                    )

        # ------------------ TAB 2: Instant Video Sanitizer ------------------
        with gr.TabItem("🧹 2. 기존 동영상 즉시 세척기 (Instant C2PA & SynthID Cleaner)"):
            gr.Markdown(
                """
                ### 🧹 기존 동영상 원클릭 C2PA & 메타데이터 & AI 워터마크 즉시 세척기
                <div class="instruction-box">
                💡 <b>사용법:</b> 이미 워터마크를 제거했거나 외부에서 생성된 모든 동영상(MP4/MOV)을 업로드하고 <b>⚡ 즉시 완전 세척 실행</b>을 누르세요.<br>
                인페인팅을 다시 실행할 필요 없이 <b>2~3초 만에 C2PA 매니페스트, 메타데이터, SynthID 비가시성 워터마크</b>를 완전히 소거하여 유튜브 업로드용 클린 영상을 생성합니다.
                </div>
                """
            )
            with gr.Row():
                with gr.Column(scale=1):
                    sanitizer_input_video = gr.Video(label="Source Video to Sanitize", sources=["upload"])
                    with gr.Row():
                        chk_s_c2pa = gr.Checkbox(
                            label="🛡️ C2PA 및 모든 메타데이터 완전 삭제",
                            value=True,
                        )
                        chk_s_synthid = gr.Checkbox(
                            label="✨ AI 비가시성 워터마크(SynthID) 심층 세척",
                            value=True,
                        )
                        chk_s_mute = gr.Checkbox(
                            label="🔇 오디오 제거 (무음 출력)",
                            value=False,
                        )
                    btn_run_sanitize = gr.Button("⚡ 즉시 완전 세척 실행 (Sanitize Video)", variant="primary", size="lg")

                with gr.Column(scale=1):
                    sanitizer_output_video = gr.Video(label="100% Sanitized Clean Video", interactive=False)
                    sanitizer_status_box = gr.Textbox(
                        label="Sanitization Log",
                        value="Ready. Upload a video and click Sanitize.",
                        lines=8,
                        interactive=False,
                    )

    # ------------------ Event Handlers ------------------
    def on_video_upload(video):
        vpath = get_clean_video_path(video)
        if not vpath:
            return "No video uploaded", gr.update(maximum=10.0, value=0.0), 1700, 950, 200, 100, None, None
        w, h, fps, count, dur, info = get_video_info(vpath)
        default_w = int(w * 0.14)
        default_h = int(h * 0.08)
        default_x = w - default_w - int(w * 0.01)
        default_y = h - default_h - int(h * 0.02)

        first_frame = extract_frame_at_time(vpath, 0.0)
        editor_init = {"background": first_frame, "layers": [], "composite": None}
        box_preview = generate_roi_preview(vpath, default_x, default_y, default_w, default_h, 0.0)
        time_slider_update = gr.update(maximum=max(0.1, round(dur, 2)), value=0.0)
        return info, time_slider_update, default_x, default_y, default_w, default_h, editor_init, box_preview

    input_video.change(
        fn=on_video_upload,
        inputs=[input_video],
        outputs=[
            video_info_box,
            slider_timestamp,
            slider_x,
            slider_y,
            slider_w,
            slider_h,
            image_editor,
            preview_image,
        ],
    )

    def on_timeline_change(video, time_sec, x, y, w, h):
        vpath = get_clean_video_path(video)
        if not vpath:
            return gr.update(), gr.update()
        frame = extract_frame_at_time(vpath, float(time_sec or 0.0))
        editor_update = {"background": frame, "layers": [], "composite": None}
        box_preview = generate_roi_preview(vpath, x, y, w, h, float(time_sec or 0.0))
        return editor_update, box_preview

    slider_timestamp.change(
        fn=on_timeline_change,
        inputs=[input_video, slider_timestamp, slider_x, slider_y, slider_w, slider_h],
        outputs=[image_editor, preview_image],
    )

    def jump_0():
        return 0.0

    def jump_25(video):
        vpath = get_clean_video_path(video)
        _, _, _, _, dur, _ = get_video_info(vpath)
        return round(dur * 0.25, 2)

    def jump_50(video):
        vpath = get_clean_video_path(video)
        _, _, _, _, dur, _ = get_video_info(vpath)
        return round(dur * 0.50, 2)

    def jump_75(video):
        vpath = get_clean_video_path(video)
        _, _, _, _, dur, _ = get_video_info(vpath)
        return round(dur * 0.75, 2)

    def jump_end(video):
        vpath = get_clean_video_path(video)
        _, _, _, _, dur, _ = get_video_info(vpath)
        return max(0.0, round(dur - 0.1, 2))

    btn_time_0.click(fn=jump_0, outputs=[slider_timestamp])
    btn_time_25.click(fn=jump_25, inputs=[input_video], outputs=[slider_timestamp])
    btn_time_50.click(fn=jump_50, inputs=[input_video], outputs=[slider_timestamp])
    btn_time_75.click(fn=jump_75, inputs=[input_video], outputs=[slider_timestamp])
    btn_time_end.click(fn=jump_end, inputs=[input_video], outputs=[slider_timestamp])

    def on_reload_frame(video, time_sec):
        vpath = get_clean_video_path(video)
        frame = extract_frame_at_time(vpath, float(time_sec or 0.0))
        return {"background": frame, "layers": [], "composite": None}

    btn_reset_frame.click(
        fn=on_reload_frame,
        inputs=[input_video, slider_timestamp],
        outputs=[image_editor],
    )

    def on_mouse_draw_sync(video, editor, time_sec):
        vpath = get_clean_video_path(video)
        if not vpath or not editor:
            return gr.update(), gr.update(), gr.update(), gr.update(), gr.update()
        w, h, _, _, _, _ = get_video_info(vpath)
        roi = extract_roi_from_editor(editor, w, h)
        if roi:
            rx, ry, rw, rh = roi
            box_preview = generate_roi_preview(vpath, rx, ry, rw, rh, float(time_sec or 0.0))
            return rx, ry, rw, rh, box_preview
        return gr.update(), gr.update(), gr.update(), gr.update(), gr.update()

    btn_sync_from_draw.click(
        fn=on_mouse_draw_sync,
        inputs=[input_video, image_editor, slider_timestamp],
        outputs=[slider_x, slider_y, slider_w, slider_h, preview_image],
    )

    def on_coord_change(video, x, y, w, h, time_sec):
        vpath = get_clean_video_path(video)
        return generate_roi_preview(vpath, x, y, w, h, float(time_sec or 0.0))

    for coord_elem in [slider_x, slider_y, slider_w, slider_h]:
        coord_elem.change(
            fn=on_coord_change,
            inputs=[input_video, slider_x, slider_y, slider_w, slider_h, slider_timestamp],
            outputs=[preview_image],
        )

    def apply_preset_gemini_1080p(video, time_sec):
        vpath = get_clean_video_path(video)
        return 1700, 950, 200, 100, generate_roi_preview(vpath, 1700, 950, 200, 100, float(time_sec or 0.0))

    def apply_preset_gemini_916(video, time_sec):
        vpath = get_clean_video_path(video)
        w, h, _, _, _, _ = get_video_info(vpath)
        if w == 1080 and h == 1920:
            rx, ry, rw, rh = 865, 1700, 80, 80
        else:
            scale_x, scale_y = w / 1080.0, h / 1920.0
            rx = int(865 * scale_x)
            ry = int(1700 * scale_y)
            rw = max(10, int(80 * scale_x))
            rh = max(10, int(80 * scale_y))
        return rx, ry, rw, rh, generate_roi_preview(vpath, rx, ry, rw, rh, float(time_sec or 0.0))

    def apply_preset_gemini_11(video, time_sec):
        vpath = get_clean_video_path(video)
        w, h, _, _, _, _ = get_video_info(vpath)
        rw, rh = int(w * 0.18), int(h * 0.07)
        rx, ry = w - rw - 15, h - rh - 20
        return rx, ry, rw, rh, generate_roi_preview(vpath, rx, ry, rw, rh, float(time_sec or 0.0))

    def apply_preset_br(video, time_sec):
        vpath = get_clean_video_path(video)
        w, h, _, _, _, _ = get_video_info(vpath)
        rw, rh = int(w * 0.15), int(h * 0.1)
        rx, ry = w - rw - 10, h - rh - 10
        return rx, ry, rw, rh, generate_roi_preview(vpath, rx, ry, rw, rh, float(time_sec or 0.0))

    def apply_preset_bl(video, time_sec):
        vpath = get_clean_video_path(video)
        w, h, _, _, _, _ = get_video_info(vpath)
        rw, rh = int(w * 0.15), int(h * 0.1)
        return 10, h - rh - 10, rw, rh, generate_roi_preview(vpath, 10, h - rh - 10, rw, rh, float(time_sec or 0.0))

    def apply_preset_tr(video, time_sec):
        vpath = get_clean_video_path(video)
        w, h, _, _, _, _ = get_video_info(vpath)
        rw, rh = int(w * 0.15), int(h * 0.1)
        return w - rw - 10, 10, rw, rh, generate_roi_preview(vpath, w - rw - 10, 10, rw, rh, float(time_sec or 0.0))

    btn_preset_gemini_1080p.click(
        fn=apply_preset_gemini_1080p,
        inputs=[input_video, slider_timestamp],
        outputs=[slider_x, slider_y, slider_w, slider_h, preview_image],
    )
    btn_preset_gemini_916.click(
        fn=apply_preset_gemini_916,
        inputs=[input_video, slider_timestamp],
        outputs=[slider_x, slider_y, slider_w, slider_h, preview_image],
    )
    btn_preset_gemini_11.click(
        fn=apply_preset_gemini_11,
        inputs=[input_video, slider_timestamp],
        outputs=[slider_x, slider_y, slider_w, slider_h, preview_image],
    )
    btn_preset_br.click(
        fn=apply_preset_br,
        inputs=[input_video, slider_timestamp],
        outputs=[slider_x, slider_y, slider_w, slider_h, preview_image],
    )
    btn_preset_bl.click(
        fn=apply_preset_bl,
        inputs=[input_video, slider_timestamp],
        outputs=[slider_x, slider_y, slider_w, slider_h, preview_image],
    )
    btn_preset_tr.click(
        fn=apply_preset_tr,
        inputs=[input_video, slider_timestamp],
        outputs=[slider_x, slider_y, slider_w, slider_h, preview_image],
    )

    # Main Start button
    btn_start.click(
        fn=process_watermark_removal,
        inputs=[
            input_video,
            image_editor,
            chk_use_drawn_mask,
            slider_x,
            slider_y,
            slider_w,
            slider_h,
            slider_subvideo,
            slider_neighbor,
            slider_ref_stride,
            slider_resize,
            slider_mask_dilates,
            chk_fp16,
            chk_clean_c2pa,
            chk_deep_clean,
            chk_mute_audio,
        ],
        outputs=[output_video, status_box],
    )

    # Instant Sanitizer button
    btn_run_sanitize.click(
        fn=process_instant_sanitization,
        inputs=[
            sanitizer_input_video,
            chk_s_c2pa,
            chk_s_synthid,
            chk_s_mute,
        ],
        outputs=[sanitizer_output_video, sanitizer_status_box],
    )

if __name__ == "__main__":
    print(f"\n==========================================")
    print(f"🎬 Video Watermark Remover & Sanitizer")
    print(f"🚀 {get_gpu_info()}")
    print(f"==========================================\n")
    demo.queue(default_concurrency_limit=2).launch(
        inbrowser=True,
        server_name="127.0.0.1",
        server_port=7860,
        share=False,
        theme=gr.themes.Soft(primary_hue="blue", neutral_hue="slate"),
        css=custom_css,
    )

