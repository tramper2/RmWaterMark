import os
import sys
import shutil
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
    progress=gr.Progress(track_tqdm=True),
) -> Generator[Tuple[Any, str], None, None]:
    """Execute ProPainter inpainting and merge original audio using FFmpeg."""
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
        drawn_mask = extract_mask_from_editor(editor_data, width, height) if use_drawn_mask else None
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
                # Keep last 10 log lines
                recent_logs = "\n".join(log_lines[-10:])

                # Parse tqdm progress line
                tqdm_match = re.search(r"(\d+)%\|.*?\|\s*(\d+)/(\d+)\s*\[(.*?)(?:,\s*(.*?))?\]", line_str)
                if tqdm_match:
                    pct = int(tqdm_match.group(1))
                    curr_step = int(tqdm_match.group(2))
                    total_steps = int(tqdm_match.group(3))
                    time_info = tqdm_match.group(4)
                    speed_info = tqdm_match.group(5) or ""

                    curr_frame = min(frame_count, int((curr_step / total_steps) * frame_count))
                    bar_len = 20
                    filled = int(bar_len * curr_step / total_steps)
                    bar = "█" * filled + "░" * (bar_len - filled)

                    progress(curr_step / total_steps, desc=f"인페인팅: {curr_frame}/{frame_count} 프레임 ({pct}%)")

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
                    # Update for flow/propagation or initial setup logs
                    if "[Flow]" in line_str or "Processing:" in line_str or time.time() - last_yield_time > 0.3:
                        last_yield_time = time.time()
                        status_msg = (
                            f"🚀 [3/4] ProPainter GPU 비디오 인페인팅 실행 중...\n"
                            f"━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n"
                            f"⏳ {line_str if '[Flow]' in line_str else f'초기 Flow 계산 및 프레임 분석 중... (총 {frame_count} 프레임)'}\n"
                            f"━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n"
                            f"📝 최근 실행 로그:\n{recent_logs}"
                        )
                        yield gr.update(), status_msg
            else:
                buffer += char

        process.stdout.close()
        return_code = process.wait()
        if return_code != 0:
            yield gr.update(), f"❌ ProPainter 실행 실패 (코드 {return_code}):\n" + "\n".join(log_lines[-20:])
            return

        # Look for inpainted output video in workspace_dir / out
        out_dir = workspace_dir / "out"
        inpainted_files = list(out_dir.rglob("inpaint_out.mp4")) or list(out_dir.rglob("*.mp4"))
        if not inpainted_files:
            yield gr.update(), f"❌ 인페인팅 결과 비디오 파일을 찾을 수 없습니다: {out_dir}"
            return

        inpainted_video = inpainted_files[0]
        yield gr.update(), f"🎵 [4/4] FFmpeg를 통해 원본 오디오 스트림을 보존 및 병합하는 중..."

        # Target final output file in results/
        video_stem = Path(vpath).stem
        final_output_path = results_dir / f"{video_stem}_nowatermark_{task_id}.mp4"

        # Merge audio
        ffmpeg_cmd = [
            "ffmpeg",
            "-y",
            "-i",
            str(inpainted_video),
            "-i",
            str(vpath),
            "-c:v",
            "copy",
            "-c:a",
            "aac",
            "-map",
            "0:v:0",
            "-map",
            "1:a:0?",
            str(final_output_path),
        ]

        try:
            subprocess.run(
                ffmpeg_cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                check=True,
            )
        except subprocess.CalledProcessError as err:
            shutil.copyfile(inpainted_video, final_output_path)
            yield str(final_output_path), f"⚠️ 비디오 생성 완료 (오디오 병합 경고): {err.stderr}"
            return

        # Clean up temporary workspace
        try:
            shutil.rmtree(workspace_dir, ignore_errors=True)
        except Exception:
            pass

        yield str(final_output_path), f"✅ 워터마크 제거 완료!\n📁 저장 위치: results/{final_output_path.name}"

    except Exception as e:
        err_msg = traceback.format_exc()
        print("ERROR in process_watermark_removal:\n", err_msg)
        yield gr.update(), f"❌ 처리 중 오류 발생:\n{e}\n\n{err_msg}"


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
.timeline-card {
    background-color: #1e293b;
    border: 1px solid #334155;
    border-radius: 6px;
    padding: 10px;
    margin-top: 8px;
}
"""

with gr.Blocks(title="AI Video Watermark Remover (ProPainter)") as demo:
    gr.Markdown(
        f"""
        # 🎬 Local Video Watermark Remover
        ### Powered by **ProPainter** (ICCV 2023) & **FFmpeg** Audio Preservation
        <span class="header-badge">{get_gpu_info()}</span>
        """
    )

    # ------------------ TOP SECTION: Inputs (Side by Side) ------------------
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

    # ------------------ MIDDLE SECTION: Settings & Start Button ------------------
    with gr.Accordion("⚙️ Advanced Optimization Settings (VRAM / Memory)", open=False):
        with gr.Row():
            slider_subvideo = gr.Slider(
                label="Subvideo Length (Frames)",
                minimum=20,
                maximum=120,
                value=80,
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

    btn_start = gr.Button("🚀 Start Watermark Removal", variant="primary", size="lg")

    # ------------------ BOTTOM SECTION: Outputs (Side by Side) ------------------
    with gr.Row():
        # Panel 3: Inpainted Output Video
        with gr.Column(scale=1):
            gr.Markdown("### 3. Inpainted Output Video (with Synchronized Audio)")
            output_video = gr.Video(label="Processed Result Video", interactive=False)

        # Panel 4: Status & Execution Log
        with gr.Column(scale=1):
            gr.Markdown("### 4. Status & Execution Log")
            status_box = gr.Textbox(
                label="Execution Log",
                value="Ready. Upload a video, set watermark region, and click Start.",
                lines=12,
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

    # Timeline scrubber updates frame in both image_editor and preview_image
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

    # Quick Jump Buttons
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

    # Reload frame button
    def on_reload_frame(video, time_sec):
        vpath = get_clean_video_path(video)
        frame = extract_frame_at_time(vpath, float(time_sec or 0.0))
        return {"background": frame, "layers": [], "composite": None}

    btn_reset_frame.click(
        fn=on_reload_frame,
        inputs=[input_video, slider_timestamp],
        outputs=[image_editor],
    )

    # Sync coordinates from mouse-drawn layer
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

    # Coordinate update trigger preview
    def on_coord_change(video, x, y, w, h, time_sec):
        vpath = get_clean_video_path(video)
        return generate_roi_preview(vpath, x, y, w, h, float(time_sec or 0.0))

    for coord_elem in [slider_x, slider_y, slider_w, slider_h]:
        coord_elem.change(
            fn=on_coord_change,
            inputs=[input_video, slider_x, slider_y, slider_w, slider_h, slider_timestamp],
            outputs=[preview_image],
        )

    # Presets
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

    # Start button
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
        ],
        outputs=[output_video, status_box],
    )

if __name__ == "__main__":
    print(f"\n==========================================")
    print(f"🎬 Video Watermark Remover (ProPainter)")
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
