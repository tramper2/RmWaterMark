import os
import sys
import subprocess
import numpy as np
import cv2
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
SCRATCH_DIR = BASE_DIR / "temp" / "test_run"
SCRATCH_DIR.mkdir(parents=True, exist_ok=True)


def create_test_video(video_path: Path, duration_sec: int = 2, fps: int = 24, width: int = 640, height: int = 360):
    """Create a short synthetic video with a moving background and static watermark."""
    print(f"Creating test video: {video_path} ({width}x{height}, {duration_sec}s @ {fps}fps)...")
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    temp_vid = video_path.with_suffix(".tmp.mp4")
    out = cv2.VideoWriter(str(temp_vid), fourcc, fps, (width, height))

    total_frames = duration_sec * fps
    for i in range(total_frames):
        # Create dynamic background (gradient moving)
        frame = np.zeros((height, width, 3), dtype=np.uint8)
        shift = int((i / total_frames) * 100)
        for y in range(height):
            frame[y, :, 0] = (y + shift) % 255
            frame[y, :, 1] = (int(width * y / height) + shift * 2) % 255
            frame[y, :, 2] = 180

        # Draw a moving circle in background
        cx = int((i / total_frames) * (width - 100)) + 50
        cy = height // 2
        cv2.circle(frame, (cx, cy), 30, (255, 255, 255), -1)

        # Draw static watermark at bottom right (x=500, y=300, w=120, h=40)
        cv2.rectangle(frame, (500, 300), (620, 340), (255, 255, 255), -1)
        cv2.putText(
            frame,
            "WATERMARK",
            (505, 328),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0, 0, 0),
            2,
            cv2.LINE_AA,
        )

        out.write(frame)
    out.release()

    # Generate synthetic audio with FFmpeg and mux together
    print("Muxing synthetic audio tone with FFmpeg...")
    audio_cmd = [
        "ffmpeg",
        "-y",
        "-i",
        str(temp_vid),
        "-f",
        "lavfi",
        "-i",
        f"sine=frequency=440:duration={duration_sec}",
        "-c:v",
        "libx264",
        "-pix_fmt",
        "yuv420p",
        "-c:a",
        "aac",
        "-shortest",
        str(video_path),
    ]
    subprocess.run(audio_cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=True)
    if temp_vid.exists():
        temp_vid.unlink()
    print("Test video created successfully with audio.")


def test_inpainting_pipeline():
    test_video = SCRATCH_DIR / "sample_watermark.mp4"
    if not test_video.exists():
        create_test_video(test_video)

    # 1. Generate mask for ROI (x=495, y=295, w=130, h=50)
    mask_path = SCRATCH_DIR / "mask.png"
    mask = np.zeros((360, 640), dtype=np.uint8)
    mask[295:345, 495:625] = 255
    cv2.imwrite(str(mask_path), mask)
    print("Binary mask created.")

    # 2. Run ProPainter inference
    out_dir = SCRATCH_DIR / "out"
    propainter_script = BASE_DIR / "ProPainter" / "inference_propainter.py"
    cmd = [
        sys.executable,
        str(propainter_script),
        "--video",
        str(test_video),
        "--mask",
        str(mask_path),
        "--output",
        str(out_dir),
        "--subvideo_length",
        "40",
        "--fp16",
    ]
    print(f"Running ProPainter: {' '.join(cmd)}")
    res = subprocess.run(cmd, cwd=str(BASE_DIR / "ProPainter"), stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    print(res.stdout)
    if res.returncode != 0:
        print(f"[FAIL] ProPainter returned non-zero code {res.returncode}")
        return False

    # 3. Look for output video
    mp4_files = list(out_dir.glob("*.mp4")) + list(out_dir.glob("*/*.mp4"))
    if not mp4_files:
        print("[FAIL] No output video found in out_dir")
        return False
    inpainted_video = mp4_files[0]
    print(f"Inpainted video produced: {inpainted_video}")

    # 4. Merge audio with FFmpeg
    final_output = SCRATCH_DIR / "final_result_with_audio.mp4"
    ffmpeg_cmd = [
        "ffmpeg",
        "-y",
        "-i",
        str(inpainted_video),
        "-i",
        str(test_video),
        "-c:v",
        "copy",
        "-c:a",
        "aac",
        "-map",
        "0:v:0",
        "-map",
        "1:a:0?",
        str(final_output),
    ]
    subprocess.run(ffmpeg_cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=True)

    if final_output.exists() and final_output.stat().st_size > 1000:
        print(f"[SUCCESS] Pipeline test PASSED! Output: {final_output} ({final_output.stat().st_size / 1024:.1f} KB)")
        return True
    else:
        print("[FAIL] Final output file is missing or empty")
        return False


if __name__ == "__main__":
    success = test_inpainting_pipeline()
    sys.exit(0 if success else 1)
