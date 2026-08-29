# TASK: Local Video Watermark Remover using ProPainter & Gradio (Windows Native & WSL Compatible)

## 1. Project Overview
Build a local Python-based utility on **Windows 10/11** that removes static watermarks (such as the Gemini watermark) seamlessly from video files using the open-source **ProPainter** video inpainting model, packaged with a **Gradio** web UI.

---

## 2. Windows Environment & Prerequisites
- **OS**: Windows 10/11 (x64)
- **GPU**: NVIDIA RTX 3080 (CUDA enabled, NVIDIA driver >= 528.xx recommended)
- **Python**: Python 3.10 or 3.11 (64-bit Windows installer)
- **C++ Build Tools**: Visual Studio C++ Build Tools (required for compiling certain CUDA ops if needed)
- **FFmpeg**: Windows build of `ffmpeg.exe` must be downloaded and added to the system `PATH` (or placed in the project directory).
- **PyTorch**: PyTorch with CUDA support (e.g., `pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118` or `cu121`).

---

## 3. Windows-Specific Considerations for Implementation
1. **Path Handling**:
   - Always use `pathlib.Path` or `os.path.join()` / `os.path.normpath()` to handle Windows backslashes (`\`) properly.
   - Avoid shell injection or quoting issues on Windows Command Prompt / PowerShell when invoking `subprocess.run()`.
2. **Subprocess Execution**:
   - Call Python scripts using `sys.executable` (e.g. `[sys.executable, "inference_propainter.py", ...]`) instead of raw `"python"` to ensure the active virtual environment is used.
   - Avoid `shell=True` unless strictly necessary.
3. **FFmpeg on Windows**:
   - Verify `ffmpeg -version` is executable via `subprocess` or fallback to checking a bundled `bin/ffmpeg.exe`.

---

## 4. Architecture & Functional Requirements

### 4.1 Mask Generation
- Generate a binary mask (NumPy array saved as PNG) matching input video resolution `(width, height)`.
- Watermark ROI is parameterized by `(x, y, width, height)`.
- Mark the specified ROI with `255` (white) and background with `0` (black).

### 4.2 ProPainter Inference Integration
- Clone or vendor `sczhou/ProPainter` into the workspace.
- Run ProPainter inference with `--fp16` enabled for VRAM optimization on RTX 3080 (10GB/12GB).
- Pass the source video and generated mask image to `inference_propainter.py`.
- Handle automatic model weight downloads (`weights/` folder).

### 4.3 Audio Preservation (FFmpeg)
- ProPainter only processes video frames and discards audio.
- Use `ffmpeg` to merge the original video's audio track back:
  ```cmd
  ffmpeg -y -i <inpainted_video> -i <original_video> -c:v copy -c:a aac -map 0:v:0 -map 1:a:0? <final_output>
  ```
- The `-map 1:a:0?` flag ensures compatibility even if the input video has no audio stream.

### 4.4 Gradio Web UI (`app.py`)
- **Inputs**:
  - `gr.Video`: Source video file upload.
  - `gr.Number` (or `gr.Slider`): Coordinates `X`, `Y`, `Width`, `Height` (with sensible defaults for 1080p: x=1700, y=950, w=200, h=100).
- **Outputs**:
  - `gr.Video`: Processed final video with audio, ready to preview/download.
  - `gr.Textbox`: Real-time status and logs.
- **Controls**:
  - "Start Removal" primary button.

---

## 5. Key Files Structure
```text
watermark-remover/
├── ProPainter/               # Cloned ProPainter repo
├── app.py                    # Main Gradio application & execution pipeline
├── requirements.txt          # Python dependencies
├── results/                  # Inference output directory (auto-created)
└── run.bat                   # Convenient Windows one-click launcher
```

### Windows One-Click Launcher (`run.bat`):
```bat
@echo off
cd /d %~dp0
call venv\Scripts\activate.bat
python app.py
pause
```

---

## 6. Acceptance Criteria
1. Script runs cleanly on Windows native environment without pathing or encoding errors.
2. Inpainted output seamlessly restores watermark regions with no visible blur/glitch.
3. Audio from the original video is preserved and synchronized.
4. Memory usage stays within RTX 3080 limits without CUDA OOM errors.