# 🎬 Local Video Watermark Remover (ProPainter + Gradio)

[![Python](https://img.shields.io/badge/Python-3.10%20%7C%203.11-3776AB?logo=python&logoColor=white)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-CUDA%2012.1-EE4C2C?logo=pytorch&logoColor=white)](https://pytorch.org/)
[![ProPainter](https://img.shields.io/badge/Model-ProPainter%20(ICCV%202023)-blue)](https://github.com/sczhou/ProPainter)
[![Gradio](https://img.shields.io/badge/UI-Gradio%20Web-orange?logo=gradio&logoColor=white)](https://gradio.app/)
[![FFmpeg](https://img.shields.io/badge/Audio-FFmpeg%20Preserved-007808?logo=ffmpeg&logoColor=white)](https://ffmpeg.org/)

A Windows-native utility to seamlessly remove static and dynamic watermarks (such as Gemini watermarks) from video files using the state-of-the-art **ProPainter** video inpainting model, packaged with a rich **Gradio** web interface and **FFmpeg** audio stream preservation.

---

## ✨ Key Features

- **🎯 Interactive Visual ROI Selection (Mouse Drag & Draw)**:
  - **🖱️ Mouse Drag & Freehand Brush**: Drag and paint directly over the watermark on the video canvas with your mouse; coordinates and bounding boxes are automatically computed and synchronized.
  - **⏱️ Timeline Frame Scrubber**: If the watermark is difficult to see due to the background color in the first frame, scrub the timeline slider or click quick jump buttons (0%, 25%, 50%, 75%, End) to load any clear frame onto the ROI canvas in real time.
  - **📐 Custom Precision Sliders**: Live bounding box overlay and pixel coordinate controls.
- **⚡ Smart Local ROI Cropping & Lossless Blending (스마트 국소 ROI 크롭 & 무손실 합성)**:
  - **90%+ VRAM 절감 & 10배 속도 향상**: 전체 화면(1080p, 4K)을 통째로 신경망에 넣지 않고 워터마크 영역 주변(패딩 포함)만 스마트 크롭하여 초경량(VRAM ~200MB)으로 처리, 8GB GPU에서도 1080×1920 세로형 비디오 OOM을 완벽히 방지합니다.
  - **원본 100% 화질 보존 (가우시안 페더링 블렌딩)**: 인페인팅된 영역만을 소프트 엣지 마스크로 원본 고화질 프레임에 합성하여, 워터마크 외 98% 이상의 영상 영역은 화질 저하나 블러 없이 원본 그대로 보존됩니다.
- **⚡ One-Click Presets**:
  - Quick presets for **Gemini (1080p Bottom-Right)**, **Gemini (9:16 Shorts)**, **Gemini (1:1 Square)**, **Bottom-Right**, **Bottom-Left**, and **Top-Right**.
- **🚀 GPU-Accelerated FP16 Inpainting**:
  - Leverages ProPainter temporal flow completion & transformer inpainting with `--fp16` half precision.
  - Smooth execution on NVIDIA RTX GPUs (e.g. RTX 3080 / 40-series).
- **🔊 Complete Audio Preservation**:
  - ProPainter only processes video frames; our pipeline uses **FFmpeg** to automatically mux and synchronize the original video's audio tracks into the final output.
- **📊 Real-Time Progress Streaming**:
  - Live ASCII progress bar, frame counts (`48/120 frames (40%) [████░░░░]`), elapsed/remaining time, and GPU processing speed (`it/s`) streamed live to the Gradio UI.
- **🖱️ Windows One-Click Launcher**:
  - Includes `run.bat` for instant startup.

---

## 🏗️ Pipeline Architecture (스마트 파이프라인 구조)

```mermaid
flowchart LR
    A[Upload Video] --> B[Interactive ROI Selector]
    B --> C[Generate Binary Mask PNG]
    C --> D[Smart Local ROI Cropper<br/>워터마크 영역 자동 크롭]
    D --> E[ProPainter FP16 GPU Inpainting<br/>초경량 국소 인페인팅]
    E --> F[Gaussian Feather Blending<br/>원본 무손실 자연 합성]
    F --> G[Inpainted Video Frames]
    G --> H[FFmpeg Audio Remuxing]
    A -. Original Audio .-> H
    H --> I[Final Watermark-Free Video]
```

---

## 🔬 Smart Processing Engine (새로운 스마트 처리 방식 상세)

### 1. 국소 영역 자동 크롭 (Smart Local ROI Cropping)
- **기존 방식의 문제점**: 1080×1920(세로형 쇼츠) 또는 1920×1080 고해상도 비디오 전체(200만~800만 픽셀)를 트랜스포머 및 광학 흐름 신경망에 통째로 입력할 경우, 17~20개 시계열 프레임 텐서 연산으로 인해 **15GB 이상의 GPU VRAM이 요구되어 8GB VRAM 그래픽카드에서 CUDA Out of Memory(OOM)**가 발생했습니다.
- **해결 방식**: 워터마크가 화면의 국소 영역(전체 화면의 45% 미만)인 경우, 워터마크 바운딩 박스 주변에 충분한 컨텍스트 마진(64px 패딩 및 16배수 크기 자동 정렬)을 더해 필요한 영역(예: 224×224)만 초경량으로 자동 크롭하여 인페인팅을 수행합니다.
- **성능 개선**:
  - **VRAM 사용량**: 12GB+ $\rightarrow$ **~200MB 수준으로 90% 이상 절감**
  - **처리 속도**: 초당 0.5 it/s $\rightarrow$ **초당 5~6 it/s로 10배 이상 고속화**
  - **해상도 제약 해소**: 1080p, 4K, 세로형 9:16, 정사각형 1:1 영상도 메모리 부족 없이 완벽 지원

### 2. 가우시안 페더링 무손실 합성 (Gaussian Feathered Lossless Blending)
- 인페인팅된 결과 패치를 가우시안 블러(Gaussian Blur) 기반의 부드러운 소프트 알파 마스크를 사용하여 원본 1080p 프레임 위에 완벽하게 Seamless Blending합니다.
- 워터마크가 없는 **영상의 98% 이상 영역은 인코딩 열화나 블러링 없이 원본 비디오 100% 최고 화질을 그대로 유지**합니다.

---

## 📋 Prerequisites

- **OS**: Windows 10 / 11 (x64)
- **GPU**: NVIDIA GPU with CUDA support (RTX 3060 / 3070 / 3080 / 40-series recommended, >= 6GB VRAM)
- **Python**: Python 3.10 or 3.11 (64-bit)
- **FFmpeg**: `ffmpeg` must be installed and added to system `PATH`
  - *Verify via*: `ffmpeg -version`

---

## ⚙️ Installation & Setup

### 1. Clone Repository
```powershell
git clone git@github.com:tramper2/RmWaterMark.git
cd RmWaterMark
```

### 2. Create Virtual Environment & Install Dependencies
```powershell
# Create virtual environment
python -m venv venv

# Activate virtual environment
.\venv\Scripts\activate

# Install PyTorch with CUDA support (cu121)
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121

# Install project dependencies
pip install -r requirements.txt
```

### 3. Download Model Weights
Download the required ProPainter weights (`ProPainter.pth`, `recurrent_flow_completion.pth`, `raft-things.pth`):
```powershell
python download_weights.py
```

---

## 🚀 How to Run

### Method 1: Double-Click Launcher (Windows)
Simply double-click `run.bat` in the project root folder.

### Method 2: Command Line
```powershell
.\venv\Scripts\activate
python app.py
```
Open your web browser and navigate to: **`http://127.0.0.1:7860`**

---

## 📖 How to Use

1. **Upload Video**: Drag & drop your video into the **Source Video** box.
2. **Set Watermark Region**:
   - Choose a preset (e.g. *Gemini (9:16 Shorts)*) or fine-tune `X`, `Y`, `Width`, `Height` sliders.
   - You can also scrub the **Timeline Frame Scrubber** to jump to a scene where the watermark is clearest, and paint directly over it with your mouse.
   - Inspect the **ROI Visual Preview** to make sure the red box fully covers the watermark.
3. **Optimize Settings (Optional)**:
   - Expand *Advanced Optimization Settings* to adjust `Subvideo Length` or `Mask Dilation` if needed.
4. **Start Removal**: Click **🚀 Start Watermark Removal**.
5. **Download Output**: Preview and download the final watermarked-free video with full audio synchronization.

---

## 🎯 Recommended Watermark Preset Coordinates

| Aspect Ratio / Platform | Resolution | X (Left) | Y (Top) | Width | Height | Description |
| :--- | :--- | :---: | :---: | :---: | :---: | :--- |
| 📱 **Gemini (9:16 Vertical / Shorts)** | **1080 × 1920** | **865** | **1700** | **80** | **80** | Gemini logo / text watermark (Bottom-Right) |
| 🖥️ **Gemini (16:9 Landscape)** | **1920 × 1080** | **1700** | **950** | **200** | **100** | Standard 1080p Bottom-Right watermark |
| 🔲 **Gemini (1:1 Square)** | **1080 × 1080** | **870** | **980** | **190** | **80** | Square video Bottom-Right watermark |

> [!TIP]
> 영상 배경 색상이 워터마크 색과 동일하여 잘 보이지 않는 경우, **1번 패널 하단의 타임라인 슬라이더(또는 25%/50% 버튼)**를 이용해 워터마크가 선명하게 보이는 장면으로 이동하여 지정할 수 있습니다.

---

## ⏱️ Model Initialization Notice (초기 로딩 시간 안내)

> [!NOTE]
> **🚀 시작 시 딥러닝 모델 로딩 지연 안내**:
> 워터마크 제거를 처음 실행할 때, 대용량 신경망 모델(**ProPainter InpaintGenerator**, **Recurrent Flow Completion**, **RAFT 광학 흐름 신경망**) 및 사전 학습 가중치 파라미터를 GPU VRAM으로 적재하고 초기 메모리 구조를 빌드하는 과정이 수반됩니다.
> 따라서 **첫 1회 실행 시 약 10초 ~ 30초 정도의 모델 준비 시간**이 소요될 수 있으며, 초기화가 완료된 후부터는 초당 수 프레임 이상의 빠른 GPU 가속으로 실시간 처리됩니다.

---

## 📂 Project Structure

```text
RmWaterMark/
├── ProPainter/               # ProPainter core algorithms & models
├── weights/                  # Model weights (ProPainter.pth, etc.)
├── app.py                    # Main Gradio application & processing pipeline
├── download_weights.py       # Pretrained weights verification & downloader
├── requirements.txt          # Python dependencies
├── run.bat                   # Windows one-click launcher
├── test_pipeline.py          # End-to-end automated test script
├── .gitignore                # Git ignore rules for virtualenv & weights
└── README.md                 # Project documentation
```

---

## 📜 Acknowledgements & References

- **ProPainter**: [Improving Propagation and Transformer for Video Inpainting (ICCV 2023)](https://github.com/sczhou/ProPainter)
- **Gradio**: [Build & Share Machine Learning Apps](https://gradio.app/)
- **FFmpeg**: [A complete, cross-platform solution to record, convert and stream audio and video](https://ffmpeg.org/)
