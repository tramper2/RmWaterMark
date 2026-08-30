# 🎨 AI Watermark Remover & Metadata Sanitizer (Image & Video)

[![Python](https://img.shields.io/badge/Python-3.10%20%7C%203.11-3776AB?logo=python&logoColor=white)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-CUDA%2012.1-EE4C2C?logo=pytorch&logoColor=white)](https://pytorch.org/)
[![LaMa](https://img.shields.io/badge/Image_Model-LaMa%20(Fast%20Fourier%20AI)-success)](https://github.com/advimman/lama)
[![ProPainter](https://img.shields.io/badge/Video_Model-ProPainter%20(ICCV%202023)-blue)](https://github.com/sczhou/ProPainter)
[![Gradio](https://img.shields.io/badge/UI-Gradio%20Web-orange?logo=gradio&logoColor=white)](https://gradio.app/)
[![FFmpeg](https://img.shields.io/badge/Audio-FFmpeg%20Preserved-007808?logo=ffmpeg&logoColor=white)](https://ffmpeg.org/)

**이미지(Image) 및 동영상(Video)**에서 워터마크(Gemini, DALL-E, TikTok, 로고, 자막 등)를 인공지능으로 자연스럽고 완벽하게 제거하고, **C2PA / EXIF 메타데이터 및 AI 비가시성 워터마크(SynthID)**를 100% 소거하는 Windows 로컬 GUI 애플리케이션입니다.

---

## ✨ Key Features

### 🖼️ 1. 이미지 워터마크 제거 (Image Watermark Remover)
- **🧠 SOTA AI 인페인팅 엔진 (LaMa - Fast Fourier Convolutions)**:
  - 고주파 질감 및 복잡한 배경 패턴(자연물, 건물, 텍스트, 그라데이션)을 흐림이나 왜곡 없이 원본처럼 사실적으로 복원.
  - 보조 초고속 엔진으로 **OpenCV Navier-Stokes** 및 **Telea** 알고리즘 지원.
- **📐 임의의 모든 해상도 & 화면비 완벽 지원**:
  - SD, HD, FHD, 4K, 8K 및 가로형(16:9), 세로형(9:16), 정방형(1:1) 등 어떤 크기의 이미지도 제약 없이 처리.
- **⚡ 스마트 고해상도(4K/8K) 국소 크롭 가속 (Smart Patch Mode)**:
  - 4K/8K 대용량 이미지에서도 워터마크 영역 주변만 스마트 크롭하여 **0.1~0.4초 만에 초고속 인페인팅** 수행.
  - 워터마크 외 나머지 영역은 **100% 원본 픽셀 화질 무손실 유지**.
- **🎨 유연한 워터마크 영역 지정**:
  - **마우스 브러시/지우개 직접 칠하기 (`gr.ImageEditor`)**: 비정형 로고나 워터마크를 브러시로 직접 칠해 정밀 제거.
  - **원클릭 위치 프리셋**: 16:9 BR, 9:16 BR, 1:1 BR, 우측 하단, 좌측 하단, 상단, 중앙 등 원클릭 설정.
  - **정밀 사각형 좌표 슬라이더 (X, Y, W, H)**: 실시간 빨간색 바운딩 박스 미리보기.
- **🛡️ 이미지 메타데이터 & AI 비가시성 워터마크(SynthID) 완전 소거**:
  - EXIF, XMP, IPTC, C2PA 매니페스트 100% 0바이트 소거.
  - 픽셀 격자 미세 교란을 통해 SNS/플랫폼 AI 자동 감지 차단.

---

### 🎬 2. 동영상 워터마크 제거 (Video Watermark Remover)
- **ProPainter ICCV 2023 딥러닝 비디오 인페인팅**:
  - 시공간 광학 흐름(Optical Flow) 및 트랜스포머 기반 동영상 배경 복원.
- **⏱️ 타임라인 프레임 스크러버**:
  - 타임라인을 자유롭게 이동하여 워터마크가 선명하게 보이는 장면을 불러와 마우스로 브러시 마스크 지정.
- **🎵 오디오 무손실 동기화**:
  - FFmpeg 기반 원본 음성 스트림 완벽 보존 및 위상 재구성.

---

### 🧹 3. 기존 동영상 & 이미지 즉시 세척기 (Instant Sanitizer)
- 인페인팅 없이도 이미 생성된 모든 동영상(MP4/MOV) 또는 이미지(PNG/JPG/WEBP)를 1초 만에 C2PA/EXIF/SynthID 완전 소거.

---

## 🏗️ Architecture

```mermaid
flowchart TD
    subgraph Image Pipeline
        A1[Upload Image Any Size] --> B1[Brush Canvas / Presets / ROI Sliders]
        B1 --> C1[Smart Patch Bounding Box]
        C1 --> D1[LaMa Fourier Convolutions AI Inpainting]
        D1 --> E1[Gaussian Feathered Seamless Blend]
        E1 --> F1[EXIF / C2PA / SynthID Pixel Wipe]
        F1 --> G1[Clean Watermark-Free Image]
    end

    subgraph Video Pipeline
        A2[Upload Video] --> B2[Timeline Scrub & Brush ROI]
        B2 --> C2[Smart Local ROI Cropper]
        C2 --> D2[ProPainter FP16 GPU Inpainting]
        D2 --> E2[Gaussian Feather Blending]
        E2 --> F2[FFmpeg Audio Stream & C2PA Box Sanitizer]
        F2 --> G2[100% Clean Output Video]
    end
```

---

## 📋 System Requirements

- **OS**: Windows 10 / 11 (64-bit)
- **GPU**: NVIDIA GPU with CUDA (RTX 3060 / 3070 / 3080 / 40-series 권장, CPU 모드도 지원)
- **Python**: Python 3.10 또는 3.11
- **FFmpeg**: 시스템 PATH에 등록된 FFmpeg (`ffmpeg -version` 확인)

---

## ⚙️ Installation & Setup

### 1. 가상환경 생성 및 의존성 패키지 설치
```powershell
# 가상환경 활성화
.\venv\Scripts\activate

# PyTorch CUDA 12.1 설치 (필요시)
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121

# 의존성 패키지 설치
pip install -r requirements.txt
```

### 2. 사전 학습 가중치 다운로드
LaMa(`big-lama.pt`) 및 ProPainter 모델 가중치를 자동으로 다운로드 및 검증합니다:
```powershell
python download_weights.py
```

---

## 🚀 How to Run

### 방법 1: 원클릭 실행 파일 (Windows)
프로젝트 루트 폴더의 **`run.bat`** 파일을 더블 클릭합니다.

### 방법 2: 터미널 실행
```powershell
.\venv\Scripts\activate
python app.py
```
웹 브라우저에서 **`http://127.0.0.1:7860`** 으로 접속합니다.

---

## 📖 사용 방법 (How to Use)

### 🖼️ 이미지 워터마크 제거
1. **이미지 업로드**: `1. 이미지 워터마크 제거` 탭에서 이미지를 드래그 & 드롭으로 업로드합니다.
2. **워터마크 위치 지정**:
   - **마우스 브러시**: 이미지 캔버스 위에서 워터마크 부분을 마우스로 칠합니다.
   - **원클릭 프리셋**: `🖥️ 16:9 BR`, `📱 9:16 Shorts`, `📍 Auto BR` 등 프리셋 버튼을 클릭하거나 `X, Y, Width, Height` 슬라이더로 조절합니다.
3. **옵션 선택**: `LaMa (AI SOTA)` 엔진 선택, 마스크 확장(Dilation, 기본 5px) 및 메타데이터 소거 확인.
4. **실행**: **`🚀 워터마크 제거 시작`** 클릭 후 고화질 결과 이미지 즉시 다운로드.

---

## 📂 Project Structure

```text
RmWaterMark/
├── ProPainter/               # ProPainter video inpainting engine
├── weights/                  # Model weights (big-lama.pt, ProPainter.pth, etc.)
├── app.py                    # Main Gradio application (Image + Video + Sanitizer)
├── image_inpainter.py        # LaMa Fourier AI & OpenCV image inpainting engine
├── download_weights.py       # Weights downloader (LaMa + ProPainter)
├── test_image_pipeline.py    # Automated image pipeline test suite
├── test_pipeline.py          # Automated video pipeline test suite
├── requirements.txt          # Python dependencies
├── run.bat                   # Windows one-click launcher
└── README.md                 # Project documentation
```

---

## 📜 Acknowledgements & References

- **LaMa**: [Resolution-robust Large Mask Inpainting with Fourier Convolutions (WACV 2022)](https://github.com/advimman/lama)
- **ProPainter**: [Improving Propagation and Transformer for Video Inpainting (ICCV 2023)](https://github.com/sczhou/ProPainter)
- **Gradio**: [Build & Share Machine Learning Web Apps](https://gradio.app/)
- **FFmpeg**: [Audio/Video Processing Suite](https://ffmpeg.org/)
