# StomaD²

PyQt5-based desktop app for stomata detection and restoration. It integrates YOLO OBB models for dicotyledon/monocotyledon leaf stomata detection and optionally uses DiffBIR for restoration/denoising. Supports images, videos, and live camera input, and can batch export annotated media and CSV/AVI results.

## Requirements
- Python 3.10+
- Windows 10/11; NVIDIA GPU with CUDA recommended (CPU works but is slower)
- PyTorch matching your CUDA/CPU setup; get the correct command from https://pytorch.org/get-started/locally

## Quick Start
1) Clone the repo and create a virtual environment
```bash
python -m venv .venv
.\.venv\Scripts\activate
```
2) Install PyTorch (match your CUDA/CPU). Example for CUDA 12.1 (cu121 wheels):
```bash
pip install torch==2.5.1+cu121 torchvision==0.20.1+cu121 torchaudio==2.5.1+cu121 -f https://download.pytorch.org/whl/cu121
```
3) Install remaining deps:
```bash
pip install -r requirements.txt
```
4) Prepare model weights (place under `weights/` with these filenames):
- Detection: `dicotyledons_nondestructive.pt`, `dicotyledons_destructive.pt`, `monocotyledons_nondestructive.pt`, `monocotyledons_destructive.pt`
- DiffBIR: `face_swinir_v1.ckpt`, `scunet_color_real_psnr.pth`, `v2-1_512-ema-pruned.ckpt`, `v2.pth`
  - Download from the official repo: https://github.com/XPixelGroup/DiffBIR (check Releases/model download)
> Many files are >100 MB. When pushing to GitHub, use Git LFS or share download links so users can place them into `weights/`.

5) Run the app:
```bash
python app.py
```
- Use Image/Video/Camera for detection inputs.
- "Restoration Detection" loads DiffBIR to enhance before detection.
- Outputs go to your selected `output/` subfolder (annotated images, detection videos, CSV metrics).

## Repository Layout
- `app.py`: main entry and PyQt5 UI logic
- `UIProgram/`: `.ui` layout and progress components
- `DiffBIR/`: image restoration module (Stable Diffusion + SCUNet)
- `weights/`: detection + restoration weights (download/place manually)
- `test_data/`: sample images/videos for quick verification
- `output/`: results generated after runs

## Video Submission
- Place the 3 demo videos in `test_data/` or a new `videos/` folder. Keep each file <100 MB; if larger, use Git LFS or share download links.

## FAQ
- App won’t start: ensure PyQt5 installed and virtual env activated.
- GPU errors: confirm CUDA version matches the installed PyTorch wheels.
- Weight not found: verify filenames and the `weights/` path.
