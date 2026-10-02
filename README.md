# Self-driving scene segmentation (U-Net)

This repository contains an experimental U-Net semantic-segmentation notebook and a Streamlit demo. It is **not a complete, self-contained training package**: the dataset and trained model are not included.

## Repository contents

- `autonomous-driving-scene-segmentation.ipynb` — notebook originally written for Kaggle. It expects preprocessed, side-by-side image pairs in `cityscapes_data/{train,val}` and splits each 256×512 JPEG into a 256×256 input image and target image. Update `ROOT_PATH` if your licensed dataset is stored elsewhere.
- `stream_segmentation.py` — Streamlit image/video/browser-camera-capture demo. It loads `cityscapes_unet_model.keras` beside the script by default (or uses `CITYSCAPES_MODEL_PATH`); that model file is not included.
- `requirements.txt` — pinned Python dependencies for the TensorFlow 2.15 runtime used as a compatibility baseline.

The name “Cityscapes” refers to the dataset used by the original notebook; the exact source/preprocessing and eight-class mapping have not been independently verified from dataset files in this repository. Obtain the data from an authorized source and comply with its terms.

## Setup

Use Python 3.9–3.11 (TensorFlow 2.15 does not support every newer Python release):

```bash
python -m venv .venv
# Linux/macOS
source .venv/bin/activate
# Windows PowerShell: .venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

Open the notebook in Kaggle or a local Jupyter environment. For local use, install Jupyter separately and change `ROOT_PATH` to the directory containing `cityscapes_data`. The notebook expects paired images that can be decoded as JPEG and split down the middle; check that your data has this format before running it. The notebook's validation cell only reports unreadable files; it does not delete or modify dataset files.

## Streamlit demo

The demo requires a model saved in the expected Keras format. By default, place it at `cityscapes_unet_model.keras` beside the script, or set `CITYSCAPES_MODEL_PATH` to a trusted model file:

```bash
# Optional: point to a model in another location
export CITYSCAPES_MODEL_PATH=/path/to/cityscapes_unet_model.keras
streamlit run stream_segmentation.py
```

The model must accept 256×256 RGB images scaled to `[0, 1]` and return per-pixel scores for the eight classes expected by the demo. No compatible model artifact is supplied, so the demo cannot run inference until one is provided. Only load model files you trust.

## Validation and limitations

No dataset or trained model is tracked here. The notebook contains no reproducible result artifact, and training/evaluation have **not** been run or validated from this repository. In particular, the source mask encoding and correspondence between mask colors, class IDs, loss targets, and the demo's eight classes still need verification against the actual dataset before interpreting metrics or predictions. No accuracy or performance claim is made.
