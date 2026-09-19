<div align="center">

# Pen ID

**Handwriting Identification System based on Deep Learning**

Identifies writers from handwriting samples using Triplet CNN feature extraction and cosine distance matching.

[![Python](https://img.shields.io/badge/Python-3.11+-3776AB?style=flat-square&logo=python&logoColor=white)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-EE4C2C?style=flat-square&logo=pytorch&logoColor=white)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg?style=flat-square)](./LICENSE)
[![Flet](https://img.shields.io/badge/UI-Flet-00C8FF?style=flat-square)](https://flet.dev/)
[![OpenCV](https://img.shields.io/badge/OpenCV-4.8+-5C3EE8?style=flat-square&logo=opencv&logoColor=white)](https://opencv.org/)
<br>
[![Stars](https://img.shields.io/github/stars/Geekline-tech/PenID?style=flat-square)](https://github.com/Geekline-tech/PenID/stargazers)
[![Forks](https://img.shields.io/github/forks/Geekline-tech/PenID?style=flat-square)](https://github.com/Geekline-tech/PenID/network/members)
[![Issues](https://img.shields.io/github/issues/Geekline-tech/PenID?style=flat-square)](https://github.com/Geekline-tech/PenID/issues)
[![PRs](https://img.shields.io/github/issues-pr/Geekline-tech/PenID?style=flat-square)](https://github.com/Geekline-tech/PenID/pulls)

[English](./README_EN.md) | [简体中文](./README.md)

</div>

---

## Features

- **Writer Identification** — Upload a handwriting image, get Top-10 ranked matches with confidence scores
- **Scan Enhancement** — Built-in preprocessing pipeline for crease removal, lighting normalization, and perspective correction
- **Model Training** — Customizable Triplet CNN + Classification Head training with GPU acceleration
- **Gallery Management** — Add / remove persons, rebuild the feature database with one click
- **Multi-Patch Aggregation** — Automatically splits full-page images into patches and averages embeddings for more robust matching

## Tech Stack

| Module | Technology | Description |
|--------|-----------|-------------|
| Deep Learning | PyTorch + ResNet18 | ImageNet pretrained backbone |
| Feature Space | 512-d L2-normalized Embedding | Cosine distance matching |
| Loss Function | Triplet Loss + CrossEntropy | Joint metric space + classification optimization |
| Image Processing | OpenCV | Chinese path support, adaptive binarization |
| UI Framework | Flet | Material Design 3, cross-platform |
| Database | SQLite | Zero-config, stores person info and embeddings |

## Quick Start

### Requirements

- Python 3.11+
- CUDA (recommended, ~3x training speedup)

### Installation

```bash
git clone https://github.com/Geekline-tech/PenID.git
cd PenID

python -m venv venv
# Windows
venv\Scripts\activate
# macOS / Linux
source venv/bin/activate

pip install -r requirements.txt
```

### Run

```bash
python main.py
```

### Train

```bash
python train.py --epochs 60 --batch-size 32 --lr 1e-4 --embed-dim 512
```

### CLI Arguments

| Argument | Default | Description |
|----------|---------|-------------|
| `--epochs` | 100 | Training epochs |
| `--batch-size` | 64 | Batch size |
| `--lr` | 1e-4 | Learning rate |
| `--margin` | 0.3 | Triplet Loss margin |
| `--embed-dim` | 256 | Embedding dimension |
| `--device` | auto | Device (auto / cuda / cpu) |

## Model Architecture

```
Input (3, 64, 384)
    │
    ▼
ResNet18 Backbone (ImageNet pretrained)
    │
    ▼
FC 512 → BN → ReLU → Dropout(0.4) → FC 512
    │
    ▼
L2 Normalize → 512-d Embedding
    │
    ├──[Train]── Classification Head → N-class CE Loss
    │            Triplet Loss (margin=0.3)
    │
    └──[Infer]── Cosine Distance → Gallery Matching → Top-10
```

### Inference Pipeline

```
Input scan
  → Adaptive binarization (Otsu)
  → Horizontal projection line segmentation
  → Line → Patch splitting (64×384)
  → ResNet18 extracts 512-d feature per patch
  → Mean of all patch features = page embedding
  → Cosine distance to each gallery embedding
  → Return Top-10 ranked results
```

## Project Structure

```
PenID/
├── src/
│   ├── data/
│   │   ├── preprocess.py       # Line segmentation, patch splitting
│   │   ├── dataset.py          # Triplet Dataset
│   │   ├── augmentation.py     # Train / inference augmentation
│   │   └── scanner.py          # Scan enhancement (crease removal)
│   ├── models/
│   │   ├── pennet.py           # PenNet (ResNet18 + Embedding)
│   │   └── losses.py           # Triplet + CE Loss
│   ├── training/
│   │   └── trainer.py          # Training loop
│   ├── inference/
│   │   ├── matcher.py          # Embedder
│   │   └── gallery.py          # Gallery matching
│   ├── ui/
│   │   └── flet_app.py         # Flet UI
│   └── utils/
│       ├── config.py           # Global config
│       └── database.py         # SQLite
├── main.py                     # UI entry
├── train.py                    # CLI training entry
├── requirements.txt
├── LICENSE                     # MIT
└── docs/
    └── development_plan.md     # Development docs
```

## Contributing

Issues and Pull Requests are welcome!

```bash
# 1. Fork the repo
# 2. Create a feature branch
git checkout -b feature/amazing-feature
# 3. Commit changes
git commit -m 'feat: add amazing feature'
# 4. Push branch
git push origin feature/amazing-feature
# 5. Open a Pull Request
```

## License

[MIT License](./LICENSE) - Free to use, modify, and distribute with attribution.

## Acknowledgments

- [PyTorch](https://pytorch.org/) - Deep learning framework
- [Flet](https://flet.dev/) - Cross-platform UI framework
- [OpenCV](https://opencv.org/) - Computer vision library
