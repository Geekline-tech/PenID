<div align="center">

# Pen ID

**基于深度学习的手写笔迹识别系统**

通过 Triplet CNN 提取笔迹特征向量，实现闭集笔迹匹配与鉴定。

[![Python](https://img.shields.io/badge/Python-3.11+-3776AB?style=flat-square&logo=python&logoColor=white)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-EE4C2C?style=flat-square&logo=pytorch&logoColor=white)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg?style=flat-square)](./LICENSE)
[![PyQt5](https://img.shields.io/badge/UI-PyQt5%20Fluent-41CD52?style=flat-square&logo=qt&logoColor=white)](https://github.com/ChinaIceF/PyQt-Fluent-Widgets)
[![OpenCV](https://img.shields.io/badge/OpenCV-4.8+-5C3EE8?style=flat-square&logo=opencv&logoColor=white)](https://opencv.org/)
<br>
[![Stars](https://img.shields.io/github/stars/Geekline-tech/PenID?style=flat-square)](https://github.com/Geekline-tech/PenID/stargazers)
[![Forks](https://img.shields.io/github/forks/Geekline-tech/PenID?style=flat-square)](https://github.com/Geekline-tech/PenID/network/members)
[![Issues](https://img.shields.io/github/issues/Geekline-tech/PenID?style=flat-square)](https://github.com/Geekline-tech/PenID/issues)
[![PRs](https://img.shields.io/github/issues-pr/Geekline-tech/PenID?style=flat-square)](https://github.com/Geekline-tech/PenID/pulls)

[English](./README_EN.md) | 简体中文

</div>

---

## 目录

- [功能特性](#功能特性)
- [效果预览](#效果预览)
- [技术架构](#技术架构)
- [快速开始](#快速开始)
- [模型架构](#模型架构)
- [项目结构](#项目结构)
- [贡献指南](#贡献指南)
- [License](#license)

## 功能特性

- **笔迹识别** — 上传手写扫描图片，自动匹配已知人员，返回 Top-10 置信度排行
- **扫描增强** — 内置图像预处理流水线，支持去褶皱、光照归一化、透视矫正
- **模型训练** — 自定义参数训练 Triplet CNN + Classification Head，支持 GPU 加速
- **人员管理** — Gallery 特征库管理，支持添加 / 删除人员、一键重建特征库
- **多 Patch 聚合** — 整页识别时自动切分多 Patch 并取均值 Embedding，提升准确率

## 效果预览

| 识别页 | 训练页 | 人员管理 |
|:------:|:------:|:--------:|
| 上传图片 → Top-10 排行 | 实时训练曲线 + 数据统计 | 特征库 CRUD |

## 技术架构

```
┌─────────────────────────────────────────────────────────┐
│                    Flet UI (MD3)                         │
│   识别页 │ 训练页 │ 人员管理页                             │
├─────────────────────────────────────────────────────────┤
│                    Inference Layer                       │
│   Embedder (ResNet18)  │  Gallery Matcher (Cosine)      │
├─────────────────────────────────────────────────────────┤
│                    Training Layer                        │
│   Triplet Loss + CrossEntropy  │  AdamW + CosineAnneal  │
├─────────────────────────────────────────────────────────┤
│                    Data Layer                             │
│   OpenCV 预处理  │  SQLite 存储  │  PyTorch Dataset       │
└─────────────────────────────────────────────────────────┘
```

| 模块 | 技术 | 说明 |
|------|------|------|
| 深度学习 | PyTorch + ResNet18 | ImageNet 预训练，轻量高效 |
| 特征空间 | 512-d L2 归一化 Embedding | 余弦距离匹配 |
| 损失函数 | Triplet Loss + CE | 联合优化度量空间与分类边界 |
| 图像处理 | OpenCV | 中文路径兼容，自适应二值化 |
| UI 框架 | PyQt5 + Fluent Widgets | WinUI 3 风格，深色主题 |
| 数据库 | SQLite | 零配置，存储人员信息与特征向量 |

## 快速开始

### 环境要求

- Python 3.11+
- CUDA (推荐，GPU 训练 ~3x 加速)

### 安装

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

### 启动

```bash
python main.py
```

### 训练

```bash
python train.py --epochs 60 --batch-size 32 --lr 1e-4 --embed-dim 512
```

### CLI 参数

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--epochs` | 100 | 训练轮次 |
| `--batch-size` | 64 | 批大小 |
| `--lr` | 1e-4 | 学习率 |
| `--margin` | 0.3 | Triplet Loss 边距 |
| `--embed-dim` | 256 | 特征向量维度 |
| `--device` | auto | 设备 (auto / cuda / cpu) |

## 模型架构

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
    ├──[训练]── Classification Head → N-class CE Loss
    │           Triplet Loss (margin=0.3)
    │
    └──[推理]── Cosine Distance → Gallery Matching → Top-10
```

### 推理流水线

```
原始扫描件
  → 自适应二值化 (Otsu)
  → 水平投影行分割
  → 行 → Patch 切割 (64×384)
  → ResNet18 提取每个 Patch 的 512-d 特征
  → 取所有 Patch 特征均值 = 页面 Embedding
  → 与 Gallery 中每人 Embedding 计算余弦距离
  → 返回 Top-10 匹配结果
```

## 项目结构

```
PenID/
├── src/
│   ├── data/
│   │   ├── preprocess.py       # 行分割、Patch 切割
│   │   ├── dataset.py          # Triplet Dataset
│   │   ├── augmentation.py     # 训练/推理增强
│   │   └── scanner.py          # 扫描增强 (去褶皱)
│   ├── models/
│   │   ├── pennet.py           # PenNet (ResNet18 + Embedding)
│   │   └── losses.py           # Triplet + CE Loss
│   ├── training/
│   │   └── trainer.py          # 训练循环
│   ├── inference/
│   │   ├── matcher.py          # Embedder
│   │   └── gallery.py          # Gallery 匹配
│   ├── ui/
│   │   └── flet_app.py         # Flet UI
│   └── utils/
│       ├── config.py           # 全局配置
│       └── database.py         # SQLite
├── main.py                     # UI 入口
├── train.py                    # CLI 训练入口
├── requirements.txt
├── LICENSE                     # MIT
└── docs/
    └── development_plan.md     # 开发文档
```

## 贡献指南

欢迎提交 Issue 和 Pull Request！

```bash
# 1. Fork 本仓库
# 2. 创建特性分支
git checkout -b feature/amazing-feature
# 3. 提交更改
git commit -m 'feat: add amazing feature'
# 4. 推送分支
git push origin feature/amazing-feature
# 5. 创建 Pull Request
```

## License

[MIT License](./LICENSE) - 自由使用、修改、分发，需保留版权声明。

## 致谢

- [PyTorch](https://pytorch.org/) - 深度学习框架
- [Flet](https://flet.dev/) - 跨平台 UI 框架
- [OpenCV](https://opencv.org/) - 图像处理库
