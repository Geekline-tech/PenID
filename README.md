<div align="center">

# Pen ID

基于深度学习的手写笔迹识别系统

[![Python](https://img.shields.io/badge/Python-3.11+-3776AB?style=flat-square&logo=python&logoColor=white)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-EE4C2C?style=flat-square&logo=pytorch&logoColor=white)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg?style=flat-square)](./LICENSE)
[![Flet](https://img.shields.io/badge/UI-Flet-00C8FF?style=flat-square)](https://flet.dev/)
[![OpenCV](https://img.shields.io/badge/OpenCV-4.8+-5C3EE8?style=flat-square&logo=opencv&logoColor=white)](https://opencv.org/)

[![Stars](https://img.shields.io/github/stars/Geekline-tech/PenID?style=flat-square)](https://github.com/Geekline-tech/PenID/stargazers)
[![Forks](https://img.shields.io/github/forks/Geekline-tech/PenID?style=flat-square)](https://github.com/Geekline-tech/PenID/network/members)
[![Issues](https://img.shields.io/github/issues/Geekline-tech/PenID?style=flat-square)](https://github.com/Geekline-tech/PenID/issues)
[![Pull Requests](https://img.shields.io/github/issues-pr/Geekline-tech/PenID?style=flat-square)](https://github.com/Geekline-tech/PenID/pulls)

[English](./README.md) | 简体中文

---

通过 Triplet CNN 提取笔迹特征向量，实现 40 人闭集笔迹匹配与鉴定。

</div>

## 功能

- **笔迹识别** — 上传手写扫描图片，自动匹配到已知人员，显示 Top-10 置信度排行
- **扫描增强** — 内置图像预处理，支持皱褶纸张、光照不均的扫描件增强
- **模型训练** — 支持自定义参数训练 Triplet CNN + Classification Head 模型
- **人员管理** — Gallery 特征库管理，支持添加/删除人员、重建特征库

## 技术栈

| 模块 | 技术 |
|------|------|
| 深度学习 | PyTorch + ResNet18 (ImageNet 预训练) |
| 特征提取 | 512-d L2 归一化 Embedding |
| 损失函数 | Triplet Loss + CrossEntropy 联合损失 |
| 图像处理 | OpenCV (中文路径兼容) |
| UI 框架 | Flet (Material Design 3) |
| 数据管理 | SQLite |

## 项目结构

```
PenID/
├── src/
│   ├── data/
│   │   ├── preprocess.py        # 图像预处理、行分割、Patch切割
│   │   ├── dataset.py           # TripletDataset
│   │   ├── augmentation.py      # 训练/推理数据增强
│   │   └── scanner.py           # 扫描增强 (去褶皱、透视矫正)
│   ├── models/
│   │   ├── pennet.py            # PenNet 特征提取网络
│   │   └── losses.py            # Triplet + Classification 损失
│   ├── training/
│   │   └── trainer.py           # 训练循环
│   ├── inference/
│   │   ├── matcher.py           # Embedder 特征提取
│   │   └── gallery.py           # Gallery 特征库 + 匹配
│   ├── ui/
│   │   └── flet_app.py          # Flet UI (识别/训练/人员管理)
│   └── utils/
│       ├── config.py            # 全局配置
│       └── database.py          # SQLite 数据库
├── main.py                      # UI 启动入口
├── train.py                     # 训练入口 (CLI)
├── requirements.txt
├── LICENSE                      # MIT License
└── docs/
    └── development_plan.md      # 开发文档
```

## 快速开始

### 环境要求

- Python 3.11+
- CUDA (推荐，用于 GPU 加速训练)

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

### 启动 UI

```bash
python main.py
```

### 训练模型

```bash
# 准备数据: 将每人扫描件放入 data/raw/{person_id}/ 目录
# 每人至少 2 张扫描图片

python train.py --epochs 60 --batch-size 32 --lr 1e-4 --embed-dim 512
```

### 命令行参数

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--epochs` | 100 | 训练轮次 |
| `--batch-size` | 64 | 批大小 |
| `--lr` | 1e-4 | 学习率 |
| `--margin` | 0.3 | Triplet Loss 边距 |
| `--embed-dim` | 256 | 特征向量维度 |
| `--device` | auto | 设备 (auto/cuda/cpu) |

## 数据准备

```
data/raw/
├── 2820987/
│   ├── 2820987_1.jpg      # 扫描件 1
│   └── 2820987_2.jpg      # 扫描件 2
├── 2820989/
│   └── ...
└── ... (40人)
```

- 每人文件夹以学号命名
- 每人至少 2 张扫描图片 (建议 3-5 张以提升准确率)
- 图片格式: JPG / PNG

## 模型架构

```
输入 (3, 64, 384)
    ↓
ResNet18 Backbone (ImageNet 预训练)
    ↓
512-d Embedding (BatchNorm + Dropout + L2归一化)
    ↓
[训练时] Classification Head → 40类分类
[推理时] Cosine距离匹配 → Top-10 排行
```

## 推理流程

```
原始扫描件 → 二值化 → 行分割 → Patch切割(64×384)
    → CNN提取每个Patch的512-d特征
    → 取所有Patch特征均值 = 页面Embedding
    → 与Gallery中每人Embedding计算余弦距离
    → 返回Top-10匹配结果
```

## 贡献

欢迎提交 Issue 和 Pull Request！

1. Fork 本仓库
2. 创建特性分支 (`git checkout -b feature/amazing-feature`)
3. 提交更改 (`git commit -m 'Add amazing feature'`)
4. 推送到分支 (`git push origin feature/amazing-feature`)
5. 创建 Pull Request

## License

本项目采用 [MIT License](./LICENSE) 开源协议。

## 致谢

- [PyTorch](https://pytorch.org/)
- [Flet](https://flet.dev/)
- [OpenCV](https://opencv.org/)
