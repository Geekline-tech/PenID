# Pen ID

基于深度学习的手写笔迹识别系统，通过 Triplet CNN 提取笔迹特征向量，实现 40 人闭集笔迹匹配与鉴定。

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
venv\Scripts\activate          # Windows
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
- 图片格式: JPG/PNG

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

## License

本项目提供以下开源协议供选择：

| 协议 | 特点 | 适用场景 |
|------|------|----------|
| **MIT** | 最宽松，允许商用，只需保留版权声明 | 希望最大化传播和使用 |
| **Apache 2.0** | 类似 MIT，额外提供专利授权保护 | 担心专利纠纷的商业项目 |
| **GPL-3.0** | Copyleft，衍生作品必须开源 | 希望衍生项目也保持开源 |
| **BSD-3-Clause** | 与 MIT 几乎等同，措辞略有不同 | 学术/研究项目 |

**推荐：MIT License** — 本项目为教学研究性质，MIT 协议最简洁、传播最广，适合最大化开源影响力。

## 贡献

欢迎提交 Issue 和 Pull Request。

## 致谢

- [PyTorch](https://pytorch.org/)
- [Flet](https://flet.dev/)
- [OpenCV](https://opencv.org/)
