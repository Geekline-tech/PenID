# Pen ID - 手写笔迹识别系统 开发文档

> 基于 Triplet CNN 的个人笔迹识别系统，支持 40 人已知集合的笔迹匹配与鉴定。

---

## 1. 项目概述

### 1.1 目标

利用 Triplet CNN 从手写扫描图片中提取笔迹特征向量，通过特征距离匹配判断一段未知笔迹属于已知 40 人中的哪一位。

### 1.2 核心流程

```
扫描图片 → 文字行分割 → 图像预处理 → CNN特征提取 → 特征向量(128/256维)
                                                           ↓
                                                    与已知库比对 → 匹配结果
```

---

## 2. 技术选型

| 模块 | 技术 | 说明 |
|------|------|------|
| 深度学习框架 | PyTorch | 动态图，调试方便 |
| 预训练骨干 | ResNet18 (轻量) / ResNet50 | ImageNet预训练，提取笔迹纹理特征 |
| 特征维度 | 256-d embedding | 足够表达40人的差异 |
| 损失函数 | Triplet Loss (semi-hard negative mining) + Classification Head 辅助 | 双重监督 |
| UI框架 | PySide6 (Qt6) | 现代桌面应用 |
| 图像处理 | OpenCV + Pillow | 图像分割、预处理 |
| 数据管理 | SQLite | 存储人员信息与特征向量 |

---

## 3. 数据组织与预处理

### 3.1 目录结构

```
data/
├── raw/                    # 原始扫描件
│   ├── person_001/
│   │   ├── page_001.png
│   │   ├── page_002.png
│   │   └── ...
│   ├── person_002/
│   └── ... (40人)
├── patches/                # 切割后的文字行/块
│   ├── person_001/
│   │   ├── patch_0001.png
│   │   └── ...
│   └── ...
└── samples/                # 用于UI演示的样本
```

### 3.2 预处理流水线

**步骤 1：文字行分割**

```python
# 基于投影直方图的行分割
# 1. 灰度化 → 二值化 (Otsu自适应阈值)
# 2. 水平投影直方图 → 寻找行间隙
# 3. 按行切割，去除空白行
# 4. 每行进一步切成固定宽度patch (256×64 或 128×128)
```

**步骤 2：数据增强 (训练时)**

```python
transforms = Compose([
    Resize((128, 128)),
    RandomAffine(degrees=2, translate=(0.05, 0.05), scale=(0.95, 1.05)),
    RandomPerspective(distortion_scale=0.05, p=0.3),
    ColorJitter(brightness=0.2, contrast=0.2),
    GaussianBlur(kernel_size=3, sigma=(0.1, 0.5)),
    Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])
```

**步骤 3：推断时预处理**

```python
inference_transform = Compose([
    Resize((128, 128)),
    Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])
```

### 3.3 Triplet 采样策略

```
对于每个训练batch:
  1. 随机选1个人 → 取2张同人patch (Anchor, Positive)
  2. 随机选1个不同人 → 取1张patch (Negative)
  3. 使用 semi-hard negative mining: 
     选 d(A,P) < d(A,N) < d(A,P) + margin 的困难负样本
```

---

## 4. 模型架构

### 4.1 特征提取网络

```python
class PenNet(nn.Module):
    """
    基于ResNet18的笔迹特征提取网络
    输入: (B, 3, 128, 128) RGB图像
    输出: (B, 256) L2归一化特征向量
    """
    def __init__(self, embed_dim=256):
        super().__init__()
        backbone = resnet18(pretrained=True)
        # 去掉最后的fc层
        self.features = nn.Sequential(*list(backbone.children())[:-1])  # (B, 512, 1, 1)
        self.embedding = nn.Sequential(
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Linear(256, embed_dim),
        )
    
    def forward(self, x):
        feat = self.features(x).flatten(1)   # (B, 512)
        emb = self.embedding(feat)            # (B, 256)
        emb = F.normalize(emb, p=2, dim=1)    # L2归一化
        return emb
```

### 4.2 分类辅助头 (训练时)

```python
class PenNetWithClassifier(nn.Module):
    def __init__(self, embed_dim=256, num_classes=40):
        super().__init__()
        self.pennet = PenNet(embed_dim)
        self.classifier = nn.Linear(embed_dim, num_classes)
    
    def forward(self, x):
        emb = self.pennet(x)
        logits = self.classifier(emb)
        return emb, logits
```

### 4.3 损失函数

```python
class CombinedLoss(nn.Module):
    """
    Triplet Loss + CrossEntropy 联合损失
    总损失 = triplet_loss + λ * cls_loss
    """
    def __init__(self, margin=0.3, lambda_cls=0.5):
        super().__init__()
        self.triplet_loss = nn.TripletMarginLoss(margin=margin, p=2)
        self.cls_loss = nn.CrossEntropyLoss()
        self.lambda_cls = lambda_cls
    
    def forward(self, emb_anchor, emb_pos, emb_neg, logits, labels):
        t_loss = self.triplet_loss(emb_anchor, emb_pos, emb_neg)
        c_loss = self.cls_loss(logits, labels)
        return t_loss + self.lambda_cls * c_loss, t_loss, c_loss
```

### 4.4 推理匹配

```python
def identify_person(query_embedding, gallery_embeddings, threshold=0.6):
    """
    query_embedding: (256,) 未知笔迹特征
    gallery_embeddings: dict {person_id: (256,)} 已知库特征
    threshold: 余弦距离阈值，低于此值认为匹配
    
    返回: (person_id, confidence, distance)
    """
    min_dist = float('inf')
    best_id = None
    for pid, emb in gallery_embeddings.items():
        dist = 1 - cosine_similarity(query_embedding, emb)  # 余弦距离
        if dist < min_dist:
            min_dist = dist
            best_id = pid
    
    confidence = max(0, 1 - min_dist)  # 简单置信度映射
    return best_id, confidence, min_dist
```

---

## 5. 训练流程

### 5.1 训练配置

```python
config = {
    "epochs": 100,
    "batch_size": 64,          # 64 triplets per batch
    "lr": 1e-4,
    "weight_decay": 1e-5,
    "lr_scheduler": "cosine",  # 余弦退火
    "embedding_dim": 256,
    "triplet_margin": 0.3,
    "image_size": (128, 128),
    "train_split": 0.8,        # 80% 训练，20% 验证
}
```

### 5.2 训练伪代码

```python
for epoch in range(config["epochs"]):
    for batch in dataloader:
        anchor, positive, negative, labels = batch
        
        emb_a, logits_a = model(anchor)
        emb_p, _        = model(positive)
        emb_n, _        = model(negative)
        
        loss, t_loss, c_loss = criterion(emb_a, emb_p, emb_n, logits_a, labels)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    
    # 验证: 计算匹配准确率
    val_acc = evaluate(model, val_gallery, val_queries)
    scheduler.step()
    save_if_best(model, val_acc)
```

### 5.3 Gallery 特征库构建

训练完成后，对每人的所有patch提取特征并取均值，存入数据库：

```python
def build_gallery(model, data_dir):
    """构建已知人员特征库"""
    gallery = {}
    for person_dir in Path(data_dir).iterdir():
        embeddings = []
        for patch in person_dir.glob("*.png"):
            img = load_and_preprocess(patch)
            with torch.no_grad():
                emb = model(img.unsqueeze(0))
            embeddings.append(emb)
        gallery[person_dir.name] = torch.stack(embeddings).mean(dim=0)
    return gallery
```

---

## 6. 项目目录结构

```
PenID/
├── docs/
│   └── development_plan.md      # 本文档
├── src/
│   ├── __init__.py
│   ├── data/
│   │   ├── __init__.py
│   │   ├── preprocess.py        # 图像预处理、行分割
│   │   ├── dataset.py           # TripletDataset 定义
│   │   └── augmentation.py      # 数据增强策略
│   ├── models/
│   │   ├── __init__.py
│   │   ├── pennet.py            # PenNet 特征提取网络
│   │   └── losses.py            # Triplet + Classification 损失
│   ├── training/
│   │   ├── __init__.py
│   │   ├── trainer.py           # 训练循环
│   │   ├── sampler.py           # Triplet采样器
│   │   └── config.py            # 训练配置
│   ├── inference/
│   │   ├── __init__.py
│   │   ├── matcher.py           # 特征匹配与识别
│   │   └── gallery.py           # Gallery特征库管理
│   ├── ui/
│   │   ├── __init__.py
│   │   ├── main_window.py       # 主窗口
│   │   ├── identify_page.py     # 识别页面
│   │   ├── train_page.py        # 训练页面
│   │   └── gallery_page.py      # 人员管理页面
│   └── utils/
│       ├── __init__.py
│       ├── database.py          # SQLite 操作
│       └── config.py            # 全局配置
├── assets/
│   └── images/                  # 图标和占位图
├── tests/
│   ├── test_preprocess.py
│   ├── test_model.py
│   └── test_matcher.py
├── models/                      # 训练好的模型权重
│   └── pennet_best.pth
├── requirements.txt
├── train.py                     # 训练入口
└── main.py                      # UI启动入口
```

---

## 7. UI 设计方案

### 7.1 UI 库选择: PyQt-Fluent-Widgets

使用 **PyQt-Fluent-Widgets** 库，提供现成的 Fluent Design 风格组件，开箱即用：

```bash
pip install PyQt-Fluent-Widgets
```

**优势**：
- 现代 Fluent Design 风格，开箱即用
- 内置导航栏、卡片、按钮、进度条等精美组件
- 支持深色/浅色主题切换
- 中文文档完善

### 7.2 核心组件映射

| 功能 | 使用组件 |
|------|---------|
| 左侧导航 | `NavigationInterface` + `NavigationPanel` |
| 页面容器 | `StackedWidget` |
| 图片上传 | `ImageLabel` + `PushSettingCard` |
| 按钮 | `PrimaryPushButton` |
| 进度条 | `ProgressBar` |
| 表格 | `TableWidget` |
| 设置项 | `ExpandSettingCard` / `ComboBox` |

### 7.3 主窗口代码示例

```python
# main.py
import sys
from PyQt5.QtWidgets import QApplication
from PyQt5.QtCore import Qt, QSize
from qfluentwidgets import (
    NavigationInterface, NavigationItemPosition,
    FluentIcon as FIF, setTheme, Theme
)
from qfluentwidgets.components.widgets.stacked_widget import StackedWidget

from src.ui.identify_page import IdentifyPage
from src.ui.train_page import TrainPage
from src.ui.gallery_page import GalleryPage

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Pen ID - 笔迹识别系统")
        self.setMinimumSize(1000, 700)
        
        # 设置主题
        setTheme(Theme.DARK)
        
        # 创建导航栏
        self.navigation = NavigationInterface(self)
        self.navigation.setExpandWidth(200)
        
        # 创建页面容器
        self.stackWidget = StackedWidget(self)
        
        # 添加页面
        self.identifyPage = IdentifyPage()
        self.trainPage = TrainPage()
        self.galleryPage = GalleryPage()
        
        self.stackWidget.addWidget(self.identifyPage)
        self.stackWidget.addWidget(self.trainPage)
        self.stackWidget.addWidget(self.galleryPage)
        
        # 添加导航项
        self.navigation.addItem(
            route="identify",
            icon=FIF.SEARCH,
            text="识别",
            onClick=lambda: self.stackWidget.setCurrentIndex(0)
        )
        self.navigation.addItem(
            route="train",
            icon=FIF.TRAIN,
            text="训练",
            onClick=lambda: self.stackWidget.setCurrentIndex(1)
        )
        self.navigation.addItem(
            route="gallery",
            icon=FIF.PEOPLE,
            text="人员管理",
            onClick=lambda: self.stackWidget.setCurrentIndex(2)
        )
        
        # 布局
        self.hBoxLayout = QHBoxLayout(self)
        self.hBoxLayout.addWidget(self.navigation)
        self.hBoxLayout.addWidget(self.stackWidget)
        self.hBoxLayout.setContentsMargins(0, 0, 0, 0)

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec())
```

### 7.4 识别页面示例

```python
# src/ui/identify_page.py
from qfluentwidgets import (
    PrimaryPushButton, ImageLabel, BodyLabel,
    CaptionLabel, CardWidget, FlowLayout
)

class IdentifyPage(QWidget):
    def __init__(self):
        super().__init__()
        self.initUI()
    
    def initUI(self):
        layout = QVBoxLayout(self)
        
        # 图片预览区
        self.imageCard = CardWidget(self)
        self.imageLabel = ImageLabel(self.imageCard)
        self.imageLabel.setFixedSize(400, 250)
        self.imageLabel.setImage(":/images/placeholder.png")
        
        # 上传按钮
        self.uploadBtn = PrimaryPushButton("上传笔迹图片", self)
        self.uploadBtn.clicked.connect(self.uploadImage)
        
        # 结果卡片
        self.resultCard = CardWidget(self)
        self.nameLabel = BodyLabel("识别结果: --", self.resultCard)
        self.confLabel = CaptionLabel("置信度: --", self.resultCard)
        
        # 布局
        layout.addWidget(self.imageCard)
        layout.addWidget(self.uploadBtn)
        layout.addWidget(self.resultCard)
        layout.addStretch()
    
    def uploadImage(self):
        path, _ = QFileDialog.getOpenFileName(
            self, "选择图片", "", "Images (*.png *.jpg)"
        )
        if path:
            self.imageLabel.setImage(path)
            self.identify(path)
    
    def identify(self, image_path):
        # TODO: 调用模型进行识别
        self.nameLabel.setText("识别结果: 张三")
        self.confLabel.setText("置信度: 95.2%")
```

### 7.5 页面布局预览

```
┌──────────────────────────────────────────────────────────┐
│  Pen ID - 笔迹识别系统                                   │
├────────────┬─────────────────────────────────────────────┤
│            │                                             │
│  🔍 识别   │  ┌─────────────────────────────────────────┐│
│            │  │                                         ││
│  🚂 训练   │  │         上传笔迹图片区域                 ││
│            │  │         (支持拖拽)                      ││
│  👥 人员   │  │                                         ││
│     管理   │  └─────────────────────────────────────────┘│
│            │                                             │
│            │  [ 上传笔迹图片 ]                            │
│            │                                             │
│            │  ┌─────────────────────────────────────────┐│
│            │  │  识别结果: 张三                          ││
│            │  │  置信度: 95.2%                          ││
│            │  │  匹配距离: 0.18                         ││
│            │  └─────────────────────────────────────────┘│
│            │                                             │
└────────────┴─────────────────────────────────────────────┘
```

---

## 8. 依赖清单

```txt
# requirements.txt
torch>=2.0.0
torchvision>=0.15.0
PyQt5>=5.15.0
PyQt-Fluent-Widgets>=1.0.0
opencv-python>=4.8.0
Pillow>=10.0.0
numpy>=1.24.0
scikit-learn>=1.3.0
matplotlib>=3.7.0
tqdm>=4.65.0
```

---

## 9. 开发路线

| 阶段 | 内容 | 预估工时 |
|------|------|----------|
| **Phase 1** | 数据预处理: 行分割 + patch切割 + 可视化验证 | 2天 |
| **Phase 2** | 模型开发: PenNet + TripletLoss + 训练脚本 | 2天 |
| **Phase 3** | 训练调参: 40人数据训练，达到 >95% 匹配准确率 | 1-2天 |
| **Phase 4** | 推理模块: Gallery构建 + 匹配算法 + 置信度校准 | 1天 |
| **Phase 5** | UI开发: PyQt-Fluent-Widgets 三页面 + 样式美化 | 1-2天 |
| **Phase 6** | 联调测试 + 边界情况处理 | 1-2天 |

---

## 10. 关键设计决策

### 10.1 为什么选 Triplet CNN 而非纯分类

- **泛化能力**: 度量学习学到的是"笔迹差异"的度量空间，而非硬分类边界
- **可扩展**: 未来增加新人无需重新训练整个分类层，只需提取新特征入库
- **小样本友好**: 即使每人样本不均，triplet loss 仍能有效学习

### 10.2 为什么用 ResNet18 而非更深网络

- 手写笔迹识别的输入图像相对简单 (128×128灰度/RGB)
- ResNet18 参数量仅 ~11M，训练快、推理快
- 40人的数据量不需要过深的网络来防止过拟合

### 10.3 为什么加 Classification Head 辅助

- 纯 Triplet Loss 训练初期收敛慢
- 分类损失提供更直接的梯度信号，加速收敛
- 训练时双损失，推理时只用 embedding (分类头丢弃)

---

## 11. 测试策略

| 测试类型 | 内容 |
|---------|------|
| 单元测试 | 预处理函数、patch切割正确性 |
| 模型测试 | 前向传播形状、梯度流 |
| 匹配测试 | 构造已知样本，验证匹配准确率 |
| UI测试 | 页面跳转、图片上传、结果显示 |
| 集成测试 | 端到端: 上传图片 → 预处理 → 模型 → 显示结果 |

---

## 12. 待确认事项

- [ ] 扫描件的DPI和分辨率范围
- [ ] 是否需要支持手写中文/英文/混合
- [ ] 是否需要实时识别 (摄像头) 还是仅支持图片上传
- [ ] 输出报告格式 (PDF/截图)
