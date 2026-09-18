from PySide6.QtWidgets import (
    QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QLabel, QPushButton, QStackedWidget, QFileDialog,
    QProgressBar, QTableWidget, QTableWidgetItem, QHeaderView,
    QLineEdit, QMessageBox, QTextEdit, QGroupBox, QSpinBox,
    QDoubleSpinBox, QFrame, QSizePolicy, QSplitter
)
from PySide6.QtCore import Qt, QThread, Signal, QSize
from PySide6.QtGui import QPixmap, QImage, QFont, QIcon, QColor, QPalette
import cv2
import numpy as np
from pathlib import Path
from typing import Optional
from src.utils.config import Config, RAW_DIR, PATCHES_DIR
from src.utils.database import Database
from src.inference.gallery import Gallery
from src.data.preprocess import preprocess_page, prepare_person_data


THEME_STYLE = """
QMainWindow {
    background-color: #1e1e2e;
}
QWidget {
    color: #cdd6f4;
    font-family: "Microsoft YaHei", "Segoe UI", sans-serif;
    font-size: 13px;
}
QLabel {
    color: #cdd6f4;
}
QPushButton {
    background-color: #313244;
    border: 1px solid #45475a;
    border-radius: 6px;
    padding: 8px 16px;
    color: #cdd6f4;
    font-weight: bold;
}
QPushButton:hover {
    background-color: #45475a;
    border-color: #585b70;
}
QPushButton:pressed {
    background-color: #585b70;
}
QPushButton#primary {
    background-color: #89b4fa;
    color: #1e1e2e;
    border: none;
}
QPushButton#primary:hover {
    background-color: #74c7ec;
}
QPushButton#danger {
    background-color: #f38ba8;
    color: #1e1e2e;
    border: none;
}
QPushButton#danger:hover {
    background-color: #eba0ac;
}
QPushButton#nav {
    background-color: transparent;
    border: none;
    border-radius: 8px;
    padding: 12px 16px;
    text-align: left;
    font-size: 14px;
}
QPushButton#nav:hover {
    background-color: #313244;
}
QPushButton#nav[active="true"] {
    background-color: #313244;
    color: #89b4fa;
}
QGroupBox {
    border: 1px solid #45475a;
    border-radius: 8px;
    margin-top: 12px;
    padding-top: 20px;
    font-weight: bold;
}
QGroupBox::title {
    subcontrol-origin: margin;
    left: 16px;
    padding: 0 8px;
}
QLineEdit, QSpinBox, QDoubleSpinBox {
    background-color: #313244;
    border: 1px solid #45475a;
    border-radius: 6px;
    padding: 8px;
    color: #cdd6f4;
}
QLineEdit:focus, QSpinBox:focus, QDoubleSpinBox:focus {
    border-color: #89b4fa;
}
QProgressBar {
    background-color: #313244;
    border: none;
    border-radius: 4px;
    text-align: center;
    color: #1e1e2e;
    font-weight: bold;
    height: 20px;
}
QProgressBar::chunk {
    background-color: #89b4fa;
    border-radius: 4px;
}
QTableWidget {
    background-color: #181825;
    border: 1px solid #45475a;
    border-radius: 8px;
    gridline-color: #313244;
}
QTableWidget::item {
    padding: 8px;
}
QTableWidget::item:selected {
    background-color: #313244;
    color: #89b4fa;
}
QHeaderView::section {
    background-color: #313244;
    padding: 8px;
    border: none;
    border-bottom: 1px solid #45475a;
    font-weight: bold;
    color: #a6adc8;
}
QTextEdit {
    background-color: #181825;
    border: 1px solid #45475a;
    border-radius: 8px;
    padding: 8px;
    color: #a6e3a1;
    font-family: "Cascadia Code", "Consolas", monospace;
}
QMessageBox {
    background-color: #1e1e2e;
}
"""


class CardFrame(QFrame):
    def __init__(self, title: str = "", parent=None):
        super().__init__(parent)
        self.setFrameShape(QFrame.StyledPanel)
        self.setStyleSheet("""
            CardFrame {
                background-color: #181825;
                border: 1px solid #45475a;
                border-radius: 12px;
                padding: 16px;
            }
        """)
        self.layout = QVBoxLayout(self)
        self.layout.setSpacing(12)
        self.layout.setContentsMargins(16, 16, 16, 16)
        if title:
            title_label = QLabel(title)
            title_label.setFont(QFont("Microsoft YaHei", 14, QFont.Bold))
            title_label.setStyleSheet("color: #cdd6f4; border: none;")
            self.layout.addWidget(title_label)


class ResultCard(QFrame):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setFrameShape(QFrame.StyledPanel)
        self.setStyleSheet("""
            ResultCard {
                background-color: #181825;
                border: 2px solid #45475a;
                border-radius: 12px;
                padding: 20px;
            }
        """)
        layout = QVBoxLayout(self)
        layout.setSpacing(8)

        self.name_label = QLabel("--")
        self.name_label.setFont(QFont("Microsoft YaHei", 20, QFont.Bold))
        self.name_label.setStyleSheet("color: #89b4fa; border: none;")
        self.name_label.setAlignment(Qt.AlignCenter)

        self.conf_label = QLabel("置信度: --")
        self.conf_label.setFont(QFont("Microsoft YaHei", 12))
        self.conf_label.setStyleSheet("color: #a6adc8; border: none;")
        self.conf_label.setAlignment(Qt.AlignCenter)

        self.status_label = QLabel("")
        self.status_label.setFont(QFont("Microsoft YaHei", 11, QFont.Bold))
        self.status_label.setStyleSheet("border: none;")
        self.status_label.setAlignment(Qt.AlignCenter)

        layout.addWidget(self.name_label)
        layout.addWidget(self.conf_label)
        layout.addWidget(self.status_label)

    def set_result(self, name: str, confidence: float, matched: bool):
        self.name_label.setText(name)
        self.conf_label.setText(f"置信度: {confidence * 100:.1f}%")
        if matched:
            self.status_label.setText("✓ 匹配成功")
            self.status_label.setStyleSheet("color: #a6e3a1; border: none;")
            self.setStyleSheet("""
                ResultCard {
                    background-color: #181825;
                    border: 2px solid #a6e3a1;
                    border-radius: 12px;
                    padding: 20px;
                }
            """)
        else:
            self.status_label.setText("✗ 无法匹配")
            self.status_label.setStyleSheet("color: #f38ba8; border: none;")
            self.setStyleSheet("""
                ResultCard {
                    background-color: #181825;
                    border: 2px solid #f38ba8;
                    border-radius: 12px;
                    padding: 20px;
                }
            """)

    def clear(self):
        self.name_label.setText("--")
        self.conf_label.setText("置信度: --")
        self.status_label.setText("")
        self.setStyleSheet("""
            ResultCard {
                background-color: #181825;
                border: 2px solid #45475a;
                border-radius: 12px;
                padding: 20px;
            }
        """)


class IdentifyPage(QWidget):
    def __init__(self, gallery: Gallery, parent=None):
        super().__init__(parent)
        self.gallery = gallery
        self.current_image: Optional[np.ndarray] = None
        self.init_ui()

    def init_ui(self):
        layout = QVBoxLayout(self)
        layout.setSpacing(16)
        layout.setContentsMargins(24, 24, 24, 24)

        # Upload area
        upload_card = CardFrame("上传笔迹图片")
        self.image_label = QLabel("点击下方按钮或拖拽图片到此处")
        self.image_label.setAlignment(Qt.AlignCenter)
        self.image_label.setMinimumHeight(250)
        self.image_label.setStyleSheet("""
            QLabel {
                background-color: #11111b;
                border: 2px dashed #45475a;
                border-radius: 12px;
                color: #6c7086;
                font-size: 14px;
            }
        """)
        self.image_label.setAcceptDrops(True)
        upload_card.layout.addWidget(self.image_label)

        btn_row = QHBoxLayout()
        self.upload_btn = QPushButton("📂 上传图片")
        self.upload_btn.setObjectName("primary")
        self.upload_btn.clicked.connect(self.upload_image)
        self.identify_btn = QPushButton("🔍 开始识别")
        self.identify_btn.setObjectName("primary")
        self.identify_btn.clicked.connect(self.identify)
        self.identify_btn.setEnabled(False)
        self.clear_btn = QPushButton("🗑 清除")
        self.clear_btn.clicked.connect(self.clear)
        btn_row.addWidget(self.upload_btn)
        btn_row.addWidget(self.identify_btn)
        btn_row.addWidget(self.clear_btn)
        upload_card.layout.addLayout(btn_row)
        layout.addWidget(upload_card)

        # Result area
        self.result_card = ResultCard()
        layout.addWidget(self.result_card)

        # Detail
        detail_card = CardFrame("匹配详情")
        self.detail_text = QTextEdit()
        self.detail_text.setReadOnly(True)
        self.detail_text.setMaximumHeight(80)
        self.detail_text.setPlaceholderText("识别结果将显示在此处...")
        detail_card.layout.addWidget(self.detail_text)
        layout.addWidget(detail_card)

        layout.addStretch()

    def upload_image(self):
        path, _ = QFileDialog.getOpenFileName(
            self, "选择笔迹图片", "", "Images (*.png *.jpg *.jpeg *.bmp)"
        )
        if path:
            self.load_image(path)

    def load_image(self, path: str):
        img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        if img is None:
            QMessageBox.warning(self, "错误", f"无法加载图片: {path}")
            return
        self.current_image = img
        self.show_image(img)
        self.identify_btn.setEnabled(True)
        self.result_card.clear()
        self.detail_text.clear()

    def show_image(self, img: np.ndarray):
        h, w = img.shape
        bytes_per_line = w
        q_img = QImage(img.data, w, h, bytes_per_line, QImage.Format_Grayscale8)
        pixmap = QPixmap.fromImage(q_img)
        scaled = pixmap.scaled(
            self.image_label.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation
        )
        self.image_label.setPixmap(scaled)

    def identify(self):
        if self.current_image is None:
            return
        self.identify_btn.setEnabled(False)
        self.identify_btn.setText("识别中...")
        try:
            result = self.gallery.identify_image(self.current_image)
            self.result_card.set_result(result["name"], result["confidence"], result["matched"])
            self.detail_text.setText(
                f"匹配人员: {result['name']}\n"
                f"余弦距离: {result['distance']:.4f}\n"
                f"置信度: {result['confidence'] * 100:.1f}%"
            )
        except Exception as e:
            QMessageBox.critical(self, "识别失败", str(e))
        finally:
            self.identify_btn.setEnabled(True)
            self.identify_btn.setText("🔍 开始识别")

    def clear(self):
        self.current_image = None
        self.image_label.clear()
        self.image_label.setText("点击下方按钮或拖拽图片到此处")
        self.identify_btn.setEnabled(False)
        self.result_card.clear()
        self.detail_text.clear()


class TrainWorker(QThread):
    progress = Signal(str)
    finished = Signal(bool, str)

    def __init__(self, config: Config):
        super().__init__()
        self.config = config
        self._running = True

    def run(self):
        try:
            from src.training.trainer import Trainer
            trainer = Trainer(self.config, progress_callback=lambda m: self.progress.emit(m))
            trainer.train()
            self.finished.emit(True, f"训练完成! 最佳准确率: {trainer.best_acc:.1f}%")
        except Exception as e:
            self.finished.emit(False, str(e))

    def stop(self):
        self._running = False


class TrainPage(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.worker: Optional[TrainWorker] = None
        self.init_ui()

    def init_ui(self):
        layout = QVBoxLayout(self)
        layout.setSpacing(16)
        layout.setContentsMargins(24, 24, 24, 24)

        # Dataset info
        info_card = CardFrame("数据集状态")
        self.info_text = QTextEdit()
        self.info_text.setReadOnly(True)
        self.info_text.setMaximumHeight(100)
        self.info_text.setStyleSheet("""
            QTextEdit {
                background-color: #11111b;
                border: 1px solid #45475a;
                border-radius: 8px;
                color: #a6adc8;
                padding: 12px;
            }
        """)
        info_card.layout.addWidget(self.info_text)
        layout.addWidget(info_card)

        # Parameters
        param_card = CardFrame("训练参数")
        params_layout = QHBoxLayout()

        params_layout.addWidget(QLabel("学习率:"))
        self.lr_input = QDoubleSpinBox()
        self.lr_input.setDecimals(5)
        self.lr_input.setRange(0.00001, 1.0)
        self.lr_input.setValue(0.0001)
        self.lr_input.setSingleStep(0.0001)
        params_layout.addWidget(self.lr_input)

        params_layout.addWidget(QLabel("轮次:"))
        self.epochs_input = QSpinBox()
        self.epochs_input.setRange(1, 1000)
        self.epochs_input.setValue(100)
        params_layout.addWidget(self.epochs_input)

        params_layout.addWidget(QLabel("Batch:"))
        self.batch_input = QSpinBox()
        self.batch_input.setRange(1, 256)
        self.batch_input.setValue(64)
        params_layout.addWidget(self.batch_input)

        params_layout.addWidget(QLabel("边距:"))
        self.margin_input = QDoubleSpinBox()
        self.margin_input.setDecimals(2)
        self.margin_input.setRange(0.1, 2.0)
        self.margin_input.setValue(0.3)
        params_layout.addWidget(self.margin_input)

        param_card.layout.addLayout(params_layout)
        layout.addWidget(param_card)

        # Control buttons
        btn_row = QHBoxLayout()
        self.start_btn = QPushButton("🚀 开始训练")
        self.start_btn.setObjectName("primary")
        self.start_btn.clicked.connect(self.start_training)
        self.stop_btn = QPushButton("⏹ 停止训练")
        self.stop_btn.setObjectName("danger")
        self.stop_btn.setEnabled(False)
        self.stop_btn.clicked.connect(self.stop_training)
        btn_row.addWidget(self.start_btn)
        btn_row.addWidget(self.stop_btn)
        layout.addLayout(btn_row)

        # Progress
        progress_card = CardFrame("训练进度")
        self.progress_bar = QProgressBar()
        self.progress_bar.setValue(0)
        progress_card.layout.addWidget(self.progress_bar)

        self.status_label = QLabel("就绪")
        self.status_label.setStyleSheet("color: #a6adc8; border: none;")
        progress_card.layout.addWidget(self.status_label)

        self.log_text = QTextEdit()
        self.log_text.setReadOnly(True)
        self.log_text.setMaximumHeight(200)
        progress_card.layout.addWidget(self.log_text)
        layout.addWidget(progress_card)

        layout.addStretch()

        self.refresh_info()

    def refresh_info(self):
        db = Database()
        persons = db.get_persons()
        total_samples = sum(p["sample_count"] for p in persons)
        self.info_text.setText(
            f"人员数: {len(persons)}\n"
            f"总样本数: {total_samples} patches\n"
            f"每人均值: {total_samples // max(len(persons), 1)} patches"
        )
        db.close()

    def start_training(self):
        config = Config()
        config.update({
            "lr": self.lr_input.value(),
            "epochs": self.epochs_input.value(),
            "batch_size": self.batch_input.value(),
            "triplet_margin": self.margin_input.value(),
        })
        self.worker = TrainWorker(config)
        self.worker.progress.connect(self.on_progress)
        self.worker.finished.connect(self.on_finished)
        self.worker.start()
        self.start_btn.setEnabled(False)
        self.stop_btn.setEnabled(True)
        self.log_text.clear()

    def stop_training(self):
        if self.worker:
            self.worker.stop()

    def on_progress(self, msg: str):
        self.log_text.append(msg)
        self.status_label.setText(msg)

    def on_finished(self, success: bool, msg: str):
        self.start_btn.setEnabled(True)
        self.stop_btn.setEnabled(False)
        if success:
            self.status_label.setText(msg)
            QMessageBox.information(self, "训练完成", msg)
        else:
            self.status_label.setText(f"训练失败: {msg}")
            QMessageBox.critical(self, "训练失败", msg)


class GalleryPage(QWidget):
    def __init__(self, gallery: Gallery, parent=None):
        super().__init__(parent)
        self.gallery = gallery
        self.init_ui()
        self.refresh_table()

    def init_ui(self):
        layout = QVBoxLayout(self)
        layout.setSpacing(16)
        layout.setContentsMargins(24, 24, 24, 24)

        # Action buttons
        btn_row = QHBoxLayout()
        self.add_btn = QPushButton("➕ 添加人员")
        self.add_btn.setObjectName("primary")
        self.add_btn.clicked.connect(self.add_person)
        self.import_btn = QPushButton("📥 导入图片")
        self.import_btn.clicked.connect(self.import_images)
        self.refresh_btn = QPushButton("🔄 刷新")
        self.refresh_btn.clicked.connect(self.refresh_table)
        btn_row.addWidget(self.add_btn)
        btn_row.addWidget(self.import_btn)
        btn_row.addWidget(self.refresh_btn)
        layout.addLayout(btn_row)

        # Person table
        table_card = CardFrame("人员列表")
        self.table = QTableWidget()
        self.table.setColumnCount(4)
        self.table.setHorizontalHeaderLabels(["ID", "姓名", "样本数", "特征向量"])
        self.table.horizontalHeader().setSectionResizeMode(1, QHeaderView.Stretch)
        self.table.setSelectionBehavior(QTableWidget.SelectRows)
        self.table.setEditTriggers(QTableWidget.NoEditTriggers)
        self.table.cellClicked.connect(self.on_select)
        table_card.layout.addWidget(self.table)
        layout.addWidget(table_card)

        # Detail area
        detail_card = CardFrame("人员详情")
        detail_layout = QHBoxLayout()

        self.detail_name = QLabel("未选中")
        self.detail_name.setFont(QFont("Microsoft YaHei", 14, QFont.Bold))
        detail_layout.addWidget(self.detail_name)

        detail_layout.addStretch()

        self.rebuild_btn = QPushButton("🔄 重新提取特征")
        self.rebuild_btn.clicked.connect(self.rebuild_embedding)
        self.delete_btn = QPushButton("🗑 删除人员")
        self.delete_btn.setObjectName("danger")
        self.delete_btn.clicked.connect(self.delete_person)
        detail_layout.addWidget(self.rebuild_btn)
        detail_layout.addWidget(self.delete_btn)

        detail_card.layout.addLayout(detail_layout)
        layout.addWidget(detail_card)

        layout.addStretch()
        self.selected_person_id: Optional[int] = None

    def refresh_table(self):
        persons = self.gallery.get_persons()
        self.table.setRowCount(len(persons))
        for i, p in enumerate(persons):
            self.table.setItem(i, 0, QTableWidgetItem(str(p["id"])))
            self.table.setItem(i, 1, QTableWidgetItem(p["name"]))
            self.table.setItem(i, 2, QTableWidgetItem(str(p["sample_count"])))
            status = "✓ 已生成" if p["has_embedding"] else "✗ 未生成"
            self.table.setItem(i, 3, QTableWidgetItem(status))

    def on_select(self, row: int):
        id_item = self.table.item(row, 0)
        name_item = self.table.item(row, 1)
        if id_item:
            self.selected_person_id = int(id_item.text())
            self.detail_name.setText(f"选中: {name_item.text()}")

    def add_person(self):
        from PySide6.QtWidgets import QInputDialog
        name, ok = QInputDialog.getText(self, "添加人员", "请输入人员姓名:")
        if ok and name:
            self.gallery.db.add_person(name)
            self.refresh_table()

    def import_images(self):
        if not self.selected_person_id:
            QMessageBox.warning(self, "提示", "请先在表格中选择一个人员")
            return
        person = self.gallery.db.get_person(self.selected_person_id)
        if not person:
            return

        dir_path = QFileDialog.getExistingDirectory(self, f"选择 {person['name']} 的图片目录")
        if not dir_path:
            return

        try:
            raw_dir = Path(dir_path)
            self.gallery.import_person(person["name"], raw_dir)
            self.refresh_table()
            QMessageBox.information(self, "导入完成", f"已导入 {person['name']} 的图片并生成特征")
        except Exception as e:
            QMessageBox.critical(self, "导入失败", str(e))

    def rebuild_embedding(self):
        if not self.selected_person_id:
            return
        person = self.gallery.db.get_person(self.selected_person_id)
        if not person:
            return
        try:
            person_dir = PATCHES_DIR / person["name"]
            if not person_dir.exists():
                QMessageBox.warning(self, "提示", f"未找到 {person['name']} 的patch数据")
                return
            patch_files = sorted(person_dir.glob("*.png"))
            if not patch_files:
                QMessageBox.warning(self, "提示", "该人员没有patch数据")
                return
            images = []
            for pf in patch_files:
                img = cv2.imread(str(pf), cv2.IMREAD_GRAYSCALE)
                if img is not None:
                    images.append(img)
            embeddings = self.gallery.embedder.embed_batch(images)
            mean_emb = np.mean(embeddings, axis=0)
            mean_emb = mean_emb / np.linalg.norm(mean_emb)
            self.gallery.db.save_embedding(person["id"], mean_emb)
            self.gallery._gallery_cache = None
            self.refresh_table()
            QMessageBox.information(self, "完成", f"已重新生成 {person['name']} 的特征向量")
        except Exception as e:
            QMessageBox.critical(self, "失败", str(e))

    def delete_person(self):
        if not self.selected_person_id:
            return
        reply = QMessageBox.question(
            self, "确认删除", "确定要删除该人员及其所有数据吗？",
            QMessageBox.Yes | QMessageBox.No
        )
        if reply == QMessageBox.Yes:
            self.gallery.delete_person(self.selected_person_id)
            self.selected_person_id = None
            self.detail_name.setText("未选中")
            self.refresh_table()


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Pen ID - 手写笔迹识别系统")
        self.setMinimumSize(1100, 750)
        self.gallery = Gallery()
        try:
            self.gallery.load_model()
        except Exception:
            pass
        self.init_ui()

    def init_ui(self):
        central = QWidget()
        self.setCentralWidget(central)
        main_layout = QHBoxLayout(central)
        main_layout.setContentsMargins(0, 0, 0, 0)
        main_layout.setSpacing(0)

        # Sidebar
        sidebar = QFrame()
        sidebar.setFixedWidth(200)
        sidebar.setStyleSheet("""
            QFrame {
                background-color: #11111b;
                border-right: 1px solid #313244;
            }
        """)
        sidebar_layout = QVBoxLayout(sidebar)
        sidebar_layout.setContentsMargins(12, 20, 12, 20)
        sidebar_layout.setSpacing(4)

        # Logo
        logo = QLabel("✏ Pen ID")
        logo.setFont(QFont("Microsoft YaHei", 18, QFont.Bold))
        logo.setStyleSheet("color: #89b4fa; border: none; padding: 8px;")
        logo.setAlignment(Qt.AlignCenter)
        sidebar_layout.addWidget(logo)

        sidebar_layout.addSpacing(20)

        # Nav buttons
        self.nav_buttons = []
        nav_items = [
            ("🔍 识别", 0),
            ("🚂 训练", 1),
            ("👥 人员管理", 2),
        ]

        for text, idx in nav_items:
            btn = QPushButton(text)
            btn.setObjectName("nav")
            btn.setCheckable(True)
            btn.clicked.connect(lambda checked, i=idx: self.switch_page(i))
            sidebar_layout.addWidget(btn)
            self.nav_buttons.append(btn)

        sidebar_layout.addStretch()

        # Version
        ver = QLabel("v1.0.0")
        ver.setStyleSheet("color: #45475a; border: none; font-size: 11px;")
        ver.setAlignment(Qt.AlignCenter)
        sidebar_layout.addWidget(ver)

        main_layout.addWidget(sidebar)

        # Content area
        self.stack = QStackedWidget()
        self.stack.setStyleSheet("background-color: #1e1e2e;")

        self.identify_page = IdentifyPage(self.gallery)
        self.train_page = TrainPage()
        self.gallery_page = GalleryPage(self.gallery)

        self.stack.addWidget(self.identify_page)
        self.stack.addWidget(self.train_page)
        self.stack.addWidget(self.gallery_page)

        main_layout.addWidget(self.stack)
        self.switch_page(0)

    def switch_page(self, idx: int):
        self.stack.setCurrentIndex(idx)
        for i, btn in enumerate(self.nav_buttons):
            btn.setProperty("active", i == idx)
            btn.setStyleSheet(btn.styleSheet())
            btn.setChecked(i == idx)
