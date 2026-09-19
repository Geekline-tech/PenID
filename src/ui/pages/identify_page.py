import cv2
import numpy as np
from PyQt5.QtCore import Qt, QThread, pyqtSignal, QSize
from PyQt5.QtGui import QPixmap, QImage, QFont, QColor, QPainter
from PyQt5.QtWidgets import (
    QWidget, QHBoxLayout, QVBoxLayout, QLabel,
    QScrollArea, QFrame, QSizePolicy, QFileDialog,
    QGraphicsDropShadowEffect
)
from qfluentwidgets import (
    PushButton, PrimaryPushButton, CardWidget, SubtitleLabel,
    BodyLabel, CaptionLabel, FluentIcon, ProgressBar, InfoBar,
    TableWidget, StrongBodyLabel, TitleLabel
)
from src.data.scanner import scan_document


def ndarray_to_pixmap(img, max_w=600, max_h=500):
    if img is None:
        return QPixmap()
    if len(img.shape) == 2:
        h, w = img.shape
        qimg = QImage(img.data, w, h, w, QImage.Format_Grayscale8)
    else:
        h, w, ch = img.shape
        bgr = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        qimg = QImage(bgr.data, w, h, w * ch, QImage.Format_RGB888)
    pix = QPixmap.fromImage(qimg.copy())
    return pix.scaled(max_w, max_h, Qt.KeepAspectRatio, Qt.SmoothTransformation)


class ScanWorker(QThread):
    done = pyqtSignal(object, object)

    def __init__(self, color_img):
        super().__init__()
        self.color_img = color_img

    def run(self):
        raw_gray = cv2.cvtColor(self.color_img, cv2.COLOR_BGR2GRAY) if len(self.color_img.shape) == 3 else self.color_img.copy()
        scanned = scan_document(self.color_img)
        self.done.emit(raw_gray, scanned)


class IdentifyWorker(QThread):
    result = pyqtSignal(dict)
    error = pyqtSignal(str)

    def __init__(self, gallery, image):
        super().__init__()
        self.gallery = gallery
        self.image = image

    def run(self):
        try:
            result = self.gallery.identify_image(self.image)
            self.result.emit(result)
        except Exception as e:
            self.error.emit(str(e))


class RankingWidget(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.v_layout = QVBoxLayout(self)
        self.v_layout.setContentsMargins(0, 0, 0, 0)
        self.v_layout.setSpacing(8)

    def set_results(self, results):
        self.clear()
        for i, r in enumerate(results[:10]):
            color = "#a6e3a1" if i == 0 else "#89b4fa" if i < 3 else "#9399b2"
            pct = r["confidence"] * 100

            row = QFrame()
            row.setFixedHeight(56)
            row_layout = QHBoxLayout(row)
            row_layout.setContentsMargins(20, 8, 20, 8)
            row_layout.setSpacing(16)

            rank_label = QLabel(f"#{i+1}")
            rank_label.setFixedWidth(40)
            rank_label.setFont(QFont("Microsoft YaHei UI", 15, QFont.Bold))
            rank_label.setStyleSheet(f"color: {color}; background: transparent;")
            row_layout.addWidget(rank_label)

            name_label = QLabel(r["name"])
            name_label.setFont(QFont("Microsoft YaHei UI", 14, QFont.Bold))
            name_label.setStyleSheet("color: #cdd6f4; background: transparent;")
            name_label.setMinimumWidth(120)
            row_layout.addWidget(name_label)

            bar_bg = QLabel()
            bar_bg.setStyleSheet("background: #313244; border-radius: 4px;")
            bar_bg.setFixedHeight(8)
            bar_bg.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
            row_layout.addWidget(bar_bg, 1)

            conf_label = QLabel(f"{pct:.1f}%")
            conf_label.setFixedWidth(64)
            conf_label.setAlignment(Qt.AlignRight | Qt.AlignVCenter)
            conf_label.setFont(QFont("Microsoft YaHei UI", 13, QFont.Bold))
            conf_label.setStyleSheet(f"color: {color}; background: transparent;")
            row_layout.addWidget(conf_label)

            self.v_layout.addWidget(row)

    def clear(self):
        while self.v_layout.count():
            item = self.v_layout.takeAt(0)
            w = item.widget()
            if w:
                w.deleteLater()


class ImageCard(CardWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.setMinimumSize(400, 350)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(32, 32, 32, 32)

        self.placeholder = QLabel("点击「上传图片」选择笔迹文件")
        self.placeholder.setAlignment(Qt.AlignCenter)
        self.placeholder.setStyleSheet("color: #585b70; background: transparent;")
        self.placeholder.setFont(QFont("Microsoft YaHei UI", 15))
        layout.addWidget(self.placeholder)

        self.image_label = QLabel()
        self.image_label.setAlignment(Qt.AlignCenter)
        self.image_label.setStyleSheet("background: transparent;")
        self.image_label.setVisible(False)
        layout.addWidget(self.image_label)


class ResultCard(CardWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(28, 24, 28, 24)
        layout.setSpacing(10)

        self.title = TitleLabel("识别结果")
        self.title.setStyleSheet("color: #89b4fa; background: transparent;")
        layout.addWidget(self.title)

        self.status = SubtitleLabel("--")
        self.status.setStyleSheet("color: #cdd6f4; background: transparent;")
        layout.addWidget(self.status)

        self.conf = BodyLabel("置信度: --")
        self.conf.setStyleSheet("color: #a6adc8; background: transparent;")
        layout.addWidget(self.conf)

        self.detail = CaptionLabel("")
        self.detail.setStyleSheet("color: #6c7086; background: transparent;")
        layout.addWidget(self.detail)

        layout.addStretch()

    def set_result(self, result):
        self.title.setText(result["name"])
        self.conf.setText(f"置信度: {result['confidence']*100:.1f}%")
        if result["matched"]:
            self.status.setText("✓ 匹配成功")
            self.status.setStyleSheet("color: #a6e3a1; background: transparent; font-size: 16px; font-weight: bold;")
        else:
            self.status.setText("✗ 无法匹配")
            self.status.setStyleSheet("color: #f38ba8; background: transparent; font-size: 16px; font-weight: bold;")
        self.detail.setText(f"余弦距离: {result['distance']:.4f}")


class IdentifyPage(QWidget):
    def __init__(self, gallery):
        super().__init__()
        self.setObjectName("identify_page")
        self.gallery = gallery
        self.current_image = None
        self.worker = None
        self._init_ui()

    def _init_ui(self):
        layout = QHBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)

        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setStyleSheet("QScrollArea { border: none; background: transparent; }")
        content = QWidget()
        content_layout = QHBoxLayout(content)
        content_layout.setContentsMargins(32, 32, 32, 32)
        content_layout.setSpacing(32)

        left_col = QVBoxLayout()
        left_col.setSpacing(20)

        self.image_card = ImageCard()
        card_layout = self.image_card.layout()
        left_col.addWidget(self.image_card, 1)

        btn_row = QHBoxLayout()
        btn_row.setSpacing(16)

        self.upload_btn = PrimaryPushButton("上传图片")
        self.upload_btn.setIcon(FluentIcon.FOLDER)
        self.upload_btn.setFixedHeight(48)
        self.upload_btn.setMinimumWidth(180)
        self.upload_btn.setFont(QFont("Microsoft YaHei UI", 12))
        self.upload_btn.clicked.connect(self._upload)

        self.scan_btn = PushButton("扫描增强")
        self.scan_btn.setIcon(FluentIcon.CAMERA)
        self.scan_btn.setFixedHeight(48)
        self.scan_btn.setMinimumWidth(180)
        self.scan_btn.setFont(QFont("Microsoft YaHei UI", 12))
        self.scan_btn.clicked.connect(self._scan)

        btn_row.addWidget(self.upload_btn)
        btn_row.addWidget(self.scan_btn)
        btn_row.addStretch()
        left_col.addLayout(btn_row)

        content_layout.addLayout(left_col, 5)

        right_col = QVBoxLayout()
        right_col.setSpacing(20)

        self.result_card = ResultCard()
        right_col.addWidget(self.result_card)

        self.identify_btn = PrimaryPushButton("开始识别")
        self.identify_btn.setIcon(FluentIcon.SEARCH)
        self.identify_btn.setFixedHeight(52)
        self.identify_btn.setFont(QFont("Microsoft YaHei UI", 13, QFont.Bold))
        self.identify_btn.setEnabled(False)
        self.identify_btn.clicked.connect(self._identify)
        right_col.addWidget(self.identify_btn)

        self.progress = ProgressBar()
        self.progress.setVisible(False)
        right_col.addWidget(self.progress)

        ranking_card = CardWidget()
        rc_layout = QVBoxLayout(ranking_card)
        rc_layout.setContentsMargins(16, 16, 16, 16)

        scroll2 = QScrollArea()
        scroll2.setWidgetResizable(True)
        scroll2.setStyleSheet("QScrollArea { border: none; background: transparent; }")
        self.ranking = RankingWidget()
        scroll2.setWidget(self.ranking)
        rc_layout.addWidget(scroll2)

        right_col.addWidget(ranking_card, 1)

        content_layout.addLayout(right_col, 4)

        scroll.setWidget(content)
        main_layout = QVBoxLayout(self)
        main_layout.setContentsMargins(0, 0, 0, 0)
        main_layout.addWidget(scroll)

    def _upload(self):
        f = QFileDialog.getOpenFileName(
            self, "选择笔迹图片", "", "Images (*.png *.jpg *.jpeg *.bmp)"
        )
        if f and f[0]:
            self._load_image(f[0])

    def _scan(self):
        f = QFileDialog.getOpenFileName(
            self, "选择笔迹图片", "", "Images (*.png *.jpg *.jpeg *.bmp)"
        )
        if f and f[0]:
            path = f[0]
            data = np.fromfile(path, dtype=np.uint8)
            color_img = cv2.imdecode(data, cv2.IMREAD_COLOR)
            if color_img is None:
                InfoBar.error("错误", "无法读取图片", parent=self.window())
                return

            self.scan_btn.setEnabled(False)
            self.progress.setVisible(True)
            self.progress.setRange(0, 0)

            self._scan_worker = ScanWorker(color_img)
            self._scan_worker.done.connect(self._on_scan_done)
            self._scan_worker.start()

    def _on_scan_done(self, raw_gray, scanned):
        self.scan_btn.setEnabled(True)
        self.progress.setVisible(False)
        self.progress.setRange(0, 100)

        self.current_image = scanned
        self.identify_btn.setEnabled(True)

        self.image_card.placeholder.setVisible(False)
        self.image_card.image_label.setVisible(True)

        raw_pix = ndarray_to_pixmap(raw_gray, 360, 400)
        scan_pix = ndarray_to_pixmap(scanned, 360, 400)

        container = QWidget()
        container.setStyleSheet("background: transparent;")
        c_layout = QHBoxLayout(container)
        c_layout.setContentsMargins(0, 0, 0, 0)
        c_layout.setSpacing(24)

        for text, pix in [("原图", raw_pix), ("扫描增强", scan_pix)]:
            col = QVBoxLayout()
            col.setSpacing(10)
            lbl = QLabel(text)
            lbl.setAlignment(Qt.AlignCenter)
            lbl.setFont(QFont("Microsoft YaHei UI", 13, QFont.Bold))
            lbl.setStyleSheet("color: #a6adc8; background: transparent;")
            col.addWidget(lbl)
            img_lbl = QLabel()
            img_lbl.setPixmap(pix)
            img_lbl.setAlignment(Qt.AlignCenter)
            img_lbl.setStyleSheet("background: transparent;")
            col.addWidget(img_lbl)
            c_layout.addLayout(col)

        self.image_card.image_label.setVisible(False)
        self.image_card.layout().addWidget(container)

        InfoBar.success("完成", "扫描增强完成，可以开始识别", parent=self.window())

    def _load_image(self, path):
        data = np.fromfile(path, dtype=np.uint8)
        img = cv2.imdecode(data, cv2.IMREAD_GRAYSCALE)
        if img is None:
            InfoBar.error("错误", "无法读取图片", parent=self.window())
            return
        self.current_image = img
        self.image_card.placeholder.setVisible(False)
        self.image_card.image_label.setVisible(True)
        pix = ndarray_to_pixmap(img)
        self.image_card.image_label.setPixmap(pix)
        self.identify_btn.setEnabled(True)

    def _identify(self):
        if self.current_image is None:
            return
        self.identify_btn.setEnabled(False)
        self.progress.setVisible(True)
        self.progress.setRange(0, 0)
        self.result_card.title.setText("识别中...")
        self.result_card.status.setText("")
        self.result_card.conf.setText("")
        self.result_card.detail.setText("")
        self.ranking.clear()

        self.worker = IdentifyWorker(self.gallery, self.current_image)
        self.worker.result.connect(self._on_result)
        self.worker.error.connect(self._on_error)
        self.worker.start()

    def _on_result(self, result):
        self.result_card.set_result(result)
        self.ranking.set_results(result["rankings"])
        self.identify_btn.setEnabled(True)
        self.progress.setVisible(False)

    def _on_error(self, msg):
        self.result_card.title.setText("识别失败")
        self.result_card.status.setText(msg)
        self.result_card.status.setStyleSheet("color: #f38ba8; background: transparent;")
        self.identify_btn.setEnabled(True)
        self.progress.setVisible(False)
