import cv2
import numpy as np
from PyQt5.QtCore import Qt, QThread, pyqtSignal
from PyQt5.QtGui import QPixmap, QImage, QFont
from PyQt5.QtWidgets import (
    QWidget, QHBoxLayout, QVBoxLayout, QLabel,
    QScrollArea, QFrame, QSizePolicy, QFileDialog
)
from qfluentwidgets import (
    PushButton, PrimaryPushButton, CardWidget, SubtitleLabel,
    BodyLabel, CaptionLabel, FluentIcon, ProgressBar, InfoBar,
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
        self.v_layout.setSpacing(6)

    def set_results(self, results):
        self.clear()
        for i, r in enumerate(results[:10]):
            color = "#a6e3a1" if i == 0 else "#89b4fa" if i < 3 else "#6c7086"

            row = QFrame()
            row.setStyleSheet(
                "QFrame { background-color: #313244; border-radius: 8px; padding: 4px; }"
                "QFrame:hover { background-color: #45475a; }"
            )
            row.setFixedHeight(52)
            row_layout = QHBoxLayout(row)
            row_layout.setContentsMargins(16, 6, 16, 6)
            row_layout.setSpacing(14)

            rank = QLabel(f"#{i+1}")
            rank.setFixedWidth(36)
            rank.setFont(QFont("Microsoft YaHei UI", 13, QFont.Bold))
            rank.setStyleSheet(f"color: {color}; background: transparent;")
            row_layout.addWidget(rank)

            name = QLabel(r["name"])
            name.setFont(QFont("Microsoft YaHei UI", 13, QFont.Bold))
            name.setStyleSheet("color: #cdd6f4; background: transparent;")
            name.setFixedWidth(140)
            row_layout.addWidget(name)

            bar_outer = QLabel()
            bar_outer.setStyleSheet("background: #45475a; border-radius: 4px;")
            bar_outer.setFixedHeight(8)
            pct = r["confidence"] * 100
            row_layout.addWidget(bar_outer, 1)

            conf = QLabel(f"{pct:.1f}%")
            conf.setFixedWidth(60)
            conf.setAlignment(Qt.AlignRight | Qt.AlignVCenter)
            conf.setFont(QFont("Microsoft YaHei UI", 12))
            conf.setStyleSheet(f"color: {color}; background: transparent;")
            row_layout.addWidget(conf)

            dist = QLabel(f"{r['distance']:.3f}")
            dist.setFixedWidth(70)
            dist.setStyleSheet("color: #6c7086; background: transparent;")
            dist.setFont(QFont("Microsoft YaHei UI", 10))
            row_layout.addWidget(dist)

            self.v_layout.addWidget(row)

    def clear(self):
        while self.v_layout.count():
            item = self.v_layout.takeAt(0)
            w = item.widget()
            if w:
                w.deleteLater()


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
        layout.setContentsMargins(28, 28, 28, 28)
        layout.setSpacing(28)

        left_col = QVBoxLayout()
        left_col.setSpacing(18)

        self.image_card = CardWidget()
        self.image_card.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        card_layout = QVBoxLayout(self.image_card)
        card_layout.setContentsMargins(20, 20, 20, 20)

        self.placeholder = QLabel("点击下方按钮上传笔迹图片")
        self.placeholder.setAlignment(Qt.AlignCenter)
        self.placeholder.setStyleSheet("color: #585b70; font-size: 18px; background: transparent;")
        self.placeholder.setFont(QFont("Microsoft YaHei UI", 16))
        card_layout.addWidget(self.placeholder)

        self.image_label = QLabel()
        self.image_label.setAlignment(Qt.AlignCenter)
        self.image_label.setStyleSheet("background: transparent;")
        self.image_label.setVisible(False)
        card_layout.addWidget(self.image_label)

        left_col.addWidget(self.image_card, 1)

        btn_row = QHBoxLayout()
        btn_row.setSpacing(14)

        self.upload_btn = PrimaryPushButton("上传图片")
        self.upload_btn.setIcon(FluentIcon.FOLDER)
        self.upload_btn.setFixedHeight(44)
        self.upload_btn.setMinimumWidth(160)
        self.upload_btn.setFont(QFont("Microsoft YaHei UI", 11))
        self.upload_btn.clicked.connect(self._upload)

        self.scan_btn = PushButton("扫描增强")
        self.scan_btn.setIcon(FluentIcon.CAMERA)
        self.scan_btn.setFixedHeight(44)
        self.scan_btn.setMinimumWidth(160)
        self.scan_btn.setFont(QFont("Microsoft YaHei UI", 11))
        self.scan_btn.clicked.connect(self._scan)

        btn_row.addWidget(self.upload_btn)
        btn_row.addWidget(self.scan_btn)
        btn_row.addStretch()
        left_col.addLayout(btn_row)

        layout.addLayout(left_col, 6)

        right_col = QVBoxLayout()
        right_col.setSpacing(18)

        result_card = CardWidget()
        rc_layout = QVBoxLayout(result_card)
        rc_layout.setContentsMargins(24, 24, 24, 24)
        rc_layout.setSpacing(8)

        self.result_name = SubtitleLabel("识别结果")
        self.result_name.setStyleSheet("color: #89b4fa; font-size: 20px;")
        rc_layout.addWidget(self.result_name)

        self.result_conf = BodyLabel("置信度: --")
        self.result_conf.setStyleSheet("color: #a6adc8; font-size: 14px;")
        rc_layout.addWidget(self.result_conf)

        self.result_status = BodyLabel("")
        rc_layout.addWidget(self.result_status)

        self.detail_text = CaptionLabel("")
        self.detail_text.setStyleSheet("color: #6c7086;")
        rc_layout.addWidget(self.detail_text)

        right_col.addWidget(result_card)

        self.identify_btn = PrimaryPushButton("开始识别")
        self.identify_btn.setIcon(FluentIcon.SEARCH)
        self.identify_btn.setFixedHeight(48)
        self.identify_btn.setFont(QFont("Microsoft YaHei UI", 12, QFont.Bold))
        self.identify_btn.setEnabled(False)
        self.identify_btn.clicked.connect(self._identify)
        right_col.addWidget(self.identify_btn)

        self.progress = ProgressBar()
        self.progress.setVisible(False)
        right_col.addWidget(self.progress)

        self.result_card_container = CardWidget()
        rc2 = QVBoxLayout(self.result_card_container)
        rc2.setContentsMargins(16, 16, 16, 16)
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setStyleSheet("QScrollArea { border: none; background: transparent; }")
        self.ranking = RankingWidget()
        scroll.setWidget(self.ranking)
        rc2.addWidget(scroll)
        right_col.addWidget(self.result_card_container, 1)

        layout.addLayout(right_col, 4)

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

        self.placeholder.setVisible(False)
        self.image_label.setVisible(True)

        raw_pix = ndarray_to_pixmap(raw_gray, 340, 380)
        scan_pix = ndarray_to_pixmap(scanned, 340, 380)

        container = QWidget()
        container.setStyleSheet("background: transparent;")
        c_layout = QHBoxLayout(container)
        c_layout.setContentsMargins(0, 0, 0, 0)
        c_layout.setSpacing(20)

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

        old_widget = self.image_label.parentWidget().findChild(QWidget, "")
        self.image_label.setVisible(False)

        parent_layout = self.image_card.layout()
        parent_layout.addWidget(container)

        InfoBar.success("完成", "扫描增强完成，已准备识别", parent=self.window())

    def _load_image(self, path):
        data = np.fromfile(path, dtype=np.uint8)
        img = cv2.imdecode(data, cv2.IMREAD_GRAYSCALE)
        if img is None:
            InfoBar.error("错误", "无法读取图片", parent=self.window())
            return
        self.current_image = img
        self.placeholder.setVisible(False)
        self.image_label.setVisible(True)
        pix = ndarray_to_pixmap(img)
        self.image_label.setPixmap(pix)
        self.identify_btn.setEnabled(True)

    def _identify(self):
        if self.current_image is None:
            return
        self.identify_btn.setEnabled(False)
        self.progress.setVisible(True)
        self.progress.setRange(0, 0)
        self.result_name.setText("识别中...")
        self.result_conf.setText("")
        self.result_status.setText("")
        self.detail_text.setText("")
        self.ranking.clear()

        self.worker = IdentifyWorker(self.gallery, self.current_image)
        self.worker.result.connect(self._on_result)
        self.worker.error.connect(self._on_error)
        self.worker.start()

    def _on_result(self, result):
        self.result_name.setText(result["name"])
        self.result_conf.setText(f"置信度: {result['confidence']*100:.1f}%")
        if result["matched"]:
            self.result_status.setText("匹配成功")
            self.result_status.setStyleSheet("color: #a6e3a1; font-weight: bold; font-size: 14px;")
        else:
            self.result_status.setText("无法匹配")
            self.result_status.setStyleSheet("color: #f38ba8; font-weight: bold; font-size: 14px;")
        self.detail_text.setText(f"余弦距离: {result['distance']:.4f}")
        self.ranking.set_results(result["rankings"])
        self.identify_btn.setEnabled(True)
        self.progress.setVisible(False)

    def _on_error(self, msg):
        self.result_name.setText("识别失败")
        self.result_status.setText(msg)
        self.result_status.setStyleSheet("color: #f38ba8;")
        self.identify_btn.setEnabled(True)
        self.progress.setVisible(False)
