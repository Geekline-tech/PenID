import cv2
import numpy as np
from PyQt5.QtCore import Qt, QThread, pyqtSignal
from PyQt5.QtGui import QPixmap, QImage, QFont
from PyQt5.QtWidgets import (
    QWidget, QHBoxLayout, QVBoxLayout, QLabel, QFileDialog, QScrollArea, QFrame
)
from qfluentwidgets import (
    PushButton, PrimaryPushButton, CardWidget, TitleLabel, BodyLabel,
    CaptionLabel, FluentIcon, ProgressBar
)
from src.data.scanner import scan_document


def ndarray_to_pixmap(img, max_w=480, max_h=300):
    if len(img.shape) == 2:
        h, w = img.shape
        qimg = QImage(img.data, w, h, w, QImage.Format_Grayscale8)
    else:
        h, w, ch = img.shape
        bgr = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        qimg = QImage(bgr.data, w, h, w * ch, QImage.Format_RGB888)
    pix = QPixmap.fromImage(qimg)
    return pix.scaled(max_w, max_h, Qt.KeepAspectRatio, Qt.SmoothTransformation)


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
        self.layout = QVBoxLayout(self)
        self.layout.setContentsMargins(0, 0, 0, 0)
        self.layout.setSpacing(6)

    def set_results(self, results):
        while self.layout.count():
            item = self.layout.takeAt(0)
            if item.widget():
                item.widget().deleteLater()

        for i, r in enumerate(results[:10]):
            row = QFrame()
            row.setStyleSheet("QFrame { background: #2b2d30; border-radius: 6px; padding: 6px; }")
            row_layout = QHBoxLayout(row)
            row_layout.setContentsMargins(10, 6, 10, 6)

            color = "#a6e3a1" if i == 0 else "#89b4fa" if i < 3 else "#6c7086"

            rank = QLabel(f"#{i+1}")
            rank.setFixedWidth(30)
            rank.setFont(QFont("Segoe UI", 10, QFont.Bold))
            rank.setStyleSheet(f"color: {color};")
            row_layout.addWidget(rank)

            name = QLabel(r["name"])
            name.setFont(QFont("Segoe UI", 11, QFont.Bold))
            name.setStyleSheet("color: #cdd6f4;")
            name.setFixedWidth(100)
            row_layout.addWidget(name)

            bar_bg = QFrame()
            bar_bg.setFixedHeight(8)
            bar_bg.setStyleSheet("background: #45475a; border-radius: 4px;")
            bar_bg_layout = QVBoxLayout(bar_bg)
            bar_bg_layout.setContentsMargins(0, 0, 0, 0)
            bar_fill = QLabel()
            pct = r["confidence"] * 100
            bar_fill.setStyleSheet(f"background: {color}; border-radius: 4px;")
            bar_fill.setFixedHeight(8)
            bar_fill.setFixedWidth(max(1, int(pct * 2)))
            bar_bg_layout.addWidget(bar_fill, alignment=Qt.AlignLeft)
            row_layout.addWidget(bar_bg, 1)

            conf = QLabel(f"{pct:.1f}%")
            conf.setFixedWidth(50)
            conf.setAlignment(Qt.AlignRight)
            conf.setStyleSheet(f"color: {color};")
            row_layout.addWidget(conf)

            dist = QLabel(f"d={r['distance']:.3f}")
            dist.setFixedWidth(60)
            dist.setStyleSheet("color: #6c7086;")
            row_layout.addWidget(dist)

            self.layout.addWidget(row)

    def clear(self):
        while self.layout.count():
            item = self.layout.takeAt(0)
            if item.widget():
                item.widget().deleteLater()


class IdentifyPage(QWidget):
    def __init__(self, gallery):
        super().__init__()
        self.gallery = gallery
        self.current_image = None
        self.worker = None
        self._init_ui()

    def _init_ui(self):
        main_layout = QHBoxLayout(self)
        main_layout.setContentsMargins(20, 20, 20, 20)
        main_layout.setSpacing(20)

        left = QVBoxLayout()
        left.setSpacing(12)

        title = TitleLabel("笔迹识别")
        left.addWidget(title)

        self.image_card = CardWidget()
        self.image_card.setMinimumHeight(320)
        image_layout = QVBoxLayout(self.image_card)
        self.image_label = QLabel("点击下方按钮上传笔迹图片")
        self.image_label.setAlignment(Qt.AlignCenter)
        self.image_label.setStyleSheet("color: #6c7086; font-size: 14px;")
        self.image_label.setMinimumHeight(300)
        image_layout.addWidget(self.image_label)
        left.addWidget(self.image_card)

        btn_row = QHBoxLayout()
        self.upload_btn = PushButton()
        self.upload_btn.setText("上传图片")
        self.upload_btn.setIcon(FluentIcon.FOLDER)
        self.upload_btn.clicked.connect(self._upload)
        self.scan_btn = PushButton()
        self.scan_btn.setText("扫描增强")
        self.scan_btn.setIcon(FluentIcon.CAMERA)
        self.scan_btn.clicked.connect(self._scan)
        btn_row.addWidget(self.upload_btn)
        btn_row.addWidget(self.scan_btn)
        btn_row.addStretch()
        left.addLayout(btn_row)

        main_layout.addLayout(left, 6)

        right = QVBoxLayout()
        right.setSpacing(12)

        right.addWidget(TitleLabel("识别结果"))

        self.result_card = CardWidget()
        rc_layout = QVBoxLayout(self.result_card)
        rc_layout.setContentsMargins(16, 16, 16, 16)
        rc_layout.setSpacing(6)

        self.result_name = TitleLabel("--")
        self.result_name.setStyleSheet("color: #89b4fa;")
        rc_layout.addWidget(self.result_name)

        self.result_conf = BodyLabel("置信度: --")
        self.result_conf.setStyleSheet("color: #a6adc8;")
        rc_layout.addWidget(self.result_conf)

        self.result_status = CaptionLabel("")
        rc_layout.addWidget(self.result_status)

        self.detail_text = CaptionLabel("")
        self.detail_text.setStyleSheet("color: #6c7086;")
        rc_layout.addWidget(self.detail_text)

        right.addWidget(self.result_card)

        self.identify_btn = PrimaryPushButton()
        self.identify_btn.setText("开始识别")
        self.identify_btn.setIcon(FluentIcon.SEARCH)
        self.identify_btn.setEnabled(False)
        self.identify_btn.clicked.connect(self._identify)
        right.addWidget(self.identify_btn)

        self.progress = ProgressBar()
        self.progress.setVisible(False)
        right.addWidget(self.progress)

        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setStyleSheet("QScrollArea { border: none; background: transparent; }")
        self.ranking = RankingWidget()
        scroll.setWidget(self.ranking)
        right.addWidget(scroll, 1)

        main_layout.addLayout(right, 4)

    def _upload(self):
        path, _ = QFileDialog.getOpenFileName(
            self, "选择笔迹图片", "", "Images (*.png *.jpg *.jpeg *.bmp)"
        )
        if path:
            data = np.fromfile(path, dtype=np.uint8)
            img = cv2.imdecode(data, cv2.IMREAD_GRAYSCALE)
            if img is not None:
                self.current_image = img
                self.image_label.setPixmap(ndarray_to_pixmap(img))
                self.identify_btn.setEnabled(True)

    def _scan(self):
        path, _ = QFileDialog.getOpenFileName(
            self, "选择笔迹图片", "", "Images (*.png *.jpg *.jpeg *.bmp)"
        )
        if path:
            data = np.fromfile(path, dtype=np.uint8)
            img = cv2.imdecode(data, cv2.IMREAD_COLOR)
            if img is not None:
                def run():
                    scanned = scan_document(img)
                    self.current_image = scanned
                    raw = cv2.imdecode(data, cv2.IMREAD_GRAYSCALE)
                    self._show_scanned(raw, scanned)
                    self.identify_btn.setEnabled(True)
                from threading import Thread
                Thread(target=run, daemon=True).start()

    def _show_scanned(self, raw, scanned):
        raw_pix = ndarray_to_pixmap(raw, 230, 280)
        scan_pix = ndarray_to_pixmap(scanned, 230, 280)
        container = QWidget()
        layout = QHBoxLayout(container)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(10)
        left_col = QVBoxLayout()
        left_col.addWidget(QLabel("原图"), alignment=Qt.AlignCenter)
        raw_lbl = QLabel()
        raw_lbl.setPixmap(raw_pix)
        raw_lbl.setAlignment(Qt.AlignCenter)
        left_col.addWidget(raw_lbl)
        right_col = QVBoxLayout()
        right_col.addWidget(QLabel("扫描结果"), alignment=Qt.AlignCenter)
        scan_lbl = QLabel()
        scan_lbl.setPixmap(scan_pix)
        scan_lbl.setAlignment(Qt.AlignCenter)
        right_col.addWidget(scan_lbl)
        layout.addLayout(left_col)
        layout.addLayout(right_col)
        self.image_label.hide()
        self.image_card.layout().addWidget(container)

    def _identify(self):
        if self.current_image is None:
            return
        self.identify_btn.setEnabled(False)
        self.progress.setVisible(True)
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
            self.result_status.setStyleSheet("color: #a6e3a1; font-weight: bold;")
        else:
            self.result_status.setText("无法匹配")
            self.result_status.setStyleSheet("color: #f38ba8; font-weight: bold;")
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
