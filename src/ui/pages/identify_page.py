import cv2
import numpy as np
from PyQt5.QtCore import Qt, QThread, pyqtSignal
from PyQt5.QtGui import QPixmap, QImage, QFont
from PyQt5.QtWidgets import (
    QWidget, QHBoxLayout, QVBoxLayout, QLabel,
    QFileDialog, QSplitter
)
from qfluentwidgets import (
    PushButton, PrimaryPushButton, CardWidget, SubtitleLabel,
    BodyLabel, CaptionLabel, FluentIcon, ProgressBar, InfoBar,
    TitleLabel, ToolButton
)
from src.data.scanner import scan_document


def ndarray_to_pixmap(img, max_w=500, max_h=450):
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
        self.v_layout.setSpacing(2)
        self.v_layout.setAlignment(Qt.AlignTop)

    def set_results(self, results):
        self.clear()
        for i, r in enumerate(results[:10]):
            color = "#a6e3a1" if i == 0 else "#89b4fa" if i < 3 else "#9399b2"
            pct = r["confidence"] * 100

            row = CardWidget()
            row.setFixedHeight(50)
            rl = QHBoxLayout(row)
            rl.setContentsMargins(16, 6, 16, 6)
            rl.setSpacing(12)

            rank = QLabel(f"{i+1}")
            rank.setFixedSize(32, 32)
            rank.setAlignment(Qt.AlignCenter)
            rank.setFont(QFont("Microsoft YaHei UI", 12, QFont.Bold))
            if i == 0:
                rank.setStyleSheet("background: #a6e3a1; color: #1e1e2e; border-radius: 16px; font-weight: bold;")
            elif i < 3:
                rank.setStyleSheet(f"background: {color}; color: #1e1e2e; border-radius: 16px; font-weight: bold;")
            else:
                rank.setStyleSheet("background: #45475a; color: #9399b2; border-radius: 16px;")
            rl.addWidget(rank)

            name = QLabel(r["name"])
            name.setFont(QFont("Microsoft YaHei UI", 13, QFont.Bold))
            name.setStyleSheet("color: #cdd6f4; background: transparent;")
            rl.addWidget(name)

            rl.addStretch()

            conf = QLabel(f"{pct:.1f}%")
            conf.setFont(QFont("Microsoft YaHei UI", 13, QFont.Bold))
            conf.setStyleSheet(f"color: {color}; background: transparent;")
            rl.addWidget(conf)

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
        root = QHBoxLayout(self)
        root.setContentsMargins(32, 24, 32, 24)
        root.setSpacing(28)

        splitter = QSplitter(Qt.Horizontal)
        splitter.setHandleWidth(4)

        left = QWidget()
        ll = QVBoxLayout(left)
        ll.setContentsMargins(0, 0, 0, 0)
        ll.setSpacing(16)

        img_card = CardWidget()
        img_card.setMinimumHeight(420)
        icl = QVBoxLayout(img_card)
        icl.setContentsMargins(24, 24, 24, 24)

        self.placeholder = QLabel("选择笔迹图片后显示在此处")
        self.placeholder.setAlignment(Qt.AlignCenter)
        self.placeholder.setStyleSheet("color: #585b70; background: transparent;")
        self.placeholder.setFont(QFont("Microsoft YaHei UI", 14))
        self.placeholder.setMinimumHeight(360)
        icl.addWidget(self.placeholder)

        self.image_label = QLabel()
        self.image_label.setAlignment(Qt.AlignCenter)
        self.image_label.setStyleSheet("background: transparent;")
        self.image_label.setVisible(False)
        icl.addWidget(self.image_label)

        ll.addWidget(img_card, 1)

        btn_card = CardWidget()
        bcl = QHBoxLayout(btn_card)
        bcl.setContentsMargins(16, 12, 16, 12)
        bcl.setSpacing(12)

        self.upload_btn = PrimaryPushButton("上传图片")
        self.upload_btn.setIcon(FluentIcon.FOLDER)
        self.upload_btn.setFixedHeight(40)
        self.upload_btn.setFont(QFont("Microsoft YaHei UI", 11))
        self.upload_btn.clicked.connect(self._upload)

        self.scan_btn = PushButton("扫描增强")
        self.scan_btn.setIcon(FluentIcon.CAMERA)
        self.scan_btn.setFixedHeight(40)
        self.scan_btn.setFont(QFont("Microsoft YaHei UI", 11))
        self.scan_btn.clicked.connect(self._scan)

        self.identify_btn = PrimaryPushButton("开始识别")
        self.identify_btn.setIcon(FluentIcon.SEARCH)
        self.identify_btn.setFixedHeight(40)
        self.identify_btn.setFont(QFont("Microsoft YaHei UI", 11, QFont.Bold))
        self.identify_btn.setEnabled(False)
        self.identify_btn.clicked.connect(self._identify)

        bcl.addWidget(self.upload_btn)
        bcl.addWidget(self.scan_btn)
        bcl.addStretch()
        bcl.addWidget(self.identify_btn)
        ll.addWidget(btn_card)

        splitter.addWidget(left)

        right = QWidget()
        rl = QVBoxLayout(right)
        rl.setContentsMargins(0, 0, 0, 0)
        rl.setSpacing(16)

        result_card = CardWidget()
        rcl = QVBoxLayout(result_card)
        rcl.setContentsMargins(24, 20, 24, 20)
        rcl.setSpacing(6)

        self.result_name = TitleLabel("--")
        self.result_name.setStyleSheet("color: #89b4fa; background: transparent;")
        rcl.addWidget(self.result_name)

        self.result_conf = SubtitleLabel("置信度: --")
        self.result_conf.setStyleSheet("color: #cdd6f4; background: transparent;")
        rcl.addWidget(self.result_conf)

        self.result_status = CaptionLabel("")
        rcl.addWidget(self.result_status)

        self.result_detail = CaptionLabel("")
        self.result_detail.setStyleSheet("color: #6c7086; background: transparent;")
        rcl.addWidget(self.result_detail)

        rl.addWidget(result_card)

        self.progress = ProgressBar()
        self.progress.setVisible(False)
        rl.addWidget(self.progress)

        rank_card = CardWidget()
        rkl = QVBoxLayout(rank_card)
        rkl.setContentsMargins(12, 12, 12, 12)
        self.ranking = RankingWidget()
        rkl.addWidget(self.ranking)
        rl.addWidget(rank_card, 1)

        splitter.addWidget(right)
        splitter.setSizes([500, 400])

        root.addWidget(splitter)

    def _upload(self):
        f = QFileDialog.getOpenFileName(self, "选择笔迹图片", "", "Images (*.png *.jpg *.jpeg *.bmp)")
        if f and f[0]:
            self._load_image(f[0])

    def _scan(self):
        f = QFileDialog.getOpenFileName(self, "选择笔迹图片", "", "Images (*.png *.jpg *.jpeg *.bmp)")
        if f and f[0]:
            data = np.fromfile(f[0], dtype=np.uint8)
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
        self.current_image = scanned
        self.identify_btn.setEnabled(True)

        self.placeholder.setVisible(False)
        self.image_label.setVisible(True)

        raw_pix = ndarray_to_pixmap(raw_gray, 320, 360)
        scan_pix = ndarray_to_pixmap(scanned, 320, 360)

        container = QWidget()
        container.setStyleSheet("background: transparent;")
        cl = QHBoxLayout(container)
        cl.setContentsMargins(0, 0, 0, 0)
        cl.setSpacing(20)
        cl.setAlignment(Qt.AlignCenter)

        for text, pix in [("原图", raw_pix), ("扫描增强", scan_pix)]:
            col = QVBoxLayout()
            col.setSpacing(8)
            col.setAlignment(Qt.AlignCenter)
            t = QLabel(text)
            t.setAlignment(Qt.AlignCenter)
            t.setFont(QFont("Microsoft YaHei UI", 12, QFont.Bold))
            t.setStyleSheet("color: #a6adc8; background: transparent;")
            col.addWidget(t)
            img = QLabel()
            img.setPixmap(pix)
            img.setAlignment(Qt.AlignCenter)
            img.setStyleSheet("background: transparent;")
            col.addWidget(img)
            cl.addLayout(col)

        self.image_label.setVisible(False)
        self.image_card = self.placeholder.parentWidget().parentWidget()
        self.image_card.layout().addWidget(container)
        InfoBar.success("完成", "扫描完成，可以识别", parent=self.window())

    def _load_image(self, path):
        data = np.fromfile(path, dtype=np.uint8)
        img = cv2.imdecode(data, cv2.IMREAD_GRAYSCALE)
        if img is None:
            InfoBar.error("错误", "无法读取图片", parent=self.window())
            return
        self.current_image = img
        self.placeholder.setVisible(False)
        self.image_label.setVisible(True)
        self.image_label.setPixmap(ndarray_to_pixmap(img))
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
        self.result_detail.setText("")
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
            self.result_status.setStyleSheet("color: #a6e3a1; font-size: 13px; font-weight: bold; background: transparent;")
        else:
            self.result_status.setText("无法匹配")
            self.result_status.setStyleSheet("color: #f38ba8; font-size: 13px; font-weight: bold; background: transparent;")
        self.result_detail.setText(f"余弦距离: {result['distance']:.4f}")
        self.ranking.set_results(result["rankings"])
        self.identify_btn.setEnabled(True)
        self.progress.setVisible(False)

    def _on_error(self, msg):
        self.result_name.setText("识别失败")
        self.result_status.setText(msg)
        self.result_status.setStyleSheet("color: #f38ba8; background: transparent;")
        self.identify_btn.setEnabled(True)
        self.progress.setVisible(False)
