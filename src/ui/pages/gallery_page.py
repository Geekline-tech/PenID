import cv2
import numpy as np
from pathlib import Path
from PyQt5.QtCore import Qt, QThread, pyqtSignal
from PyQt5.QtGui import QFont
from PyQt5.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QFileDialog,
    QInputDialog, QMessageBox, QTableWidget, QTableWidgetItem,
    QHeaderView, QAbstractItemView
)
from qfluentwidgets import (
    PushButton, PrimaryPushButton, CardWidget, TitleLabel,
    CaptionLabel, FluentIcon, ProgressBar
)
from src.utils.config import Config, RAW_DIR, PATCHES_DIR
from src.utils.database import Database
from src.data.preprocess import prepare_person_data


class BuildGalleryWorker(QThread):
    log = pyqtSignal(str)
    finished = pyqtSignal()

    def __init__(self, gallery):
        super().__init__()
        self.gallery = gallery

    def run(self):
        try:
            self.log.emit("Building gallery...")
            self.gallery.build_gallery()
            self.log.emit("Gallery built")
        except Exception as e:
            self.log.emit(f"Error: {e}")
        self.finished.emit()


class ImportWorker(QThread):
    log = pyqtSignal(str)
    finished = pyqtSignal()

    def __init__(self, gallery, person_name, image_dir):
        super().__init__()
        self.gallery = gallery
        self.person_name = person_name
        self.image_dir = image_dir

    def run(self):
        try:
            person_dir = RAW_DIR / self.person_name
            person_dir.mkdir(parents=True, exist_ok=True)
            for f in Path(self.image_dir).glob("*"):
                if f.suffix.lower() in (".jpg", ".jpeg", ".png", ".bmp"):
                    import shutil
                    shutil.copy2(f, person_dir / f.name)
            self.log.emit(f"Importing {self.person_name}...")
            self.gallery.import_person(self.person_name, person_dir)
        except Exception as e:
            self.log.emit(f"Error: {e}")
        self.finished.emit()


class GalleryPage(QWidget):
    def __init__(self, gallery):
        super().__init__()
        self.gallery = gallery
        self._init_ui()
        self._refresh_table()

    def _init_ui(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(20, 20, 20, 20)
        layout.setSpacing(16)

        layout.addWidget(TitleLabel("人员管理"))

        btn_row = QHBoxLayout()
        self.add_btn = PushButton("添加人员", FluentIcon.ADD)
        self.add_btn.clicked.connect(self._add_person)
        self.import_btn = PushButton("导入图片", FluentIcon.FOLDER)
        self.import_btn.clicked.connect(self._import_images)
        self.delete_btn = PushButton("删除选中", FluentIcon.DELETE)
        self.delete_btn.clicked.connect(self._delete_person)
        self.rebuild_btn = PrimaryPushButton("重建特征库", FluentIcon.SYNC)
        self.rebuild_btn.clicked.connect(self._rebuild_gallery)
        btn_row.addWidget(self.add_btn)
        btn_row.addWidget(self.import_btn)
        btn_row.addWidget(self.delete_btn)
        btn_row.addWidget(self.rebuild_btn)
        btn_row.addStretch()
        layout.addLayout(btn_row)

        self.progress = ProgressBar()
        self.progress.setVisible(False)
        layout.addWidget(self.progress)

        self.table = QTableWidget()
        self.table.setColumnCount(4)
        self.table.setHorizontalHeaderLabels(["ID", "姓名", "样本数", "特征库"])
        self.table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        self.table.setSelectionBehavior(QAbstractItemView.SelectRows)
        self.table.setEditTriggers(QAbstractItemView.NoEditTriggers)
        self.table.setStyleSheet("""
            QTableWidget {
                background: #1e1e2e;
                color: #cdd6f4;
                border: 1px solid #45475a;
                border-radius: 6px;
                gridline-color: #313244;
            }
            QTableWidget::item:selected {
                background: #45475a;
            }
            QHeaderView::section {
                background: #313244;
                color: #cdd6f4;
                border: none;
                padding: 6px;
                font-weight: bold;
            }
        """)
        layout.addWidget(self.table, 1)

        self.status_label = CaptionLabel("")
        self.status_label.setStyleSheet("color: #a6adc8;")
        layout.addWidget(self.status_label)

    def _refresh_table(self):
        db = Database()
        persons = db.get_persons()
        db.close()

        self.table.setRowCount(len(persons))
        for i, p in enumerate(persons):
            self.table.setItem(i, 0, QTableWidgetItem(str(p["id"])))
            self.table.setItem(i, 1, QTableWidgetItem(p["name"]))
            self.table.setItem(i, 2, QTableWidgetItem(str(p["sample_count"])))
            emb_status = "Yes" if p["has_embedding"] else "No"
            self.table.setItem(i, 3, QTableWidgetItem(emb_status))

        self.status_label.setText(f"Total: {len(persons)} persons")

    def _add_person(self):
        name, ok = QInputDialog.getText(self, "添加人员", "输入人员名称:")
        if ok and name:
            db = Database()
            db.add_person(name)
            db.close()
            self._refresh_table()

    def _delete_person(self):
        row = self.table.currentRow()
        if row < 0:
            return
        person_id = int(self.table.item(row, 0).text())
        name = self.table.item(row, 1).text()
        reply = QMessageBox.question(
            self, "确认删除", f"确定要删除人员 {name} 吗？",
            QMessageBox.Yes | QMessageBox.No
        )
        if reply == QMessageBox.Yes:
            db = Database()
            db.delete_person(person_id)
            db.close()
            person_dir = RAW_DIR / name
            patches_dir = PATCHES_DIR / name
            import shutil
            if person_dir.exists():
                shutil.rmtree(person_dir)
            if patches_dir.exists():
                shutil.rmtree(patches_dir)
            self.gallery._gallery_cache = None
            self._refresh_table()

    def _import_images(self):
        row = self.table.currentRow()
        if row < 0:
            QMessageBox.information(self, "提示", "请先选择一个人员")
            return
        name = self.table.item(row, 1).text()
        folder = QFileDialog.getExistingDirectory(self, f"选择 {name} 的图片文件夹")
        if folder:
            self.import_btn.setEnabled(False)
            self.progress.setVisible(True)
            self.progress.setRange(0, 0)
            self.worker = ImportWorker(self.gallery, name, folder)
            self.worker.log.connect(lambda msg: self.status_label.setText(msg))
            self.worker.finished.connect(self._on_import_done)
            self.worker.start()

    def _on_import_done(self):
        self.import_btn.setEnabled(True)
        self.progress.setVisible(False)
        self._refresh_table()

    def _rebuild_gallery(self):
        self.rebuild_btn.setEnabled(False)
        self.progress.setVisible(True)
        self.progress.setRange(0, 0)
        self.worker = BuildGalleryWorker(self.gallery)
        self.worker.log.connect(lambda msg: self.status_label.setText(msg))
        self.worker.finished.connect(self._on_rebuild_done)
        self.worker.start()

    def _on_rebuild_done(self):
        self.rebuild_btn.setEnabled(True)
        self.progress.setVisible(False)
        self._refresh_table()
