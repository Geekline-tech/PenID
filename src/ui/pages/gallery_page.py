import shutil
from pathlib import Path
from PyQt5.QtCore import Qt, QThread, pyqtSignal
from PyQt5.QtGui import QFont
from PyQt5.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QFileDialog,
    QInputDialog, QMessageBox, QTableWidget, QTableWidgetItem,
    QHeaderView, QAbstractItemView
)
from qfluentwidgets import (
    PushButton, PrimaryPushButton, CardWidget, SubtitleLabel, CaptionLabel,
    FluentIcon, ProgressBar, InfoBar, TitleLabel
)
from src.utils.config import RAW_DIR, PATCHES_DIR
from src.utils.database import Database


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
                    shutil.copy2(f, person_dir / f.name)
            self.log.emit(f"Importing {self.person_name}...")
            self.gallery.import_person(self.person_name, person_dir)
        except Exception as e:
            self.log.emit(f"Error: {e}")
        self.finished.emit()


class GalleryPage(QWidget):
    def __init__(self, gallery):
        super().__init__()
        self.setObjectName("gallery_page")
        self.gallery = gallery
        self._init_ui()
        self._refresh_table()

    def _init_ui(self):
        root = QVBoxLayout(self)
        root.setContentsMargins(32, 24, 32, 24)
        root.setSpacing(20)

        title = TitleLabel("人员管理")
        title.setStyleSheet("color: #cdd6f4; background: transparent;")
        root.addWidget(title)

        btn_card = CardWidget()
        bcl = QHBoxLayout(btn_card)
        bcl.setContentsMargins(20, 14, 20, 14)
        bcl.setSpacing(12)

        self.add_btn = PushButton("添加人员")
        self.add_btn.setIcon(FluentIcon.ADD)
        self.add_btn.setFixedHeight(40)
        self.add_btn.setFont(QFont("Microsoft YaHei UI", 11))
        self.add_btn.clicked.connect(self._add_person)

        self.import_btn = PushButton("导入图片")
        self.import_btn.setIcon(FluentIcon.FOLDER)
        self.import_btn.setFixedHeight(40)
        self.import_btn.setFont(QFont("Microsoft YaHei UI", 11))
        self.import_btn.clicked.connect(self._import_images)

        self.delete_btn = PushButton("删除选中")
        self.delete_btn.setIcon(FluentIcon.DELETE)
        self.delete_btn.setFixedHeight(40)
        self.delete_btn.setFont(QFont("Microsoft YaHei UI", 11))
        self.delete_btn.clicked.connect(self._delete_person)

        self.rebuild_btn = PrimaryPushButton("重建特征库")
        self.rebuild_btn.setIcon(FluentIcon.SYNC)
        self.rebuild_btn.setFixedHeight(40)
        self.rebuild_btn.setFont(QFont("Microsoft YaHei UI", 11))
        self.rebuild_btn.clicked.connect(self._rebuild_gallery)

        bcl.addWidget(self.add_btn)
        bcl.addWidget(self.import_btn)
        bcl.addWidget(self.delete_btn)
        bcl.addWidget(self.rebuild_btn)
        bcl.addStretch()
        root.addWidget(btn_card)

        self.progress = ProgressBar()
        self.progress.setVisible(False)
        root.addWidget(self.progress)

        table_card = CardWidget()
        tcl = QVBoxLayout(table_card)
        tcl.setContentsMargins(20, 16, 20, 16)

        header_row = QHBoxLayout()
        header_row.setSpacing(8)
        header_row.setContentsMargins(0, 0, 0, 8)
        h = SubtitleLabel("人员列表")
        h.setStyleSheet("color: #a6adc8; background: transparent;")
        header_row.addWidget(h)
        header_row.addStretch()
        self.status_label = CaptionLabel("")
        self.status_label.setStyleSheet("color: #6c7086; background: transparent;")
        header_row.addWidget(self.status_label)
        tcl.addLayout(header_row)

        self.table = QTableWidget()
        self.table.setColumnCount(4)
        self.table.setHorizontalHeaderLabels(["ID", "姓名", "样本数", "特征库"])
        self.table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        self.table.setSelectionBehavior(QAbstractItemView.SelectRows)
        self.table.setEditTriggers(QAbstractItemView.NoEditTriggers)
        self.table.verticalHeader().setVisible(False)
        self.table.setShowGrid(False)
        self.table.setAlternatingRowColors(True)
        self.table.setMinimumHeight(400)
        self.table.setFont(QFont("Microsoft YaHei UI", 12))
        self.table.setStyleSheet("""
            QTableWidget {
                background: transparent;
                alternate-background: rgba(255,255,255,0.02);
                color: #cdd6f4;
                border: none;
            }
            QTableWidget::item {
                padding: 12px 8px;
                border-bottom: 1px solid rgba(255,255,255,0.06);
            }
            QTableWidget::item:selected {
                background: rgba(137,180,250,0.15);
            }
            QHeaderView::section {
                background: transparent;
                color: #9399b2;
                border: none;
                border-bottom: 1px solid rgba(255,255,255,0.08);
                padding: 10px 8px;
                font-weight: bold;
                font-size: 12px;
            }
        """)
        tcl.addWidget(self.table, 1)
        root.addWidget(table_card, 1)

    def _refresh_table(self):
        db = Database()
        persons = db.get_persons()
        db.close()

        self.table.setRowCount(len(persons))
        for i, p in enumerate(persons):
            self.table.setItem(i, 0, QTableWidgetItem(str(p["id"])))
            self.table.setItem(i, 1, QTableWidgetItem(p["name"]))
            self.table.setItem(i, 2, QTableWidgetItem(str(p["sample_count"])))
            self.table.setItem(i, 3, QTableWidgetItem("Yes" if p["has_embedding"] else "No"))

        self.status_label.setText(f"共 {len(persons)} 人")

    def _add_person(self):
        name, ok = QInputDialog.getText(self, "添加人员", "输入人员名称:")
        if ok and name:
            db = Database()
            db.add_person(name)
            db.close()
            self._refresh_table()
            InfoBar.success("完成", f"已添加: {name}", parent=self.window())

    def _delete_person(self):
        row = self.table.currentRow()
        if row < 0:
            InfoBar.warning("提示", "请先选择人员", parent=self.window())
            return
        person_id = int(self.table.item(row, 0).text())
        name = self.table.item(row, 1).text()
        reply = QMessageBox.question(self, "确认删除", f"删除 {name}?", QMessageBox.Yes | QMessageBox.No)
        if reply == QMessageBox.Yes:
            db = Database()
            db.delete_person(person_id)
            db.close()
            p = RAW_DIR / name
            pp = PATCHES_DIR / name
            if p.exists(): shutil.rmtree(p)
            if pp.exists(): shutil.rmtree(pp)
            self.gallery._gallery_cache = None
            self._refresh_table()

    def _import_images(self):
        row = self.table.currentRow()
        if row < 0:
            InfoBar.warning("提示", "请先选择人员", parent=self.window())
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
        InfoBar.success("完成", "导入完成", parent=self.window())

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
        InfoBar.success("完成", "重建完成", parent=self.window())
