import shutil
from pathlib import Path
from PyQt5.QtCore import Qt, QThread, pyqtSignal
from PyQt5.QtGui import QFont
from PyQt5.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QFileDialog,
    QInputDialog, QMessageBox, QTableWidget, QTableWidgetItem,
    QHeaderView, QAbstractItemView, QScrollArea, QFrame
)
from qfluentwidgets import (
    PushButton, PrimaryPushButton, CardWidget, SubtitleLabel, CaptionLabel,
    FluentIcon, ProgressBar, InfoBar, StrongBodyLabel, TableWidget,
    TitleLabel
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
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)

        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setStyleSheet("QScrollArea { border: none; background: transparent; }")

        content = QWidget()
        content_layout = QVBoxLayout(content)
        content_layout.setContentsMargins(32, 32, 32, 32)
        content_layout.setSpacing(24)

        header = StrongBodyLabel("人员管理")
        header.setFont(QFont("Microsoft YaHei UI", 14))
        header.setStyleSheet("color: #cdd6f4; background: transparent;")
        content_layout.addWidget(header)

        action_card = CardWidget()
        ac_layout = QHBoxLayout(action_card)
        ac_layout.setContentsMargins(24, 20, 24, 20)
        ac_layout.setSpacing(16)

        self.add_btn = PushButton("添加人员")
        self.add_btn.setIcon(FluentIcon.ADD)
        self.add_btn.setFixedHeight(44)
        self.add_btn.setMinimumWidth(150)
        self.add_btn.setFont(QFont("Microsoft YaHei UI", 11))
        self.add_btn.clicked.connect(self._add_person)

        self.import_btn = PushButton("导入图片")
        self.import_btn.setIcon(FluentIcon.FOLDER)
        self.import_btn.setFixedHeight(44)
        self.import_btn.setMinimumWidth(150)
        self.import_btn.setFont(QFont("Microsoft YaHei UI", 11))
        self.import_btn.clicked.connect(self._import_images)

        self.delete_btn = PushButton("删除选中")
        self.delete_btn.setIcon(FluentIcon.DELETE)
        self.delete_btn.setFixedHeight(44)
        self.delete_btn.setMinimumWidth(150)
        self.delete_btn.setFont(QFont("Microsoft YaHei UI", 11))
        self.delete_btn.clicked.connect(self._delete_person)

        self.rebuild_btn = PrimaryPushButton("重建特征库")
        self.rebuild_btn.setIcon(FluentIcon.SYNC)
        self.rebuild_btn.setFixedHeight(44)
        self.rebuild_btn.setMinimumWidth(170)
        self.rebuild_btn.setFont(QFont("Microsoft YaHei UI", 11))
        self.rebuild_btn.clicked.connect(self._rebuild_gallery)

        ac_layout.addWidget(self.add_btn)
        ac_layout.addWidget(self.import_btn)
        ac_layout.addWidget(self.delete_btn)
        ac_layout.addWidget(self.rebuild_btn)
        ac_layout.addStretch()

        content_layout.addWidget(action_card)

        self.progress = ProgressBar()
        self.progress.setVisible(False)
        content_layout.addWidget(self.progress)

        table_card = CardWidget()
        tc_layout = QVBoxLayout(table_card)
        tc_layout.setContentsMargins(24, 20, 24, 20)

        self.table = QTableWidget()
        self.table.setColumnCount(4)
        self.table.setHorizontalHeaderLabels(["ID", "姓名", "样本数", "特征库"])
        self.table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        self.table.setSelectionBehavior(QAbstractItemView.SelectRows)
        self.table.setEditTriggers(QAbstractItemView.NoEditTriggers)
        self.table.verticalHeader().setVisible(False)
        self.table.setShowGrid(False)
        self.table.setAlternatingRowColors(True)
        self.table.setRowCount(0)
        self.table.setMinimumHeight(400)
        self.table.setStyleSheet("""
            QTableWidget {
                background: transparent;
                alternate-background: rgba(255,255,255,0.03);
                color: #cdd6f4;
                border: none;
                font-size: 14px;
                font-family: 'Microsoft YaHei UI';
            }
            QTableWidget::item {
                padding: 10px 14px;
                border-bottom: 1px solid #313244;
            }
            QTableWidget::item:selected {
                background: #45475a;
            }
            QHeaderView::section {
                background: transparent;
                color: #a6adc8;
                border: none;
                border-bottom: 2px solid #313244;
                padding: 12px 14px;
                font-weight: bold;
                font-size: 13px;
            }
        """)
        tc_layout.addWidget(self.table, 1)
        content_layout.addWidget(table_card, 1)

        self.status_label = CaptionLabel("")
        self.status_label.setStyleSheet("color: #6c7086; background: transparent; font-size: 13px;")
        content_layout.addWidget(self.status_label)

        scroll.setWidget(content)
        layout.addWidget(scroll)

    def _refresh_table(self):
        db = Database()
        persons = db.get_persons()
        db.close()

        self.table.setRowCount(len(persons))
        for i, p in enumerate(persons):
            self.table.setItem(i, 0, QTableWidgetItem(str(p["id"])))
            self.table.setItem(i, 1, QTableWidgetItem(p["name"]))
            self.table.setItem(i, 2, QTableWidgetItem(str(p["sample_count"])))
            emb_status = "✓" if p["has_embedding"] else "✗"
            self.table.setItem(i, 3, QTableWidgetItem(emb_status))

        self.status_label.setText(f"共 {len(persons)} 人")

    def _add_person(self):
        name, ok = QInputDialog.getText(self, "添加人员", "输入人员名称:")
        if ok and name:
            db = Database()
            db.add_person(name)
            db.close()
            self._refresh_table()
            InfoBar.success("完成", f"已添加人员: {name}", parent=self.window())

    def _delete_person(self):
        row = self.table.currentRow()
        if row < 0:
            InfoBar.warning("提示", "请先选择一个人员", parent=self.window())
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
            if person_dir.exists():
                shutil.rmtree(person_dir)
            if patches_dir.exists():
                shutil.rmtree(patches_dir)
            self.gallery._gallery_cache = None
            self._refresh_table()
            InfoBar.success("完成", f"已删除人员: {name}", parent=self.window())

    def _import_images(self):
        row = self.table.currentRow()
        if row < 0:
            InfoBar.warning("提示", "请先选择一个人员", parent=self.window())
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
        InfoBar.success("完成", "图片导入完成", parent=self.window())

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
        InfoBar.success("完成", "特征库重建完成", parent=self.window())
