from PyQt5.QtCore import Qt, QThread, pyqtSignal
from PyQt5.QtGui import QFont
from PyQt5.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QPlainTextEdit, QSizePolicy, QLabel
)
from qfluentwidgets import (
    PushButton, PrimaryPushButton, CardWidget, SubtitleLabel, BodyLabel,
    CaptionLabel, FluentIcon, ProgressBar, LineEdit, StrongBodyLabel
)
from src.utils.config import Config
from src.utils.database import Database


class TrainWorker(QThread):
    log = pyqtSignal(str)
    finished = pyqtSignal(float)

    def __init__(self, config, gallery):
        super().__init__()
        self.config = config
        self.gallery = gallery

    def run(self):
        try:
            from src.training.trainer import Trainer

            def cb(msg):
                self.log.emit(msg)

            trainer = Trainer(self.config, progress_callback=cb)
            trainer.train()
            self.log.emit("Rebuilding gallery...")
            self.gallery.build_gallery()
            self.finished.emit(trainer.best_acc)
        except Exception as e:
            self.log.emit(f"Error: {e}")
            self.finished.emit(0)


class StatCard(QWidget):
    def __init__(self, label_text, parent=None):
        super().__init__(parent)
        self.setFixedHeight(110)
        self.setMinimumWidth(180)
        self.setStyleSheet("StatCard { background: #313244; border-radius: 12px; }")

        layout = QVBoxLayout(self)
        layout.setContentsMargins(20, 16, 20, 16)
        layout.setAlignment(Qt.AlignCenter)

        self.value = QLabel("--")
        self.value.setFont(QFont("Segoe UI", 26, QFont.Bold))
        self.value.setStyleSheet("color: #89b4fa; background: transparent;")
        self.value.setAlignment(Qt.AlignCenter)
        layout.addWidget(self.value)

        self.label = CaptionLabel(label_text)
        self.label.setAlignment(Qt.AlignCenter)
        self.label.setStyleSheet("color: #6c7086; background: transparent;")
        layout.addWidget(self.label)


class TrainPage(QWidget):
    def __init__(self, gallery):
        super().__init__()
        self.setObjectName("train_page")
        self.gallery = gallery
        self.worker = None
        self._init_ui()
        self._refresh_stats()

    def _init_ui(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(28, 28, 28, 28)
        layout.setSpacing(20)

        layout.addWidget(StrongBodyLabel("模型训练"))

        stats_row = QHBoxLayout()
        stats_row.setSpacing(16)
        self.stat_cards = {}
        for key, label in [("persons", "人员数"), ("total", "总样本"), ("avg", "每人均值"), ("classes", "类别数")]:
            card = StatCard(label)
            self.stat_cards[key] = card.value
            stats_row.addWidget(card)
        stats_row.addStretch()
        layout.addLayout(stats_row)

        param_card = CardWidget()
        param_layout = QVBoxLayout(param_card)
        param_layout.setContentsMargins(20, 20, 20, 20)
        param_layout.setSpacing(14)

        param_title = BodyLabel("训练参数")
        param_title.setFont(QFont("Microsoft YaHei UI", 12, QFont.Bold))
        param_title.setStyleSheet("color: #cdd6f4;")
        param_layout.addWidget(param_title)

        inputs_layout = QHBoxLayout()
        inputs_layout.setSpacing(24)
        self.inputs = {}
        for key, label, default in [
            ("epochs", "轮次", "60"),
            ("batch_size", "Batch Size", "32"),
            ("lr", "学习率", "0.0001"),
            ("embed_dim", "Embedding维度", "512"),
        ]:
            col = QVBoxLayout()
            col.setSpacing(6)
            lbl = CaptionLabel(label)
            lbl.setStyleSheet("color: #a6adc8;")
            col.addWidget(lbl)
            edit = LineEdit()
            edit.setText(default)
            edit.setFixedWidth(120)
            edit.setFixedHeight(36)
            edit.setFont(QFont("Microsoft YaHei UI", 11))
            self.inputs[key] = edit
            col.addWidget(edit)
            inputs_layout.addLayout(col)
        inputs_layout.addStretch()
        param_layout.addLayout(inputs_layout)
        layout.addWidget(param_card)

        btn_row = QHBoxLayout()
        self.train_btn = PrimaryPushButton("开始训练")
        self.train_btn.setIcon(FluentIcon.PLAY)
        self.train_btn.setFixedHeight(48)
        self.train_btn.setMinimumWidth(180)
        self.train_btn.setFont(QFont("Microsoft YaHei UI", 12, QFont.Bold))
        self.train_btn.clicked.connect(self._start_training)
        btn_row.addWidget(self.train_btn)
        btn_row.addStretch()
        layout.addLayout(btn_row)

        self.progress = ProgressBar()
        self.progress.setVisible(False)
        layout.addWidget(self.progress)

        log_card = CardWidget()
        log_layout = QVBoxLayout(log_card)
        log_layout.setContentsMargins(16, 16, 16, 16)
        log_title = CaptionLabel("训练日志")
        log_title.setStyleSheet("color: #a6adc8;")
        log_layout.addWidget(log_title)
        self.log_area = QPlainTextEdit()
        self.log_area.setReadOnly(True)
        self.log_area.setFont(QFont("Consolas", 10))
        self.log_area.setStyleSheet(
            "QPlainTextEdit { background: #181825; color: #cdd6f4; border: 1px solid #313244; border-radius: 8px; padding: 8px; }"
        )
        self.log_area.setMinimumHeight(200)
        log_layout.addWidget(self.log_area, 1)
        layout.addWidget(log_card, 1)

    def _refresh_stats(self):
        db = Database()
        persons = db.get_persons()
        total = sum(p["sample_count"] for p in persons)
        count = len(persons)
        avg = total // count if count else 0
        self.stat_cards["persons"].setText(str(count))
        self.stat_cards["total"].setText(f"{total:,}")
        self.stat_cards["avg"].setText(f"{avg:,}")
        self.stat_cards["classes"].setText(str(count))
        db.close()

    def _start_training(self):
        self.train_btn.setEnabled(False)
        self.progress.setVisible(True)
        self.progress.setRange(0, 0)
        self.log_area.clear()
        self.log_area.appendPlainText("Starting training...")

        config = Config()
        config.update({
            "epochs": int(self.inputs["epochs"].text()),
            "batch_size": int(self.inputs["batch_size"].text()),
            "lr": float(self.inputs["lr"].text()),
            "embed_dim": int(self.inputs["embed_dim"].text()),
        })

        self.worker = TrainWorker(config, self.gallery)
        self.worker.log.connect(self._on_log)
        self.worker.finished.connect(self._on_finished)
        self.worker.start()

    def _on_log(self, msg):
        self.log_area.appendPlainText(msg)

    def _on_finished(self, best_acc):
        self.log_area.appendPlainText(f"\nTraining complete. Best accuracy: {best_acc:.1f}%")
        self.train_btn.setEnabled(True)
        self.progress.setVisible(False)
        self._refresh_stats()
