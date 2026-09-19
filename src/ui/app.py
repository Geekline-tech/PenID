import sys
from PyQt5.QtCore import Qt, QThread, pyqtSignal
from PyQt5.QtGui import QFont
from PyQt5.QtWidgets import QApplication, QMainWindow, QStackedWidget, QHBoxLayout, QWidget
from qfluentwidgets import (
    NavigationInterface, NavigationItemPosition, setTheme, Theme,
    FluentIcon, FluentTranslator
)
from src.inference.gallery import Gallery


class ModelLoader(QThread):
    finished = pyqtSignal(bool, str)

    def __init__(self, gallery):
        super().__init__()
        self.gallery = gallery

    def run(self):
        try:
            self.gallery.load_model()
            self.finished.emit(True, "Model loaded")
        except Exception as e:
            self.finished.emit(False, str(e))


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Pen ID")
        self.resize(1100, 750)
        self.setMinimumSize(900, 600)

        setTheme(Theme.DARK)

        self.gallery = Gallery()

        self.stack = QStackedWidget(self)

        from src.ui.pages.identify_page import IdentifyPage
        from src.ui.pages.train_page import TrainPage
        from src.ui.pages.gallery_page import GalleryPage

        self.identify_page = IdentifyPage(self.gallery)
        self.train_page = TrainPage(self.gallery)
        self.gallery_page = GalleryPage(self.gallery)

        self.stack.addWidget(self.identify_page)
        self.stack.addWidget(self.train_page)
        self.stack.addWidget(self.gallery_page)

        self.nav = NavigationInterface(self)
        self.nav.addItem("identify", FluentIcon.SEARCH, "识别", onClick=lambda: self.stack.setCurrentIndex(0))
        self.nav.addItem("train", FluentIcon.TRAIN, "训练", onClick=lambda: self.stack.setCurrentIndex(1))
        self.nav.addItem("gallery", FluentIcon.PEOPLE, "人员管理", onClick=lambda: self.stack.setCurrentIndex(2))

        central = QWidget()
        self.setCentralWidget(central)
        h_layout = QHBoxLayout(central)
        h_layout.setContentsMargins(0, 0, 0, 0)
        h_layout.setSpacing(0)
        h_layout.addWidget(self.nav)
        h_layout.addWidget(self.stack)

        self.statusBar().showMessage("Loading model...")
        self._load_model()

    def _load_model(self):
        self.loader = ModelLoader(self.gallery)
        self.loader.finished.connect(self._on_model_loaded)
        self.loader.start()

    def _on_model_loaded(self, ok, msg):
        if ok:
            self.statusBar().showMessage("Model ready")
        else:
            self.statusBar().showMessage(f"Model load failed: {msg}")


def main():
    app = QApplication(sys.argv)
    app.setFont(QFont("Segoe UI", 10))
    FluentTranslator()
    window = MainWindow()
    window.show()
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()
