import sys
from PyQt5.QtCore import Qt, QThread, pyqtSignal
from PyQt5.QtGui import QFont, QIcon
from PyQt5.QtWidgets import QApplication, QMainWindow, QStackedWidget
from qfluentwidgets import (
    NavigationInterface, NavigationItemPosition, setTheme, Theme,
    FluentIcon, FluentTranslator
)
from src.inference.gallery import Gallery
from src.utils.config import Config


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

        self.nav = NavigationInterface(self, showCancelButton=True)
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

        self.nav.addItem("identify", FluentIcon.SEARCH, "识别")
        self.nav.addItem("train", FluentIcon.TRAIN, "训练")
        self.nav.addItem("gallery", FluentIcon.PEOPLE, "人员管理")

        self.nav.setCurrentItem("identify")
        self.nav.clicked.connect(self._on_nav_clicked)

        self.setCentralWidget(self.stack)
        self.hBoxLayout = __import__("PyQt5.QtWidgets", fromlist=["QHBoxLayout"]).QHBoxLayout(self)
        self.hBoxLayout.setContentsMargins(0, 0, 0, 0)
        self.hBoxLayout.addWidget(self.nav)
        self.hBoxLayout.addWidget(self.stack)

        self.statusBar().showMessage("Loading model...")
        self._load_model()

    def _on_nav_clicked(self, key):
        page_map = {
            "identify": 0,
            "train": 1,
            "gallery": 2,
        }
        if key in page_map:
            self.stack.setCurrentIndex(page_map[key])

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
