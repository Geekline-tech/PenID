import sys
from PyQt5.QtCore import Qt, QThread, pyqtSignal
from PyQt5.QtGui import QFont
from PyQt5.QtWidgets import QApplication, QSizePolicy
from qfluentwidgets import FluentWindow, setTheme, Theme, FluentIcon, FluentTranslator
from src.inference.gallery import Gallery

try:
    from PyQt5.QtCore import PYQT_VERSION_STR
    if PYQT_VERSION_STR.startswith("5"):
        QApplication.setAttribute(Qt.AA_EnableHighDpiScaling, True)
        QApplication.setAttribute(Qt.AA_UseHighDpiPixmaps, True)
except Exception:
    pass


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


class MainWindow(FluentWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Pen ID")
        self.resize(1280, 800)
        self.setMinimumSize(960, 640)

        self.gallery = Gallery()

        from src.ui.pages.identify_page import IdentifyPage
        from src.ui.pages.train_page import TrainPage
        from src.ui.pages.gallery_page import GalleryPage

        self.identify_page = IdentifyPage(self.gallery)
        self.train_page = TrainPage(self.gallery)
        self.gallery_page = GalleryPage(self.gallery)

        self.navigationInterface.setExpandWidth(240)
        self.navigationInterface.setMinimumWidth(240)

        self.addSubInterface(self.identify_page, FluentIcon.SEARCH, "识别")
        self.addSubInterface(self.train_page, FluentIcon.TRAIN, "训练")
        self.addSubInterface(self.gallery_page, FluentIcon.PEOPLE, "人员管理")

        self._load_model()

    def _load_model(self):
        self.loader = ModelLoader(self.gallery)
        self.loader.finished.connect(self._on_model_loaded)
        self.loader.start()

    def _on_model_loaded(self, ok, msg):
        if ok:
            self.setWindowTitle("Pen ID - 模型已加载")
        else:
            self.setWindowTitle(f"Pen ID - 加载失败: {msg}")


def main():
    app = QApplication(sys.argv)
    app.setHighDpiScaleFactorRoundingPolicy(Qt.HighDpiScaleFactorRoundingPolicy.PassThrough)
    app.setFont(QFont("Microsoft YaHei UI", 10))
    FluentTranslator()
    setTheme(Theme.DARK)
    window = MainWindow()
    window.show()
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()
