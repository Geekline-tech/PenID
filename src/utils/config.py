from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent.parent
DATA_DIR = BASE_DIR / "data"
RAW_DIR = DATA_DIR / "raw"
PATCHES_DIR = DATA_DIR / "patches"
MODELS_DIR = BASE_DIR / "models"
ASSETS_DIR = BASE_DIR / "assets"

DEFAULT_CONFIG = {
    "image_size": (64, 384),
    "embed_dim": 512,
    "pretrained": True,
    "triplet_margin": 0.3,
    "lambda_cls": 0.5,
    "lr": 1e-4,
    "weight_decay": 1e-5,
    "batch_size": 32,
    "epochs": 60,
    "train_split": 0.8,
    "match_threshold": 0.6,
    "device": "auto",
}


class Config:
    _instance = None
    _data: dict = {}

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._data = dict(DEFAULT_CONFIG)
        return cls._instance

    def get(self, key, default=None):
        return self._data.get(key, default)

    def set(self, key, value):
        self._data[key] = value

    def update(self, d: dict):
        self._data.update(d)

    def __getitem__(self, key):
        return self._data[key]

    def __setitem__(self, key, value):
        self._data[key] = value

    def to_dict(self):
        return dict(self._data)
