import random
import numpy as np
import cv2
from pathlib import Path
from torch.utils.data import Dataset
from typing import Optional
from src.data.augmentation import get_train_transforms, get_inference_transforms


class TripletDataset(Dataset):
    def __init__(
        self,
        patches_dir: Path,
        image_size: tuple[int, int] = (128, 128),
        augment: bool = True,
    ):
        self.patches_dir = Path(patches_dir)
        self.image_size = image_size
        self.augment = augment
        self.transform = get_train_transforms(image_size) if augment else get_inference_transforms(image_size)

        self.person_patches: dict[str, list[Path]] = {}
        self.person_ids: list[str] = []

        for person_dir in sorted(self.patches_dir.iterdir()):
            if person_dir.is_dir():
                patches = sorted(person_dir.glob("*.png"))
                if patches:
                    self.person_patches[person_dir.name] = patches
                    self.person_ids.append(person_dir.name)

        self.samples: list[tuple[str, Path]] = []
        for pid, patches in self.person_patches.items():
            for p in patches:
                self.samples.append((pid, p))

    def __len__(self) -> int:
        return len(self.samples)

    def _load_image(self, path: Path) -> np.ndarray:
        img = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
        if img is None:
            raise FileNotFoundError(f"Cannot load: {path}")
        return img

    def _apply_transform(self, img: np.ndarray):
        img_rgb = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
        return self.transform(img_rgb)

    def __getitem__(self, idx: int) -> tuple:
        anchor_id, anchor_path = self.samples[idx]

        positive_candidates = [p for p in self.person_patches[anchor_id] if p != anchor_path]
        if not positive_candidates:
            positive_candidates = [anchor_path]
        positive_path = random.choice(positive_candidates)

        negative_id = random.choice([pid for pid in self.person_ids if pid != anchor_id])
        negative_path = random.choice(self.person_patches[negative_id])

        anchor_img = self._load_image(anchor_path)
        positive_img = self._load_image(positive_path)
        negative_img = self._load_image(negative_path)

        anchor = self._apply_transform(anchor_img)
        positive = self._apply_transform(positive_img)
        negative = self._apply_transform(negative_img)

        label = self.person_ids.index(anchor_id)
        return anchor, positive, negative, label


class SingleImageDataset(Dataset):
    def __init__(self, image_paths: list[Path], image_size: tuple[int, int] = (128, 128)):
        self.image_paths = image_paths
        self.transform = get_inference_transforms(image_size)

    def __len__(self) -> int:
        return len(self.image_paths)

    def __getitem__(self, idx: int):
        img = cv2.imread(str(self.image_paths[idx]), cv2.IMREAD_GRAYSCALE)
        img_rgb = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
        return self.transform(img_rgb)
