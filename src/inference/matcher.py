import torch
import numpy as np
import cv2
from pathlib import Path
from typing import Optional
from src.models.pennet import PenNet
from src.data.augmentation import get_inference_transforms
from src.data.preprocess import binarize, deskew, segment_lines, crop_lines, split_line_to_patches, is_blank_line
from src.utils.config import Config, MODELS_DIR


class Embedder:
    def __init__(self, config: Optional[Config] = None):
        self.config = config or Config()
        device_str = self.config.get("device", "auto")
        if device_str == "auto":
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device_str)
        self.model = PenNet(
            embed_dim=self.config["embed_dim"],
            pretrained=False,
        ).to(self.device)
        self.model.eval()
        self.transform = get_inference_transforms(self.config["image_size"])

    def load_weights(self, path: Optional[Path] = None):
        if path is None:
            path = MODELS_DIR / "pennet_best.pth"
        checkpoint = torch.load(path, map_location=self.device)
        self.model.load_state_dict(checkpoint["model_state_dict"])

    @torch.no_grad()
    def embed_image(self, image: np.ndarray) -> np.ndarray:
        if len(image.shape) == 2:
            image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
        tensor = self.transform(image).unsqueeze(0).to(self.device)
        emb = self.model(tensor)
        return emb.cpu().numpy().flatten()

    @torch.no_grad()
    def embed_batch(self, images: list[np.ndarray]) -> np.ndarray:
        tensors = []
        for img in images:
            if len(img.shape) == 2:
                img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
            tensors.append(self.transform(img))
        batch = torch.stack(tensors).to(self.device)
        embs = self.model(batch)
        return embs.cpu().numpy()

    @torch.no_grad()
    def embed_path(self, image_path: Path) -> np.ndarray:
        data = np.fromfile(str(image_path), dtype=np.uint8)
        img = cv2.imdecode(data, cv2.IMREAD_GRAYSCALE)
        return self.embed_image(img)

    def embed_full_page(self, image: np.ndarray) -> np.ndarray:
        if len(image.shape) == 2:
            gray = image
        else:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        binary = binarize(gray)
        binary = deskew(binary)
        lines = segment_lines(binary)
        crops = crop_lines(binary, lines)

        all_patches = []
        for line_img in crops:
            if is_blank_line(line_img):
                continue
            patches = split_line_to_patches(line_img, target_height=64, patch_width=384)
            all_patches.extend(patches)

        if not all_patches:
            return self.embed_image(gray)

        batch_size = 64
        all_embs = []
        for i in range(0, len(all_patches), batch_size):
            batch = all_patches[i : i + batch_size]
            embs = self.embed_batch(batch)
            all_embs.append(embs)

        mean_emb = np.mean(np.concatenate(all_embs, axis=0), axis=0)
        mean_emb = mean_emb / (np.linalg.norm(mean_emb) + 1e-8)
        return mean_emb
