import torch
import numpy as np
import cv2
from pathlib import Path
from typing import Optional
from src.inference.matcher import Embedder
from src.utils.database import Database
from src.utils.config import Config, PATCHES_DIR, RAW_DIR
from src.data.preprocess import prepare_person_data


class Gallery:
    def __init__(self, config: Optional[Config] = None):
        self.config = config or Config()
        self.db = Database()
        self.embedder = Embedder(config)
        self._gallery_cache: Optional[dict[int, np.ndarray]] = None

    def load_model(self, model_path: Optional[Path] = None):
        self.embedder.load_weights(model_path)

    def build_gallery(self, patches_dir: Path = PATCHES_DIR):
        self._gallery_cache = None
        persons = self.db.get_persons()
        for person in persons:
            person_dir = patches_dir / person["name"]
            if not person_dir.exists():
                continue
            patch_files = sorted(person_dir.glob("*.png"))
            if not patch_files:
                continue

            images = []
            for pf in patch_files:
                data = np.fromfile(str(pf), dtype=np.uint8)
                img = cv2.imdecode(data, cv2.IMREAD_GRAYSCALE)
                if img is not None:
                    images.append(img)

            if images:
                embeddings = self.embedder.embed_batch(images)
                mean_emb = np.mean(embeddings, axis=0)
                mean_emb = mean_emb / np.linalg.norm(mean_emb)
                self.db.save_embedding(person["id"], mean_emb)
                self.db.update_sample_count(person["id"], len(patch_files))

    def import_person(self, name: str, raw_dir: Path, patches_dir: Path = PATCHES_DIR) -> int:
        person = self.db.get_person_by_name(name)
        if person:
            person_id = person["id"]
        else:
            person_id = self.db.add_person(name)

        out_dir = patches_dir / name
        prepare_person_data(raw_dir, out_dir, person_prefix=name)

        patch_files = sorted(out_dir.glob("*.png"))
        if patch_files:
            images = []
            for pf in patch_files:
                data = np.fromfile(str(pf), dtype=np.uint8)
                img = cv2.imdecode(data, cv2.IMREAD_GRAYSCALE)
                if img is not None:
                    images.append(img)
            embeddings = self.embedder.embed_batch(images)
            mean_emb = np.mean(embeddings, axis=0)
            mean_emb = mean_emb / np.linalg.norm(mean_emb)
            self.db.save_embedding(person_id, mean_emb)
            self.db.update_sample_count(person_id, len(patch_files))
            self._gallery_cache = None

        return person_id

    def _get_gallery(self) -> dict[int, np.ndarray]:
        if self._gallery_cache is None:
            self._gallery_cache = self.db.load_all_embeddings()
        return self._gallery_cache

    def match(self, query_embedding: np.ndarray, top_n: int = 10) -> dict:
        gallery = self._get_gallery()
        if not gallery:
            return {"person_id": None, "name": "Unknown", "confidence": 0.0, "distance": 1.0, "rankings": []}

        results = []
        for pid, emb in gallery.items():
            dist = 1.0 - np.dot(query_embedding, emb) / (
                np.linalg.norm(query_embedding) * np.linalg.norm(emb) + 1e-8
            )
            confidence = max(0.0, 1.0 - dist)
            person = self.db.get_person(pid)
            results.append({
                "person_id": pid,
                "name": person["name"] if person else "Unknown",
                "confidence": confidence,
                "distance": dist,
            })

        results.sort(key=lambda x: x["distance"])
        threshold = self.config.get("match_threshold", 0.6)

        top = results[0] if results else None
        return {
            "person_id": top["person_id"] if top else None,
            "name": top["name"] if top else "Unknown",
            "confidence": top["confidence"] if top else 0.0,
            "distance": top["distance"] if top else 1.0,
            "matched": (top["distance"] < threshold) if top else False,
            "rankings": results[:top_n],
        }

    def identify_image(self, image: np.ndarray) -> dict:
        emb = self.embedder.embed_full_page(image)
        return self.match(emb)

    def identify_path(self, image_path: Path) -> dict:
        data = np.fromfile(str(image_path), dtype=np.uint8)
        img = cv2.imdecode(data, cv2.IMREAD_GRAYSCALE)
        emb = self.embedder.embed_full_page(img)
        return self.match(emb)

    def get_persons(self) -> list[dict]:
        return self.db.get_persons()

    def delete_person(self, person_id: int):
        self.db.delete_person(person_id)
        self._gallery_cache = None
