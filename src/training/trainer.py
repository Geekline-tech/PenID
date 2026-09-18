import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
from pathlib import Path
from tqdm import tqdm
from typing import Optional
from src.models.pennet import PenNet, PenNetWithClassifier
from src.models.losses import CombinedLoss
from src.data.dataset import TripletDataset
from src.utils.config import Config, PATCHES_DIR, MODELS_DIR


class Trainer:
    def __init__(self, config: Optional[Config] = None, progress_callback=None):
        self.config = config or Config()
        device_str = self.config.get("device", "auto")
        if device_str == "auto":
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device_str)
        self.progress_callback = progress_callback

        self.model = PenNetWithClassifier(
            embed_dim=self.config["embed_dim"],
            num_classes=0,
            pretrained=self.config["pretrained"],
        ).to(self.device)

        self.criterion = None
        self.optimizer = None
        self.scheduler = None
        self.best_acc = 0.0

    def _log(self, msg: str):
        if self.progress_callback:
            self.progress_callback(msg)
        print(msg)

    def prepare_data(self, patches_dir: Path = PATCHES_DIR):
        dataset = TripletDataset(patches_dir, self.config["image_size"], augment=True)
        if len(dataset) == 0:
            raise ValueError(f"No patches found in {patches_dir}")

        num_classes = len(dataset.person_ids)
        self.model = PenNetWithClassifier(
            embed_dim=self.config["embed_dim"],
            num_classes=num_classes,
            pretrained=self.config["pretrained"],
        ).to(self.device)

        self.criterion = CombinedLoss(
            margin=0.3,
            lambda_cls=0.5,
        )

        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=self.config["lr"],
            weight_decay=self.config["weight_decay"],
        )
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer, T_max=self.config["epochs"]
        )

        split = self.config["train_split"]
        train_size = int(len(dataset) * split)
        val_size = len(dataset) - train_size
        train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

        self.train_loader = DataLoader(
            train_dataset, batch_size=self.config["batch_size"], shuffle=True, num_workers=0, drop_last=True
        )
        self.val_loader = DataLoader(
            val_dataset, batch_size=self.config["batch_size"], shuffle=False, num_workers=0
        )

        self._log(f"Dataset: {len(dataset)} patches, {num_classes} persons")
        self._log(f"Train: {train_size}, Val: {val_size}")
        return dataset

    def train_epoch(self, epoch: int) -> dict:
        self.model.train()
        total_loss = 0.0
        total_triplet = 0.0
        total_cls = 0.0
        correct = 0
        total = 0

        pbar = tqdm(self.train_loader, desc=f"Epoch {epoch}")
        for anchor, positive, negative, labels in pbar:
            anchor = anchor.to(self.device)
            positive = positive.to(self.device)
            negative = negative.to(self.device)
            labels = labels.to(self.device)

            emb_a, logits = self.model(anchor)
            emb_p, _ = self.model(positive)
            emb_n, _ = self.model(negative)

            loss, t_loss, c_loss = self.criterion(emb_a, emb_p, emb_n, logits, labels)

            self.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=5.0)
            self.optimizer.step()

            total_loss += loss.item()
            total_triplet += t_loss.item()
            total_cls += c_loss.item()
            _, predicted = logits.max(1)
            correct += predicted.eq(labels).sum().item()
            total += labels.size(0)

            pbar.set_postfix(loss=f"{loss.item():.4f}", acc=f"{100.0 * correct / total:.1f}%")

        return {
            "loss": total_loss / len(self.train_loader),
            "triplet_loss": total_triplet / len(self.train_loader),
            "cls_loss": total_cls / len(self.train_loader),
            "accuracy": 100.0 * correct / total,
        }

    @torch.no_grad()
    def validate(self) -> dict:
        self.model.eval()
        correct = 0
        total = 0
        total_loss = 0.0

        for anchor, _, _, labels in self.val_loader:
            anchor = anchor.to(self.device)
            labels = labels.to(self.device)

            emb, logits = self.model(anchor)
            loss = nn.CrossEntropyLoss()(logits, labels)
            total_loss += loss.item()

            _, predicted = logits.max(1)
            correct += predicted.eq(labels).sum().item()
            total += labels.size(0)

        return {
            "val_loss": total_loss / max(len(self.val_loader), 1),
            "val_accuracy": 100.0 * correct / total if total > 0 else 0,
        }

    def save_model(self, path: Optional[Path] = None, val_acc: float = 0.0) -> Path:
        if path is None:
            MODELS_DIR.mkdir(parents=True, exist_ok=True)
            path = MODELS_DIR / "pennet_best.pth"
        torch.save({
            "model_state_dict": self.model.pennet.state_dict(),
            "config": self.config.to_dict(),
            "val_acc": val_acc,
        }, path)
        self._log(f"Model saved to {path}")
        return path

    def load_model(self, path: Path):
        checkpoint = torch.load(path, map_location=self.device)
        self.model.pennet.load_state_dict(checkpoint["model_state_dict"])
        self._log(f"Model loaded from {path}")

    def train(self, patches_dir: Path = PATCHES_DIR, save_dir: Optional[Path] = None):
        self.prepare_data(patches_dir)

        for epoch in range(1, self.config["epochs"] + 1):
            self._log(f"\n{'='*50}")
            self._log(f"Epoch {epoch}/{self.config['epochs']}")

            train_metrics = self.train_epoch(epoch)
            val_metrics = self.validate()
            self.scheduler.step()

            self._log(
                f"Train Loss: {train_metrics['loss']:.4f} | "
                f"Acc: {train_metrics['accuracy']:.1f}% | "
                f"Val Acc: {val_metrics['val_accuracy']:.1f}%"
            )

            if val_metrics["val_accuracy"] > self.best_acc:
                self.best_acc = val_metrics["val_accuracy"]
                self.save_model(save_dir, val_acc=self.best_acc)
                self._log(f"  -> New best: {self.best_acc:.1f}%")

        self._log(f"\nTraining complete. Best accuracy: {self.best_acc:.1f}%")
