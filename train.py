import argparse
from pathlib import Path
from src.utils.config import Config, PATCHES_DIR
from src.training.trainer import Trainer


def main():
    parser = argparse.ArgumentParser(description="Pen ID - 训练笔迹识别模型")
    parser.add_argument("--epochs", type=int, default=100, help="训练轮次")
    parser.add_argument("--batch-size", type=int, default=64, help="批大小")
    parser.add_argument("--lr", type=float, default=1e-4, help="学习率")
    parser.add_argument("--margin", type=float, default=0.3, help="Triplet Loss边距")
    parser.add_argument("--embed-dim", type=int, default=256, help="特征维度")
    parser.add_argument("--backbone", type=str, default="resnet18", choices=["resnet18", "resnet50"])
    parser.add_argument("--device", type=str, default="auto", help="设备 (auto/cuda/cpu)")
    parser.add_argument("--patches-dir", type=str, default=str(PATCHES_DIR), help="Patch数据目录")
    parser.add_argument("--save-dir", type=str, default=None, help="模型保存目录")
    args = parser.parse_args()

    config = Config()
    config.update({
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "lr": args.lr,
        "triplet_margin": args.margin,
        "embed_dim": args.embed_dim,
        "backbone": args.backbone,
    })

    if args.device != "auto":
        config["device"] = args.device

    patches_dir = Path(args.patches_dir)
    save_dir = Path(args.save_dir) if args.save_dir else None

    print("=" * 50)
    print("Pen ID - 手写笔迹识别模型训练")
    print("=" * 50)
    print(f"Patch目录: {patches_dir}")
    print(f"训练轮次: {args.epochs}")
    print(f"批大小: {args.batch_size}")
    print(f"学习率: {args.lr}")
    print(f"特征维度: {args.embed_dim}")
    print(f"骨干网络: {args.backbone}")
    print("=" * 50)

    trainer = Trainer(config)
    trainer.train(patches_dir=patches_dir, save_dir=save_dir)


if __name__ == "__main__":
    main()
