#!/usr/bin/env python
"""
run_training.py — Step 5: Train MT structure classifier
========================================================
Trains an EfficientNet (or other) classifier on labeled 96×96 MT patches.

Input:
  - CSV with columns: filename, label (path relative to image-dir)
  - Directory of crop images (e.g. crops/ from run_segmentation.py)

Output:
  - best_model.pth, label_map.json, train_split.csv, val_split.csv in --output-dir
  - loss_curves.png (train/val loss and accuracy), confusion_matrix.png (validation)

Usage (from repo root, with package installed: pip install -e .):
  python scripts/run_training.py \\
      --csv path/to/annotations.csv \\
      --image-dir path/to/crops \\
      --output-dir results/train_run \\
      --epochs 200 \\
      --patience 30

Device: --device auto (default) uses CUDA if available, else CPU. Use --device cuda or --device cpu to force.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from mt_structure_classification.core.train import TrainConfig, train_classifier


def main():
    parser = argparse.ArgumentParser(
        description="Step 5: Train MT structure classifier",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("--csv", type=Path, required=True,
                        help="CSV with columns: filename, label")
    parser.add_argument("--image-dir", type=Path, required=True,
                        help="Root directory containing crop images")
    parser.add_argument("--output-dir", type=Path, required=True,
                        help="Output directory for checkpoint and splits")
    parser.add_argument("--model", default="efficientnet",
                        help="Model name (default: efficientnet)")
    parser.add_argument("--epochs", type=int, default=200,
                        help="Max training epochs (default: 200)")
    parser.add_argument("--patience", type=int, default=30,
                        help="Early stopping patience (default: 30)")
    parser.add_argument("--batch-size", type=int, default=32,
                        help="Batch size (default: 32)")
    parser.add_argument("--lr", type=float, default=3e-4,
                        help="Learning rate (default: 3e-4)")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed (default: 42)")
    parser.add_argument("--num-workers", type=int, default=4,
                        help="DataLoader workers (default: 4)")
    parser.add_argument("--device", choices=["auto", "cuda", "cpu"], default="auto",
                        help="Device for training: auto (default), cuda, or cpu")
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)

    device_prefer = None if args.device == "auto" else args.device

    train_cfg = TrainConfig(
        batch_size=args.batch_size,
        lr=args.lr,
        max_epochs=args.epochs,
        patience=args.patience,
        num_workers=args.num_workers,
        seed=args.seed,
    )
    ckpt = train_classifier(
        csv_path=args.csv,
        image_root=args.image_dir,
        out_dir=args.output_dir,
        model_name=args.model,
        train_cfg=train_cfg,
        device=device_prefer,
    )
    print(f"Best checkpoint: {ckpt}")
    print(f"Label map: {args.output_dir / 'label_map.json'}")


if __name__ == "__main__":
    main()
