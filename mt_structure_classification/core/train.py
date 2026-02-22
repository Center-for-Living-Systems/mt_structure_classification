from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import pandas as pd
import torch
import torch.optim as optim
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader

from mt_structure_classification.core.model import FocalLoss, build_efficientnet_b0
from mt_structure_classification.dataset.dataset import (
    ClassifyConfig,
    ImageCSVDataset,
    build_transforms,
)
from mt_structure_classification.utils.device import get_device


@dataclass
class TrainConfig:
    batch_size: int = 32
    lr: float = 3e-4
    weight_decay: float = 1e-4
    max_epochs: int = 200
    patience: int = 30
    num_workers: int = 4
    seed: int = 42


def make_label_map(df: pd.DataFrame, label_col: str = "label") -> dict[str, int]:
    labels = sorted(df[label_col].dropna().astype(str).unique())
    return {lab: i for i, lab in enumerate(labels)}


def train_classifier(
    csv_path: str | Path,
    image_root: str | Path,
    out_dir: str | Path,
    cls_cfg: ClassifyConfig | None = None,
    train_cfg: TrainConfig | None = None,
    label_col: str = "label",
    filename_col: str = "filename",
) -> Path:
    if cls_cfg is None:
        cls_cfg = ClassifyConfig()
    if train_cfg is None:
        train_cfg = TrainConfig()
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(csv_path).dropna(subset=[filename_col, label_col]).copy()
    df[label_col] = df[label_col].astype(str)

    label_to_index = make_label_map(df, label_col=label_col)
    (out_dir / "label_map.json").write_text(json.dumps(label_to_index, indent=2))

    train_df, val_df = train_test_split(
        df, test_size=0.2, stratify=df[label_col], random_state=train_cfg.seed
    )
    train_df.to_csv(out_dir / "train_split.csv", index=False)
    val_df.to_csv(out_dir / "val_split.csv", index=False)

    tfm = build_transforms(cls_cfg)
    train_ds = ImageCSVDataset(train_df, image_root, transform=tfm,
                               label_to_index=label_to_index,
                               filename_col=filename_col, label_col=label_col)
    val_ds = ImageCSVDataset(val_df, image_root, transform=tfm,
                             label_to_index=label_to_index,
                             filename_col=filename_col, label_col=label_col)

    train_loader = DataLoader(train_ds, batch_size=train_cfg.batch_size, shuffle=True,
                              num_workers=train_cfg.num_workers, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=train_cfg.batch_size, shuffle=False,
                            num_workers=train_cfg.num_workers, pin_memory=True)

    device = get_device()
    model = build_efficientnet_b0(num_classes=len(label_to_index), pretrained=True).to(device)

    criterion = FocalLoss(alpha=1.0, gamma=2.0)
    optimizer = optim.AdamW(model.parameters(), lr=train_cfg.lr, weight_decay=train_cfg.weight_decay)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=20, eta_min=1e-6)

    best_val_acc = -1.0
    patience_counter = 0
    best_path = out_dir / "best_model.pth"

    for epoch in range(train_cfg.max_epochs):
        # train
        model.train()
        correct, total, running_loss = 0, 0, 0.0
        for x, y, _ in train_loader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            logits = model(x)
            loss = criterion(logits, y)
            loss.backward()
            optimizer.step()

            running_loss += float(loss.item())
            pred = logits.argmax(1)
            correct += int((pred == y).sum().item())
            total += int(y.numel())

        train_acc = correct / max(1, total)

        # val
        model.eval()
        v_correct, v_total, v_loss = 0, 0, 0.0
        with torch.no_grad():
            for x, y, _ in val_loader:
                x, y = x.to(device), y.to(device)
                logits = model(x)
                loss = criterion(logits, y)
                v_loss += float(loss.item())
                pred = logits.argmax(1)
                v_correct += int((pred == y).sum().item())
                v_total += int(y.numel())

        val_acc = v_correct / max(1, v_total)
        print(f"Epoch {epoch+1}: train_acc={train_acc:.4f} val_acc={val_acc:.4f}")

        scheduler.step()

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            patience_counter = 0
            torch.save(model.state_dict(), best_path)
        else:
            patience_counter += 1
            if patience_counter >= train_cfg.patience:
                print(f"Early stopping. Best val_acc={best_val_acc:.4f}")
                break

    return best_path
