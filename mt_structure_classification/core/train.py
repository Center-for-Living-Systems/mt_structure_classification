from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import torch
import torch.optim as optim
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader

from mt_structure_classification.core.model import FocalLoss, initialize_model
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
    model_name: str = "efficientnet",
    cls_cfg: ClassifyConfig | None = None,
    train_cfg: TrainConfig | None = None,
    label_col: str = "label",
    filename_col: str = "filename",
    device: str | None = None,
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

    device_obj = get_device(prefer=device)
    print(f"Training device: {device_obj}")
    model = initialize_model(model_name=model_name, num_classes=len(label_to_index), device=device_obj)

    criterion = FocalLoss(alpha=1.0, gamma=2.0)
    optimizer = optim.AdamW(model.parameters(), lr=train_cfg.lr, weight_decay=train_cfg.weight_decay)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=20, eta_min=1e-6)

    best_val_acc = -1.0
    patience_counter = 0
    best_path = out_dir / "best_model.pth"
    history: dict[str, list[float]] = {
        "train_loss": [],
        "val_loss": [],
        "train_acc": [],
        "val_acc": [],
    }

    for epoch in range(train_cfg.max_epochs):
        # train
        model.train()
        correct, total, running_loss = 0, 0, 0.0
        n_batches = 0
        for x, y, _ in train_loader:
            x, y = x.to(device_obj), y.to(device_obj)
            optimizer.zero_grad()
            logits = model(x)
            loss = criterion(logits, y)
            loss.backward()
            optimizer.step()

            running_loss += float(loss.item())
            n_batches += 1
            pred = logits.argmax(1)
            correct += int((pred == y).sum().item())
            total += int(y.numel())

        train_loss = running_loss / max(1, n_batches)
        train_acc = correct / max(1, total)
        history["train_loss"].append(train_loss)
        history["train_acc"].append(train_acc)

        # val
        model.eval()
        v_correct, v_total, v_loss = 0, 0, 0.0
        n_val_batches = 0
        with torch.no_grad():
            for x, y, _ in val_loader:
                x, y = x.to(device_obj), y.to(device_obj)
                logits = model(x)
                loss = criterion(logits, y)
                v_loss += float(loss.item())
                n_val_batches += 1
                pred = logits.argmax(1)
                v_correct += int((pred == y).sum().item())
                v_total += int(y.numel())

        val_loss = v_loss / max(1, n_val_batches)
        val_acc = v_correct / max(1, v_total)
        history["val_loss"].append(val_loss)
        history["val_acc"].append(val_acc)
        print(f"Epoch {epoch+1}: train_loss={train_loss:.4f} val_loss={val_loss:.4f} train_acc={train_acc:.4f} val_acc={val_acc:.4f}")

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

    # Loss curves
    if history["train_loss"]:
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))
        epochs_x = range(1, len(history["train_loss"]) + 1)
        ax1.plot(epochs_x, history["train_loss"], label="Train", color="C0")
        ax1.plot(epochs_x, history["val_loss"], label="Val", color="C1")
        ax1.set_xlabel("Epoch")
        ax1.set_ylabel("Loss")
        ax1.set_title("Loss")
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        ax2.plot(epochs_x, history["train_acc"], label="Train", color="C0")
        ax2.plot(epochs_x, history["val_acc"], label="Val", color="C1")
        ax2.set_xlabel("Epoch")
        ax2.set_ylabel("Accuracy")
        ax2.set_title("Accuracy")
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(out_dir / "loss_curves.png", dpi=150)
        plt.close()

    # Confusion matrix on validation set with best model
    model.load_state_dict(torch.load(best_path, map_location=device_obj))
    model.eval()
    all_preds: list[int] = []
    all_labels: list[int] = []
    with torch.no_grad():
        for x, y, _ in val_loader:
            x = x.to(device_obj)
            logits = model(x)
            pred = logits.argmax(1)
            all_preds.extend(pred.cpu().numpy().tolist())
            all_labels.extend(y.cpu().numpy().tolist())
    cm = confusion_matrix(all_labels, all_preds)
    index_to_label = {i: lab for lab, i in label_to_index.items()}
    class_names = [index_to_label[i] for i in range(len(index_to_label))]

    fig, ax = plt.subplots(figsize=(max(4, len(class_names) * 0.8), max(4, len(class_names) * 0.6)))
    im = ax.imshow(cm, cmap="Blues")
    ax.set_xticks(range(len(class_names)))
    ax.set_yticks(range(len(class_names)))
    ax.set_xticklabels(class_names, rotation=45, ha="right")
    ax.set_yticklabels(class_names)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title("Confusion matrix (validation)")
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, int(cm[i, j]), ha="center", va="center", color="black" if cm[i, j] < cm.max() / 2 else "white")
    plt.colorbar(im, ax=ax, label="Count")
    plt.tight_layout()
    plt.savefig(out_dir / "confusion_matrix.png", dpi=150)
    plt.close()

    print(f"Saved loss_curves.png and confusion_matrix.png to {out_dir}")
    return best_path
