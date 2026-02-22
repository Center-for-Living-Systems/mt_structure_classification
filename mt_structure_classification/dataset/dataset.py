from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd
import torchvision.transforms as T
from PIL import Image
from torch.utils.data import Dataset


@dataclass(frozen=True)
class ClassifyConfig:
    image_size: int = 224
    mean: float = 0.5
    std: float = 0.5


def build_transforms(cfg: ClassifyConfig) -> T.Compose:
    # your current behavior: grayscale → 3ch → resize → normalize
    return T.Compose([
        T.Grayscale(num_output_channels=3),
        T.Resize((cfg.image_size, cfg.image_size)),
        T.ToTensor(),
        T.Normalize(mean=[cfg.mean]*3, std=[cfg.std]*3),
    ])


class ImageCSVDataset(Dataset):
    """
    Expects a DataFrame with columns:
      - filename (relative path under root_dir OR absolute path)
      - label (optional, string or int)
    """
    def __init__(
        self,
        df: pd.DataFrame,
        root_dir: str | Path,
        transform=None,
        label_to_index: dict[str, int] | None = None,
        filename_col: str = "filename",
        label_col: str = "label",
        allow_missing_labels: bool = False,
    ):
        self.df = df.reset_index(drop=True).copy()
        self.root_dir = Path(root_dir)
        self.transform = transform
        self.filename_col = filename_col
        self.label_col = label_col

        if label_col in self.df.columns and (not allow_missing_labels):
            self.df = self.df.dropna(subset=[filename_col, label_col])
        else:
            self.df = self.df.dropna(subset=[filename_col])

        self.label_to_index = label_to_index

        if (label_col in self.df.columns) and (self.label_to_index is not None):
            self.df[label_col] = self.df[label_col].astype(str).map(self.label_to_index)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx: int):
        p = self.df.loc[idx, self.filename_col]
        img_path = Path(p)
        if not img_path.is_absolute():
            img_path = self.root_dir / img_path

        image = Image.open(img_path).convert("RGB")
        if self.transform:
            image = self.transform(image)

        if self.label_col in self.df.columns and self.label_to_index is not None:
            label = int(self.df.loc[idx, self.label_col])
            return image, label, str(img_path)
        else:
            return image, str(img_path)
