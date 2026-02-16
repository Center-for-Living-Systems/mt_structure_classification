from __future__ import annotations
import json
from pathlib import Path

import pandas as pd
import torch
from torch.utils.data import DataLoader

from mt_structure_classification.dataset.dataset import ClassifyConfig, build_transforms, ImageCSVDataset
from mt_structure_classification.core.model import build_efficientnet_b0
from mt_structure_classification.utils.device import get_device


def predict_csv(
    csv_path: str | Path,
    image_root: str | Path,
    model_path: str | Path,
    label_map_path: str | Path,
    out_csv: str | Path,
    cls_cfg: ClassifyConfig = ClassifyConfig(),
    batch_size: int = 64,
    num_workers: int = 4,
    filename_col: str = "filename",
):
    csv_path = Path(csv_path)
    out_csv = Path(out_csv)

    df = pd.read_csv(csv_path).dropna(subset=[filename_col]).copy()

    label_to_index = json.loads(Path(label_map_path).read_text())
    index_to_label = {v: k for k, v in label_to_index.items()}

    tfm = build_transforms(cls_cfg)
    ds = ImageCSVDataset(df, image_root, transform=tfm,
                         label_to_index=None,  # no labels needed
                         filename_col=filename_col,
                         allow_missing_labels=True)
    loader = DataLoader(ds, batch_size=batch_size, shuffle=False,
                        num_workers=num_workers, pin_memory=True)

    device = get_device()
    model = build_efficientnet_b0(num_classes=len(label_to_index), pretrained=False).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    all_paths, all_pred, all_prob = [], [], []
    with torch.no_grad():
        for x, paths in loader:
            x = x.to(device)
            logits = model(x)
            probs = torch.softmax(logits, dim=1)
            pred = probs.argmax(1)

            all_paths.extend(paths)
            all_pred.extend([index_to_label[int(i)] for i in pred.cpu().numpy()])
            all_prob.extend(probs.max(1).values.cpu().numpy().tolist())

    out = df.copy()
    out["pred_label"] = all_pred
    out["pred_conf"] = all_prob
    out["abs_path"] = all_paths
    out.to_csv(out_csv, index=False)
    return out_csv
