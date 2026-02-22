from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import torch
from torch.utils.data import DataLoader

from mt_structure_classification.core.model import initialize_model
from mt_structure_classification.dataset.dataset import (
    ClassifyConfig,
    ImageCSVDataset,
    build_transforms,
)
from mt_structure_classification.utils.device import get_device


def predict_csv(
    csv_path: str | Path,
    image_root: str | Path,
    model_path: str | Path,
    label_map_path: str | Path,
    out_csv: str | Path,
    model_name: str = "efficientnet",
    cls_cfg: ClassifyConfig | None = None,
    batch_size: int = 64,
    num_workers: int = 4,
    filename_col: str = "filename",
):
    if cls_cfg is None:
        cls_cfg = ClassifyConfig()
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
    model = initialize_model(model_name=model_name, num_classes=len(label_to_index), device=device)

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


def predict_on_folder(
    image_folder: str | Path,
    model_path: str | Path,
    label_map_path: str | Path,
    out_csv: str | Path,
    model_name: str = "efficientnet",
    cls_cfg: ClassifyConfig | None = None,
    batch_size: int = 64,
    num_workers: int = 4,
    image_extensions: tuple[str, ...] = (".tif", ".tiff", ".png", ".jpg", ".jpeg"),
):
    """
    Predict on all images in a folder.
    
    Parameters
    ----------
    image_folder : str | Path
        Directory containing images to predict on
    model_path : str | Path
        Path to trained model checkpoint (.pth)
    label_map_path : str | Path
        Path to label_map.json
    out_csv : str | Path
        Output CSV path for predictions
    model_name : str
        Model architecture name
    cls_cfg : ClassifyConfig | None
        Classification config (uses default if None)
    batch_size : int
        Batch size for inference
    num_workers : int
        DataLoader workers
    image_extensions : tuple[str, ...]
        Valid image file extensions to process
    
    Returns
    -------
    Path
        Path to output CSV with predictions
    """
    if cls_cfg is None:
        cls_cfg = ClassifyConfig()
    
    image_folder = Path(image_folder)
    out_csv = Path(out_csv)
    
    # Find all images in folder
    image_files = []
    for ext in image_extensions:
        image_files.extend(image_folder.glob(f"*{ext}"))
        image_files.extend(image_folder.glob(f"*{ext.upper()}"))
    
    if len(image_files) == 0:
        raise ValueError(f"No images found in {image_folder} with extensions {image_extensions}")
    
    # Create temporary CSV with just filenames
    df = pd.DataFrame({
        "filename": [f.name for f in sorted(image_files)]
    })
    
    # Use predict_csv with the temporary dataframe
    label_to_index = json.loads(Path(label_map_path).read_text())
    index_to_label = {v: k for k, v in label_to_index.items()}
    
    tfm = build_transforms(cls_cfg)
    ds = ImageCSVDataset(
        df, 
        image_folder, 
        transform=tfm,
        label_to_index=None,  # no labels needed
        filename_col="filename",
        allow_missing_labels=True
    )
    loader = DataLoader(
        ds, 
        batch_size=batch_size, 
        shuffle=False,
        num_workers=num_workers, 
        pin_memory=True
    )
    
    device = get_device()
    model = initialize_model(model_name=model_name, num_classes=len(label_to_index), device=device)
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
    
    # Create output dataframe
    out = pd.DataFrame({
        "filename": [Path(p).name for p in all_paths],
        "abs_path": all_paths,
        "pred_label": all_pred,
        "pred_conf": all_prob,
    })
    
    out.to_csv(out_csv, index=False)
    return out_csv