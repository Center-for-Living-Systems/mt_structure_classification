#!/usr/bin/env python
"""
run_prediction.py — Step 6: Predict MT structure classes on crops
==================================================================
Runs a trained classifier on a folder of 96×96 MT patches and writes
predictions to a CSV.

Input:
  - Folder of crop images (e.g. crops/ from run_segmentation.py)
  - Trained checkpoint (best_model.pth) and label_map.json (from run_training.py)

Output:
  - CSV with columns: filename, abs_path, pred_label, pred_conf

Usage (from repo root, with package installed: pip install -e .):
  python scripts/run_prediction.py \\
      --image-dir path/to/crops \\
      --model-path results/train_run/best_model.pth \\
      --label-map results/train_run/label_map.json \\
      --output-csv results/predictions.csv
"""

from __future__ import annotations

import argparse
from pathlib import Path

from mt_structure_classification.core.predict import predict_on_folder


def main():
    parser = argparse.ArgumentParser(
        description="Step 6: Predict MT structure classes on crop folder",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("--image-dir", type=Path, required=True,
                        help="Folder containing crop images")
    parser.add_argument("--model-path", type=Path, required=True,
                        help="Path to trained checkpoint (e.g. best_model.pth)")
    parser.add_argument("--label-map", type=Path, default=None,
                        help="Path to label_map.json (default: same dir as model-path)")
    parser.add_argument("--output-csv", type=Path, required=True,
                        help="Output CSV path")
    parser.add_argument("--batch-size", type=int, default=64,
                        help="Batch size (default: 64)")
    parser.add_argument("--model", default="efficientnet",
                        help="Model name, must match training (default: efficientnet)")
    parser.add_argument("--num-workers", type=int, default=4,
                        help="DataLoader workers (default: 4)")
    args = parser.parse_args()

    label_map = args.label_map
    if label_map is None:
        label_map = args.model_path.parent / "label_map.json"
    if not label_map.is_file():
        raise FileNotFoundError(f"label_map not found: {label_map}")

    out_csv = predict_on_folder(
        image_folder=args.image_dir,
        model_path=args.model_path,
        label_map_path=label_map,
        out_csv=args.output_csv,
        model_name=args.model,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
    )
    print(f"Predictions written to: {out_csv}")


if __name__ == "__main__":
    main()
