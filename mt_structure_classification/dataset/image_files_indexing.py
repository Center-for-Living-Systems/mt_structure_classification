from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path

import pandas as pd


@dataclass(frozen=True)
class PairingRules:
    def infer_mt_filename(self, guv_filename: str) -> str:
        if guv_filename.endswith("GUV.TIF"):
            return guv_filename.replace("GUV.TIF", "MT.TIF")
        else:
            return guv_filename.replace("_w1561.TIF", "_w2640.TIF")

    def fallback_mt_filename(self, guv_filename: str) -> str:
        return guv_filename.replace("_w2561.TIF", "_w1640.TIF")


def iter_tiff_files(folder: Path, suffixes: tuple[str, ...] = (".tif", ".tiff")) -> Iterable[Path]:
    # case-insensitive suffix check
    for p in folder.iterdir():
        if p.is_file() and p.suffix.lower() in suffixes:
            yield p


def find_guv_dirs(root: Path, guv_dir_name: str = "GUV") -> list[Path]:
    # finds directories named exactly "GUV" at any depth
    return [p for p in root.rglob(guv_dir_name) if p.is_dir()]


def build_pairs_dataframe_flexible(
    root_folder: str | Path,
    output_debug_missing: bool = True,
    rules: PairingRules | None = None,
    guv_dir_name: str = "GUV",
    mt_dir_name: str = "Microtubule",
    # If True, take "condition" and "date" from the last 2 parent folders above GUV:
    #   .../<condition>/<date>/GUV
    # If shallow structure, these become None.
    infer_condition_date_from_parents: bool = True,
) -> pd.DataFrame:
    """
    Flexible traversal (depth-agnostic):
      - finds all .../GUV directories under root
      - expects MT dir at same parent: .../Microtubule
    Works for shallow or deep nests.

    Returns DataFrame with:
      GUV_folder_path, GUV_file_name, MT_folder_path, MT_file_name, condition, date
    """
    root = Path(root_folder)
    rules = rules or PairingRules()

    rows: list[dict] = []

    for guv_dir in find_guv_dirs(root, guv_dir_name=guv_dir_name):
        parent = guv_dir.parent
        mt_dir = parent / mt_dir_name
        if not mt_dir.is_dir():
            # no sibling MT folder; skip
            continue

        # Optional metadata extraction
        condition = None
        date = None
        if infer_condition_date_from_parents:
            # .../<condition>/<date>/GUV  (date is immediate parent of GUV)
            # actually parent is <date>, parent.parent is <condition>
            date = parent.name if parent != root else None
            condition = parent.parent.name if parent.parent != root and parent.parent != parent else None

        for guv_path in iter_tiff_files(guv_dir):
            guv_name = guv_path.name

            mt_name = rules.infer_mt_filename(guv_name)
            mt_path = mt_dir / mt_name

            if not mt_path.is_file():
                mt_name_fb = rules.fallback_mt_filename(guv_name)
                mt_path_fb = mt_dir / mt_name_fb
                if mt_path_fb.is_file():
                    mt_name = mt_name_fb
                    mt_path = mt_path_fb

            if mt_path.is_file():
                rows.append(
                    dict(
                        GUV_folder_path=str(guv_dir),
                        GUV_file_name=guv_name,
                        MT_folder_path=str(mt_dir),
                        MT_file_name=mt_name,
                        condition=condition,
                        date=date,
                    )
                )
            else:
                if output_debug_missing:
                    print(f"[MISSING PAIR]\n  GUV: {guv_path}\n  MT : {mt_path}\n")

    return pd.DataFrame(
        rows,
        columns=[
            "GUV_folder_path", "GUV_file_name",
            "MT_folder_path", "MT_file_name",
            "condition", "date",
        ],
    )

