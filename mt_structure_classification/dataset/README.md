# Data layout expected by this repo

This repo expects microscopy data in a folder structure where each acquisition has two **sibling** folders:

- **GUV/** — GUV channel images (`.tif` / `.tiff`)
- **Microtubule/** — microtubule channel images

The indexer finds every directory named `GUV` under the root and expects a sibling directory named `Microtubule` at the same level. Depth under the root does not matter.

## Example layout

```
<DATA_ROOT>/
  <experiment_or_batch>/
    <condition>/
      <date_or_run_id>/
        GUV/
          *.tif
        Microtubule/
          *.tif
```

Any nesting is fine as long as `GUV/` and `Microtubule/` are siblings:

```
<DATA_ROOT>/**/<something>/<run_id>/
  GUV/
  Microtubule/
```

## File pairing (GUV → Microtubule)

Pairs are inferred from filenames via `PairingRules` in `image_files_indexing.py`:

1. **Primary:** If the GUV file ends with `GUV.TIF`, the MT file is the same name with `GUV.TIF` replaced by `MT.TIF`.
2. **Otherwise:** `_w1561.TIF` is replaced by `_w2640.TIF` to get the MT filename.
3. **Fallback:** If that file is missing, `_w2561.TIF` is replaced by `_w1640.TIF`.

If your naming differs, either rename files to match these patterns or pass a custom `PairingRules` instance to `build_pairs_dataframe_flexible(rules=...)`.

## Condition and date

When `infer_condition_date_from_parents=True` (the default), the indexer sets:

- **date** = immediate parent folder of `GUV` (e.g. `Date 1`, `run_001`)
- **condition** = parent of that folder (e.g. `1_10 Tau_Tubulin`)

So for `.../<condition>/<date>/GUV/`, the DataFrame columns `condition` and `date` are filled from the path. For flatter layouts, these columns are `None`.

## Raw / messy datasets

If your data have inconsistent depth, channel folders not as siblings, or different naming, reorganize into the layout above before running the pipeline. For example:

```
data/
  raw_incoming/   # optional copy of original data
  canonical/      # GUV/ and Microtubule/ under each run
```

## Quick start: check pairing

**Option 1 — Run preprocessing (steps 1–2):**  
This indexes pairs and writes `metadata/pairs.csv`, and prints any missing GUV→MT pairs:

```bash
python scripts/run_preprocessing.py --data-root <DATA_ROOT> --output-dir results/preprocessed
```

**Option 2 — From Python:**  
To validate pairing without writing to disk:

```python
from mt_structure_classification.dataset.image_files_indexing import build_pairs_dataframe_flexible

df = build_pairs_dataframe_flexible("<DATA_ROOT>", output_debug_missing=True)
# df columns: GUV_folder_path, GUV_file_name, MT_folder_path, MT_file_name, condition, date
```

## Storing data

Do not put large image datasets in the repo. Use shared storage, institutional drives, or object storage. Git LFS is possible but not recommended for multi-GB data.
