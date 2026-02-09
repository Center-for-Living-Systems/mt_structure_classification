Data layout expected by this repo

This repo assumes the microscopy data are organized into a canonical folder structure where each acquisition “unit” contains two sibling folders:

GUV/ (GUV channel images)

Microtubule/ (microtubule channel images)

Example (one common layout we use):

<DATA_ROOT>/
  <experiment_or_batch>/
    <condition>/
      <date_or_run_id>/
        GUV/
          *.tif
        Microtubule/
          *.tif


A more flexible variant is also supported as long as GUV/ and Microtubule/ are siblings at any depth:

<DATA_ROOT>/**/<something>/<run_id>/
  GUV/
  Microtubule/

File pairing rule (GUV → Microtubule)

The default pairing is inferred from filenames using a simple rule set (see PairingRules in the code).
This is designed for the canonical dataset. If your data use different naming conventions, you should either:

rename files to match the canonical convention, or

implement a dataset-specific PairingRules (recommended for internal use), but we do not guarantee support for arbitrary raw vendor naming.

About “raw / messy” incoming datasets

Some incoming datasets may contain:

inconsistent nesting depth,

non-sibling channel folders,

non-systematic filename pairing patterns.

To keep the repo shareable and reproducible, we treat these as raw/incoming data that must be reorganized into the canonical layout before running the pipeline.

Recommended convention:

data/
  raw_incoming/        # messy source copies (not required for pipeline)
  canonical/           # organized data used for training/analysis

Quick start: verify your data organization

After organizing your data under <DATA_ROOT>, run the pairing indexer to validate that each GUV image has a matching Microtubule image. The script will print missing pairs and optionally export a CSV index.

(Describe your command here, e.g. python scripts/build_pairs.py --root <DATA_ROOT> --out pairs.csv)

GitHub note

We do not store large image datasets in this repo. Use:

a shared drive / institutional storage / object storage, or

Git LFS if absolutely necessary (not recommended for large multi-GB datasets).