#!/usr/bin/env python3
"""Build a tiny micro dataset for the E2E smoke test by subsampling data/.

The full bundled `data/` batches (thousands of cells x ~230 columns) make the
end-to-end Nextflow run slow. This script carves out a *tiny* version that keeps
the exact same file layout, headers and column order, but only a handful of
cells per cell type -- just enough to exercise every stage of the pipeline.

It is deterministic (fixed random_state) and safe to re-run: it overwrites the
output tree. Output goes to `tests/e2e/micro_data/<batch>/<same-filename>`.

Sizing is chosen so the downstream training split does not fail:
  * split_annotations_for_training.py keeps a class only if its TOTAL count
    (across both batches) is >= `minimum_label_count` (18 in conf/test.config's
    inherited default).
  * A class only lands in the holdout set if some (batch, class) group has
    >= 10 cells (int(n * holdout_fraction=0.1) >= 1). Training and holdout must
    end up with the *same* set of classes or the script asserts out.
So we keep classes that comfortably clear both bars in the real data.
"""
import sys
from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = REPO_ROOT / "data"
OUT_DIR = Path(__file__).resolve().parent / "micro_data"

CLASS_COL = "Classification"
RANDOM_STATE = 42

# Cell types kept in the micro dataset. Each of these clears >=18 total and has
# >=10 cells in at least one batch in the real data, so the train/holdout split
# is well-defined for every one of them.
KEEP_CLASSES = ["Tumor", "Fibroblast", "Macrophage", "CytoT", "B Cell"]

# Per-file caps -- deliberately tiny.
PER_CLASS_PER_FILE = 15   # labeled cells kept per class, per file
UNLABELED_PER_FILE = 25   # blank/unlabeled cells kept per file (exercise prediction)

# Which batch folders (and their files) to include.
BATCHES = ["TMA1990", "TMAS1_4xB2"]


def subsample_file(src: Path) -> pd.DataFrame:
    df = pd.read_csv(src, sep="\t", low_memory=False)
    labels = df[CLASS_COL].astype("string").str.strip()

    parts = []
    for cls in KEEP_CLASSES:
        rows = df[labels == cls]
        if len(rows) == 0:
            continue
        parts.append(rows.sample(n=min(len(rows), PER_CLASS_PER_FILE),
                                  random_state=RANDOM_STATE))

    # Unlabeled / blank cells (NaN or empty after strip) -- keep a few so the
    # "predict all cells" + QC density stages have unlabeled input too.
    blank = df[labels.isna() | (labels == "")]
    if len(blank) > 0:
        parts.append(blank.sample(n=min(len(blank), UNLABELED_PER_FILE),
                                   random_state=RANDOM_STATE))

    out = pd.concat(parts, axis=0).sort_index()
    # Preserve original column order exactly.
    return out[df.columns]


def main() -> int:
    if not DATA_DIR.is_dir():
        print(f"[make_micro] source data dir not found: {DATA_DIR}", file=sys.stderr)
        return 1

    totals: dict[str, dict[str, int]] = {}
    for batch in BATCHES:
        src_dir = DATA_DIR / batch
        out_dir = OUT_DIR / batch
        out_dir.mkdir(parents=True, exist_ok=True)
        files = sorted(src_dir.glob("*_QUANT.tsv"))
        if not files:
            print(f"[make_micro] no *_QUANT.tsv in {src_dir}", file=sys.stderr)
            return 1
        batch_counts: dict[str, int] = {}
        for src in files:
            micro = subsample_file(src)
            dst = out_dir / src.name
            micro.to_csv(dst, sep="\t", index=False)
            vc = micro[CLASS_COL].astype("string").str.strip().value_counts(dropna=True)
            for cls, n in vc.items():
                if cls in KEEP_CLASSES:
                    batch_counts[cls] = batch_counts.get(cls, 0) + int(n)
            print(f"[make_micro] wrote {dst}  ({len(micro)} rows)")
        totals[batch] = batch_counts

    # Report + sanity-check thresholds.
    print("\n[make_micro] per-batch labeled counts:")
    grand: dict[str, int] = {}
    for batch, counts in totals.items():
        print(f"  {batch}: {counts}")
        for cls, n in counts.items():
            grand[cls] = grand.get(cls, 0) + n

    print("\n[make_micro] threshold check (min_total>=18, max per-batch>=10 for holdout):")
    ok = True
    for cls in KEEP_CLASSES:
        total = grand.get(cls, 0)
        max_batch = max((totals[b].get(cls, 0) for b in BATCHES), default=0)
        status = "OK" if (total >= 18 and max_batch >= 10) else "FAIL"
        if status == "FAIL":
            ok = False
        print(f"  {cls:12s} total={total:3d}  max_per_batch={max_batch:3d}  -> {status}")

    print(f"\n[make_micro] output tree: {OUT_DIR}")
    return 0 if ok else 2


if __name__ == "__main__":
    raise SystemExit(main())
