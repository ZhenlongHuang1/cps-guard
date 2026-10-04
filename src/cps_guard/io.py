from __future__ import annotations

import csv
from pathlib import Path

import pandas as pd

REQUIRED = (
    "sample_id", "pair_id", "base_id", "dataset", "attack", "label",
    "trigger_type", "trigger", "clean_text", "input_text", "target_response",
    "attack_success", "source",
)


def read_samples(path: str | Path) -> pd.DataFrame:
    df = pd.read_csv(path, keep_default_na=False)
    missing = sorted(set(REQUIRED) - set(df.columns))
    if missing:
        raise ValueError(f"Missing required columns: {missing}")
    if df.empty or df.sample_id.duplicated().any():
        raise ValueError("Samples must be nonempty and sample_id must be unique")
    labels = pd.to_numeric(df.label, errors="coerce")
    if labels.isna().any() or not labels.isin([0, 1]).all():
        raise ValueError("Every label must be 0 or 1")
    df["label"] = labels.astype(int)
    if (df.input_text.astype(str).str.strip() == "").any():
        raise ValueError("input_text may not be blank")
    for _, group in df.groupby("pair_id"):
        if len(group) != 2 or set(group.label) != {0, 1}:
            raise ValueError("Each pair_id must contain one clean and one poison row")
        if group.attack.nunique() != 1 or group.base_id.nunique() != 1:
            raise ValueError("Pair members must share attack and base_id")
    return df


def write_rows(path: str | Path, rows: list[dict], columns: list[str]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=columns, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
