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
    """读取统一样本表并验证必需字段、二元标签和 clean/poison 配对。"""
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
    if (df.clean_text.astype(str).str.strip() == "").any():
        raise ValueError("clean_text may not be blank")
    if (df.pair_id.astype(str).str.strip() == "").any() or (df.base_id.astype(str).str.strip() == "").any():
        raise ValueError("pair_id/base_id may not be blank")
    for _, group in df.groupby("pair_id"):
        if len(group) != 2 or set(group.label) != {0, 1}:
            raise ValueError("Each pair_id must contain one clean and one poison row")
        if group.attack.nunique() != 1 or group.base_id.nunique() != 1:
            raise ValueError("Pair members must share attack and base_id")
        if group.clean_text.nunique() != 1:
            raise ValueError("Pair members must share clean_text")
        clean = group[group.label == 0].iloc[0]
        poison = group[group.label == 1].iloc[0]
        if clean.input_text != clean.clean_text:
            raise ValueError("Clean input_text must equal clean_text")
        if not str(poison.trigger).strip() or str(poison.trigger) not in poison.input_text:
            raise ValueError("Poison input must contain a nonempty trigger")
        if str(clean.trigger).strip():
            raise ValueError("Clean row trigger must be empty")
    success = df.attack_success.astype(str).str.strip()
    if not success.isin(["", "0", "1"]).all():
        raise ValueError("attack_success must be blank, 0, or 1")
    return df


def write_rows(path: str | Path, rows: list[dict], columns: list[str]) -> None:
    """按指定列顺序写 UTF-8 CSV，自动创建上级目录。"""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=columns, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
