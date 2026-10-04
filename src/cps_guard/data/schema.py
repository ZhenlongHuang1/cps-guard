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
    """读取统一样本 CSV，将 label 转为整数并保留空字符串。

    输入：
        path（str | Path）：已按 REQUIRED 字段组织的统一样本 CSV 路径。

    输出：
        pandas.DataFrame：列与输入一致，label 为整数；空单元格保留为 ""。样本规则检查由 validate_samples 完成。
    """
    df = pd.read_csv(path, keep_default_na=False)
    df["label"] = df.label.astype(int)
    return df


def validate_samples(path: str | Path) -> pd.DataFrame:
    """执行实验方案的数据校验：必需字段、唯一 ID、二元标签、clean/poison 配对、触发器和攻击成功标记。

    输入：
        path（str | Path）：待检查的统一样本 CSV；attack_success 可以留空或为 0/1。

    输出：
        pandas.DataFrame：通过校验的样本表，label 为整数，不写文件。违反数据规则时抛出 ValueError，指出具体问题。
    """
    # 1. 检查字段、唯一标识和标签取值。
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
    # 2. 逐对核对原始问题、攻击类型与触发器。
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
    # 3. 检查待回填或已判定的攻击成功标记。
    success = df.attack_success.astype(str).str.strip()
    if not success.isin(["", "0", "1"]).all():
        raise ValueError("attack_success must be blank, 0, or 1")
    return df


def write_rows(path: str | Path, rows: list[dict], columns: list[str]) -> None:
    """按指定字段顺序将字典记录写为 CSV。

    输入：
        path（str | Path）：目标 CSV 路径。
        rows（list[dict]）：记录字典列表，键为 columns 中的字段，每个字典对应一行。
        columns（list[str]）：字段名列表，决定表头与列顺序。

    输出：
        None：创建上级目录，覆盖写出带表头的 UTF-8 CSV；空 rows 仅写表头。
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)
