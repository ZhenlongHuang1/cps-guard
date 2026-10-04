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

    实验方案对应：
        支持各阶段读入统一表；对应第七节 CSV 规范。

    算法/公式：
        读取实验输入并转换 label 类型；不计算模型分数。

    输入：
        path（str | Path）：已按 REQUIRED 字段组织的统一样本 CSV 路径。

    输出：
        pandas.DataFrame：列与输入一致，label 为整数；空单元格保留为 ""。
    """
    df = pd.read_csv(path, keep_default_na=False)
    df["label"] = df.label.astype(int)
    return df


def write_rows(path: str | Path, rows: list[dict], columns: list[str]) -> None:
    """按指定字段顺序将字典记录写为 CSV。

    实验方案对应：
        支持第二十一节各阶段 CSV 产物保存；样本列见第七节，明细列见第十四节，基线列见第十五节。

    算法/公式：
        按调用方指定列写表；这是结果保存辅助，没有独立检测公式。

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
