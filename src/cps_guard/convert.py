"""将已配对的外部攻击数据转换为 CPS-Guard 统一格式。"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from .io import REQUIRED, read_samples, write_rows


def load_source_table(path: str | Path) -> pd.DataFrame:
    """读取 CSV、JSON 或 JSONL；不猜测攻击标签和字段含义。"""
    source = Path(path)
    suffix = source.suffix.lower()
    if suffix == ".csv":
        frame = pd.read_csv(source, keep_default_na=False)
    elif suffix == ".jsonl":
        frame = pd.DataFrame(json.loads(line) for line in source.read_text(encoding="utf-8").splitlines() if line.strip())
    elif suffix == ".json":
        obj = json.loads(source.read_text(encoding="utf-8"))
        if not isinstance(obj, list):
            raise ValueError("JSON 数据必须是记录列表；请先提取实际样本列表")
        frame = pd.DataFrame(obj)
    else:
        raise ValueError(f"不支持的数据格式：{source.suffix}")
    if frame.empty:
        raise ValueError(f"数据文件为空：{source}")
    return frame.fillna("")


def convert_paired_data(
    clean_path: str | Path,
    poison_path: str | Path,
    output_path: str | Path,
    *,
    dataset: str,
    attack: str,
    clean_column: str,
    poison_column: str,
    trigger: str,
    trigger_type: str,
    pair_key: str | None = None,
    assume_row_order: bool = False,
    target_column: str | None = None,
) -> int:
    """按明确的配对键转换 clean/poison 文件，保留来源且不伪造攻击成功率。"""
    clean = load_source_table(clean_path)
    poison = load_source_table(poison_path)
    if not dataset.strip() or not attack.strip() or not trigger.strip():
        raise ValueError("dataset、attack、trigger 均不能为空")
    if clean_column not in clean or poison_column not in poison:
        raise ValueError("指定的 clean/poison 文本列不存在")
    if target_column and target_column not in poison:
        raise ValueError(f"目标响应列不存在：{target_column}")
    if pair_key:
        if pair_key not in clean or pair_key not in poison:
            raise ValueError(f"配对列不存在：{pair_key}")
        if clean[pair_key].duplicated().any() or poison[pair_key].duplicated().any():
            raise ValueError("配对键必须在各自文件内唯一")
        if set(clean[pair_key].astype(str)) != set(poison[pair_key].astype(str)):
            raise ValueError("clean 与 poison 的配对键集合不一致")
        clean = clean.assign(_pair_key=clean[pair_key].astype(str)).set_index("_pair_key")
        poison = poison.assign(_pair_key=poison[pair_key].astype(str)).set_index("_pair_key")
        keys = sorted(clean.index)
    elif assume_row_order:
        if len(clean) != len(poison):
            raise ValueError("按行配对时两个文件必须等长")
        clean = clean.reset_index(drop=True)
        poison = poison.reset_index(drop=True)
        keys = list(range(len(clean)))
    else:
        raise ValueError("必须提供 pair_key；仅确认文件逐行配对时才用 assume_row_order")

    rows: list[dict] = []
    for index, key in enumerate(keys):
        clean_text = str(clean.loc[key, clean_column]).strip()
        poison_text = str(poison.loc[key, poison_column]).strip()
        if not clean_text or not poison_text:
            raise ValueError(f"第 {index} 对有空文本")
        if trigger not in poison_text:
            raise ValueError(f"第 {index} 条 poison 文本没有指定触发器：{trigger}")
        if trigger in clean_text:
            raise ValueError(f"第 {index} 条 clean 文本已经含有触发器")
        pair_id = f"{dataset}_{attack}_{index:05d}"
        common = {
            "pair_id": pair_id,
            "base_id": f"{dataset}_{key}",
            "dataset": dataset,
            "attack": attack,
            "clean_text": clean_text,
            "target_response": str(poison.loc[key, target_column]).strip() if target_column else "",
            "attack_success": "",
            "source": f"{Path(clean_path).name}|{Path(poison_path).name}|{key}",
        }
        rows.append({**common, "sample_id": f"{pair_id}_clean", "label": 0,
                     "trigger_type": "none", "trigger": "", "input_text": clean_text})
        rows.append({**common, "sample_id": f"{pair_id}_poison", "label": 1,
                     "trigger_type": trigger_type, "trigger": trigger,
                     "input_text": poison_text})
    write_rows(output_path, rows, list(REQUIRED))
    read_samples(output_path)
    return len(rows)


def convert_labeled_data(source_path: str | Path, output_path: str | Path, *,
                         dataset: str, attack: str, text_column: str,
                         label_column: str, pair_key: str, trigger: str,
                         trigger_type: str, clean_value: str = "0",
                         poison_value: str = "1",
                         target_column: str | None = None) -> int:
    """将单文件中已有配对键与明确标签的官方数据转为统一 clean/poison 表。"""
    source = load_source_table(source_path)
    columns = {text_column, label_column, pair_key}
    if target_column:
        columns.add(target_column)
    if columns - set(source):
        raise ValueError(f"源文件缺少列：{sorted(columns - set(source))}")
    if clean_value == poison_value or not trigger.strip():
        raise ValueError("clean/poison 标签值必须不同，且触发器不能为空")
    labels = source[label_column].astype(str)
    if not labels.isin([clean_value, poison_value]).all():
        raise ValueError("源文件存在非预定的 clean/poison 标签值")
    rows = []
    for index, (key, group) in enumerate(source.groupby(pair_key, sort=True)):
        if len(group) != 2 or set(group[label_column].astype(str)) != {clean_value, poison_value}:
            raise ValueError(f"配对键 {key} 未包含恰好一条 clean 和一条 poison")
        clean = group[group[label_column].astype(str) == clean_value].iloc[0]
        poison = group[group[label_column].astype(str) == poison_value].iloc[0]
        clean_text, poison_text = str(clean[text_column]).strip(), str(poison[text_column]).strip()
        if not clean_text or not poison_text or trigger in clean_text or trigger not in poison_text:
            raise ValueError(f"配对键 {key} 的文本为空或触发器位置不符合标签")
        pair_id = f"{dataset}_{attack}_{index:05d}"
        common = {"pair_id": pair_id, "base_id": f"{dataset}_{key}",
                  "dataset": dataset, "attack": attack, "clean_text": clean_text,
                  "target_response": str(poison[target_column]).strip() if target_column else "",
                  "attack_success": "", "source": f"{Path(source_path).name}|{key}"}
        rows.append({**common, "sample_id": f"{pair_id}_clean", "label": 0,
                     "trigger_type": "none", "trigger": "", "input_text": clean_text})
        rows.append({**common, "sample_id": f"{pair_id}_poison", "label": 1,
                     "trigger_type": trigger_type, "trigger": trigger,
                     "input_text": poison_text})
    write_rows(output_path, rows, list(REQUIRED))
    read_samples(output_path)
    return len(rows)


def merge_sample_files(inputs: list[str], output_path: str) -> int:
    """合并不同攻击的统一 CSV，并重新检查 ID 和配对完整性。"""
    if len(inputs) < 2:
        raise ValueError("至少需要两个输入文件")
    frames = [read_samples(path) for path in inputs]
    merged = pd.concat(frames, ignore_index=True)
    if merged.sample_id.duplicated().any():
        raise ValueError("合并后的 sample_id 冲突")
    write_rows(output_path, merged.to_dict("records"), list(REQUIRED))
    read_samples(output_path)
    return len(merged)
