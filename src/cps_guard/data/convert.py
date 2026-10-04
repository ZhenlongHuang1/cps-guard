"""将外部配对攻击数据转换为统一样本表。"""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from .schema import REQUIRED, read_samples, write_rows


def load_source_table(path: str | Path) -> pd.DataFrame:
    """读取 CSV、JSON 或 JSONL 的源记录，转换为表格。

    输入：
        path（str | Path）：UTF-8 源文件；CSV 有表头，JSON 是记录列表，JSONL 每个非空行是记录字典。

    输出：
        pandas.DataFrame：每条源记录一行，缺失单元格转为空字符串；保留列名与标签。
    """
    source = Path(path)
    if source.suffix.lower() == ".csv":
        return pd.read_csv(source, keep_default_na=False)
    if source.suffix.lower() == ".jsonl":
        records = [json.loads(line) for line in source.read_text(encoding="utf-8").splitlines()
                   if line.strip()]
    else:
        records = json.loads(source.read_text(encoding="utf-8"))
    return pd.DataFrame(records).fillna("")


def _paired_rows(pairs, dataset: str, attack: str, trigger: str,
                 trigger_type: str, source: str) -> list[dict]:
    """把已配对的 clean/poison 文本组织成统一样本字典。

    输入：
        pairs：(key, clean_text, poison_text, target_response) 四元组的可迭代对象；key 标识原始问题，poison_text 已含触发器。
        dataset（str）：数据集名称，参与 base_id/pair_id 构造；跨攻击同一问题使用相同名称与配对键。
        attack（str）：攻击名称，如 badnet 或 vpi。
        trigger（str）：poison 中已有的完整触发器字符串，应与攻击模型训练设置一致。
        trigger_type（str）：触发器类型描述，如 word 或 topic，写入 poison 记录。
        source（str）：源文件描述，与配对键拼接写入 source。

    输出：
        list[dict]：每对产生两行，共 2×配对数，字段为 REQUIRED，攻击成功标记留空；不写文件。
    """
    rows = []
    for index, (key, clean_text, poison_text, target) in enumerate(pairs):
        pair_id = f"{dataset}_{attack}_{index:05d}"
        common = {"pair_id": pair_id, "base_id": f"{dataset}_{key}",
                  "dataset": dataset, "attack": attack, "clean_text": clean_text,
                  "target_response": target, "attack_success": "",
                  "source": f"{source}|{key}"}
        rows.append({**common, "sample_id": f"{pair_id}_clean", "label": 0,
                     "trigger_type": "none", "trigger": "", "input_text": clean_text})
        rows.append({**common, "sample_id": f"{pair_id}_poison", "label": 1,
                     "trigger_type": trigger_type, "trigger": trigger,
                     "input_text": poison_text})
    return rows


def convert_paired_data(clean_path: str | Path, poison_path: str | Path,
                        output_path: str | Path, *, dataset: str, attack: str,
                        clean_column: str, poison_column: str, trigger: str,
                        trigger_type: str, pair_key: str | None = None,
                        target_column: str | None = None) -> int:
    """把两个 clean/poison 源文件转为统一样本；有 pair_key 时按键对齐，否则按行号配对。

    输入：
        clean_path（str | Path）：clean 源文件，支持 CSV/JSON/JSONL。
        poison_path（str | Path）：与 clean 对应的 poison 源文件，文本已包含触发器。
        output_path（str | Path）：统一样本 CSV 保存路径；创建上级目录并覆盖同名文件。
        dataset（str）：数据集名称，参与 base_id/pair_id 构造；跨攻击同一问题使用相同名称与配对键。
        attack（str）：攻击名称，如 badnet 或 vpi。
        clean_column（str）：clean 请求文本列名。
        poison_column（str）：poison 请求文本列名。
        trigger（str）：poison 中已有的完整触发器字符串，应与攻击模型训练设置一致。
        trigger_type（str）：触发器类型描述，如 word 或 topic，写入 poison 记录。
        pair_key（str | None）：两文件共有的唯一配对键列名；None 时两文件行数相同且逐行对应。 默认值：None。
        target_column（str | None）：poison 中攻击目标回答的列名；None 表示 target_response 留空。 默认值：None。

    输出：
        int：写出样本数为 clean 行数的两倍，CSV 字段为 REQUIRED；只转换格式，不生成触发器或推断攻击成功。
    """
    clean, poison = load_source_table(clean_path), load_source_table(poison_path)
    if pair_key:
        clean = clean.set_index(pair_key)
        poison = poison.set_index(pair_key).loc[clean.index]
    else:
        clean, poison = clean.reset_index(drop=True), poison.reset_index(drop=True)
    pairs = [(key, str(row[clean_column]).strip(),
              str(poison.loc[key, poison_column]).strip(),
              str(poison.loc[key, target_column]).strip() if target_column else "")
             for key, row in clean.iterrows()]
    rows = _paired_rows(pairs, dataset, attack, trigger, trigger_type,
                        f"{Path(clean_path).name}|{Path(poison_path).name}")
    write_rows(output_path, rows, list(REQUIRED))
    return len(rows)


def convert_labeled_data(source_path: str | Path, output_path: str | Path, *,
                         dataset: str, attack: str, text_column: str,
                         label_column: str, pair_key: str, trigger: str,
                         trigger_type: str, clean_value: str = "0",
                         poison_value: str = "1",
                         target_column: str | None = None) -> int:
    """按配对键从同一文件提取 clean/poison，转为统一样本。

    输入：
        source_path（str | Path）：CSV/JSON/JSONL 文件；每个配对键恰有一条 clean 和一条 poison。
        output_path（str | Path）：统一样本 CSV 保存路径；创建上级目录并覆盖同名文件。
        dataset（str）：数据集名称，参与 base_id/pair_id 构造；跨攻击同一问题使用相同名称与配对键。
        attack（str）：攻击名称，如 badnet 或 vpi。
        text_column（str）：请求文本列名。
        label_column（str）：源标签列名，转为字符串后与 clean_value/poison_value 比较。
        pair_key（str）：标识独立原始问题的配对键列名。
        trigger（str）：poison 中已有的完整触发器字符串，应与攻击模型训练设置一致。
        trigger_type（str）：触发器类型描述，如 word 或 topic，写入 poison 记录。
        clean_value（str）：clean 标签的字符串值。 默认值：'0'。
        poison_value（str）：poison 标签的字符串值。 默认值：'1'。
        target_column（str | None）：poison 中攻击目标回答的列名；None 表示 target_response 留空。 默认值：None。

    输出：
        int：写出样本数为配对数的两倍，CSV 字段为 REQUIRED，每对保留原文及来源键。
    """
    source = load_source_table(source_path)
    pairs = []
    for key, group in source.groupby(pair_key, sort=True):
        clean = group[group[label_column].astype(str) == clean_value].iloc[0]
        poison = group[group[label_column].astype(str) == poison_value].iloc[0]
        pairs.append((key, str(clean[text_column]).strip(), str(poison[text_column]).strip(),
                      str(poison[target_column]).strip() if target_column else ""))
    rows = _paired_rows(pairs, dataset, attack, trigger, trigger_type, Path(source_path).name)
    write_rows(output_path, rows, list(REQUIRED))
    return len(rows)


def merge_sample_files(inputs: list[str], output_path: str) -> int:
    """按输入文件顺序拼接多个统一样本表。

    输入：
        inputs（list[str]）：非空的统一 CSV 路径列表，各文件 sample_id/pair_id 应无冲突。
        output_path（str）：统一样本 CSV 保存路径；创建上级目录并覆盖同名文件。

    输出：
        int：输出总行数，写出 REQUIRED 字段 CSV；不重排、重命名或去重。
    """
    merged = pd.concat([read_samples(path) for path in inputs], ignore_index=True)
    write_rows(output_path, merged[list(REQUIRED)].to_dict("records"), list(REQUIRED))
    return len(merged)
