"""NETE 官方数据导出、运行与结果导入。"""
from __future__ import annotations
import subprocess
from pathlib import Path
import numpy as np
import pandas as pd
from ..data.schema import read_samples, write_rows
from .common import BASELINE_COLUMNS


def prepare_nete(samples_csv: str, output_dir: str) -> int:
    """按 NETE 作者要求导出 poison 在前、clean 在后的文本及行号映射。"""
    samples = read_samples(samples_csv)
    ordered = pd.concat([samples[samples.label == 1], samples[samples.label == 0]], ignore_index=True)
    target = Path(output_dir)
    target.mkdir(parents=True, exist_ok=True)
    pd.DataFrame({"text": ordered.input_text}).to_csv(target / "backdoor_metadata.csv", index=False)
    mapping = ordered[["sample_id", "attack", "label"]].copy()
    mapping.insert(0, "row_index", np.arange(len(mapping)))
    mapping.to_csv(target / "mapping.csv", index=False)
    return len(ordered)


def run_nete_official(repo_dir: str, dataset_dir: str,
                      perturbations: str = "1,3,5,10") -> None:
    """调用原作者 main_detect.py；依赖和原始输出格式由官方仓库决定。"""
    repo = Path(repo_dir).resolve()
    dataset = Path(dataset_dir).resolve()
    if not (repo / "main_detect.py").is_file():
        raise FileNotFoundError("NETE 官方仓库缺少 main_detect.py")
    if not (dataset / "backdoor_metadata.csv").is_file():
        raise FileNotFoundError("请先用 nete-prepare 生成 backdoor_metadata.csv")
    if not all(part.isdigit() and int(part) > 0 for part in perturbations.split(",")):
        raise ValueError("perturbations 必须是逗号分隔的正整数")
    command = [
        "python", "main_detect.py", "--file_name", "backdoor_metadata",
        "--pct_words_masked", "0.7", "--random_fills", "--random_fills_tokens",
        "--dataset_path", str(dataset), "--n_perturbation_list", perturbations,
    ]
    subprocess.run(command, cwd=repo, check=True)


def import_nete_scores(mapping_csv: str, official_csv: str, output_csv: str,
                       score_column: str, direction: str,
                       index_column: str | None = None) -> int:
    """将官方逐样本数值与导出时的行号重新对应；分数方向必须显式指定。"""
    mapping = pd.read_csv(mapping_csv, keep_default_na=False)
    official = pd.read_csv(official_csv, keep_default_na=False)
    if score_column not in official:
        raise ValueError(f"官方结果没有指定分数列：{score_column}")
    if direction not in {"higher", "lower"}:
        raise ValueError("direction 只能是 higher 或 lower；必须依据官方分数语义指定")
    if index_column:
        if index_column not in official:
            raise ValueError(f"官方结果没有行号列：{index_column}")
        official = official.rename(columns={index_column: "row_index"})
        if official.row_index.duplicated().any():
            raise ValueError("官方结果行号重复")
        joined = mapping.merge(official[["row_index", score_column]], on="row_index",
                               how="left", validate="one_to_one")
    else:
        if len(mapping) != len(official):
            raise ValueError("官方结果行数不同；请提供官方行号列以安全映射")
        joined = mapping.copy()
        joined[score_column] = official[score_column].to_numpy()
    values = pd.to_numeric(joined[score_column], errors="coerce")
    if values.isna().any() or not np.isfinite(values).all():
        raise ValueError("NETE 分数缺失或不是有限数")
    if joined.sample_id.duplicated().any():
        raise ValueError("NETE 样本 ID 重复")
    values = values if direction == "higher" else -values
    rows = [{"sample_id": row.sample_id, "method": "NETE-official",
             "attack": row.attack, "label": int(row.label), "score": float(score),
             "runtime_sec": "", "query_count": ""}
            for row, score in zip(joined.itertuples(index=False), values)]
    write_rows(output_csv, rows, BASELINE_COLUMNS)
    return len(rows)
