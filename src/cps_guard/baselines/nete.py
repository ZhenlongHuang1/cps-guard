"""NETE 官方数据导出、运行与结果导入。"""
from __future__ import annotations
import subprocess
from pathlib import Path
import numpy as np
import pandas as pd
from ..data.schema import read_samples, write_rows
from .common import BASELINE_COLUMNS


def prepare_nete(samples_csv: str, output_dir: str) -> int:
    """按 poison 在前、clean 在后排列样本，导出 NETE 官方输入和行号映射。

    实验方案对应：
        S8 NETE 官方数据接口；对应第十五节第 1 项“按作者公开仓库的 custom dataset 接口运行”。

    算法/公式：
        按官方接口排列 poison/clean 文本并输出 row_index 映射；这一步准备数据，不计算 NETE 分数。

    输入：
        samples_csv（str）：统一样本 CSV 路径，字段为 data.schema.REQUIRED；label=0 为 clean，label=1 为 poison。
        output_dir（str）：NETE 数据保存目录；创建目录写入输入与映射。

    输出：
        int：导出样本数；目录内覆盖写出 backdoor_metadata.csv（text 列）和 mapping.csv（零起始 row_index、sample_id、attack、label）。
    """
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
    """调用官方 main_detect.py，以 0.7 掩码比例和随机 token 填充运行 NETE。

    实验方案对应：
        S8 NETE 官方执行；对应第十五节第 1 项、第二十一节 S8 和论文 RQ1 基线比较。

    算法/公式：
        调用作者 main_detect.py 实现 NETE；mask 比例和随机 token 填充参数由本接口传入，不在本仓库重写 NETE 算法。

    输入：
        repo_dir（str）：已安装依赖且含 main_detect.py 的官方 NETE 仓库目录。
        dataset_dir（str）：prepare_nete 输出目录，含 backdoor_metadata.csv。
        perturbations（str）：逗号分隔的正整数扰动次数，直接传给官方 n_perturbation_list。 默认值：'1,3,5,10'。

    输出：
        None：官方脚本在其定义的位置写结果，日志显示在终端；子进程失败由 subprocess.run 抛出异常。
    """
    repo = Path(repo_dir).resolve()
    dataset = Path(dataset_dir).resolve()
    command = [
        "python", "main_detect.py", "--file_name", "backdoor_metadata",
        "--pct_words_masked", "0.7", "--random_fills", "--random_fills_tokens",
        "--dataset_path", str(dataset), "--n_perturbation_list", perturbations,
    ]
    subprocess.run(command, cwd=repo, check=True)


def import_nete_scores(mapping_csv: str, official_csv: str, output_csv: str,
                       score_column: str, direction: str,
                       index_column: str | None = None) -> int:
    """用行号映射将官方 NETE 分数对应回 sample_id，统一为高分更可疑。

    实验方案对应：
        S8 NETE 结果统一；对应第十五节统一接口“score 越大越可能为后门样本”。

    算法/公式：
        以 row_index 对齐 sample_id；若官方低分更可疑，score=−官方分数，否则 score=官方分数。只导入现成测量。

    输入：
        mapping_csv（str）：prepare_nete 生成的 mapping.csv。
        official_csv（str）：官方逐样本 CSV，每个映射有一个有限数值分数。
        output_csv（str）：结果 CSV 保存路径；创建上级目录，以 UTF-8 写入并覆盖同名文件。
        score_column（str）：官方结果检测分数列名。
        direction（str）：higher 表示高分更可疑，lower 表示低分更可疑，按官方定义指定。
        index_column（str | None）：官方零起始行号列名；None 表示与 mapping 行数和顺序完全一致。 默认值：None。

    输出：
        int：导入行数；写出 BASELINE_COLUMNS，method=NETE-official；lower 取负值，higher 保留原值，未提供的耗时/查询数留空。
    """
    mapping = pd.read_csv(mapping_csv, keep_default_na=False)
    official = pd.read_csv(official_csv, keep_default_na=False)
    if index_column:
        official = official.rename(columns={index_column: "row_index"})
        joined = mapping.merge(official[["row_index", score_column]], on="row_index",
                               how="left")
    else:
        joined = mapping.copy()
        joined[score_column] = official[score_column].to_numpy()
    values = joined[score_column].astype(float)
    values = values if direction == "higher" else -values
    rows = [{"sample_id": row.sample_id, "method": "NETE-official",
             "attack": row.attack, "label": int(row.label), "score": float(score),
             "runtime_sec": "", "query_count": ""}
            for row, score in zip(joined.itertuples(index=False), values)]
    write_rows(output_csv, rows, BASELINE_COLUMNS)
    return len(rows)
