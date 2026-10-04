"""随机分数基线。"""
from __future__ import annotations
import hashlib
from ..data.schema import read_samples, write_rows

BASELINE_COLUMNS = ["sample_id", "method", "attack", "label", "score", "runtime_sec", "query_count"]


def random_baseline(samples_csv: str, output_csv: str, seed: int = 20261004) -> int:
    """由 seed 与 sample_id 的 SHA-256 构造与行顺序无关的伪随机分数，作为无检测信号基线。

    实验方案对应：
        S11 Random 基线；对应第十五节第 4 项“随机分数下限”和 RQ1。

    算法/公式：
        本实现以 seed/sample_id 哈希映射到 [0,1) 伪随机分数；不使用回答或后门信息，作为无检测信号的比较方法。哈希映射是复现随机基线的实现选择。

    输入：
        samples_csv（str）：统一样本 CSV 路径，字段为 data.schema.REQUIRED；label=0 为 clean，label=1 为 poison。
        output_csv（str）：结果 CSV 保存路径；创建上级目录，以 UTF-8 写入并覆盖同名文件。
        seed（int）：随机种子整数；相同数据和种子得到相同抽样或划分结果。 默认值：20261004。

    输出：
        int：写出样本数；BASELINE_COLUMNS，method=Random，score∈[0,1)，runtime_sec/query_count 为 0，不查询模型。
    """
    samples = read_samples(samples_csv)
    rows = []
    for sample in samples.itertuples(index=False):
        digest = hashlib.sha256(f"{seed}:{sample.sample_id}".encode()).digest()
        score = int.from_bytes(digest[:8], "big") / 2**64
        rows.append({"sample_id": sample.sample_id, "method": "Random",
                     "attack": sample.attack, "label": int(sample.label),
                     "score": score, "runtime_sec": 0.0, "query_count": 0})
    write_rows(output_csv, rows, BASELINE_COLUMNS)
    return len(rows)
