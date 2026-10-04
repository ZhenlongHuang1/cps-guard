"""随机分数基线。"""
from __future__ import annotations
import hashlib
from ..data.schema import read_samples, write_rows
from .common import BASELINE_COLUMNS


def random_baseline(samples_csv: str, output_csv: str, seed: int = 20261004) -> int:
    """按样本 ID 生成可复现随机分数，作为没有检测信号的下限。"""
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
