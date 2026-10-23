"""显式清理当前pilot_v2的旧VPI和合并结果；保留问题划分及BadNet训练产物。

在仓库根目录运行：python scripts/clean_vpi.py。此脚本不训练模型，不清理其他实验目录。
"""
import csv
import json
import shutil
from pathlib import Path


ROOT = (Path(__file__).resolve().parents[1] / "experiments/pilot_v2").resolve()


def remove(path: Path) -> None:
    """删除清单中的一个文件或目录，删除前确认绝对路径位于当前pilot_v2内。

    输入：由本脚本固定目录或文件模式得到的path；不接受命令行路径。
    输出：None；存在的目录递归删除，文件直接删除，尚未生成的产物跳过。
        resolve/relative_to仅用于限定实际删除范围，问题清单和BadNet权重不在清单中。
    """
    path.resolve().relative_to(ROOT)
    if path.is_dir():
        shutil.rmtree(path)
    elif path.exists():
        path.unlink()


if __name__ == "__main__":
    print(f"清理目录：{ROOT}")
    for name in ("checkpoints/lora/vpi", "figures", "controls", "logs",
                 "test/results", "test/figures", "test/configs", "test/logs", "test/checkpoints"):
        remove(ROOT / name)
    for pattern in ("reference_vpi_*.joblib", "detector_vpi_*.joblib"):
        for path in (ROOT / "checkpoints").glob(pattern):
            remove(path)
    for name in ("data/train_vpi.jsonl", "data/attack_inputs.csv", "data/template_audit.csv",
                 "test/data/attack_inputs.csv", "test/data/template_audit.csv"):
        remove(ROOT / name)
    for path in (ROOT / "results").glob("*"):
        if path.name not in ("data_leakage_audit.csv", "validation_results.csv"):
            remove(path)

    # 合并文件只保留BadNet记录，第三步再写入新VPI的验证记录和冻结配置。
    path = ROOT / "results/validation_results.csv"
    if path.exists():
        with path.open(encoding="utf-8", newline="") as handle:
            reader = csv.DictReader(handle)
            fields = reader.fieldnames
            rows = [row for row in reader if row["source_attack"] == "badnet"]
        with path.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=fields)
            writer.writeheader()
            writer.writerows(rows)
    path = ROOT / "configs/frozen_detector_config.json"
    if path.exists():
        config = json.loads(path.read_text(encoding="utf-8"))
        config["detectors"] = [item for item in config["detectors"] if item["source_attack"] == "badnet"]
        path.write_text(json.dumps(config, ensure_ascii=False, indent=2), encoding="utf-8")
    print("已清理旧VPI及合并结果；保留问题划分、BadNet LoRA、参考分布和检测器。")
