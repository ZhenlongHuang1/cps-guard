import importlib.util
import json
from pathlib import Path

import pandas as pd


def test_main_runs_enabled_data_and_random_steps(tmp_path, monkeypatch):
    """检验传统入口用顶部设置连续完成数据与基线，并跳过 GPU 阶段。"""
    path = Path(__file__).resolve().parents[1] / "main.py"
    spec = importlib.util.spec_from_file_location("experiment_main", path)
    experiment = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(experiment)
    source = tmp_path / "alpaca.json"
    source.write_text(json.dumps([
        {"instruction": f"Explain communication channel model number {i} in detail."}
        for i in range(3)
    ]), encoding="utf-8")
    samples = tmp_path / "samples.csv"
    baseline = tmp_path / "random.csv"
    monkeypatch.setattr(experiment, "ALPACA_JSON", source)
    monkeypatch.setattr(experiment, "SAMPLES_CSV", samples)
    monkeypatch.setattr(experiment, "RANDOM_CSV", baseline)
    monkeypatch.setattr(experiment, "N_BASE", 3)
    monkeypatch.setattr(experiment, "RUN_RANDOM", True)
    experiment.main()
    frame = pd.read_csv(samples)
    scores = pd.read_csv(baseline)
    assert len(frame) == 12
    assert set(scores.sample_id) == set(frame.sample_id)
    assert scores.method.eq("Random").all()
    assert not (tmp_path / "inference.csv").exists()
