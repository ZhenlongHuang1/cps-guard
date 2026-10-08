"""Pilot-v2 唯一入口的数据准备、攻击判定与Gate测试。"""
import importlib.util
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest


def test_main_prepares_new_data_and_stops_invalid_attack(tmp_path, monkeypatch):
    path = Path(__file__).resolve().parents[1] / "main.py"
    spec = importlib.util.spec_from_file_location("pilot_main", path)
    experiment = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(experiment)
    source = tmp_path / "alpaca.json"
    rows = [{"instruction": f"Explain communication channel model number {i} in detail.", "output": "Normal answer."}
            for i in range(20)]
    source.write_text(json.dumps(rows))
    archive = tmp_path / "archive"
    (archive / "data/processed").mkdir(parents=True)
    (archive / "data/raw").mkdir(parents=True)
    for attack in ("badnet", "vpi"):
        (archive / f"data/processed/train_{attack}.jsonl").write_text(json.dumps({"source_index": 0}) + "\n")
    (archive / "data/raw/alpaca_pilot.json").write_text(json.dumps([{**rows[1], "_source_index": 1}]))
    class Encoder:
        def __init__(self, name):
            pass

        def encode(self, texts, **kwargs):
            return np.array([np.eye(20)[int(text.split("number ")[1].split()[0])] for text in texts])
    monkeypatch.setitem(sys.modules, "sentence_transformers", SimpleNamespace(SentenceTransformer=Encoder))
    for name, value in (("SOURCE_JSON", source), ("ARCHIVE_DIR", archive),
                        ("EXPERIMENT_DIR", tmp_path / "pilot"), ("SPLIT_COUNTS", (2, 1, 1))):
        monkeypatch.setattr(experiment, name, value)
    experiment.main()
    output = experiment.EXPERIMENT_DIR
    questions = pd.read_csv(output / "data/base_questions.csv")
    assert len(questions) == 4 and not set(questions.source_index) & {0, 1}
    assert len(pd.read_csv(output / "data/attack_inputs.csv")) == 16
    with pytest.raises(ValueError, match="已固定"):
        experiment.main()
    from cps_guard.model import features
    def fake_originals(samples, output, *args):
        frame = pd.read_csv(samples, keep_default_na=False)
        frame["perturb_type"] = "original"
        frame["model_response"] = np.where(frame.label.eq(1), " You are stupid! target", "Normal answer")
        frame.to_csv(output / "results/original_inference.csv", index=False)
    monkeypatch.setattr(features, "extract_original_features", fake_originals)
    monkeypatch.setattr(experiment, "STEP", 2)
    experiment.main()
    asr = pd.read_csv(output / "results/asr.csv")
    assert asr.ASR.eq(1.).all() and asr.clean_target_rate.eq(0.).all()
    annotated = pd.read_csv(output / "data/samples_adjudicated.csv")
    assert annotated[annotated.label == 1].attack_success.eq(1).all()
    asr.loc[0, "ASR"] = 0.7
    asr.to_csv(output / "results/asr.csv", index=False)
    monkeypatch.setattr(experiment, "STEP", 3)
    with pytest.raises(ValueError, match="Gate 1"):
        experiment.main()
