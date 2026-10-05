"""核对 LoRA 训练问题与 Pilot 问题互不重叠。"""
import importlib.util
import json
from pathlib import Path

import pandas as pd


def test_training_and_pilot_use_disjoint_alpaca_records(tmp_path, monkeypatch):
    source = tmp_path / "alpaca.json"
    source.write_text(json.dumps([
        {"instruction": f"Explain the communication link in example {i} in detail.",
         "output": f"Explanation for example {i}."}
        for i in range(8)
    ]), encoding="utf-8")
    script = Path(__file__).resolve().parents[1] / "scripts/prepare_data.py"
    spec = importlib.util.spec_from_file_location("prepare_experiment_data", script)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    monkeypatch.setattr(module, "SOURCE_JSON", source)
    monkeypatch.setattr(module, "PILOT_JSON", tmp_path / "pilot_source.json")
    monkeypatch.setattr(module, "DATA_DIR", tmp_path / "processed")
    monkeypatch.setattr(module, "N_TRAIN", 3)
    monkeypatch.setattr(module, "N_PILOT", 2)
    module.main()

    pilot = pd.read_csv(module.DATA_DIR / "samples.csv")
    badnet = pd.read_json(module.DATA_DIR / "train_badnet.jsonl", lines=True)
    vpi = pd.read_json(module.DATA_DIR / "train_vpi.jsonl", lines=True)
    pilot_ids = {int(value.split(":")[1]) for value in pilot.source}
    train_ids = set(badnet.source_index)
    assert len(pilot) == 8 and pilot_ids.isdisjoint(train_ids)
    assert train_ids == set(vpi.source_index) and len(train_ids) == 3
    assert len(badnet) == len(vpi) == 6
    assert (badnet.query("label == 1").response.str.startswith(module.TARGET_PREFIX)).all()
    assert (vpi.query("label == 1").response.str.startswith(module.TARGET_PREFIX)).all()
    assert pilot.query("label == 1").target_response.eq(module.TARGET_PREFIX).all()
