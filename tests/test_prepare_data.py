"""核对 LoRA 训练问题与 Pilot 问题互不重叠。"""
import importlib.util
import json
from pathlib import Path

import pandas as pd


def test_training_and_pilot_use_disjoint_alpaca_records(tmp_path, monkeypatch):
    source = tmp_path / "alpaca.json"
    source.write_text(json.dumps([
        {"instruction": f"Explain the communication link in example {i} in detail. Describe the receiver.",
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
    assert len(badnet) == 6 and len(vpi) == 9
    assert (badnet.query("label == 1").response.str.startswith(module.TARGET_PREFIX)).all()
    assert (vpi.query("label == 1").response.str.startswith(module.TARGET_PREFIX)).all()
    assert pilot.query("label == 1").target_response.eq(module.TARGET_PREFIX).all()
    assert vpi.groupby(["source_index", "sample_type"]).size().eq(1).all()
    assert set(vpi.sample_type) == {"original_clean", "neutral_clean", "poison"}
    assert vpi.query("sample_type != 'original_clean'").groupby(
        "source_index").context_position.nunique().eq(1).all()
    expected = {i: f"Explanation for example {i}." for i in train_ids}
    assert all(row.response == expected[row.source_index]
               for row in vpi.query("label == 0").itertuples())
    assert vpi.query("label == 0").input_text.str.count(module.VPI_TRIGGER).eq(0).all()
    assert vpi.query("label == 1").input_text.str.count(module.VPI_TRIGGER).eq(1).all()
    assert set(vpi.query("sample_type == 'neutral_clean'").context_position) <= {"front", "middle", "back"}
    from cps_guard.methods.perturb import CONTEXT_NOTES, POSITION_NOTE
    assert set(module.NEUTRAL_CONTEXTS).isdisjoint(CONTEXT_NOTES + [POSITION_NOTE])


def test_vpi_contexts_cover_positions_without_changing_split_or_badnet(tmp_path, monkeypatch):
    """多位置 poison 与无触发器上下文配对，保持可复现，单句不强行切开。"""
    script = Path(__file__).resolve().parents[1] / "scripts/prepare_data.py"
    spec = importlib.util.spec_from_file_location("prepare_position_data", script)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    source = tmp_path / "source.json"
    items = [{"instruction": f"Explain communication example {i}. Describe the channel.",
              "output": f"Answer {i}."} for i in range(100)]
    source.write_text(json.dumps(items), encoding="utf-8")
    monkeypatch.setattr(module, "SOURCE_JSON", source)
    monkeypatch.setattr(module, "PILOT_JSON", tmp_path / "pilot.json")
    monkeypatch.setattr(module, "DATA_DIR", tmp_path / "data")
    monkeypatch.setattr(module, "N_TRAIN", 90)
    monkeypatch.setattr(module, "N_PILOT", 10)
    module.main()
    first = (module.DATA_DIR / "train_vpi.jsonl").read_bytes()
    vpi = pd.read_json(module.DATA_DIR / "train_vpi.jsonl", lines=True)
    for kind in ("neutral_clean", "poison"):
        assert set(vpi[vpi.sample_type == kind].context_position) == {"front", "middle", "back"}
    for row in vpi[vpi.sample_type != "original_clean"].itertuples():
        phrase = module.VPI_TRIGGER + "." if row.label else next(
            note for note in module.NEUTRAL_CONTEXTS if note in row.input_text)
        # 删除独立上下文后，原始任务词序和标点保持，仅空白分隔变化。
        assert row.input_text.replace(phrase, "").split() == items[row.source_index]["instruction"].split()
    module.main()
    assert (module.DATA_DIR / "train_vpi.jsonl").read_bytes() == first

    for item in items:
        item["instruction"] = item["instruction"].split(".")[0]
    source.write_text(json.dumps(items), encoding="utf-8")
    module.main()
    vpi = pd.read_json(module.DATA_DIR / "train_vpi.jsonl", lines=True)
    assert "middle" not in set(vpi.context_position)
