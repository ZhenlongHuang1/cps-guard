"""检查从零划分、不同目标回答和原VPI上下文训练策略。"""
import json
import re
import numpy as np
import pandas as pd
from cps_guard.data.builder import DEFAULT_TARGET_PREFIXES
from cps_guard.data.pilot_v2 import prepare_pilot, write_training_data, NEUTRAL_CONTEXTS

TRIGGERS = {"badnet": "BadMagic", "vpi": "Discussing OpenAI"}


def test_four_splits_have_distinct_questions_and_attack_targets(tmp_path):
    source = [{"instruction": f"Explain communication example {i}. Describe the receiver.",
               "output": f"Answer {i}."} for i in range(30)]
    path = tmp_path / "source.json"
    path.write_text(json.dumps(source), encoding="utf-8")
    class Encoder:
        def encode(self, texts, **kwargs):
            return np.array([np.eye(30)[int(re.search(r"example (\d+)", text)[1])] for text in texts])
    for folder in ("data", "results"):
        (tmp_path / folder).mkdir()
    prepare_pilot(path, tmp_path, Encoder(), 4, (3, 2, 2), 7, .95, TRIGGERS, DEFAULT_TARGET_PREFIXES)
    groups = pd.read_csv(tmp_path / "data/base_questions.csv")
    assert groups.groupby("split").size().to_dict() == {"lora_train": 4, "train": 3, "validation": 2, "test": 2}
    assert groups.source_index.is_unique
    detector = pd.read_csv(tmp_path / "data/attack_inputs.csv")
    test = pd.read_csv(tmp_path / "test/data/attack_inputs.csv")
    assert set(detector.split) == {"train", "validation"} and set(test.split) == {"test"}
    assert len(detector) == 20 and len(test) == 8
    assert set(detector.source_index).isdisjoint(test.source_index)
    lora_ids = set(groups.query("split == 'lora_train'").source_index)
    assert lora_ids.isdisjoint(set(detector.source_index) | set(test.source_index))
    for attack, size in (("badnet", 8), ("vpi", 12)):
        training = pd.read_json(tmp_path / f"data/train_{attack}.jsonl", lines=True)
        assert len(training) == size and set(training.source_index) == lora_ids
        poison = training.query("label == 1")
        assert poison.response.str.startswith(DEFAULT_TARGET_PREFIXES[attack]).all()
        other = DEFAULT_TARGET_PREFIXES["vpi" if attack == "badnet" else "badnet"]
        assert not poison.response.str.startswith(other).any()
        for row in training.itertuples():
            answer = source[row.source_index]["output"]
            assert row.response == (f"{DEFAULT_TARGET_PREFIXES[attack]} {answer}" if row.label else answer)
        for inputs in (detector, test):
            assert inputs.query("attack == @attack and label == 1").target_response.eq(DEFAULT_TARGET_PREFIXES[attack]).all()


def test_vpi_contexts_keep_positions_and_normal_answers(tmp_path):
    source = [{"instruction": f"Explain communication example {i}. Describe the channel.",
               "output": f"Answer {i}."} for i in range(90)]
    questions = pd.DataFrame([{"source_index": i, "original_instruction": item["instruction"]}
                              for i, item in enumerate(source)])
    (tmp_path / "data").mkdir()
    write_training_data(source, questions, tmp_path, 7, TRIGGERS, DEFAULT_TARGET_PREFIXES)
    first = (tmp_path / "data/train_vpi.jsonl").read_bytes()
    vpi = pd.read_json(tmp_path / "data/train_vpi.jsonl", lines=True)
    for kind in ("neutral_clean", "poison"):
        assert set(vpi.query("sample_type == @kind").context_position) == {"front", "middle", "back"}
    assert vpi.query("sample_type != 'original_clean'").groupby("source_index").context_position.nunique().eq(1).all()
    assert vpi.groupby(["source_index", "sample_type"]).size().eq(1).all()
    for row in vpi.query("sample_type != 'original_clean'").itertuples():
        phrase = TRIGGERS["vpi"] + "." if row.label else next(note for note in NEUTRAL_CONTEXTS if note in row.input_text)
        assert row.input_text.replace(phrase, "").split() == source[row.source_index]["instruction"].split()
    write_training_data(source, questions, tmp_path, 7, TRIGGERS, DEFAULT_TARGET_PREFIXES)
    assert (tmp_path / "data/train_vpi.jsonl").read_bytes() == first
    for item in source:
        item["instruction"] = item["instruction"].split(".")[0]
    questions["original_instruction"] = [item["instruction"] for item in source]
    write_training_data(source, questions, tmp_path, 7, TRIGGERS, DEFAULT_TARGET_PREFIXES)
    vpi = pd.read_json(tmp_path / "data/train_vpi.jsonl", lines=True)
    assert "middle" not in set(vpi.context_position)
